import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np

def plot_aqi_trend(df, region_name):
    """
    Plots AQI trend over time for a specific region.
    """
    if 'date' not in df.columns:
        return None
        
    target_col = 'aqi' if 'aqi' in df.columns else 'value'
    if target_col not in df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Aggregate by date if needed
    df_plot = df.groupby('date')[target_col].mean().reset_index()
    
    ax.plot(df_plot['date'], df_plot[target_col])
    ax.set_title(f"{target_col.upper()} Trend for {region_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col.upper())
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_pollutant_distribution(df):
    """
    Plots distribution of different pollutants with clear labeling.
    """
    if df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # The main pollutant in our data is PM2.5 (stored in 'value' column)
    if 'value' in df.columns:
        # Create comparison by location type if available
        if 'location_type' in df.columns:
            urban_data = df[df['location_type'] == 'Urban']['value'].dropna()
            rural_data = df[df['location_type'] == 'Rural']['value'].dropna()
            
            data_to_plot = []
            labels = []
            
            if len(urban_data) > 0:
                data_to_plot.append(urban_data)
                labels.append(f'Urban PM2.5\n({len(urban_data)} samples)')
            
            if len(rural_data) > 0:
                data_to_plot.append(rural_data)
                labels.append(f'Rural PM2.5\n({len(rural_data)} samples)')
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # Color code the boxes
                colors = ['lightblue', 'lightgreen']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                
                ax.set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=12)
                ax.set_title("PM2.5 Distribution: Urban vs Rural", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add mean lines
                for i, data in enumerate(data_to_plot, 1):
                    mean_val = data.mean()
                    ax.hlines(mean_val, i-0.4, i+0.4, colors='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}' if i == 1 else '')
                
                ax.legend()
        else:
            # Single distribution
            data_clean = df['value'].dropna()
            if len(data_clean) > 0:
                ax.boxplot([data_clean], labels=['PM2.5'])
                ax.set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=12)
                ax.set_title("PM2.5 Distribution", fontsize=14, fontweight='bold')
    else:
        return None
    
    plt.tight_layout()
    return fig

def create_map(df):
    """
    Creates a Folium map with AQI markers/heatmap.
    """
    # Filter out rows with NaN coordinates
    df_valid = df.dropna(subset=['latitude', 'longitude'])
    
    if df_valid.empty:
        print("Warning: No valid coordinates for map")
        # Return a default map centered on Chennai
        return folium.Map(location=[13.0827, 80.2707], zoom_start=10)
    
    # Check if coordinates look like real lat/lon (within reasonable bounds for Chennai region)
    lat_min, lat_max = df_valid['latitude'].min(), df_valid['latitude'].max()
    lon_min, lon_max = df_valid['longitude'].min(), df_valid['longitude'].max()
    
    # Valid coordinate ranges for Chennai + surrounding rural areas: lat 12.5-14, lon 79.5-81
    coords_look_valid = (12.5 <= lat_min <= 14 and 12.5 <= lat_max <= 14 and
                        79.5 <= lon_min <= 81 and 79.5 <= lon_max <= 81)
    
    if not coords_look_valid:
        # Coordinates appear normalized or invalid, use Chennai center
        center_lat = 13.0827
        center_lon = 80.2707
        print(f"Warning: Coordinates out of expected range (lat: {lat_min:.2f}-{lat_max:.2f}, lon: {lon_min:.2f}-{lon_max:.2f})")
    else:
        # Use actual coordinates
        center_lat = df_valid['latitude'].mean()
        center_lon = df_valid['longitude'].mean()  
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Only add markers if we have valid coordinates
    if coords_look_valid:
        # Add HeatMap if we have AQI data
        if 'aqi' in df_valid.columns:
            heat_data = [[row['latitude'], row['longitude'], row['aqi']] 
                        for index, row in df_valid.iterrows() 
                        if not pd.isna(row['aqi'])]
            if heat_data:
                HeatMap(heat_data, radius=15, blur=20, max_zoom=13).add_to(m)
        
        # Add markers for specific locations with location type color coding
        for index, row in df_valid.head(100).iterrows():  # Limit to 100 markers
            aqi_val = row.get('aqi', 50)
            location_type = row.get('location_type', 'Unknown')
            location_name = row.get('location', 'Unknown Location')
            
            # Color based on both AQI level and location type
            if aqi_val > 200:
                color = 'darkred'
            elif aqi_val > 100:
                color = 'orange'
            else:
                color = 'blue' if location_type == 'Urban' else 'green'
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                popup=f"<b>{location_name}</b><br>Type: {location_type}<br>AQI: {aqi_val:.1f}" if not pd.isna(aqi_val) else f"<b>{location_name}</b><br>Type: {location_type}",
                tooltip=f"{location_name}: AQI {aqi_val:.1f}",
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
    
    return m

def plot_model_comparison(results):
    """
    Plots model comparison metrics.
    results: dict like {'ModelName': {'MSE': val, 'R2': val}}
    """
    if not results:
        return None
        
    df_res = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.barplot(data=df_res, x='Model', y='MSE', ax=ax[0])
    ax[0].set_title("Model MSE Comparison")
    
    sns.barplot(data=df_res, x='Model', y='R2', ax=ax[1])
    ax[1].set_title("Model R2 Comparison")
    
    return fig
