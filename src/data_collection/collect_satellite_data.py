import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def fetch_satellite_data(start_date, end_date, region_bounds):
    """
    Placeholder for fetching real satellite data from NASA Earthdata or Google Earth Engine.
    In a real scenario, this would use earthengine-api or similar.
    """
    print("Fetching satellite data (Mocking for demonstration)...")
    
    # Generate mock data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    
    # Define both URBAN areas (for training with ground truth) and RURAL areas (for prediction)
    all_locations = [
        # URBAN areas (match ground stations for training)
        {'name': 'Manali (Urban)', 'lat': 13.23, 'lon': 80.32, 'type': 'Urban'},
        {'name': 'Adyar (Urban)', 'lat': 13.10, 'lon': 80.29, 'type': 'Urban'},
        {'name': 'Anna Nagar (Urban)', 'lat': 13.06, 'lon': 80.25, 'type': 'Urban'},
        {'name': 'Velachery (Urban)', 'lat': 13.09, 'lon': 80.20, 'type': 'Urban'},
        # RURAL areas (for prediction via transfer learning)
        {'name': 'Poonamallee (Rural)', 'lat': 13.05, 'lon': 80.09, 'type': 'Rural'},
        {'name': 'Thiruvallur (Rural)', 'lat': 13.13, 'lon': 79.91, 'type': 'Rural'},
        {'name': 'Sriperumbudur (Rural)', 'lat': 12.97, 'lon': 79.94, 'type': 'Rural'},
        {'name': 'Kanchipuram (Rural)', 'lat': 12.84, 'lon': 79.70, 'type': 'Rural'}
    ]
    
    for date in dates:
        for loc in all_locations:
            # Simulate some spatial variation around each location
            for offset_lat in [-0.005, 0, 0.005]:
                for offset_lon in [-0.005, 0, 0.005]:
                    lat = loc['lat'] + offset_lat
                    lon = loc['lon'] + offset_lon
                    
                    # Generate satellite values
                    # Urban areas have slightly higher pollution
                    base_aod = 0.4 if loc['type'] == 'Urban' else 0.25
                    base_no2 = 30 if loc['type'] == 'Urban' else 18
                    base_co = 0.8 if loc['type'] == 'Urban' else 0.5
                    
                    aod_value = np.random.uniform(base_aod * 0.7, base_aod * 1.3)
                    no2_value = np.random.uniform(base_no2 * 0.7, base_no2 * 1.3)
                    co_value = np.random.uniform(base_co * 0.7, base_co * 1.3)
                    
                    data.append({
                        'date': date,
                        'latitude': lat,
                        'longitude': lon,
                        'location': loc['name'],
                        'location_type': loc['type'],
                        'aod_550': max(0, aod_value),  # MODIS/VIIRS AOD
                        'trop_no2': max(0, no2_value),  # Sentinel-5P NO2
                        'trop_co': max(0, co_value)  # Sentinel-5P CO
                    })
    
    print(f"Generated {len(data)} satellite records")
    return pd.DataFrame(data)

def save_data(df, filename):
    if not df.empty:
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Satellite data saved to {filepath}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    # Define bounds (min_lat, max_lat, min_lon, max_lon)
    bounds = (config.RURAL_LAT_RANGE[0], config.RURAL_LAT_RANGE[1], 
              config.RURAL_LON_RANGE[0], config.RURAL_LON_RANGE[1])
    
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    df = fetch_satellite_data(start_date, end_date, bounds)
    save_data(df, "satellite_data_rural.csv")
