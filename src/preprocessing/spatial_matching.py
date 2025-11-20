import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def match_satellite_to_ground(ground_df, satellite_df):
    """
    Matches satellite pixels to ground station coordinates using nearest neighbor.
    """
    print("Matching satellite data to ground coordinates...")
    
    if ground_df.empty or satellite_df.empty:
        print("Empty dataframes provided for matching.")
        return pd.DataFrame()

    # Ensure unique satellite locations for KDTree
    sat_coords = satellite_df[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    ground_coords = ground_df[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    
    # Build KDTree
    tree = cKDTree(sat_coords.values)
    
    # Query nearest neighbors
    distances, indices = tree.query(ground_coords.values, k=1)
    
    # Map ground coordinates to nearest satellite coordinates
    ground_coords['nearest_sat_lat'] = sat_coords.iloc[indices]['latitude'].values
    ground_coords['nearest_sat_lon'] = sat_coords.iloc[indices]['longitude'].values
    
    # Merge back to original dataframes
    # This is a simplified merge strategy. In reality, you'd merge on date AND location.
    
    # Create a mapping dictionary
    coord_map = ground_coords.set_index(['latitude', 'longitude'])[['nearest_sat_lat', 'nearest_sat_lon']].to_dict('index')
    
    def get_nearest(row):
        key = (row['latitude'], row['longitude'])
        if key in coord_map:
            return coord_map[key]
        return None, None

    # Apply mapping (this part needs to be robust for large datasets)
    # For efficiency, we merge on date and nearest coordinates
    
    # First, add nearest coords to ground_df
    ground_df['nearest_sat_lat'] = ground_df.apply(lambda row: coord_map.get((row['latitude'], row['longitude']), {}).get('nearest_sat_lat'), axis=1)
    ground_df['nearest_sat_lon'] = ground_df.apply(lambda row: coord_map.get((row['latitude'], row['longitude']), {}).get('nearest_sat_lon'), axis=1)
    
    # Merge
    merged_df = pd.merge(
        ground_df,
        satellite_df,
        left_on=['date', 'nearest_sat_lat', 'nearest_sat_lon'],
        right_on=['date', 'latitude', 'longitude'],
        suffixes=('_ground', '_sat'),
        how='left'
    )
    
    # Rename ground coordinates back to original if they were suffixed
    if 'latitude_ground' in merged_df.columns:
        merged_df = merged_df.rename(columns={'latitude_ground': 'latitude', 'longitude_ground': 'longitude'})
        
    # Drop redundant satellite coordinates if needed, or keep them for reference
    # We can drop the right_on keys if they became latitude_sat/longitude_sat
    # But since we used them as keys, pandas might handle them differently.
    # Let's just ensure 'latitude' and 'longitude' exist.
    
    return merged_df

if __name__ == "__main__":
    # Example usage (requires existing data files)
    try:
        ground_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "ground_aqi_chennai.csv"))
        sat_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "satellite_data_rural.csv")) # Using rural sat data for demo
        
        matched_df = match_satellite_to_ground(ground_df, sat_df)
        if not matched_df.empty:
            matched_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "matched_data.csv"), index=False)
            print("Matched data saved.")
    except FileNotFoundError:
        print("Data files not found. Run collection scripts first.")
