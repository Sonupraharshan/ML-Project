import os
import sys
from src.data_collection import collect_ground_data, collect_satellite_data, collect_weather_data
from src.preprocessing import spatial_matching, temporal_alignment, feature_engineering
from src.models import urban_model, transfer_learning
from config import config
import pandas as pd
import joblib

def main():
    print("Starting Granular Rural Air Quality Estimation Pipeline...")
    
    # 1. Data Collection
    print("\n--- Step 1: Data Collection ---")
    
    # Check if data already exists
    ground_data_file = os.path.join(config.RAW_DATA_DIR, "ground_aqi_chennai.csv")
    sat_data_file = os.path.join(config.RAW_DATA_DIR, "satellite_data_rural.csv")
    weather_data_file = os.path.join(config.RAW_DATA_DIR, "weather_data_chennai.csv")
    
    # Ground Data
    if os.path.exists(ground_data_file):
        print(f"✓ Ground data already exists: {ground_data_file}")
    else:
        print("Collecting ground data...")
        ground_df = collect_ground_data.fetch_openaq_data()
        collect_ground_data.save_data(ground_df, "ground_aqi_chennai.csv")
    
    # Satellite Data
    if os.path.exists(sat_data_file):
        print(f"✓ Satellite data already exists: {sat_data_file}")
    else:
        print("Collecting satellite data...")
        bounds = (config.RURAL_LAT_RANGE[0], config.RURAL_LAT_RANGE[1], 
                  config.RURAL_LON_RANGE[0], config.RURAL_LON_RANGE[1])
        sat_df = collect_satellite_data.fetch_satellite_data("2023-01-01", "2023-01-30", bounds)
        collect_satellite_data.save_data(sat_df, "satellite_data_rural.csv")
    
    # Weather Data
    if os.path.exists(weather_data_file):
        print(f"✓ Weather data already exists: {weather_data_file}")
    else:
        print("Collecting weather data...")
        weather_df = collect_weather_data.fetch_weather_data(config.CHENNAI_LAT, config.CHENNAI_LON, "2023-01-01", "2023-01-30")
        collect_weather_data.save_data(weather_df, "weather_data_chennai.csv")
    
    # 2. Preprocessing
    print("\n--- Step 2: Preprocessing ---")
    # Load raw data
    ground_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "ground_aqi_chennai.csv"))
    sat_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "satellite_data_rural.csv"))
    
    print(f"Ground data: {len(ground_df)} rows")
    print(f"Satellite data: {len(sat_df)} rows")
    
    # Spatial Matching
    matched_df = spatial_matching.match_satellite_to_ground(ground_df, sat_df)
    print(f"After spatial matching: {len(matched_df)} rows")
    
    if matched_df.empty:
        print("Matching failed or empty data. Exiting.")
        return
        
    # Temporal Alignment
    aligned_df = temporal_alignment.align_temporal_data(matched_df)
    print(f"After temporal alignment: {len(aligned_df)} rows")
    
    # Feature Engineering
    final_df = feature_engineering.create_features(aligned_df)
    print(f"After feature creation: {len(final_df)} rows")
    
    final_df = feature_engineering.normalize_features(final_df)
    print(f"After normalization: {len(final_df)} rows")
    
    final_path = os.path.join(config.PROCESSED_DATA_DIR, "final_training_data.csv")
    final_df.to_csv(final_path, index=False)
    print(f"Processed data saved to {final_path}")
    
    if final_df.empty:
        print("ERROR: Final dataset is empty! Cannot train models.")
        return
    
    # 3. Model Training
    print("\n--- Step 3: Model Training ---")
    urban_model.train_urban_models(final_path)
    
    # 4. Transfer Learning (Demo)
    print("\n--- Step 4: Transfer Learning (Demo) ---")
    # In a real scenario, we would split final_df into source (urban) and target (rural)
    # Here we just mock the input for demonstration
    import numpy as np
    X_s = np.random.rand(100, 10)
    y_s = np.random.rand(100)
    X_t = np.random.rand(100, 10)
    
    transfer_learning.train_dann((X_s, y_s), X_t)
    
    print("\n--- Pipeline Complete ---")
    print("To run the dashboard: streamlit run src/dashboard/app.py")

if __name__ == "__main__":
    main()
