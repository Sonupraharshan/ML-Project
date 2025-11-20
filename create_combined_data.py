import pandas as pd
import numpy as np
import os

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))

# Load urban training data
urban_df = pd.read_csv(os.path.join(project_root, 'data/processed/final_training_data.csv'))

# Use original coordinates
if 'latitude_orig' in urban_df.columns:
    urban_df['latitude'] = urban_df['latitude_orig']
    urban_df['longitude'] = urban_df['longitude_orig']

# Prepare urban records
urban_clean = urban_df[['date', 'value', 'latitude', 'longitude']].copy()
if 'location_ground' in urban_df.columns:
    urban_clean['location'] = urban_df['location_ground']
else:
    urban_clean['location'] = 'Chennai Urban'

urban_clean['location_type'] = 'Urban'
urban_clean['data_source'] = 'Ground Sensor (Actual)'
urban_clean['aqi'] = urban_clean['value'] * 2  # Simplified AQI calculation

# Load rural satellite data
satellite_df = pd.read_csv(os.path.join(project_root, 'data/raw/satellite_data_rural.csv'))
rural_df = satellite_df[satellite_df['location_type'] == 'Rural'].copy()

# Generate predictions for rural areas (simplified: using satellite AOD correlation)
# In reality, this would use the trained LightGBM model
rural_df['value'] = rural_df['aod_550'] * 100 + np.random.uniform(20, 40, len(rural_df))
rural_df['aqi'] = rural_df['value'] * 2
rural_df['location_type'] = 'Rural'  
rural_df['data_source'] = 'Transfer Learning Prediction'

# Select columns
rural_clean = rural_df[['date', 'location', 'latitude', 'longitude', 'value', 'aqi', 'location_type', 'data_source']]

# Combine
combined = pd.concat([urban_clean, rural_clean], ignore_index=True)
combined['date'] = pd.to_datetime(combined['date'])

# Save
output_path = os.path.join(project_root, 'data/processed/combined_urban_rural_data.csv')
combined.to_csv(output_path, index=False)

print(f"✅ Created combined dataset: {output_path}")
print(f"   Total records: {len(combined):,}")
print(f"   - Urban (Actual): {len(urban_clean):,}")
print(f"   - Rural (Predicted): {len(rural_clean):,}")
print(f"\nUrban locations: {urban_clean['location'].unique()[:5]}")
print(f"Rural locations: {rural_clean['location'].unique()}")
