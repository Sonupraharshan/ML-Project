"""
Generate rural AQI predictions using trained urban model and satellite data.
This demonstrates transfer learning: urban model → rural prediction.
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import config
from src.models import aqi_calculator

def generate_rural_predictions():
    """
    Apply trained urban model to rural satellite data to generate AQI predictions.
    """
    print("Generating rural AQI predictions...")
    
    # Load trained model
    model_path = os.path.join(config.PROCESSED_DATA_DIR, 'best_urban_model.pkl')
    if not os.path.exists(model_path):
        print("Error: Trained model not found. Run main.py first.")
        return None
    
    model = joblib.load(model_path)
    print(f"Loaded trained model from {model_path}")
    
    # Load rural satellite data
    sat_data_path = os.path.join(config.RAW_DATA_DIR, "satellite_data_rural.csv")
    sat_df = pd.read_csv(sat_data_path)
    
    # Filter for rural locations only
    rural_df = sat_df[sat_df['location_type'] == 'Rural'].copy()
    print(f"Loaded {len(rural_df)} rural satellite records")
    
    # Add timestamp features for consistency with training data
    rural_df['date'] = pd.to_datetime(rural_df['date'])
    rural_df['month'] = rural_df['date'].dt.month
    rural_df['day_of_week'] = rural_df['date'].dt.dayofweek
    rural_df['hour'] = 12  # Default to noon
    
    # Create lag and rolling features (using zeros as placeholders)
    for col in ['aod_550', 'trop_no2', 'trop_co']:
        if col in rural_df.columns:
            rural_df[f'{col}_lag1'] = 0
            rural_df[f'{col}_lag3'] = 0
            rural_df[f'{col}_rolling_mean_3'] = rural_df[col]
            rural_df[f'{col}_rolling_mean_7'] = rural_df[col]
    
    # Select features matching training data
    # Features used in training: month, day_of_week, hour, and various rolling/lag features
    feature_cols = ['month', 'day_of_week', 'hour',
                    'aod_550_rolling_mean_3', 'aod_550_rolling_mean_7',
                    'trop_no2_rolling_mean_3', 'trop_no2_rolling_mean_7']
    
    # Filter to available features
    available_features = [f for f in feature_cols if f in rural_df.columns]
    
    if not available_features:
        print("Warning: No matching features found")
        return None
    
    print(f"Using {len(available_features)} features for prediction")
    
    # Make predictions
    X_rural = rural_df[available_features].fillna(0)
    rural_df['predicted_pm25'] = model.predict(X_rural)
    
    # Calculate AQI from predicted PM2.5
    rural_df['predicted_aqi'] = rural_df['predicted_pm25'].apply(
        lambda x: aqi_calculator.calculate_aqi(x, 'pm25')
    )
    rural_df['aqi_category'] = rural_df['predicted_aqi'].apply(
        lambda x: aqi_calculator.get_aqi_category(x) if pd.notna(x) else 'Unknown'
    )
    
    # Prepare final output
    rural_predictions = rural_df[[
        'date', 'location', 'location_type', 
        'latitude', 'longitude',
        'predicted_pm25', 'predicted_aqi', 'aqi_category'
    ]].copy()
    
    # Rename for consistency
    rural_predictions = rural_predictions.rename(columns={
        'predicted_pm25': 'value',
        'predicted_aqi': 'aqi'
    })
    
    # Add metadata
    rural_predictions['data_source'] = 'Transfer Learning Prediction'
    rural_predictions['parameter'] = 'pm25'
    rural_predictions['unit'] = 'µg/m³'
    
    print(f"\nGenerated {len(rural_predictions)} rural predictions")
    print(f"AQI range: {rural_predictions['aqi'].min():.1f} - {rural_predictions['aqi'].max():.1f}")
    print(f"\nAQI Category distribution:")
    print(rural_predictions['aqi_category'].value_counts())
    
    return rural_predictions

def combine_urban_rural_data():
    """
    Combine urban actual measurements with rural predictions for dashboard.
    """
    print("\n" + "="*60)
    print("Combining Urban (Actual) + Rural (Predicted) Data")
    print("="*60)
    
    # Load urban actual data
    urban_path = os.path.join(config.PROCESSED_DATA_DIR, "final_training_data.csv")
    urban_df = pd.read_csv(urban_path)
    
    # Use original coordinates if available
    if 'latitude_orig' in urban_df.columns:
        urban_df['latitude'] = urban_df['latitude_orig']
        urban_df['longitude'] = urban_df['longitude_orig']
    
    # Prepare urban data
    urban_display = urban_df[['date', 'value', 'latitude', 'longitude']].copy()
    if 'location_ground' in urban_df.columns:
        urban_display['location'] = urban_df['location_ground']
    else:
        urban_display['location'] = 'Chennai Urban'
    
    urban_display['location_type'] = 'Urban'
    urban_display['data_source'] = 'Ground Sensor (Actual)'
    urban_display['parameter'] = 'pm25'
    urban_display['unit'] = 'µg/m³'
    
    # Calculate AQI for urban data
    urban_display['aqi'] = urban_display['value'].apply(
        lambda x: aqi_calculator.calculate_aqi(x, 'pm25')
    )
    urban_display['aqi_category'] = urban_display['aqi'].apply(
        lambda x: aqi_calculator.get_aqi_category(x) if pd.notna(x) else 'Unknown'
    )
    
    # Generate rural predictions
    rural_display = generate_rural_predictions()
    
    if rural_display is None:
        print("Could not generate rural predictions")
        return urban_display
    
    # Combine
    combined_df = pd.concat([urban_display, rural_display], ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    
    # Save combined data
    output_path = os.path.join(config.PROCESSED_DATA_DIR, "combined_urban_rural_data.csv")
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Combined data saved to {output_path}")
    print(f"   Total records: {len(combined_df)}")
    print(f"   - Urban (Actual): {len(urban_display)}")
    print(f"   - Rural (Predicted): {len(rural_display)}")
    
    return combined_df

if __name__ == "__main__":
    combined_data = combine_urban_rural_data()
    
    if combined_data is not None:
        print("\n🎉 Rural predictions generated successfully!")
        print("Update the dashboard to load 'combined_urban_rural_data.csv' to see both urban and rural data.")
