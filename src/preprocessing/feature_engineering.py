import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def create_features(df):
    """
    Creates lag features, rolling means, and seasonal indicators.
    """
    print("Creating features...")
    
    if df.empty:
        return df
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['latitude', 'longitude', 'date'])
    
    # Seasonal indicators
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    
    # Lag features and rolling means for pollutants/weather
    # Assuming 'value' is the target pollutant (e.g., PM2.5)
    # In a real scenario, we'd do this for specific columns
    target_cols = ['value', 'temperature', 'humidity'] # Example columns
    
    for col in target_cols:
        if col in df.columns:
            # Lag features
            df[f'{col}_lag1'] = df.groupby(['latitude', 'longitude'])[col].shift(1)
            df[f'{col}_lag3'] = df.groupby(['latitude', 'longitude'])[col].shift(3)
            
            # Rolling means
            df[f'{col}_rolling_mean_3'] = df.groupby(['latitude', 'longitude'])[col].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            df[f'{col}_rolling_mean_7'] = df.groupby(['latitude', 'longitude'])[col].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # Drop rows with NaNs only in critical columns (target and lag features)
    # Keep rows where we have at least the current value, even if lags are missing
    critical_cols = ['value'] if 'value' in df.columns else []
    if critical_cols:
        df = df.dropna(subset=critical_cols)
    
    # Fill remaining NaNs with forward fill for lag features
    lag_cols = [c for c in df.columns if '_lag' in c or '_rolling' in c]
    if lag_cols:
        df[lag_cols] = df[lag_cols].ffill().bfill().fillna(0)
    
    return df

def normalize_features(df, target_col='value'):
    """
    Normalizes features using StandardScaler.
    """
    print("Normalizing features...")
    
    if df.empty:
        print("Warning: Empty dataframe provided for normalization. Skipping.")
        return df
    
    # First, handle any remaining NaN values before normalization
    print(f"Rows before NaN cleanup: {len(df)}")
    
    # PRESERVE original coordinates for mapping (don't normalize these!)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['latitude_orig'] = df['latitude']
        df['longitude_orig'] = df['longitude']
    
    # Only select numeric columns for features, excluding target and identifier columns
    exclude_cols = ['date', 'latitude', 'longitude', target_col, 'unit', 'location', 'parameter', 
                    'location_type', 'latitude_orig', 'longitude_orig',
                    'location_ground', 'location_sat',
                    'nearest_sat_lat', 'nearest_sat_lon', 'latitude_sat', 'longitude_sat']
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter out excluded columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if not feature_cols:
        print("Warning: No feature columns found for normalization.")
        return df
    
    print(f"Normalizing {len(feature_cols)} features: {feature_cols[:5]}...")  # Show first 5
    
    # Fill any remaining NaNs in feature columns with 0
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Also ensure target column has no NaNs
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
    
    print(f"Rows after NaN cleanup: {len(df)}")
    
    if df.empty:
        print("Warning: All rows were dropped during NaN cleanup.")
        return df
    
    
    scaler = StandardScaler()
    
    # Check for constant columns before scaling
    constant_cols = []
    for col in feature_cols:
        if df[col].std() == 0:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Warning: Found {len(constant_cols)} constant columns (will be excluded): {constant_cols[:5]}")
        feature_cols = [c for c in feature_cols if c not in constant_cols]
    
    if not feature_cols:
        print("Warning: No variable features to normalize.")
        return df
    
    # Normalize only non-constant columns
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save scaler for later use
    joblib.dump(scaler, os.path.join(config.PROCESSED_DATA_DIR, 'scaler.pkl'))
    
    # Check for inf/nan after normalization
    inf_count = np.isinf(df[feature_cols]).sum().sum()
    nan_count = df[feature_cols].isna().sum().sum()
    
    if inf_count > 0:
        print(f"Warning: {inf_count} inf values after normalization, replacing with 0")
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values after normalization, filling with 0")
        df[feature_cols] = df[feature_cols].fillna(0)
    
    print(f"Final row count: {len(df)}")
    
    return df

if __name__ == "__main__":
    try:
        df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "aligned_data.csv"))
        df = create_features(df)
        df = normalize_features(df)
        df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_training_data.csv"), index=False)
        print("Feature engineering complete. Data saved.")
    except FileNotFoundError:
        print("Aligned data file not found.")
