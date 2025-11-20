import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def train_urban_models(data_path):
    """
    Trains multiple models on urban data and saves the best one.
    """
    print("Training urban models...")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Data file not found.")
        return

    # Define features and target
    # Assuming 'value' is the target (PM2.5) and others are features
    target_col = 'value'
    
    # Exclude non-feature columns (identifiers, strings, target)
    exclude_cols = ['date', 'latitude', 'longitude', 'location', 'parameter', 'unit', 
                    target_col, 'location_ground', 'location_sat', 'location_type',
                    'nearest_sat_lat', 'nearest_sat_lon', 'latitude_sat', 'longitude_sat',
                    'latitude_orig', 'longitude_orig']
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter to get feature columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    if not feature_cols:
        print("Error: No feature columns found!")
        return
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42)
    }
    
    best_model = None
    best_score = -float('inf')
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MSE': mse, 'R2': r2}
        print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            
    # Save best model
    if best_model:
        model_path = os.path.join(config.PROCESSED_DATA_DIR, 'best_urban_model.pkl')
        joblib.dump(best_model, model_path)
        print(f"Best model saved to {model_path}")
        
    return results

if __name__ == "__main__":
    # Example usage
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "final_training_data.csv")
    train_urban_models(data_path)
