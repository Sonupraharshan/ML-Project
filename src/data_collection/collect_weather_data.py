import requests
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def fetch_weather_data(lat, lon, start_date, end_date):
    """
    Fetches weather data. Can use Open-Meteo (free, no key) for real data.
    """
    print(f"Fetching weather data for {lat}, {lon}...")
    
    # Using Open-Meteo Historical Weather API
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get('hourly', {})
        df = pd.DataFrame({
            'date': hourly.get('time'),
            'temperature': hourly.get('temperature_2m'),
            'humidity': hourly.get('relative_humidity_2m'),
            'wind_speed': hourly.get('wind_speed_10m'),
            'pressure': hourly.get('surface_pressure')
        })
        return df
        
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        # Fallback to mock data
        print("Generating mock weather data...")
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        return pd.DataFrame({
            'date': dates,
            'temperature': np.random.uniform(25, 35, size=len(dates)),
            'humidity': np.random.uniform(40, 80, size=len(dates)),
            'wind_speed': np.random.uniform(0, 15, size=len(dates)),
            'pressure': np.random.uniform(1000, 1010, size=len(dates))
        })

def save_data(df, filename):
    if not df.empty:
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Weather data saved to {filepath}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch for Chennai center
    df = fetch_weather_data(config.CHENNAI_LAT, config.CHENNAI_LON, start_date, end_date)
    save_data(df, "weather_data_chennai.csv")
