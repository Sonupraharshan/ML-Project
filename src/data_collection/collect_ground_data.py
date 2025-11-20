import requests
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def fetch_openaq_data_v3(city="Chennai", limit=1000):
    """
    Fetches air quality data from OpenAQ API v3.
    """
    print(f"Fetching data for {city} from OpenAQ API v3...")
    
    # OpenAQ v3 API endpoint
    url = "https://api.openaq.org/v3/locations"
    
    headers = {
        'Accept': 'application/json'
    }
    
    params = {
        'limit': 100,
        'page': 1,
        'offset': 0,
        'sort': 'desc',
        'radius': 50000,  # 50km radius
        'coordinates': f"{config.CHENNAI_LAT},{config.CHENNAI_LON}",
        'order_by': 'lastUpdated'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            print(f"Found {len(data['results'])} locations")
            
            # Now fetch measurements for each location
            all_records = []
            measurements_url = "https://api.openaq.org/v3/measurements"
            
            for location in data['results'][:5]:  # Limit to first 5 locations
                location_id = location.get('id')
                if not location_id:
                    continue
                    
                params_m = {
                    'locations_id': location_id,
                    'limit': 1000,
                    'date_from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'date_to': datetime.now().strftime('%Y-%m-%d')
                }
                
                try:
                    resp_m = requests.get(measurements_url, headers=headers, params=params_m, timeout=10)
                    resp_m.raise_for_status()
                    meas_data = resp_m.json()
                    
                    if 'results' in meas_data:
                        for meas in meas_data['results']:
                            record = {
                                'location': location.get('name', 'Unknown'),
                                'parameter': meas.get('parameter', {}).get('name'),
                                'value': meas.get('value'),
                                'unit': meas.get('parameter', {}).get('units'),
                                'date': meas.get('datetime', {}).get('utc'),
                                'latitude': location.get('coordinates', {}).get('latitude'),
                                'longitude': location.get('coordinates', {}).get('longitude')
                            }
                            all_records.append(record)
                except Exception as e:
                    print(f"Error fetching measurements for location {location_id}: {e}")
                    continue
            
            if all_records:
                print(f"Collected {len(all_records)} measurement records")
                return pd.DataFrame(all_records)
            else:
                print("No measurements found")
                return pd.DataFrame()
        else:
            print("No locations found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching OpenAQ data v3: {e}")
        return pd.DataFrame()

def fetch_openaq_data(city="Chennai", limit=1000):
    """
    Main function to fetch air quality data, tries v3 API first.
    """
    print(f"Fetching data for {city}...")
    
    # Try v3 API first
    df = fetch_openaq_data_v3(city, limit)
    
    if not df.empty:
        return df
    
    # If v3 fails, fall back to mock data
    print("API fetch failed, falling back to mock data generation...")
    return generate_mock_data(city)

def generate_mock_data(city="Chennai"):
    """
    Generates mock ground air quality data.
    """
    print(f"Generating mock data for {city}...")
    
    dates = pd.date_range(start=(datetime.now() - timedelta(days=30)), end=datetime.now(), freq='h')
    records = []
    
    # Center coordinates for Chennai
    lat_center = 13.0827
    lon_center = 80.2707
    
    # Simulate a few stations with realistic names
    stations = [
        {'location': 'Manali (Urban)', 'lat': lat_center + 0.15, 'lon': lon_center + 0.05},
        {'location': 'Adyar (Urban)', 'lat': lat_center + 0.02, 'lon': lon_center + 0.02},
        {'location': 'Anna Nagar (Urban)', 'lat': lat_center - 0.02, 'lon': lon_center - 0.02},
        {'location': 'Velachery (Urban)', 'lat': lat_center + 0.01, 'lon': lon_center - 0.01}
    ]
    
    for date in dates:
        for station in stations:
            for pollutant in ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']:
                # Random values with some realistic ranges and daily patterns
                hour = date.hour
                base_multiplier = 1.5 if (7 <= hour <= 10 or 18 <= hour <= 21) else 1.0  # Rush hours
                
                if pollutant == 'pm25': 
                    value = np.random.uniform(20, 100) * base_multiplier
                elif pollutant == 'pm10': 
                    value = np.random.uniform(40, 150) * base_multiplier
                elif pollutant == 'no2':
                    value = np.random.uniform(10, 60) * base_multiplier
                elif pollutant == 'so2':
                    value = np.random.uniform(5, 30)
                elif pollutant == 'o3':
                    value = np.random.uniform(10, 80)
                else:  # co
                    value = np.random.uniform(0.1, 2.5) * base_multiplier
                
                records.append({
                    'location': station['location'],
                    'parameter': pollutant,
                    'value': value,
                    'unit': 'µg/m³' if pollutant != 'co' else 'mg/m³',
                    'date': date,
                    'latitude': station['lat'],
                    'longitude': station['lon']
                })
                
    return pd.DataFrame(records)

def save_data(df, filename):
    if not df.empty:
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath} ({len(df)} records)")
    else:
        print("No data to save.")

if __name__ == "__main__":
    df = fetch_openaq_data()
    save_data(df, "ground_aqi_chennai.csv")

