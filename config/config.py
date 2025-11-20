import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Data Sources
OPENAQ_API_URL = "https://api.openaq.org/v2/measurements"
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# Placeholders for API Keys
OPENWEATHER_API_KEY = "YOUR_API_KEY_HERE"
NASA_EARTHDATA_TOKEN = "YOUR_TOKEN_HERE"

# Region of Interest (Chennai)
CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707
RURAL_LAT_RANGE = (12.5, 13.5) # Example range
RURAL_LON_RANGE = (79.5, 80.5) # Example range

# Pollutants
POLLUTANTS = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
