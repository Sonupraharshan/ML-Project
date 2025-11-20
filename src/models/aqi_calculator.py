import numpy as np

def calculate_aqi(concentration, pollutant):
    """
    Calculates AQI based on CPCB standards.
    """
    # CPCB Breakpoints (simplified for demo)
    breakpoints = {
        'pm25': [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200), (90, 120, 201, 300), (120, 250, 301, 400), (250, 5000, 401, 500)],
        'pm10': [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200), (250, 350, 201, 300), (350, 430, 301, 400), (430, 5000, 401, 500)],
        # Add other pollutants...
    }
    
    if pollutant not in breakpoints:
        return None
        
    for (low_c, high_c, low_i, high_i) in breakpoints[pollutant]:
        if low_c <= concentration <= high_c:
            aqi = ((high_i - low_i) / (high_c - low_c)) * (concentration - low_c) + low_i
            return round(aqi)
            
    return 500 # Cap at 500

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"
