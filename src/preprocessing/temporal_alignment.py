import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def align_temporal_data(df, freq='h'):
    """
    Aligns data to a specific temporal frequency (e.g., Hourly 'h', Daily 'D').
    Handles missing data via interpolation.
    """
    print(f"Aligning data to frequency: {freq}...")
    
    if df.empty:
        return df
        
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by location and resample
    aligned_dfs = []
    for (lat, lon), group in df.groupby(['latitude', 'longitude']):
        # Separate numeric and non-numeric columns
        numeric_cols = group.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = [col for col in group.columns if col not in numeric_cols and col != 'date']
        
        group = group.set_index('date').sort_index()
        
        # Resample only numeric columns
        resampled = group[numeric_cols].resample(freq).mean()
        
        # Interpolate missing values (linear for short gaps, forward fill for others)
        resampled = resampled.interpolate(method='linear', limit=6) # Interpolate up to 6 hours/days
        resampled = resampled.ffill().bfill() # Fill remaining
        
        # Add back the non-numeric columns (take first value of each resample period)
        for col in non_numeric_cols:
            resampled[col] = group[col].resample(freq).first()
        
        resampled['latitude'] = lat
        resampled['longitude'] = lon
        aligned_dfs.append(resampled.reset_index())
        
    if not aligned_dfs:
        return pd.DataFrame()
        
    return pd.concat(aligned_dfs, ignore_index=True)

if __name__ == "__main__":
    try:
        df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "matched_data.csv"))
        aligned_df = align_temporal_data(df, freq='D') # Daily alignment (use 'h' for hourly)
        aligned_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "aligned_data.csv"), index=False)
        print("Aligned data saved.")
    except FileNotFoundError:
        print("Matched data file not found.")
