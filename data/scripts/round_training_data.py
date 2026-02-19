"""
Round numeric columns to sensible precision
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

TRAINING_DATA = PROCESSED_DIR / 'training_data.csv'
TRAINING_DATA_ROUNDED = PROCESSED_DIR / 'training_data_rounded.csv'


def round_training_data():
    """Round numeric columns to appropriate precision"""
    
    print("Loading training data...")
    df = pd.read_csv(TRAINING_DATA)
    
    print(f"Loaded {len(df):,} rows")
    
    # Define precision for each column type
    rounding_rules = {
        # Coordinates (4 decimals = ~10 meter precision)
        'Longitude': 4,
        'latitude': 4,
        
        # Terrain (1-2 decimals is plenty)
        'elevation': 1,           # 3421.2 m (no need for cm precision)
        'slope': 1,               # 38.2° (tenth of degree is fine)
        'aspect_degrees': 1,      # 315.4° (tenth of degree is fine)
        
        # Weather (1-2 decimals)
        'snow_depth': 1,          # 126.3 cm (no need for mm precision)
        'new_snow_24h': 1,        # 27.4 cm
        'swe': 1,                 # 31.2 cm
        'temp': 1,                # -5.6°C (tenth of degree)
        'wind_speed': 1,          # Would be 12.3 m/s (but all None anyway)
        
        # Avalanche size (1 decimal - matches D-scale)
        'avalanche_size': 1,      # 1.5, 2.0, etc.
    }
    
    print("\nRounding numeric columns...")
    
    for column, decimals in rounding_rules.items():
        if column in df.columns:
            # Only round numeric columns (skip if already int or object)
            if pd.api.types.is_numeric_dtype(df[column]):
                before = df[column].iloc[0] if len(df) > 0 else None
                df[column] = df[column].round(decimals)
                after = df[column].iloc[0] if len(df) > 0 else None
                print(f"  {column:20s}: {decimals} decimals (e.g., {before} → {after})")
    
    # Show file size comparison
    import os
    
    # Save rounded version
    df.to_csv(TRAINING_DATA_ROUNDED, index=False)
    
    # Compare sizes
    original_size = os.path.getsize(TRAINING_DATA) / 1024 / 1024
    rounded_size = os.path.getsize(TRAINING_DATA_ROUNDED) / 1024 / 1024
    savings = ((original_size - rounded_size) / original_size) * 100
    
    print(f"\n✓ Rounded data saved to: {TRAINING_DATA_ROUNDED}")
    print(f"\nFile size:")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Rounded:  {rounded_size:.2f} MB")
    print(f"  Savings:  {savings:.1f}%")
    
    # Show sample
    print("\nSample data (first 3 rows):")
    print(df[['elevation', 'slope', 'aspect_degrees', 'snow_depth', 'temp']].head(3))
    
    return df


if __name__ == "__main__":
    df = round_training_data()