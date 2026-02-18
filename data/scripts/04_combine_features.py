"""
Combine CAIC data with terrain and weather features

This script:
1. Loads cleaned CAIC avalanche data
2. Merges terrain features (elevation, slope, aspect)
3. Merges weather features (snow, temp, wind)
4. Creates unique IDs based on location + timestamp
5. Creates final training dataset with all features
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

CAIC_CLEAN = PROCESSED_DIR / 'caic_clean.csv'
TERRAIN_CACHE = PROCESSED_DIR / 'terrain_cache.csv'
WEATHER_CACHE = PROCESSED_DIR / 'weather_cache.csv'
POSITIVE_EXAMPLES = PROCESSED_DIR / 'positive_examples_enriched.csv'


def combine_features():
    """Combine CAIC, terrain, and weather data into final dataset"""
    
    print("="*60)
    print("COMBINING FEATURES")
    print("="*60)
    
    # Load CAIC data
    print("\nLoading CAIC data...")
    caic_df = pd.read_csv(CAIC_CLEAN)
    caic_df['Date'] = pd.to_datetime(caic_df['Date'])
    print(f"Loaded {len(caic_df):,} avalanche observations")
    
    # Load terrain data
    print("\nLoading terrain data...")
    terrain_df = pd.read_csv(TERRAIN_CACHE)
    print(f"Loaded terrain for {len(terrain_df):,} unique locations")
    
    # Load weather data
    print("\nLoading weather data...")
    weather_df = pd.read_csv(WEATHER_CACHE)
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    print(f"Loaded weather for {len(weather_df):,} observations")
    
    # Merge terrain features
    print("\nMerging terrain features...")
    df = caic_df.merge(
        terrain_df,
        on=['Longitude', 'latitude'],
        how='left',
        suffixes=('', '_terrain')
    )
    print(f"After terrain merge: {len(df):,} rows")
    
    # Merge weather features
    print("\nMerging weather features...")
    df = df.merge(
        weather_df,
        on=['Longitude', 'latitude', 'Date'],
        how='left',
        suffixes=('', '_weather')
    )
    print(f"After weather merge: {len(df):,} rows")
    
    # Create truly unique IDs (CAIC IDs may be reused/duplicated)
    print("\nCreating unique observation IDs...")
    df['unique_id'] = df.apply(
        lambda row: f"{row['Longitude']:.6f}_{row['latitude']:.6f}_{row['Date'].strftime('%Y%m%d_%H%M')}",
        axis=1
    )
    
    # Check for duplicates with new ID
    duplicates = df.duplicated(subset=['unique_id']).sum()
    print(f"True duplicates found: {duplicates}")
    
    if duplicates > 0:
        # Keep first occurrence
        df = df.drop_duplicates(subset=['unique_id'], keep='first').reset_index(drop=True)
        print(f"Removed {duplicates} duplicate observations")
    
    # Replace Observation ID with unique ID
    df['Observation ID'] = df['unique_id']
    df = df.drop(columns=['unique_id'])
    
    # Select and order final columns
    final_columns = [
        # Identifiers
        'Observation ID',
        'Date',
        'Longitude',
        'latitude',
        
        # CAIC features
        'Area',
        'Aspect',  # From CAIC (98.6% complete)
        'Type',
        'Trigger',
        
        # Target variables
        'avalanche_size',
        'avalanche_occurred',
        
        # Terrain features
        'elevation',
        'slope',
        'aspect_degrees',
        'aspect_cardinal',
        
        # Weather features
        'snow_depth',
        'new_snow_24h',
        'swe',
        'temp',
        'wind_speed',
        
        # Metadata
        'nearest_stations',
        'nearest_distances'
    ]
    
    # Keep only columns that exist
    final_columns = [col for col in final_columns if col in df.columns]
    df_final = df[final_columns].copy()
    
    # Data quality summary
    print("\n" + "="*60)
    print("DATA QUALITY SUMMARY")
    print("="*60)
    
    print(f"\nFinal dataset: {len(df_final):,} observations")
    print(f"\nCompleteness by feature:")
    
    key_features = [
        'elevation', 'slope', 'aspect_degrees',
        'snow_depth', 'new_snow_24h', 'swe', 'temp', 'wind_speed'
    ]
    
    for feature in key_features:
        if feature in df_final.columns:
            complete = df_final[feature].notna().sum()
            pct = complete / len(df_final) * 100
            print(f"  {feature:20s}: {complete:6,} ({pct:5.1f}%)")
    
    # Statistics
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    
    numeric_features = [
        'elevation', 'slope', 'avalanche_size',
        'snow_depth', 'new_snow_24h', 'swe', 'temp'
    ]
    
    for feature in numeric_features:
        if feature in df_final.columns and df_final[feature].notna().sum() > 0:
            print(f"\n{feature}:")
            stats = df_final[feature].describe()
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std:  {stats['std']:.2f}")
            print(f"  Min:  {stats['min']:.2f}")
            print(f"  Max:  {stats['max']:.2f}")
    
    # Save
    df_final.to_csv(POSITIVE_EXAMPLES, index=False)
    print(f"\n✓ Saved final dataset to: {POSITIVE_EXAMPLES}")
    
    # Preview
    print("\n" + "="*60)
    print("DATASET PREVIEW")
    print("="*60)
    print("\nFirst 5 rows:")
    print(df_final.head())
    
    print("\nColumn data types:")
    print(df_final.dtypes)
    
    return df_final


if __name__ == "__main__":
    df = combine_features()