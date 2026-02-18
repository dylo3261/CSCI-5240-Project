import pandas as pd
import numpy as np

def load_caic_data(filepath, natural_only=True):
    """Load and clean CAIC avalanche data"""
    
    df = pd.read_csv(filepath)
    
    # Strip whitespace from all string columns FIRST
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].str.strip()
    
    # Replace placeholder values with NaN (after stripping whitespace)
    placeholder_values = ['-', '']
    df = df.replace(placeholder_values, np.nan)
    
    # Filter to valid coordinates
    df = df.dropna(subset=['Longitude', 'latitude'])
    
    # Convert types
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df = df.dropna(subset=['Longitude', 'latitude'])
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract avalanche size
    df['avalanche_size'] = df['Destructive Size'].str.extract(r'D(\d+\.?\d*)')[0]
    df['avalanche_size'] = pd.to_numeric(df['avalanche_size'], errors='coerce')
    
    # Keep useful columns
    df_final = df[[
        'Observation ID',
        'Date',
        'Longitude',
        'latitude',
        'Aspect',
        'Area',
        'avalanche_size',
        'Type',
        'Trigger'
    ]].copy()
    
    # Add label
    df_final['avalanche_occurred'] = 1
    
    # Remove rows missing critical columns (including Area now)
    df_final = df_final.dropna(subset=['Date', 'Longitude', 'latitude', 'avalanche_size', 'Area'])
    
    # DEBUG: Check Trigger values before filtering
    print("\nDEBUG - Trigger values before filtering:")
    print(df_final['Trigger'].value_counts(dropna=False))
    print(f"\nUnique Trigger values: {df_final['Trigger'].unique()}")
    
    # Filter to natural avalanches only (N) and unknown triggers (U)
    if natural_only:
        print(f"\nFiltering to natural (N) and unknown (U) triggers...")
        before_count = len(df_final)
        # Include NaN in Trigger as well (treat missing trigger as potentially natural)
        df_final = df_final[df_final['Trigger'].isin(['N', 'U']) | df_final['Trigger'].isna()]
        after_count = len(df_final)
        print(f"Filtered from {before_count:,} to {after_count:,} observations ({before_count - after_count:,} removed)")
    
    print(f"\nLoaded {len(df_final):,} avalanche observations")
    print(f"Date range: {df_final['Date'].min()} to {df_final['Date'].max()}")
    print(f"\nAvalanche sizes:\n{df_final['avalanche_size'].value_counts().sort_index()}")
    print(f"\nTrigger distribution:\n{df_final['Trigger'].value_counts(dropna=False)}")
    print(f"\nRegion distribution:\n{df_final['Area'].value_counts()}")
    
    return df_final


if __name__ == "__main__":
    df = load_caic_data('data/raw/avalanche-record-CAIC-2016-2026.csv', natural_only=True)
    df.to_csv('data/processed/caic_clean.csv', index=False)
    print(f"\nSaved to data/processed/caic_clean.csv")