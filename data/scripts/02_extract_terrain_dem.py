"""
Extract terrain features from local DEM file
No API calls needed - fast and offline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import rasterio
from rasterio.transform import rowcol

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
EXTERNAL_DIR = DATA_DIR / 'external'

CAIC_CLEAN = PROCESSED_DIR / 'caic_clean.csv'
TERRAIN_CACHE = PROCESSED_DIR / 'terrain_cache.csv'
DEM_PATH = EXTERNAL_DIR / 'colorado_dem.tif'


def calculate_slope_aspect(elevation_array, transform, row, col, lat):
    """Calculate slope and aspect from elevation array"""
    try:
        # Get cell size in degrees
        cell_size_deg = abs(transform[0])
        
        # Convert degrees to meters at this latitude
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        # 1 degree latitude ≈ 111,320 meters
        lat_rad = np.radians(lat)
        meters_per_deg_lon = 111320 * np.cos(lat_rad)
        meters_per_deg_lat = 111320
        
        # Average meters per degree (approximation for small areas)
        cell_size_m = (meters_per_deg_lon + meters_per_deg_lat) / 2 * cell_size_deg
        
        # Check bounds
        if row < 1 or row >= elevation_array.shape[0] - 1 or \
           col < 1 or col >= elevation_array.shape[1] - 1:
            return None, None
        
        # Get 3x3 window
        window = elevation_array[row-1:row+2, col-1:col+2]
        
        # Check for no-data
        if np.any(np.isnan(window)) or np.any(window == -32768):
            return None, None
        
        # Calculate gradients (Horn's method)
        dz_dx = ((window[0, 2] + 2*window[1, 2] + window[2, 2]) - 
                 (window[0, 0] + 2*window[1, 0] + window[2, 0])) / (8 * cell_size_m)
        
        dz_dy = ((window[2, 0] + 2*window[2, 1] + window[2, 2]) - 
                 (window[0, 0] + 2*window[0, 1] + window[0, 2])) / (8 * cell_size_m)
        
        # Calculate slope
        slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_degrees = np.degrees(slope_radians)
        
        # Calculate aspect
        aspect_radians = np.arctan2(dz_dy, -dz_dx)
        aspect_degrees = np.degrees(aspect_radians)
        
        # Convert to compass bearing
        if aspect_degrees < 0:
            aspect_degrees = 90.0 - aspect_degrees
        elif aspect_degrees > 90.0:
            aspect_degrees = 360.0 - aspect_degrees + 90.0
        else:
            aspect_degrees = 90.0 - aspect_degrees
        
        return float(slope_degrees), float(aspect_degrees)
        
    except Exception:
        return None, None


def aspect_degrees_to_cardinal(degrees):
    """Convert aspect degrees to cardinal direction"""
    if degrees is None or np.isnan(degrees):
        return None
    
    degrees = degrees % 360
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = int((degrees + 22.5) / 45) % 8
    return directions[index]


def extract_terrain_features():
    """Extract terrain features from local DEM"""
    
    print("="*60)
    print("TERRAIN FEATURE EXTRACTION (Local DEM)")
    print("="*60)
    
    # Check DEM exists
    if not DEM_PATH.exists():
        print(f"\n✗ DEM file not found: {DEM_PATH}")
        print("\nPlease download Colorado DEM first:")
        print("1. Go to https://portal.opentopography.org/")
        print("2. Download SRTM data for Colorado")
        print(f"3. Save as: {DEM_PATH}")
        return None
    
    print(f"\n✓ Found DEM: {DEM_PATH}")
    print(f"  Size: {DEM_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Load CAIC data
    print("\nLoading CAIC data...")
    df = pd.read_csv(CAIC_CLEAN)
    print(f"Loaded {len(df):,} avalanche observations")
    
    # Get unique locations
    unique_locations = df[['Longitude', 'latitude']].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_locations):,} unique locations")
    
    # Open DEM
    print("\nOpening DEM...")
    src = rasterio.open(DEM_PATH)
    elevation_data = src.read(1)
    print(f"DEM shape: {elevation_data.shape}")
    print(f"DEM bounds: {src.bounds}")
    
    # Initialize results
    terrain_data = {
        'Longitude': [],
        'latitude': [],
        'elevation': [],
        'slope': [],
        'aspect_degrees': [],
        'aspect_cardinal': []
    }
    
    print(f"\nExtracting terrain features...")
    
    failed_count = 0
    
    for idx, row in tqdm(unique_locations.iterrows(), total=len(unique_locations), desc="Processing locations"):
        lon = row['Longitude']
        lat = row['latitude']
        
        try:
            # Convert lat/lon to row/col
            py, px = rowcol(src.transform, lon, lat)
            
            # Check bounds
            if py < 0 or py >= src.height or px < 0 or px >= src.width:
                elevation = None
                slope = None
                aspect_deg = None
            else:
                # Get elevation
                elevation = float(elevation_data[py, px])
                if elevation == src.nodata or elevation < -999:
                    elevation = None
                
                # Calculate slope and aspect (pass latitude for proper conversion)
                slope, aspect_deg = calculate_slope_aspect(elevation_data, src.transform, py, px, lat)
            
        except Exception:
            elevation = None
            slope = None
            aspect_deg = None
        
        aspect_card = aspect_degrees_to_cardinal(aspect_deg) if aspect_deg is not None else None
        
        # Record results
        terrain_data['Longitude'].append(lon)
        terrain_data['latitude'].append(lat)
        terrain_data['elevation'].append(elevation)
        terrain_data['slope'].append(slope)
        terrain_data['aspect_degrees'].append(aspect_deg)
        terrain_data['aspect_cardinal'].append(aspect_card)
        
        if elevation is None:
            failed_count += 1
    
    src.close()
    
    # Create DataFrame
    terrain_df = pd.DataFrame(terrain_data)
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"\nTotal locations: {len(terrain_df):,}")
    print(f"Successful elevation: {terrain_df['elevation'].notna().sum():,} ({terrain_df['elevation'].notna().sum()/len(terrain_df)*100:.1f}%)")
    print(f"Successful slope: {terrain_df['slope'].notna().sum():,} ({terrain_df['slope'].notna().sum()/len(terrain_df)*100:.1f}%)")
    print(f"Successful aspect: {terrain_df['aspect_degrees'].notna().sum():,} ({terrain_df['aspect_degrees'].notna().sum()/len(terrain_df)*100:.1f}%)")
    print(f"Failed extractions: {failed_count}")
    
    print(f"\nElevation statistics:")
    print(terrain_df['elevation'].describe())
    
    print(f"\nSlope statistics:")
    print(terrain_df['slope'].describe())
    
    # Save cache
    terrain_df.to_csv(TERRAIN_CACHE, index=False)
    print(f"\n✓ Terrain cache saved to: {TERRAIN_CACHE}")
    
    return terrain_df


if __name__ == "__main__":
    terrain_df = extract_terrain_features()