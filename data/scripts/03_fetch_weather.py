"""
Fetch weather data from SNOTEL stations using inverse distance weighting

This script:
1. Loads all Colorado SNOTEL stations
2. For each avalanche, finds 3 nearest stations
3. Queries weather data from those stations
4. Applies IDW to interpolate values
5. Caches results for efficiency
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
from metloom.pointdata import SnotelPointData

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.weather_utils import (
    haversine_distance,
    find_nearest_stations,
    inverse_distance_weighting,
    get_station_weather,
    calculate_24h_snow_change
)

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

CAIC_CLEAN = PROCESSED_DIR / 'caic_clean.csv'
WEATHER_CACHE = PROCESSED_DIR / 'weather_cache.csv'
STATIONS_LIST = PROCESSED_DIR / 'snotel_stations.csv'


def fetch_weather_data():
    """
    Fetch weather data for all avalanche locations using IDW from 3 nearest stations
    """
    
    print("="*60)
    print("WEATHER DATA EXTRACTION (SNOTEL + IDW)")
    print("="*60)
    
    # Load CAIC data
    print("\nLoading CAIC data...")
    df = pd.read_csv(CAIC_CLEAN)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df):,} avalanche observations")
    
    # Get or load SNOTEL stations
    if STATIONS_LIST.exists():
        print("\nLoading SNOTEL stations from cache...")
        stations_df = pd.read_csv(STATIONS_LIST)
    else:
        print("Error: SNOTEL stations list not found!")
        print("Please run: python scripts/build_snotel_stations.py")
        return None
    
    print(f"Using {len(stations_df)} SNOTEL stations")
    
    # Initialize results
    weather_data = {
        'Longitude': [],
        'latitude': [],
        'Date': [],
        'snow_depth': [],
        'new_snow_24h': [],
        'swe': [],
        'temp': [],
        'wind_speed': [],
        'nearest_stations': [],
        'nearest_distances': []
    }
    
    # Cache for station queries (key: (station_triplet, date_str))
    station_cache = {}
    
    print(f"\nFetching weather data with 3-station IDW interpolation...")
    
    failed_count = 0
    cache_hits = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing avalanches"):
        lon = row['Longitude']
        lat = row['latitude']
        date = row['Date']
        date_str = date.strftime('%Y-%m-%d')
        
        # Find 3 nearest stations
        nearest = find_nearest_stations(lon, lat, stations_df, n=3)
        
        if not nearest:
            # No stations found
            weather_data['Longitude'].append(lon)
            weather_data['latitude'].append(lat)
            weather_data['Date'].append(date)
            weather_data['snow_depth'].append(None)
            weather_data['new_snow_24h'].append(None)
            weather_data['swe'].append(None)
            weather_data['temp'].append(None)
            weather_data['wind_speed'].append(None)
            weather_data['nearest_stations'].append(None)
            weather_data['nearest_distances'].append(None)
            failed_count += 1
            continue
        
        # Extract station triplets and distances
        station_triplets = []
        distances = []
        for station_id, dist in nearest:
            # Look up the triplet for this station_id
            station_row = stations_df[stations_df['station_id'] == int(station_id)]
            if len(station_row) > 0:
                triplet = station_row['station_triplet'].iloc[0]
                station_triplets.append(triplet)
                distances.append(dist)
        
        if not station_triplets:
            # No valid triplets found
            weather_data['Longitude'].append(lon)
            weather_data['latitude'].append(lat)
            weather_data['Date'].append(date)
            weather_data['snow_depth'].append(None)
            weather_data['new_snow_24h'].append(None)
            weather_data['swe'].append(None)
            weather_data['temp'].append(None)
            weather_data['wind_speed'].append(None)
            weather_data['nearest_stations'].append(None)
            weather_data['nearest_distances'].append(None)
            failed_count += 1
            continue
        
        # Query weather from each station (with caching)
        snow_depths = []
        new_snows = []
        swes = []
        temps = []
        wind_speeds = []
        
        for station_triplet in station_triplets:
            cache_key = (station_triplet, date_str)
            
            if cache_key in station_cache:
                # Use cached data
                station_data = station_cache[cache_key]
                cache_hits += 1
            else:
                # Query SNOTEL with triplet
                station_data = get_station_weather(station_triplet, date.to_pydatetime())
                
                # Calculate 24h snow change
                new_snow = calculate_24h_snow_change(station_triplet, date.to_pydatetime())
                station_data['new_snow_24h'] = new_snow
                
                # Cache it
                station_cache[cache_key] = station_data
            
            # Collect values
            snow_depths.append(station_data['snow_depth'])
            new_snows.append(station_data.get('new_snow_24h'))
            swes.append(station_data['swe'])
            temps.append(station_data['temp'])
            wind_speeds.append(station_data['wind_speed'])
        
        # Apply IDW to interpolate values
        snow_depth_idw = inverse_distance_weighting(snow_depths, distances)
        new_snow_idw = inverse_distance_weighting(new_snows, distances)
        swe_idw = inverse_distance_weighting(swes, distances)
        temp_idw = inverse_distance_weighting(temps, distances)
        wind_speed_idw = inverse_distance_weighting(wind_speeds, distances)
        
        # Record results
        weather_data['Longitude'].append(lon)
        weather_data['latitude'].append(lat)
        weather_data['Date'].append(date)
        weather_data['snow_depth'].append(snow_depth_idw)
        weather_data['new_snow_24h'].append(new_snow_idw)
        weather_data['swe'].append(swe_idw)
        weather_data['temp'].append(temp_idw)
        weather_data['wind_speed'].append(wind_speed_idw)
        weather_data['nearest_stations'].append(','.join(station_triplets))
        weather_data['nearest_distances'].append(','.join([f"{d:.1f}" for d in distances]))
        
        if snow_depth_idw is None:
            failed_count += 1
    
    # Create DataFrame
    weather_df = pd.DataFrame(weather_data)
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"\nTotal observations: {len(weather_df):,}")
    print(f"Successful snow_depth: {weather_df['snow_depth'].notna().sum():,} ({weather_df['snow_depth'].notna().sum()/len(weather_df)*100:.1f}%)")
    print(f"Successful new_snow_24h: {weather_df['new_snow_24h'].notna().sum():,} ({weather_df['new_snow_24h'].notna().sum()/len(weather_df)*100:.1f}%)")
    print(f"Successful SWE: {weather_df['swe'].notna().sum():,} ({weather_df['swe'].notna().sum()/len(weather_df)*100:.1f}%)")
    print(f"Successful temp: {weather_df['temp'].notna().sum():,} ({weather_df['temp'].notna().sum()/len(weather_df)*100:.1f}%)")
    print(f"Failed extractions: {failed_count}")
    print(f"\nCache hits: {cache_hits:,} (saved API calls)")
    print(f"Unique station+date queries: {len(station_cache):,}")
    
    print(f"\nWeather statistics:")
    print(weather_df[['snow_depth', 'new_snow_24h', 'swe', 'temp', 'wind_speed']].describe())
    
    # Save cache
    weather_df.to_csv(WEATHER_CACHE, index=False)
    print(f"\n✓ Weather cache saved to: {WEATHER_CACHE}")
    
    return weather_df


if __name__ == "__main__":
    weather_df = fetch_weather_data()