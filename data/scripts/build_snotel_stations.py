"""
Build comprehensive list of Colorado SNOTEL stations using REST API
"""

import pandas as pd
import requests
from pathlib import Path
import time

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STATIONS_LIST = PROCESSED_DIR / 'snotel_stations.csv'


def get_colorado_snotel_stations():
    """
    Fetch all Colorado SNOTEL stations from AWDB REST API
    
    Returns:
        DataFrame with station information
    """
    print("="*60)
    print("FETCHING COLORADO SNOTEL STATIONS")
    print("="*60)
    
    # Use REST API endpoint
    url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations"
    
    params = {
        'stationTriplets': '*:CO:SNTL',  # All SNOTEL in Colorado
    }
    
    print("\nQuerying AWDB REST API...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print("✗ No stations returned")
            return pd.DataFrame()
        
        print(f"✓ Found {len(data)} stations")
        
        # Parse station data
        stations = []
        for station in data:
            try:
                # Get station triplet and parse ID
                triplet = station.get('stationTriplet', '')
                if not triplet:
                    continue
                
                parts = triplet.split(':')
                if len(parts) != 3:
                    continue
                
                station_id = str(parts[0]) 
                
                # Extract coordinates
                lat = station.get('latitude')
                lon = station.get('longitude')
                
                if lat is None or lon is None:
                    continue
                
                # Build station info
                station_info = {
                    'station_id': station_id,
                    'station_triplet': triplet,
                    'name': station.get('name', ''),
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'elevation_ft': station.get('elevation'),
                    'elevation_m': station.get('elevation') * 0.3048 if station.get('elevation') else None,
                    'county': station.get('countyName', ''),
                    'huc': station.get('huc', ''),
                    'active': True
                }
                
                stations.append(station_info)
                
            except Exception as e:
                print(f"Warning: Could not parse station: {e}")
                continue
        
        if not stations:
            print("✗ No valid stations parsed")
            return pd.DataFrame()
        
        stations_df = pd.DataFrame(stations)
        
        # Filter to Colorado bounds
        co_bounds = {
            'min_lat': 36.9,
            'max_lat': 41.1,
            'min_lon': -109.1,
            'max_lon': -102.0
        }
        
        in_bounds = (
            (stations_df['latitude'] >= co_bounds['min_lat']) &
            (stations_df['latitude'] <= co_bounds['max_lat']) &
            (stations_df['longitude'] >= co_bounds['min_lon']) &
            (stations_df['longitude'] <= co_bounds['max_lon'])
        )
        
        stations_df = stations_df[in_bounds].reset_index(drop=True)
        
        print(f"✓ {len(stations_df)} stations in Colorado bounds")
        
        return stations_df
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def save_stations(stations_df):
    """Save and summarize stations"""
    
    if stations_df.empty:
        print("\n✗ No stations to save")
        return None
    
    # Sort by name
    stations_df = stations_df.sort_values('name').reset_index(drop=True)
    
    # Save
    stations_df.to_csv(STATIONS_LIST, index=False)
    print(f"\n✓ Saved to: {STATIONS_LIST}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTotal stations: {len(stations_df)}")
    print(f"\nElevation range: {stations_df['elevation_ft'].min():.0f} - {stations_df['elevation_ft'].max():.0f} ft")
    print(f"Latitude range: {stations_df['latitude'].min():.2f}° - {stations_df['latitude'].max():.2f}°")
    print(f"Longitude range: {stations_df['longitude'].min():.2f}° - {stations_df['longitude'].max():.2f}°")
    
    print(f"\nFirst 10 stations:")
    print(stations_df[['station_id', 'name', 'station_triplet', 'elevation_ft']].head(10).to_string(index=False))
    
    return stations_df


if __name__ == "__main__":
    stations_df = get_colorado_snotel_stations()
    
    if not stations_df.empty:
        save_stations(stations_df)
    else:
        print("\nFailed to fetch stations")