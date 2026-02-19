"""
Generate negative examples by querying real SNOTEL data

Strategy:
1. Use avalanche terrain locations
2. Sample random dates where NO avalanche occurred
3. Query SNOTEL for actual weather on those dates
4. Use same IDW interpolation as positive examples
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import sys
from tqdm import tqdm

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.weather_utils import (
    find_nearest_stations,
    inverse_distance_weighting,
    get_station_weather,
    calculate_24h_snow_change
)

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

POSITIVE_EXAMPLES = PROCESSED_DIR / 'positive_examples_enriched.csv'
TERRAIN_CACHE = PROCESSED_DIR / 'terrain_cache.csv'
STATIONS_LIST = PROCESSED_DIR / 'snotel_stations.csv'
TRAINING_DATA = PROCESSED_DIR / 'training_data.csv'


def generate_negative_examples(ratio=2.0):
    """
    Generate negative examples with real SNOTEL weather data
    
    Args:
        ratio: Ratio of negatives to positives (default 2:1)
    
    Returns:
        DataFrame with negative examples
    """
    
    print("="*60)
    print("GENERATING NEGATIVE EXAMPLES (QUERYING SNOTEL)")
    print("="*60)
    
    # Load positive examples
    print("\nLoading positive examples...")
    positives = pd.read_csv(POSITIVE_EXAMPLES)
    positives['Date'] = pd.to_datetime(positives['Date'])
    print(f"Loaded {len(positives):,} avalanche observations")
    
    # Load terrain and stations
    terrain_df = pd.read_csv(TERRAIN_CACHE)
    stations_df = pd.read_csv(STATIONS_LIST)
    print(f"Loaded {len(stations_df)} SNOTEL stations")
    
    # Calculate target
    n_negatives = int(len(positives) * ratio)
    print(f"\nTarget: {n_negatives:,} negative examples ({ratio}:1 ratio)")
    
    # Get unique locations
    unique_locations = positives[['Longitude', 'latitude', 'Area']].drop_duplicates()
    print(f"Unique avalanche locations: {len(unique_locations):,}")
    
    # Get date range
    min_date = positives['Date'].min()
    max_date = positives['Date'].max()
    # Exclude recent dates (SNOTEL lag)
    max_date = max_date - timedelta(days=10)
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    
    # Create set of actual avalanche (location, date) pairs
    avalanche_pairs = set(
        positives.apply(lambda row: (
            round(row['Longitude'], 6), 
            round(row['latitude'], 6), 
            row['Date'].date()
        ), axis=1)
    )
    print(f"Avalanche events to avoid: {len(avalanche_pairs):,}")
    
    # Generate negative samples
    print("\nGenerating negative examples...")
    negatives = []
    cache = {}  # Cache SNOTEL queries
    cache_hits = 0
    
    attempts = 0
    max_attempts = n_negatives * 10
    
    pbar = tqdm(total=n_negatives, desc="Generating negatives")
    
    while len(negatives) < n_negatives and attempts < max_attempts:
        attempts += 1
        
        # Sample random location
        location = unique_locations.sample(1).iloc[0]
        lon = location['Longitude']
        lat = location['latitude']
        area = location['Area']
        
        # Sample random date
        date_range_days = (max_date - min_date).days
        random_days = random.randint(0, date_range_days)
        sample_date = min_date + timedelta(days=random_days)
        
        # Check if this is an actual avalanche
        if (round(lon, 6), round(lat, 6), sample_date.date()) in avalanche_pairs:
            continue
        
        # Check if we already have this negative
        if any(n['Longitude'] == lon and n['latitude'] == lat and n['Date'] == sample_date for n in negatives):
            continue
        
        # Get terrain
        terrain = terrain_df[(terrain_df['Longitude'] == lon) & 
                             (terrain_df['latitude'] == lat)]
        if terrain.empty:
            continue
        
        # Find nearest SNOTEL stations
        nearest = find_nearest_stations(lon, lat, stations_df, n=3)
        if not nearest:
            continue
        
        # Get station triplets
        station_triplets = []
        distances = []
        for station_id, dist in nearest:
            station_row = stations_df[stations_df['station_id'] == int(station_id)]
            if len(station_row) > 0:
                triplet = station_row['station_triplet'].iloc[0]
                station_triplets.append(triplet)
                distances.append(dist)
        
        if not station_triplets:
            continue
        
        # Query weather from stations (with caching)
        snow_depths = []
        new_snows = []
        swes = []
        temps = []
        
        date_str = sample_date.strftime('%Y-%m-%d')
        
        for triplet in station_triplets:
            cache_key = (triplet, date_str)
            
            if cache_key in cache:
                station_data = cache[cache_key]
                cache_hits += 1
            else:
                # Query SNOTEL
                station_data = get_station_weather(triplet, sample_date.to_pydatetime())
                new_snow = calculate_24h_snow_change(triplet, sample_date.to_pydatetime())
                station_data['new_snow_24h'] = new_snow
                cache[cache_key] = station_data
            
            snow_depths.append(station_data['snow_depth'])
            new_snows.append(station_data.get('new_snow_24h'))
            swes.append(station_data['swe'])
            temps.append(station_data['temp'])
        
        # Apply IDW
        snow_depth_idw = inverse_distance_weighting(snow_depths, distances)
        new_snow_idw = inverse_distance_weighting(new_snows, distances)
        swe_idw = inverse_distance_weighting(swes, distances)
        temp_idw = inverse_distance_weighting(temps, distances)
        
        # Skip if no weather data available
        if snow_depth_idw is None:
            continue
        
        # Create negative example
        negative = {
            'Observation ID': f'NEG_{len(negatives):06d}',
            'Date': sample_date,
            'Longitude': lon,
            'latitude': lat,
            'Area': area,
            'Aspect': None,
            'Type': None,
            'Trigger': None,
            'avalanche_size': 0.0,
            'avalanche_occurred': 0,
            'elevation': terrain.iloc[0]['elevation'],
            'slope': terrain.iloc[0]['slope'],
            'aspect_degrees': terrain.iloc[0]['aspect_degrees'],
            'aspect_cardinal': terrain.iloc[0]['aspect_cardinal'],
            'snow_depth': snow_depth_idw,
            'new_snow_24h': new_snow_idw,
            'swe': swe_idw,
            'temp': temp_idw,
            'wind_speed': None,
            'nearest_stations': ','.join(station_triplets),
            'nearest_distances': ','.join([f"{d:.1f}" for d in distances])
        }
        
        negatives.append(negative)
        pbar.update(1)
    
    pbar.close()
    
    negatives_df = pd.DataFrame(negatives)
    
    print(f"\n✓ Generated {len(negatives_df):,} negative examples")
    print(f"  Attempts: {attempts:,}")
    print(f"  Cache hits: {cache_hits:,}")
    print(f"  Unique SNOTEL queries: {len(cache):,}")
    print(f"  Success rate: {len(negatives_df)/attempts*100:.1f}%")
    
    return negatives_df


def combine_and_save(positives_path, negatives_df):
    """Combine positive and negative examples into training dataset"""
    
    print("\n" + "="*60)
    print("CREATING FINAL TRAINING DATASET")
    print("="*60)
    
    # Load positives
    positives = pd.read_csv(positives_path)
    print(f"\nPositive examples: {len(positives):,}")
    print(f"Negative examples: {len(negatives_df):,}")
    
    # Combine
    training_data = pd.concat([positives, negatives_df], ignore_index=True)
    
    # Shuffle
    training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal training dataset: {len(training_data):,} rows")
    
    # Class balance
    class_counts = training_data['avalanche_occurred'].value_counts()
    print(f"\nClass distribution:")
    print(f"  No avalanche (0): {class_counts[0]:,} ({class_counts[0]/len(training_data)*100:.1f}%)")
    print(f"  Avalanche (1):    {class_counts[1]:,} ({class_counts[1]/len(training_data)*100:.1f}%)")
    
    # Weather completeness
    print(f"\nWeather data completeness (negatives):")
    print(f"  Snow depth:   {negatives_df['snow_depth'].notna().sum():,} ({negatives_df['snow_depth'].notna().sum()/len(negatives_df)*100:.1f}%)")
    print(f"  New snow 24h: {negatives_df['new_snow_24h'].notna().sum():,} ({negatives_df['new_snow_24h'].notna().sum()/len(negatives_df)*100:.1f}%)")
    print(f"  SWE:          {negatives_df['swe'].notna().sum():,} ({negatives_df['swe'].notna().sum()/len(negatives_df)*100:.1f}%)")
    print(f"  Temperature:  {negatives_df['temp'].notna().sum():,} ({negatives_df['temp'].notna().sum()/len(negatives_df)*100:.1f}%)")
    
    # Save
    training_data.to_csv(TRAINING_DATA, index=False)
    print(f"\n✓ Saved training data to: {TRAINING_DATA}")
    
    return training_data


def main():
    """Generate negatives and create final training dataset"""
    
    # Generate negatives (2:1 ratio)
    negatives_df = generate_negative_examples(ratio=2.0)
    
    # Combine with positives
    training_data = combine_and_save(POSITIVE_EXAMPLES, negatives_df)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTraining dataset ready!")
    print(f"  Total samples: {len(training_data):,}")
    print(f"  Features: {len(training_data.columns)}")
    print(f"  File: {TRAINING_DATA}")
    
    return training_data


if __name__ == "__main__":
    training_data = main()