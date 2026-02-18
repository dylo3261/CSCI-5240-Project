"""
Utility functions for weather data extraction from SNOTEL
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from metloom.pointdata import SnotelPointData
from metloom.variables import SnotelVariables


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate haversine distance between two points in kilometers
    
    Args:
        lon1, lat1: First point coordinates
        lon2, lat2: Second point coordinates
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    
    return c * r


def find_nearest_stations(lon: float, lat: float, 
                          stations_df: pd.DataFrame, 
                          n: int = 3) -> List[Tuple[str, float]]:
    """
    Find n nearest SNOTEL stations to a point
    
    Args:
        lon: Longitude of point
        lat: Latitude of point
        stations_df: DataFrame with columns ['station_id', 'longitude', 'latitude']
        n: Number of nearest stations to return
    
    Returns:
        List of tuples (station_id, distance_km)
    """
    # Calculate distances to all stations
    distances = []
    for _, station in stations_df.iterrows():
        dist = haversine_distance(lon, lat, station['longitude'], station['latitude'])
        distances.append((str(station['station_id']), dist))
    
    # Sort by distance and return top n
    distances.sort(key=lambda x: x[1])
    return distances[:n]


def inverse_distance_weighting(values: List[float], 
                                distances: List[float], 
                                power: float = 2.0) -> Optional[float]:
    """
    Calculate inverse distance weighted average
    
    Args:
        values: List of values from different stations
        distances: List of distances to those stations (in km)
        power: Power parameter for IDW (default 2 = inverse square)
    
    Returns:
        Weighted average value, or None if no valid values
    """
    # Filter out None values
    valid_pairs = [(v, d) for v, d in zip(values, distances) if v is not None and not np.isnan(v)]
    
    if not valid_pairs:
        return None
    
    values_valid, distances_valid = zip(*valid_pairs)
    
    # Handle case where station is exactly at the point (distance = 0)
    if any(d == 0 for d in distances_valid):
        # Return value from the station at distance 0
        idx = distances_valid.index(0)
        return values_valid[idx]
    
    # Calculate weights: 1 / distance^power
    weights = [1 / (d ** power) for d in distances_valid]
    total_weight = sum(weights)
    
    # Weighted average
    weighted_avg = sum(v * w for v, w in zip(values_valid, weights)) / total_weight
    
    return weighted_avg


def get_station_weather(station_triplet: str, 
                        date: datetime,
                        variables: List = None) -> Dict[str, Optional[float]]:
    """
    Get weather data from a SNOTEL station for a specific date
    
    Args:
        station_triplet: SNOTEL station triplet (e.g., "663:CO:SNTL")
        date: Date to query
        variables: List of variables to query
    
    Returns:
        Dictionary with variable values (in metric units)
    """
    if variables is None:
        variables = [
            SnotelVariables.SNOWDEPTH,
            SnotelVariables.SWE,
            SnotelVariables.TEMP,
        ]
    
    try:
        # Query data (go back a bit since SNOTEL has reporting lag)
        start_date = date - timedelta(days=3)
        end_date = date + timedelta(days=1)
        
        # Initialize SNOTEL point data with triplet
        point = SnotelPointData(station_triplet, "SNOTEL")
        
        # Get data
        df = point.get_daily_data(
            start_date=start_date,
            end_date=end_date,
            variables=variables
        )
        
        if df.empty:
            return {
                'snow_depth': None,
                'swe': None,
                'temp': None,
                'wind_speed': None
            }
        
        # Get data for the specific date (or closest available)
        target_date = pd.Timestamp(date.date())
        
        if target_date in df.index:
            row = df.loc[target_date]
        else:
            # Use most recent available date
            row = df.iloc[-1]
        
        # Extract values from known column names
        result = {
            'snow_depth': None,
            'swe': None,
            'temp': None,
            'wind_speed': None
        }
        
        # Snow depth (convert inches to cm: 1 inch = 2.54 cm)
        if 'SNOWDEPTH' in df.columns:
            value = row['SNOWDEPTH']
            if pd.notna(value) and value != '':
                result['snow_depth'] = float(value) * 2.54  # inches to cm
        
        # SWE (convert inches to cm)
        if 'SWE' in df.columns:
            value = row['SWE']
            if pd.notna(value) and value != '':
                result['swe'] = float(value) * 2.54  # inches to cm
        
        # Temperature (convert Fahrenheit to Celsius: C = (F - 32) * 5/9)
        if 'AIR TEMP' in df.columns:
            value = row['AIR TEMP']
            if pd.notna(value) and value != '':
                temp_f = float(value)
                result['temp'] = (temp_f - 32) * 5 / 9  # F to C
        
        return result
        
    except Exception as e:
        # Return None for all values if query fails
        return {
            'snow_depth': None,
            'swe': None,
            'temp': None,
            'wind_speed': None
        }


def calculate_24h_snow_change(station_triplet: str, 
                               date: datetime) -> Optional[float]:
    """
    Calculate 24-hour snow depth change
    
    Args:
        station_triplet: SNOTEL station triplet (e.g., "663:CO:SNTL")
        date: Date to calculate change for
    
    Returns:
        Snow depth change in cm (positive = accumulation)
    """
    try:
        # Query 4 days of data (account for reporting lag)
        start_date = date - timedelta(days=4)
        end_date = date
        
        point = SnotelPointData(station_triplet, "SNOTEL")
        df = point.get_daily_data(
            start_date=start_date,
            end_date=end_date,
            variables=[SnotelVariables.SNOWDEPTH]
        )
        
        if len(df) < 2 or 'SNOWDEPTH' not in df.columns:
            return None
        
        # Get snow depth for date and previous day
        target_date = pd.Timestamp(date.date())
        prev_date = pd.Timestamp((date - timedelta(days=1)).date())
        
        # If exact dates not available, use most recent two dates
        if target_date not in df.index or prev_date not in df.index:
            if len(df) >= 2:
                current = df['SNOWDEPTH'].iloc[-1]
                previous = df['SNOWDEPTH'].iloc[-2]
            else:
                return None
        else:
            current = df.loc[target_date, 'SNOWDEPTH']
            previous = df.loc[prev_date, 'SNOWDEPTH']
        
        # Convert to float and calculate change (inches to cm)
        if pd.notna(current) and pd.notna(previous) and current != '' and previous != '':
            current_cm = float(current) * 2.54
            previous_cm = float(previous) * 2.54
            return current_cm - previous_cm
        
        return None
        
    except Exception:
        return None