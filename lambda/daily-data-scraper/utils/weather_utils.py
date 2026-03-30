import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from metloom.pointdata import SnotelPointData
from metloom.variables import SnotelVariables

logger = logging.getLogger(__name__)


def get_station_weather(station_triplet: str, 
                        date: datetime,
                        variables: list = None) -> Dict[str, Optional[float]]:
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
        }
        
        # Snow depth (convert inches to cm: 1 inch = 2.54 cm)
        if 'SNOWDEPTH' in df.columns:
            value = row['SNOWDEPTH']
            if pd.notna(value) and value != '':
                result['snow_depth'] = round(float(value) * 2.54, 3)  # inches to cm
        
        # SWE (convert inches to cm)
        if 'SWE' in df.columns:
            value = row['SWE']
            if pd.notna(value) and value != '':
                result['swe'] = round(float(value) * 2.54, 3)  # inches to cm
        
        # Temperature (convert Fahrenheit to Celsius: C = (F - 32) * 5/9)
        if 'AIR TEMP' in df.columns:
            value = row['AIR TEMP']
            if pd.notna(value) and value != '':
                temp_f = float(value)
                result['temp'] = round((temp_f - 32) * 5 / 9, 3)  # F to C
        
        return result
        
    except Exception as e:
        logger.debug(f"Failed to fetch weather for {station_triplet} on {date.date()}: {e}")
        return {
            'snow_depth': None,
            'swe': None,
            'temp': None,
        }