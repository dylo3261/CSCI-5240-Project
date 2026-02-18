"""
Utility functions for terrain feature extraction using OpenTopography API
"""
import numpy as np
import requests
import rasterio
from rasterio.io import MemoryFile
from typing import Tuple, Optional
import time

# Get your API key from: https://opentopography.org/
OPENTOPO_API_KEY = "9e0bf5f011976ecb305509672566a156"  # Replace with your own key


def get_terrain_from_opentopo(lon: float, lat: float, 
                                buffer_deg: float = 0.005) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Get elevation, slope, and aspect from OpenTopography API
    """
    
    # Create bounding box around point
    south = lat - buffer_deg
    north = lat + buffer_deg
    west = lon - buffer_deg
    east = lon + buffer_deg
    
    # API endpoint
    url = "https://portal.opentopography.org/API/globaldem"
    
    params = {
        'demtype': 'SRTMGL3',  # Changed to GL3 for larger quota
        'south': south,
        'north': north,
        'west': west,
        'east': east,
        'outputFormat': 'GTiff',
        'API_Key': OPENTOPO_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        # Log status code for debugging (temporary)
        if response.status_code != 200:
            # Only print first 10 errors to avoid spam
            import random
            if random.random() < 0.01:  # Print ~1% of errors
                print(f"Status {response.status_code} for ({lat:.4f}, {lon:.4f})")
        
        # Check for specific error codes
        if response.status_code in [400, 401, 204]:
            return None, None, None
        
        response.raise_for_status()
        
        # Read GeoTIFF from response
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                # Read elevation data
                elevation_array = dataset.read(1)
                
                # Get center pixel (our point of interest)
                center_row = elevation_array.shape[0] // 2
                center_col = elevation_array.shape[1] // 2
                elevation = float(elevation_array[center_row, center_col])
                
                # Handle no-data values
                if elevation == dataset.nodata or np.isnan(elevation):
                    return None, None, None
                
                # Calculate slope and aspect from the elevation patch
                slope, aspect = calculate_slope_aspect_from_array(
                    elevation_array, 
                    dataset.transform,
                    center_row, 
                    center_col
                )
                
                return elevation, slope, aspect
                
    except requests.exceptions.RequestException:
        return None, None, None
    except Exception:
        return None, None, None


def calculate_slope_aspect_from_array(elevation_array: np.ndarray, 
                                        transform, 
                                        row: int, 
                                        col: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate slope and aspect from elevation array using gradient method
    
    Args:
        elevation_array: 2D array of elevation values
        transform: Rasterio affine transform
        row: Row index of center point
        col: Column index of center point
    
    Returns:
        Tuple of (slope_degrees, aspect_degrees)
    """
    try:
        # Get cell size from transform
        cell_size = abs(transform[0])  # meters per pixel
        
        # Get 3x3 window around center point
        if row < 1 or row >= elevation_array.shape[0] - 1 or \
           col < 1 or col >= elevation_array.shape[1] - 1:
            return None, None
        
        window = elevation_array[row-1:row+2, col-1:col+2]
        
        # Check for no-data values in window
        if np.any(np.isnan(window)):
            return None, None
        
        # Calculate gradients using Horn's method (same as GDAL)
        dz_dx = ((window[0, 2] + 2*window[1, 2] + window[2, 2]) - 
                 (window[0, 0] + 2*window[1, 0] + window[2, 0])) / (8 * cell_size)
        
        dz_dy = ((window[2, 0] + 2*window[2, 1] + window[2, 2]) - 
                 (window[0, 0] + 2*window[0, 1] + window[0, 2])) / (8 * cell_size)
        
        # Calculate slope in degrees
        slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_degrees = np.degrees(slope_radians)
        
        # Calculate aspect in degrees (0-360, 0=North, 90=East)
        aspect_radians = np.arctan2(dz_dy, -dz_dx)
        aspect_degrees = np.degrees(aspect_radians)
        
        # Convert to compass bearing (0-360)
        if aspect_degrees < 0:
            aspect_degrees = 90.0 - aspect_degrees
        elif aspect_degrees > 90.0:
            aspect_degrees = 360.0 - aspect_degrees + 90.0
        else:
            aspect_degrees = 90.0 - aspect_degrees
        
        return float(slope_degrees), float(aspect_degrees)
        
    except Exception:
        return None, None


def aspect_degrees_to_cardinal(degrees: float) -> str:
    """
    Convert aspect in degrees to cardinal direction
    
    Args:
        degrees: Aspect in degrees (0-360)
    
    Returns:
        Cardinal direction (N, NE, E, SE, S, SW, W, NW)
    """
    if degrees is None or np.isnan(degrees):
        return None
    
    # Normalize to 0-360
    degrees = degrees % 360
    
    # Convert to 8 cardinal directions
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = int((degrees + 22.5) / 45) % 8
    
    return directions[index]