"""
Terrain feature extraction from Colorado DEM.

Samples elevation at a (lat, lon) point, then derives slope and aspect
using Horn's method on a 3x3 window — identical to 02_extract_terrain_dem.py.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import rowcol
from utils.s3_helpers import (
    IS_LAMBDA, DATA_DIR, S3_DEM_KEY,
    download_model_file,
)

logger = logging.getLogger(__name__)

DEM_FILE = os.path.join(DATA_DIR, "colorado_dem.tif")

# Cache: keep DEM bytes in memory across warm invocations
_dem_bytes = None


def _load_dem() -> bytes:
    """Load DEM into memory (from disk cache or S3)."""
    global _dem_bytes
    if _dem_bytes is not None:
        return _dem_bytes

    if IS_LAMBDA or not os.path.exists(DEM_FILE):
        download_model_file(S3_DEM_KEY, DEM_FILE)

    with open(DEM_FILE, "rb") as f:
        _dem_bytes = f.read()

    logger.info(f"DEM loaded into memory ({len(_dem_bytes) / 1e6:.1f} MB)")
    return _dem_bytes


def calculate_slope_aspect(elevation_array, transform, row, col, lat):
    """
    Calculate slope and aspect from DEM using Horn's method.
    Directly from 02_extract_terrain_dem.py.
    """
    try:
        cell_size_deg = abs(transform[0])

        lat_rad = np.radians(lat)
        meters_per_deg_lon = 111320 * np.cos(lat_rad)
        meters_per_deg_lat = 111320
        cell_size_m = (meters_per_deg_lon + meters_per_deg_lat) / 2 * cell_size_deg

        if (row < 1 or row >= elevation_array.shape[0] - 1 or
                col < 1 or col >= elevation_array.shape[1] - 1):
            return None, None

        window = elevation_array[row - 1:row + 2, col - 1:col + 2]

        if np.any(np.isnan(window)) or np.any(window == -32768):
            return None, None

        dz_dx = ((window[0, 2] + 2 * window[1, 2] + window[2, 2]) -
                 (window[0, 0] + 2 * window[1, 0] + window[2, 0])) / (8 * cell_size_m)
        dz_dy = ((window[2, 0] + 2 * window[2, 1] + window[2, 2]) -
                 (window[0, 0] + 2 * window[0, 1] + window[0, 2])) / (8 * cell_size_m)

        slope_degrees = float(np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))))

        aspect_degrees = float(np.degrees(np.arctan2(dz_dy, -dz_dx)))
        if aspect_degrees < 0:
            aspect_degrees = 90.0 - aspect_degrees
        elif aspect_degrees > 90.0:
            aspect_degrees = 360.0 - aspect_degrees + 90.0
        else:
            aspect_degrees = 90.0 - aspect_degrees

        return slope_degrees, aspect_degrees

    except Exception:
        return None, None


def extract_terrain(lat: float, lon: float) -> dict:
    """
    Sample elevation, slope, aspect at a (lat, lon) point from the Colorado DEM.

    Returns:
        {"elevation": float, "slope": float, "aspect_degrees": float}

    Raises:
        ValueError if point is outside DEM or data is missing.
    """
    dem_data = _load_dem()

    with MemoryFile(dem_data) as memfile:
        with memfile.open() as src:
            py, px = rowcol(src.transform, lon, lat)

            if py < 0 or py >= src.height or px < 0 or px >= src.width:
                raise ValueError(
                    f"Point ({lat}, {lon}) is outside the DEM coverage area. "
                    f"DEM bounds: {src.bounds}"
                )

            elevation_data = src.read(1)

            elevation = float(elevation_data[py, px])
            if elevation == src.nodata or elevation < -999:
                raise ValueError(f"No elevation data at ({lat}, {lon})")

            slope, aspect_deg = calculate_slope_aspect(
                elevation_data, src.transform, py, px, lat
            )

            if slope is None:
                raise ValueError(
                    f"Location ({lat}, {lon}) is at the edge of the DEM or in a "
                    f"no-data zone where slope/aspect cannot be calculated. "
                    f"Elevation at this point is {elevation:.0f}m. "
                    f"Try a location further inside Colorado's mountain terrain."
                )

            return {
                "elevation": round(elevation, 1),
                "slope": round(slope, 2),
                "aspect_degrees": round(aspect_deg, 2),
            }