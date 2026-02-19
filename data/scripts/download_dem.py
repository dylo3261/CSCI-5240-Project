"""
Download Colorado DEM using OpenTopography API
One-time download for offline terrain extraction
"""

import requests
from pathlib import Path
import os

# Your OpenTopography API key
API_KEY = os.getenv('OPENTOPO_API_KEY', 'api_key_here')

# Output path
DATA_DIR = Path(__file__).parent.parent / 'data'
EXTERNAL_DIR = DATA_DIR / 'external'
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

DEM_PATH = EXTERNAL_DIR / 'colorado_dem.tif'

# Colorado bounds
COLORADO_BOUNDS = {
    'south': 37.0,
    'north': 41.0,
    'west': -109.1,
    'east': -102.0
}

print("Downloading Colorado DEM from OpenTopography...")
print(f"Bounds: {COLORADO_BOUNDS}")
print(f"This may take 10-30 minutes depending on connection...")

url = "https://portal.opentopography.org/API/globaldem"

params = {
    'demtype': 'SRTMGL3',  # 90m resolution, smaller file
    'south': COLORADO_BOUNDS['south'],
    'north': COLORADO_BOUNDS['north'],
    'west': COLORADO_BOUNDS['west'],
    'east': COLORADO_BOUNDS['east'],
    'outputFormat': 'GTiff',
    'API_Key': API_KEY
}

try:
    response = requests.get(url, params=params, stream=True, timeout=300)
    response.raise_for_status()
    
    # Download with progress
    total_size = int(response.headers.get('content-length', 0))
    
    with open(DEM_PATH, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                f.write(chunk)
                done = int(50 * downloaded / total_size)
                print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB", end='')
    
    print(f"\n✓ DEM downloaded to: {DEM_PATH}")
    print(f"File size: {DEM_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nAlternative: Download manually from OpenTopography:")
    print("https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3")