# Avalanche Risk Model

Machine learning model to predict avalanche probability and size across Colorado. Built for Big Data Architecture (CSCI5214) course at CU Boulder.

## Project Overview

Combines historical avalanche data with terrain and weather features to predict natural avalanche risk:
- **Avalanche data:** 9,541 unique natural avalanche events from Colorado Avalanche Information Center (CAIC)
- **Terrain features:** Elevation, slope, aspect from SRTM DEM data
- **Weather features:** Snow depth, temperature, SWE from 118 SNOTEL stations using inverse distance weighting

## Prerequisites

- Python 3.11
- ~50 MB disk space (DEM file)
- Internet connection for SNOTEL API queries

## Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/avalanche-risk-model.git
cd avalanche-risk-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Note:** All dependencies work on macOS (including Apple Silicon), Linux, and Windows.

## Data Pipeline

### Step 1: Load and Clean CAIC Data ✅
```bash
python scripts/01_load_and_clean_caic.py
```

- Loads raw CAIC avalanche data
- Filters to natural/unknown triggers only
- Removes invalid coordinates and missing critical fields
- **Output:** `data/processed/caic_clean.csv` (15,826 observations)

### Step 2: Build SNOTEL Station List ✅
```bash
python scripts/build_snotel_stations.py
```

- Queries NRCS AWDB REST API for all Colorado SNOTEL stations
- Validates coordinates within Colorado bounds
- **Output:** `data/processed/snotel_stations.csv` (118 stations)

### Step 3: Download DEM ✅
```bash
python scripts/download_dem.py
```

- Downloads SRTM GL3 (90m) DEM for Colorado from OpenTopography
- Requires OpenTopography API key (free)
- **Output:** `data/external/colorado_dem.tif` (~48 MB)

### Step 4: Extract Terrain Features ✅
```bash
python scripts/02_extract_terrain_dem.py
```

- Extracts elevation, slope, aspect from local DEM for 6,080 unique locations
- Uses Horn's method for slope/aspect calculation
- Converts geographic coordinates to proper metric units
- **Runtime:** ~1 minute
- **Output:** `data/processed/terrain_cache.csv` (100% success rate)

### Step 5: Fetch Weather Data ✅
```bash
python scripts/03_fetch_weather.py
```

- Queries SNOTEL API for 15,826 avalanche observations
- Uses 3-station inverse distance weighting (IDW) for spatial interpolation
- Fetches: snow depth, 24h snow change, SWE, temperature
- Implements intelligent caching to minimize API calls
- **Runtime:** ~13 hours
- **Output:** `data/processed/weather_cache.csv` (100% completeness!)

### Step 6: Combine Features ✅
```bash
python scripts/04_combine_features.py
```

- Merges CAIC + terrain + weather data
- Creates unique observation IDs based on location + timestamp
- Removes duplicate observations (59,114 → 9,541 unique events)
- **Output:** `data/processed/positive_examples_enriched.csv` (9,541 unique avalanches)

### Step 7: Generate Negative Examples 🔄
```bash
python scripts/05_generate_negatives.py
```

- Samples safe days (no avalanches) from avalanche terrain
- Queries SNOTEL for actual weather conditions on those dates
- Uses same 3-station IDW interpolation as positive examples
- Creates 2:1 ratio (negatives:positives)
- **Runtime:** ~60 hours estimated
- **Output:** `data/processed/training_data.csv` (~28,623 total rows)

### Step 8: Train Model ⏳

- Train multi-output model (avalanche probability + size prediction)
- Evaluate on test set with cross-validation
- Feature importance analysis
- Save trained model for deployment

## Project Structure
```
avalanche-risk-model/
│
├── data/
│   ├── raw/                           # Original CAIC CSV (not in git)
│   ├── processed/                     # Processed datasets
│   │   ├── caic_clean.csv            # ✓ 15,826 avalanche obs
│   │   ├── snotel_stations.csv       # ✓ 118 stations
│   │   ├── terrain_cache.csv         # ✓ 6,080 locations
│   │   ├── weather_cache.csv         # ✓ 15,826 obs (100% complete)
│   │   ├── positive_examples_enriched.csv  # ✓ 9,541 unique avalanches
│   │   └── training_data.csv         # 🔄 In progress (~28k rows)
│   └── external/
│       └── colorado_dem.tif          # ✓ 48 MB DEM (not in git)
│
├── scripts/
│   ├── 01_load_and_clean_caic.py     # ✓ Data cleaning
│   ├── 02_extract_terrain_dem.py     # ✓ Terrain extraction
│   ├── 03_fetch_weather.py           # ✓ Weather fetching (IDW)
│   ├── 04_combine_features.py        # ✓ Feature combination
│   ├── 05_generate_negatives.py      # 🔄 Negative sampling
│   ├── build_snotel_stations.py      # ✓ Station list builder
│   ├── download_dem.py               # ✓ DEM downloader
│   └── utils/
│       ├── terrain_utils.py          # DEM processing + slope calculation
│       └── weather_utils.py          # SNOTEL query + IDW functions
│
├── notebooks/
│   └── 01_eda_caic_avalanches.ipynb  # ✓ Exploratory data analysis
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Current Progress

- ✅ **Data collection:** CAIC avalanche data loaded and cleaned (15,826 observations)
- ✅ **EDA:** Comprehensive analysis of temporal, geographic, and avalanche patterns
- ✅ **SNOTEL stations:** 118 Colorado stations queried from NRCS AWDB API
- ✅ **DEM download:** SRTM 90m elevation data acquired (48 MB)
- ✅ **Terrain extraction:** Elevation, slope, aspect computed for 6,080 locations (100% success)
- ✅ **Weather fetching:** SNOTEL data extracted for all avalanches (100% completeness, 13 hours)
- ✅ **Feature combination:** All features merged, duplicates removed (9,541 unique avalanches)
- 🔄 **Negative sampling:** Generating safe-day examples with real SNOTEL queries (~60 hours)
- ⏳ **Model training:** Planned after negative sampling completes

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Positive Examples (Avalanches)** | |
| Total observations | 15,826 raw → 9,541 unique |
| Date range | 2016-02-06 to 2026-02-13 (10 years) |
| Unique locations | 6,080 coordinates |
| Unique dates | 1,661 days |
| CAIC regions | 10 forecast zones |
| Peak season | December - March |
| Avalanche sizes | D1.0 - D5.0 (mean: D1.74) |
| **Terrain Features** | |
| Elevation range | 1,515 - 4,385 m (mean: 3,482 m) |
| Slope range | 0 - 60° (mean: 33°) |
| Data completeness | 100% |
| **Weather Features** | |
| SNOTEL stations used | 118 across Colorado |
| Snow depth | 100% complete (mean: 106 cm) |
| New snow (24h) | 100% complete (mean: +4.9 cm) |
| SWE | 100% complete (mean: 27 cm) |
| Temperature | 100% complete (mean: -9.6°C) |
| **Training Dataset** | |
| Positive examples | 9,541 (33%) |
| Negative examples | ~19,082 target (67%) |
| Total samples | ~28,623 |

## Key Technical Decisions

**Inverse Distance Weighting (IDW):**
- Uses 3 nearest SNOTEL stations per location
- Power parameter = 2 (inverse square weighting)
- Handles sparse weather station coverage across Colorado mountains
- Gracefully handles missing data from individual stations

**Terrain Extraction:**
- SRTM GL3 (90m resolution) downloaded via OpenTopography API
- Uses rasterio for DEM reading and numpy for calculations
- Slope calculated using Horn's method with proper lat/lon → meters conversion
- Geographic coordinates converted to metric: 111,320m per degree latitude, varies by latitude for longitude
- Aspect converted to both degrees (0-360°) and cardinal directions (N, NE, E, etc.)

**Data Deduplication:**
- CAIC observation IDs are not unique (reused across events)
- Created unique IDs based on: `longitude_latitude_datetime`
- Reduced from 59,114 rows to 9,541 unique avalanche events
- Preserves spatial-temporal uniqueness

**Negative Sampling Strategy:**
- Uses same avalanche terrain locations (prevents location bias)
- Samples different dates where no avalanche occurred
- Queries real SNOTEL data for those dates (not synthetic)
- Model learns conditions that lead to avalanches vs. safe conditions at same locations

**Data Filtering:**
- Only natural (N) and unknown (U) triggers included
- Human-triggered avalanches excluded (not relevant for natural avalanche forecasting)
- Invalid coordinates and missing critical fields removed
- Recent dates (last 10 days) excluded from negatives due to SNOTEL reporting lag

## Known Issues

- **SNOTEL API:** Occasional timeouts and missing data - script uses try-catch and caching
- **DEM file size:** 48 MB - excluded from git (download script provided)
- **Weather data lag:** SNOTEL reports with 1-3 day delay (not real-time)
- **Long runtimes:** Weather fetching takes 13+ hours, negative sampling ~60 hours
- **Wind speed:** Not consistently available from SNOTEL, excluded from model

## Team Collaboration

**Before starting:**
1. Use virtual environment for dependency isolation
2. Never commit: large data files, API keys, raw CAIC CSV, DEM files
3. Run scripts from project root: `python scripts/script_name.py`
4. Check `.gitignore` before committing new files

**Git workflow:**
```bash
# Pull latest changes
git pull origin main

# Make your changes, then:
git add .
git commit -m "Description of changes"
git push origin main
```

## API Keys

**OpenTopography API:**
- Required for DEM download
- Get free key: https://opentopography.org/
- Set in `scripts/download_dem.py` or use environment variable:
```bash
export OPENTOPO_API_KEY="your_key_here"
```

**SNOTEL/Metloom:**
- No API key required
- Uses NRCS public AWDB web service
- Station triplet format: `{id}:CO:SNTL` (e.g., "663:CO:SNTL")

## Resources

- [CAIC Avalanche Database](https://avalanche.state.co.us/)
- [SNOTEL Network](https://www.nrcs.usda.gov/wps/portal/wcc/home/snowClimateMonitoring/snowpack/)
- [OpenTopography](https://opentopography.org/)
- [Metloom Documentation](https://metloom.readthedocs.io/)
- [NRCS AWDB API](https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html)

## License

Educational use only - CU Boulder CSCI5214 class project.