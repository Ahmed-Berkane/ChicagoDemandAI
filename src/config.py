from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default modeling window (extend via --start on CLI)
HISTORY_START = date(2019, 1, 1)
DATA_DIR = PROJECT_ROOT / "data"

# Raw inputs stay local; the final modeling table lives under data/processed/ (tracked in git).
PROCESSED_DATA_DIR = DATA_DIR / "processed"
UNIFIED_DATASET_PARQUET_FILE = PROCESSED_DATA_DIR / "unified_daily_demand.parquet"

# Downtown Chicago (matches city crime/event geography)
CHICAGO_LAT = 41.8781
CHICAGO_LON = -87.6298
CHICAGO_TIMEZONE = "America/Chicago"

DATASETS = {
    "business_licenses": "Business_Licenses_-_Current_Active_20260517.csv",
    "food_inspections": "Food_Inspections_20260517.csv",
    "crimes_one_year": "Crimes_-_One_year_prior_to_present_20260517.csv",
    "crimes_full": "Crimes_-_2001_to_Present_20260517.csv",
    "special_events": "Special_Events_20260517.csv",
}

CHICAGO_PERMITS_API = "https://data.cityofchicago.org/resource/ydr8-5enu.json"
CHICAGO_CTA_RIDERSHIP_API = "https://data.cityofchicago.org/resource/6iiy-9s97.json"
# Official portal dataset — only publishes *upcoming* events (~Oct 2025+), not historical 2019
CHICAGO_SPECIAL_EVENTS_API = "https://data.cityofchicago.org/resource/xgse-8eg7.json"
MAJOR_EVENTS_FILE = DATA_DIR / "major_chicago_events.csv"
OPEN_METEO_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
NASA_POWER_DAILY_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
# Optional offline cache (created after first successful fetch; speeds up rebuilds).
WEATHER_REFERENCE_FILE = DATA_DIR / "weather_chicago_daily.csv"

MLB_STATS_API = "https://statsapi.mlb.com/api/v1/schedule"
MLB_CUBS_TEAM_ID = 112
ESPN_CUBS_TEAM_ID = 16
ESPN_BULLS_TEAM_ID = 4

# Synthetic demand defaults (coffee-shop MVP)
SYNTHETIC_BASE_TRAFFIC = 120
SYNTHETIC_RANDOM_SEED = 42

# License descriptions / activities useful for coffee-shop MVP filtering
COFFEE_SHOP_LICENSE_KEYWORDS = (
    "coffee",
    "cafe",
    "café",
    "espresso",
    "tea house",
    "bakery",
)
