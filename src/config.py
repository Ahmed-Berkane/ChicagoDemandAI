from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_DIR = PROJECT_ROOT / "DataSourceCSV"

UNIFIED_PARQUET = DATA_DIR / "unified_df.parquet"
TRAIN_PARQUET = DATA_DIR / "train_df.parquet"
VAL_PARQUET = DATA_DIR / "val_df.parquet"
TEST_PARQUET = DATA_DIR / "test_df.parquet"
REGION_MAP_PARQUET = DATA_DIR / "region_map.parquet"
FEATURE_MANIFEST = DATA_DIR / "feature_manifest.json"

WEATHER_REFERENCE_FILE = RAW_DATA_DIR / "weather_chicago_daily.csv"
EVENTS_REFERENCE_FILE = RAW_DATA_DIR / "chicago_events_2023_2025_expanded.csv"

CHICAGO_LAT = 41.8781
CHICAGO_LON = -87.6298
CHICAGO_TIMEZONE = "America/Chicago"
OPEN_METEO_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
CHICAGO_SPECIAL_EVENTS_API = "https://data.cityofchicago.org/resource/xgse-8eg7.json"
CHICAGO_CTA_RIDERSHIP_API = "https://data.cityofchicago.org/resource/6iiy-9s97.json"
MLB_STATS_API = "https://statsapi.mlb.com/api/v1/schedule"
MLB_CUBS_TEAM_ID = 112
ESPN_CUBS_TEAM_ID = 16
ESPN_BULLS_TEAM_ID = 4

FUTURE_FORECAST_DAYS = 14
FUTURE_MAX_HORIZON_DAYS = 14
PAST_LOOKBACK_DAYS = 90

TARGET = "customer_traffic"
RANDOM_STATE = 42

NUM_COLS = [
    "year",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "city_special_events",
    "temperature_f",
    "precipitation_in",
    "snowfall_in",
    "humidity_pct",
    "wind_speed_mph",
    "cta_total_rides",
    "traffic_lag1",
    "traffic_lag7",
    "traffic_roll7",
]

BOOL_COLS = [
    "cubs_home_game",
    "is_weekend",
    "bulls_home_game",
    "is_holiday",
    "is_payweek",
    "is_major_festival",
    "is_semimonthly_payday",
]

CAT_COLS = ["demand_type", "region", "business_category"]

SERIES_COLS = ["COMMUNITY AREA", "demand_type", "business_category"]
LAG_COLS = ["traffic_lag1", "traffic_lag7", "traffic_roll7"]
