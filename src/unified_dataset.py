"""Build the roadmap unified daily ML table (one row per day, demo coffee shop)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.business_features import (
    business_static_columns,
    daily_inspection_features,
    select_demo_business,
)
from src.city_features import build_city_daily_features
from src.config import UNIFIED_DATASET_PARQUET_FILE
from src.data_loader import load_crimes
from src.events_data import fetch_bulls_home_games, fetch_cubs_home_games
from src.external_data import (
    default_date_range,
    fetch_building_permits,
    fetch_calendar_features,
    fetch_weather,
)
from src.pay_cycle import add_pay_cycle_features
from src.synthetic_demand import generate_synthetic_traffic
from src.transit_data import fetch_cta_ridership


def build_unified_daily_dataset(
    start: date | str | None = None,
    end: date | str | None = None,
    *,
    license_number: int | str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Join weather, calendar, pay cycles, sports, CTA, city aggregates,
    business license, food inspections, and synthetic target.

    Each row = one calendar day for one real licensed Chicago coffee shop.
    """
    if start is None or end is None:
        default_start, default_end = default_date_range()
        start = start or default_start
        end = end or default_end

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    spine = pd.DataFrame({"date": pd.date_range(start_ts, end_ts, freq="D")})

    weather = fetch_weather(start_ts, end_ts)
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    weather = weather[(weather["date"] >= start_ts) & (weather["date"] <= end_ts)]

    calendar = fetch_calendar_features(start_ts, end_ts)
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.normalize()
    calendar = calendar[(calendar["date"] >= start_ts) & (calendar["date"] <= end_ts)]
    calendar = add_pay_cycle_features(calendar)

    try:
        cubs = fetch_cubs_home_games(start_ts, end_ts)
    except Exception:
        cubs = pd.DataFrame(columns=["date"])

    try:
        bulls = fetch_bulls_home_games(start_ts, end_ts)
    except Exception:
        bulls = pd.DataFrame(columns=["date"])

    use_full_crimes = (end_ts - start_ts).days > 400
    crimes = load_crimes(full_history=use_full_crimes)
    crimes = crimes[(crimes["date"] >= start_ts) & (crimes["date"] <= end_ts)]

    try:
        permits = fetch_building_permits(start_ts, end_ts)
    except Exception:
        permits = pd.DataFrame(columns=["issue_date"])

    city = build_city_daily_features(crimes=crimes, permits=permits, cubs=cubs, bulls=bulls)
    city["date"] = pd.to_datetime(city["date"]).dt.normalize()

    try:
        cta = fetch_cta_ridership(start_ts, end_ts)
        cta["date"] = pd.to_datetime(cta["date"]).dt.normalize()
        cta = cta[(cta["date"] >= start_ts) & (cta["date"] <= end_ts)]
    except Exception:
        cta = pd.DataFrame(columns=["date", "cta_total_rides"])

    business = select_demo_business(license_number=license_number)
    inspections_daily = daily_inspection_features(
        start_ts, end_ts, business["LICENSE NUMBER"]
    )

    df = spine.merge(weather, on="date", how="left")
    df = df.merge(calendar, on="date", how="left", suffixes=("", "_cal"))
    df = df.merge(city, on="date", how="left")
    df = df.merge(cta, on="date", how="left", suffixes=("", "_cta"))
    df = df.merge(inspections_daily, on="date", how="left")

    for key, value in business_static_columns(business).items():
        df[key] = value
    df["day_of_week"] = df["date"].dt.day_name()

    for col in ("cubs_home_game", "bulls_home_game", "is_major_festival"):
        if col not in df.columns:
            df[col] = False
        else:
            df[col] = df[col].fillna(False).infer_objects(copy=False).astype(bool)
    for col in ("cubs_home_games", "bulls_home_games"):
        if col not in df.columns:
            df[col] = 0
    for col in (
        "crime_count",
        "permit_count",
        "city_special_events",
        "inspections_count",
        "inspections_pass_count",
        "inspections_fail_count",
        "inspections_7d_count",
    ):
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df = generate_synthetic_traffic(df)

    if save:
        UNIFIED_DATASET_PARQUET_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(UNIFIED_DATASET_PARQUET_FILE, index=False)

    return df


def load_unified_daily_dataset() -> pd.DataFrame:
    if UNIFIED_DATASET_PARQUET_FILE.exists():
        return pd.read_parquet(UNIFIED_DATASET_PARQUET_FILE)

    raise FileNotFoundError(
        f"Modeling data missing. Run: python scripts/build_unified_dataset.py"
    )


def load_modeling_data() -> pd.DataFrame:
    """Load the final training table as ``data`` (alias for notebooks)."""
    return load_unified_daily_dataset()
