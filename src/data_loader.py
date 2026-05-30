"""Load Chicago Data Portal CSVs for the demand-intelligence pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import COFFEE_SHOP_LICENSE_KEYWORDS, DATASETS, RAW_DATA_DIR

DatasetName = Literal[
    "business_licenses",
    "food_inspections",
    "crimes",
    "special_events",
]


def _path(name: str) -> Path:
    return RAW_DATA_DIR / DATASETS[name]


def load_business_licenses(*, coffee_shops_only: bool = False) -> pd.DataFrame:
    """Active business licenses with geocoded locations."""
    df = pd.read_csv(
        _path("business_licenses"),
        parse_dates=["APPLICATION CREATED DATE", "DATE ISSUED", "LICENSE TERM START DATE"],
        low_memory=False,
    )
    if coffee_shops_only:
        pattern = "|".join(COFFEE_SHOP_LICENSE_KEYWORDS)
        mask = (
            df["LICENSE DESCRIPTION"].str.contains(pattern, case=False, na=False)
            | df["DOING BUSINESS AS NAME"].str.contains(pattern, case=False, na=False)
            | df["BUSINESS ACTIVITY"].str.contains(pattern, case=False, na=False)
        )
        df = df.loc[mask].copy()
    return df


def load_food_inspections() -> pd.DataFrame:
    """Restaurant / food-service inspection history (activity proxy)."""
    return pd.read_csv(
        _path("food_inspections"),
        parse_dates=["Inspection Date"],
        low_memory=False,
    )


def load_crimes(*, full_history: bool = False) -> pd.DataFrame:
    """
    Crime reports for demand-shock features.

    Defaults to the one-year extract (~50 MB). Set full_history=True for
    the 2001–present file (~470 MB).
    """
    key = "crimes_full" if full_history else "crimes_one_year"
    df = pd.read_csv(_path(key), low_memory=False)

    if full_history:
        df = df.rename(
            columns={
                "Date": "date",
                "Primary Type": "primary_type",
                "Latitude": "latitude",
                "Longitude": "longitude",
                "Community Area": "community_area",
                "Ward": "ward",
            }
        )
        date_col = "date"
    else:
        df = df.rename(
            columns={
                "DATE  OF OCCURRENCE": "date",
                " PRIMARY DESCRIPTION": "primary_type",
                "LATITUDE": "latitude",
                "LONGITUDE": "longitude",
                "WARD": "ward",
            }
        )
        date_col = "date"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def expand_special_events_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Keep event_name, intensity, demand_type, category; one row per day from start_date to end_date."""
    out = df.copy()
    out["start_date"] = pd.to_datetime(out["start_date"])
    out["end_date"] = pd.to_datetime(out["end_date"])
    out = out[["event_name", "intensity", "demand_type", "category", "start_date", "end_date"]]
    out["date"] = out.apply(
        lambda r: pd.date_range(r["start_date"], r["end_date"], freq="D").tolist(),
        axis=1,
    )
    daily = (
        out.explode("date")[["event_name", "intensity", "demand_type", "category", "date"]]
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily


def load_special_events(*, daily: bool = False) -> pd.DataFrame:
    """City special events (festivals, street closures, etc.)."""
    df = pd.read_csv(_path("special_events"))
    if "start_date" in df.columns and "end_date" in df.columns:
        if daily:
            return expand_special_events_daily(df)
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_all(
    *,
    crimes_full_history: bool = False,
    coffee_shops_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load every Chicago portal dataset used in Phase 1."""
    return {
        "business_licenses": load_business_licenses(coffee_shops_only=coffee_shops_only),
        "food_inspections": load_food_inspections(),
        "crimes": load_crimes(full_history=crimes_full_history),
        "special_events": load_special_events(),
    }


def summarize(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Quick row/column/memory overview after loading."""
    rows = []
    for name, df in datasets.items():
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "columns": len(df.columns),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            }
        )
    return pd.DataFrame(rows)
