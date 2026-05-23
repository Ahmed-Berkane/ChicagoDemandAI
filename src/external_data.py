"""Fetch external data for the unified dataset build."""

from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests

from src.config import (
    CHICAGO_LAT,
    CHICAGO_LON,
    CHICAGO_PERMITS_API,
    CHICAGO_TIMEZONE,
    HISTORY_START,
    OPEN_METEO_ARCHIVE_API,
)

# Roadmap weather fields → Open-Meteo daily archive (free; no API key)
WEATHER_DAILY_PARAMS = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
)


def _wind_chill_f(temp_f: float, wind_mph: float) -> float | None:
    """NWS wind chill; returns None when not applicable."""
    if pd.isna(temp_f) or pd.isna(wind_mph) or temp_f > 50 or wind_mph <= 3:
        return None
    v = wind_mph**0.16
    return (
        35.74
        + 0.6215 * temp_f
        - 35.75 * v
        + 0.4275 * temp_f * v
    )


def fetch_weather(
    start: date | str,
    end: date | str,
    *,
    latitude: float = CHICAGO_LAT,
    longitude: float = CHICAGO_LON,
) -> pd.DataFrame:
    """
    Daily Chicago weather from Open-Meteo historical archive.

    Covers roadmap NOAA-style fields: temperature, precipitation, snowfall,
    humidity, and computed wind chill.
    """
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_s,
        "end_date": end_s,
        "daily": ",".join(WEATHER_DAILY_PARAMS),
        "timezone": CHICAGO_TIMEZONE,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "snowfall_unit": "inch",
    }
    
    # Retry with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(OPEN_METEO_ARCHIVE_API, params=params, timeout=180)
            resp.raise_for_status()
            daily = resp.json()["daily"]
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                print(f"Weather API timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    df = pd.DataFrame(daily)
    df = df.rename(
        columns={
            "time": "date",
            "temperature_2m_mean": "temperature_f",
            "temperature_2m_max": "temperature_max_f",
            "temperature_2m_min": "temperature_min_f",
            "precipitation_sum": "precipitation_in",
            "snowfall_sum": "snowfall_in",
            "relative_humidity_2m_mean": "humidity_pct",
            "wind_speed_10m_max": "wind_speed_mph",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["wind_chill_f"] = df.apply(
        lambda r: _wind_chill_f(r["temperature_f"], r["wind_speed_mph"]),
        axis=1,
    )
    return df


def _fetch_permits_page(
    *,
    where: str | None,
    limit: int,
    offset: int,
    max_retries: int = 5,
) -> list[dict]:
    params: dict[str, str | int] = {"$limit": limit, "$offset": offset}
    if where:
        params["$where"] = where

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(CHICAGO_PERMITS_API, params=params, timeout=180)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Permits API failed after {max_retries} retries") from last_error


def fetch_building_permits(
    start: date | str | None = None,
    end: date | str | None = None,
    *,
    page_size: int = 5_000,
) -> pd.DataFrame:
    """Building permits from Chicago Data Portal (Socrata API, dataset ydr8-5enu)."""
    clauses: list[str] = []
    if start is not None:
        clauses.append(f"issue_date >= '{pd.Timestamp(start).strftime('%Y-%m-%d')}T00:00:00'")
    if end is not None:
        clauses.append(f"issue_date < '{pd.Timestamp(end).strftime('%Y-%m-%d')}T00:00:00'")
    where = " AND ".join(clauses) if clauses else None

    rows: list[dict] = []
    offset = 0
    while True:
        page = _fetch_permits_page(where=where, limit=page_size, offset=offset)
        if not page:
            break
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    drop_cols = [c for c in df.columns if c.startswith(":")]
    df = df.drop(columns=drop_cols, errors="ignore")

    for col in ("issue_date", "application_start_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df


def fetch_calendar_features(
    start: date | str,
    end: date | str,
) -> pd.DataFrame:
    """Weekend and US holiday flags (roadmap Section 3D)."""
    import holidays

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    us_holidays = holidays.US(state="IL")

    dates = pd.date_range(start_ts, end_ts, freq="D")
    df = pd.DataFrame({"date": dates})
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    df["is_holiday"] = df["date"].dt.date.map(lambda d: d in us_holidays)
    df["holiday_name"] = df["date"].dt.date.map(lambda d: us_holidays.get(d))
    return df


def default_date_range() -> tuple[date, date]:
    """Default modeling window: HISTORY_START (config) through today."""
    return HISTORY_START, date.today()
