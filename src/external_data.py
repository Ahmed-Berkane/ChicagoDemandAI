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
    NASA_POWER_DAILY_API,
    OPEN_METEO_ARCHIVE_API,
    WEATHER_REFERENCE_FILE,
)

# Roadmap weather fields -> Open-Meteo daily archive (free; no API key)
WEATHER_DAILY_PARAMS = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
)

# Open-Meteo archive often 504s on large ranges; use smaller chunks + NASA fallback.
WEATHER_CHUNK_DAYS = 90
WEATHER_RETRYABLE_STATUS = {429, 502, 503, 504}

WEATHER_COLUMNS = (
    "date",
    "temperature_f",
    "temperature_max_f",
    "temperature_min_f",
    "precipitation_in",
    "snowfall_in",
    "humidity_pct",
    "wind_speed_mph",
    "wind_chill_f",
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


def _weather_date_chunks(
    start: date | str,
    end: date | str,
    *,
    chunk_days: int = WEATHER_CHUNK_DAYS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start_ts
    while cur <= end_ts:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), end_ts)
        chunks.append((cur, chunk_end))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks


def _finalize_weather_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["wind_chill_f"] = out.apply(
        lambda r: _wind_chill_f(r["temperature_f"], r["wind_speed_mph"]),
        axis=1,
    )
    return out[list(WEATHER_COLUMNS)]


def _load_weather_reference(
    start: date | str,
    end: date | str,
) -> pd.DataFrame | None:
    if not WEATHER_REFERENCE_FILE.exists():
        return None

    df = pd.read_csv(WEATHER_REFERENCE_FILE, parse_dates=["date"])
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    expected_days = (end_ts - start_ts).days + 1
    if len(df) < expected_days:
        return None

    print(f"Using cached weather from {WEATHER_REFERENCE_FILE.name} ({len(df):,} days)")
    return _finalize_weather_df(df)


def _save_weather_reference(df: pd.DataFrame) -> None:
    WEATHER_REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if WEATHER_REFERENCE_FILE.exists():
        existing = pd.read_csv(WEATHER_REFERENCE_FILE, parse_dates=["date"])
        df = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
        )
    df.to_csv(WEATHER_REFERENCE_FILE, index=False)


def _fetch_weather_open_meteo_chunk(
    start: date | str,
    end: date | str,
    *,
    latitude: float,
    longitude: float,
    max_retries: int = 4,
) -> pd.DataFrame:
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

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(OPEN_METEO_ARCHIVE_API, params=params, timeout=90)
            resp.raise_for_status()
            return pd.DataFrame(resp.json()["daily"])
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status not in WEATHER_RETRYABLE_STATUS or attempt >= max_retries - 1:
                raise
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            if attempt >= max_retries - 1:
                raise

        wait_time = 2**attempt
        print(
            f"Open-Meteo error for {start_s} -> {end_s} "
            f"(attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s..."
        )
        time.sleep(wait_time)

    raise RuntimeError(f"Open-Meteo failed for {start_s} -> {end_s}") from last_error


def _fetch_weather_open_meteo(
    start: date | str,
    end: date | str,
    *,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    chunks = _weather_date_chunks(start, end)
    if len(chunks) > 1:
        print(f"Fetching weather from Open-Meteo in {len(chunks)} chunks ({start} -> {end})...")

    frames: list[pd.DataFrame] = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        if len(chunks) > 1:
            print(
                f"  chunk {i + 1}/{len(chunks)}: "
                f"{chunk_start.date()} -> {chunk_end.date()}"
            )
        frames.append(
            _fetch_weather_open_meteo_chunk(
                chunk_start,
                chunk_end,
                latitude=latitude,
                longitude=longitude,
            )
        )
        if i < len(chunks) - 1:
            time.sleep(0.5)

    df = pd.concat(frames, ignore_index=True)
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
    return df


def _fetch_weather_nasa_power_chunk(
    start: date | str,
    end: date | str,
    *,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """NASA POWER fallback when Open-Meteo archive is down (504 overload)."""
    start_s = pd.Timestamp(start).strftime("%Y%m%d")
    end_s = pd.Timestamp(end).strftime("%Y%m%d")
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS10M",
        "community": "AG",
        "latitude": latitude,
        "longitude": longitude,
        "start": start_s,
        "end": end_s,
        "format": "JSON",
    }
    resp = requests.get(NASA_POWER_DAILY_API, params=params, timeout=300)
    resp.raise_for_status()
    raw = resp.json()["properties"]["parameter"]

    rows: list[dict] = []
    for day_key, temp_c in raw["T2M"].items():
        if day_key.endswith("MISSING"):
            continue
        temp_c = float(temp_c)
        temp_max_c = float(raw["T2M_MAX"][day_key])
        temp_min_c = float(raw["T2M_MIN"][day_key])
        precip_mm = float(raw["PRECTOTCORR"][day_key])
        humidity = float(raw["RH2M"][day_key])
        wind_m_s = float(raw["WS10M"][day_key])

        temp_f = temp_c * 9 / 5 + 32
        temp_max_f = temp_max_c * 9 / 5 + 32
        temp_min_f = temp_min_c * 9 / 5 + 32
        precip_in = precip_mm / 25.4
        wind_mph = wind_m_s * 2.237
        snowfall_in = precip_in if temp_min_f <= 32 and precip_in > 0.01 else 0.0

        rows.append(
            {
                "date": pd.Timestamp(day_key),
                "temperature_f": temp_f,
                "temperature_max_f": temp_max_f,
                "temperature_min_f": temp_min_f,
                "precipitation_in": precip_in,
                "snowfall_in": snowfall_in,
                "humidity_pct": humidity,
                "wind_speed_mph": wind_mph,
            }
        )

    return pd.DataFrame(rows)


def _fetch_weather_nasa_power(
    start: date | str,
    end: date | str,
    *,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    # NASA responds in ~3 min per request regardless of range; use yearly chunks.
    chunks = _weather_date_chunks(start, end, chunk_days=366)
    print(
        f"Open-Meteo unavailable; using NASA POWER fallback "
        f"({len(chunks)} year-chunks, ~3 min each)..."
    )

    frames: list[pd.DataFrame] = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        print(
            f"  NASA chunk {i + 1}/{len(chunks)}: "
            f"{chunk_start.date()} -> {chunk_end.date()}"
        )
        frames.append(
            _fetch_weather_nasa_power_chunk(
                chunk_start,
                chunk_end,
                latitude=latitude,
                longitude=longitude,
            )
        )
    return pd.concat(frames, ignore_index=True)


def fetch_weather(
    start: date | str,
    end: date | str,
    *,
    latitude: float = CHICAGO_LAT,
    longitude: float = CHICAGO_LON,
    use_cache: bool = True,
    save_cache: bool = True,
) -> pd.DataFrame:
    """
    Daily Chicago weather for the modeling window.

    Tries, in order: local CSV cache, Open-Meteo archive (fast), NASA POWER (slow fallback).
    """
    if use_cache:
        cached = _load_weather_reference(start, end)
        if cached is not None:
            return cached

    try:
        df = _fetch_weather_open_meteo(start, end, latitude=latitude, longitude=longitude)
        source = "Open-Meteo"
    except requests.RequestException as exc:
        print(f"Open-Meteo weather fetch failed ({exc}).")
        df = _fetch_weather_nasa_power(start, end, latitude=latitude, longitude=longitude)
        source = "NASA POWER"

    result = _finalize_weather_df(df)
    if save_cache:
        _save_weather_reference(result)
        print(f"Weather cached to {WEATHER_REFERENCE_FILE} ({source})")
    return result


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
