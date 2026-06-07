"""Fetch or estimate city-wide features for future prediction dates."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from src.config import (
    CHICAGO_CTA_RIDERSHIP_API,
    CHICAGO_LAT,
    CHICAGO_LON,
    CHICAGO_SPECIAL_EVENTS_API,
    CHICAGO_TIMEZONE,
    ESPN_BULLS_TEAM_ID,
    EVENTS_REFERENCE_FILE,
    FUTURE_FORECAST_DAYS,
    MLB_CUBS_TEAM_ID,
    MLB_STATS_API,
    OPEN_METEO_FORECAST_API,
    WEATHER_REFERENCE_FILE,
)
from src.pay_cycle import pay_cycle_flags

FESTIVAL_KEYWORDS = (
    "lollapalooza",
    "lolla",
    "taste of chicago",
    "chicago marathon",
    "air and water show",
    "pride parade",
    "st. patrick",
    "christkindlmarket",
    "festival",
    "parade",
    "marathon",
)

WEATHER_COLS = (
    "temperature_f",
    "precipitation_in",
    "snowfall_in",
    "humidity_pct",
    "wind_speed_mph",
)


def _c_to_f(value: float | None) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value) * 9 / 5 + 32


def _cm_snow_to_in(value: float | None) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value) / 2.54


def _mm_to_in(value: float | None) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value) / 25.4


@lru_cache(maxsize=1)
def _load_weather_reference() -> pd.DataFrame:
    if not WEATHER_REFERENCE_FILE.exists():
        return pd.DataFrame(columns=["date", *WEATHER_COLS])
    df = pd.read_csv(WEATHER_REFERENCE_FILE, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_events_reference() -> pd.DataFrame:
    if not EVENTS_REFERENCE_FILE.exists():
        return pd.DataFrame(columns=["event_name", "start_date", "end_date", "intensity"])
    df = pd.read_csv(EVENTS_REFERENCE_FILE, parse_dates=["start_date", "end_date"])
    return df


def _weather_from_reference(target_date: pd.Timestamp) -> dict[str, float] | None:
    ref = _load_weather_reference()
    row = ref.loc[ref["date"] == target_date.normalize()]
    if row.empty:
        return None
    values = row.iloc[0]
    return {col: float(values[col]) for col in WEATHER_COLS}


def _weather_climatology(target_date: pd.Timestamp) -> dict[str, float]:
    ref = _load_weather_reference()
    month, day = target_date.month, target_date.day
    subset = ref[(ref["date"].dt.month == month) & (ref["date"].dt.day == day)]
    if subset.empty and month == 2 and day == 29:
        subset = ref[(ref["date"].dt.month == 2) & (ref["date"].dt.day == 28)]
    if subset.empty:
        return {col: np.nan for col in WEATHER_COLS}
    return {col: float(subset[col].mean()) for col in WEATHER_COLS}


@lru_cache(maxsize=4)
def _fetch_open_meteo_forecast(forecast_days: int) -> pd.DataFrame:
    params = {
        "latitude": CHICAGO_LAT,
        "longitude": CHICAGO_LON,
        "timezone": CHICAGO_TIMEZONE,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "forecast_days": forecast_days,
        "daily": ",".join(
            [
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "snowfall_sum",
                "relative_humidity_2m_mean",
                "wind_speed_10m_max",
            ]
        ),
    }
    resp = requests.get(OPEN_METEO_FORECAST_API, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    daily = payload.get("daily", {})
    if not daily.get("time"):
        return pd.DataFrame(columns=["date", *WEATHER_COLS])

    temp_mean = daily.get("temperature_2m_mean")
    temp_max = daily.get("temperature_2m_max")
    temp_min = daily.get("temperature_2m_min")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temperature_f": temp_mean
            if temp_mean is not None
            else [
                (mx + mn) / 2 for mx, mn in zip(temp_max or [], temp_min or [], strict=False)
            ],
            "precipitation_in": daily.get("precipitation_sum", []),
            "snowfall_in": [_mm_to_in(v) for v in daily.get("snowfall_sum", [])],
            "humidity_pct": daily.get("relative_humidity_2m_mean", []),
            "wind_speed_mph": daily.get("wind_speed_10m_max", []),
        }
    )
    return df


def fetch_weather_features(target_date: pd.Timestamp) -> tuple[dict[str, float], str]:
    target_date = target_date.normalize()
    today = pd.Timestamp(date.today()).normalize()
    days_ahead = (target_date - today).days

    if days_ahead < 0:
        cached = _weather_from_reference(target_date)
        if cached is not None:
            return cached, "weather_cache"
        return _weather_climatology(target_date), "weather_climatology"

    if days_ahead <= FUTURE_FORECAST_DAYS:
        try:
            forecast = _fetch_open_meteo_forecast(FUTURE_FORECAST_DAYS)
            row = forecast.loc[forecast["date"] == target_date]
            if not row.empty:
                values = row.iloc[0]
                return (
                    {
                        "temperature_f": float(values["temperature_f"]),
                        "precipitation_in": float(values["precipitation_in"]),
                        "snowfall_in": float(values.get("snowfall_in", 0.0) or 0.0),
                        "humidity_pct": float(values["humidity_pct"]),
                        "wind_speed_mph": float(values["wind_speed_mph"]),
                    },
                    "open_meteo_forecast",
                )
        except requests.RequestException:
            pass

    raise ValueError(
        f"Weather forecast unavailable for {target_date.date()}. "
        f"Only the next {FUTURE_FORECAST_DAYS} days are supported."
    )


def fetch_calendar_features(target_date: pd.Timestamp) -> dict[str, int]:
    import holidays

    us_il = holidays.country_holidays("US", subdiv="IL", years=target_date.year)
    pay_flags = pay_cycle_flags(target_date)
    return {
        "is_weekend": int(target_date.dayofweek >= 5),
        "is_holiday": int(target_date.normalize() in us_il),
        **pay_flags,
    }


def _has_cubs_home_game(target_date: pd.Timestamp) -> int:
    try:
        resp = requests.get(
            MLB_STATS_API,
            params={
                "sportId": 1,
                "teamId": MLB_CUBS_TEAM_ID,
                "startDate": target_date.strftime("%Y-%m-%d"),
                "endDate": target_date.strftime("%Y-%m-%d"),
            },
            timeout=30,
        )
        resp.raise_for_status()
        for day in resp.json().get("dates", []):
            for game in day.get("games", []):
                home_id = game.get("teams", {}).get("home", {}).get("team", {}).get("id")
                if home_id == MLB_CUBS_TEAM_ID:
                    return 1
    except requests.RequestException:
        pass
    return 0


def _has_bulls_home_game(target_date: pd.Timestamp) -> int:
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{ESPN_BULLS_TEAM_ID}/schedule"
        resp = requests.get(url, params={"season": target_date.year}, timeout=30)
        resp.raise_for_status()
        for event in resp.json().get("events", []):
            ts = pd.Timestamp(event["date"])
            if ts.tzinfo is not None:
                ts = ts.tz_convert(CHICAGO_TIMEZONE).tz_localize(None)
            if ts.normalize() != target_date.normalize():
                continue
            comp = event["competitions"][0]
            for competitor in comp["competitors"]:
                if competitor.get("homeAway") == "home":
                    return 1
    except requests.RequestException:
        pass
    return 0


def fetch_sports_features(target_date: pd.Timestamp) -> dict[str, int]:
    return {
        "cubs_home_game": _has_cubs_home_game(target_date),
        "bulls_home_game": _has_bulls_home_game(target_date),
    }


def _festival_match(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in FESTIVAL_KEYWORDS)


def _events_from_reference(target_date: pd.Timestamp) -> tuple[float, int]:
    ref = _load_events_reference()
    if ref.empty:
        return 0.0, 0
    mask = (ref["start_date"] <= target_date) & (ref["end_date"] >= target_date)
    day_events = ref.loc[mask]
    if day_events.empty:
        return 0.0, 0
    is_major = int(day_events["event_name"].astype(str).map(_festival_match).any())
    return float(len(day_events)), is_major


def _events_from_portal(target_date: pd.Timestamp) -> tuple[float, int]:
    where = (
        f"date >= '{target_date.strftime('%Y-%m-%d')}T00:00:00' "
        f"AND date <= '{target_date.strftime('%Y-%m-%d')}T23:59:59'"
    )
    try:
        resp = requests.get(
            CHICAGO_SPECIAL_EVENTS_API,
            params={"$limit": 5000, "$where": where},
            timeout=60,
        )
        resp.raise_for_status()
        rows = resp.json()
    except requests.RequestException:
        return 0.0, 0

    if not rows:
        return 0.0, 0

    df = pd.DataFrame(rows)
    text_cols = [c for c in df.columns if "event" in c.lower() or "detail" in c.lower()]
    text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1) if text_cols else pd.Series("", index=df.index)
    is_major = int(text.map(_festival_match).any()) if not text.empty else 0
    return float(len(df)), is_major


def fetch_event_features(target_date: pd.Timestamp) -> tuple[dict[str, float | int], str]:
    portal_count, portal_major = _events_from_portal(target_date)
    ref_count, ref_major = _events_from_reference(target_date)
    source = "events_portal" if portal_count else "events_reference"
    return (
        {
            "city_special_events": portal_count + ref_count,
            "is_major_festival": int(max(portal_major, ref_major)),
        },
        source,
    )


def estimate_cta_ridership(target_date: pd.Timestamp, city_history: pd.DataFrame) -> tuple[float, str]:
    if city_history.empty or "cta_total_rides" not in city_history.columns:
        return np.nan, "cta_unavailable"

    hist = city_history.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist["month"] = hist["date"].dt.month
    hist["dow"] = hist["date"].dt.dayofweek
    match = hist[
        (hist["month"] == target_date.month) & (hist["dow"] == target_date.dayofweek)
    ]
    if not match.empty:
        return float(match["cta_total_rides"].median()), "cta_seasonal_profile"

    return float(hist["cta_total_rides"].median()), "cta_global_median"


def fetch_city_features_for_date(
    target_date: pd.Timestamp | str,
    *,
    city_history: pd.DataFrame | None = None,
) -> tuple[dict[str, float | int], dict[str, str]]:
    target_date = pd.Timestamp(target_date).normalize()
    city_history = city_history if city_history is not None else pd.DataFrame()

    weather, weather_source = fetch_weather_features(target_date)
    calendar = fetch_calendar_features(target_date)
    sports = fetch_sports_features(target_date)
    events, events_source = fetch_event_features(target_date)
    cta, cta_source = estimate_cta_ridership(target_date, city_history)

    features = {
        **weather,
        **calendar,
        **sports,
        **events,
        "cta_total_rides": cta,
    }
    sources = {
        "weather": weather_source,
        "calendar": "computed",
        "sports": "live_schedule",
        "events": events_source,
        "cta": cta_source,
    }
    return features, sources
