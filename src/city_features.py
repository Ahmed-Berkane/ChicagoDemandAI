"""Aggregate Chicago portal datasets to daily city-level features."""

from __future__ import annotations

import pandas as pd

from src.data_loader import load_crimes, load_special_events
from src.events_data import daily_city_event_features


def daily_crime_counts(crimes: pd.DataFrame) -> pd.DataFrame:
    df = crimes.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return (
        df.groupby("date")
        .size()
        .reset_index(name="crime_count")
    )


def daily_permit_counts(permits: pd.DataFrame) -> pd.DataFrame:
    df = permits.copy()
    if "issue_date" not in df.columns:
        return pd.DataFrame(columns=["date", "permit_count"])
    df["date"] = pd.to_datetime(df["issue_date"], errors="coerce").dt.normalize()
    return (
        df.groupby("date")
        .size()
        .reset_index(name="permit_count")
    )


def daily_sports_flags(cubs: pd.DataFrame, bulls: pd.DataFrame) -> pd.DataFrame:
    cubs_d = cubs.copy()
    cubs_d["date"] = pd.to_datetime(cubs_d["date"]).dt.normalize()
    bulls_d = bulls.copy()
    bulls_d["date"] = pd.to_datetime(bulls_d["date"]).dt.normalize()

    cubs_days = cubs_d.groupby("date").size().reset_index(name="cubs_home_games")
    bulls_days = bulls_d.groupby("date").size().reset_index(name="bulls_home_games")

    bounds = []
    if not cubs_d.empty:
        bounds.extend([cubs_d["date"].min(), cubs_d["date"].max()])
    if not bulls_d.empty:
        bounds.extend([bulls_d["date"].min(), bulls_d["date"].max()])
    if not bounds:
        return pd.DataFrame(
            columns=[
                "date",
                "cubs_home_game",
                "bulls_home_game",
                "cubs_home_games",
                "bulls_home_games",
            ]
        )
    dmin, dmax = min(bounds), max(bounds)
    all_dates = pd.DataFrame({"date": pd.date_range(dmin, dmax, freq="D")})
    out = all_dates.merge(cubs_days, on="date", how="left").merge(bulls_days, on="date", how="left")
    out["cubs_home_game"] = out["cubs_home_games"].fillna(0).astype(int).gt(0)
    out["bulls_home_game"] = out["bulls_home_games"].fillna(0).astype(int).gt(0)
    return out[
        ["date", "cubs_home_game", "bulls_home_game", "cubs_home_games", "bulls_home_games"]
    ]


def build_city_daily_features(
    *,
    crimes: pd.DataFrame | None = None,
    permits: pd.DataFrame | None = None,
    special_events: pd.DataFrame | None = None,
    cubs: pd.DataFrame | None = None,
    bulls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    crimes = crimes if crimes is not None else load_crimes()
    permits = permits if permits is not None else pd.DataFrame(columns=["issue_date"])
    special_events = special_events if special_events is not None else load_special_events()

    parts = [
        daily_crime_counts(crimes),
        daily_permit_counts(permits),
        daily_city_event_features(special_events, include_major_calendar=True),
    ]

    if cubs is not None and not cubs.empty:
        bulls_df = bulls if bulls is not None and not bulls.empty else pd.DataFrame(columns=["date"])
        if bulls_df.empty:
            bulls_df = pd.DataFrame({"date": pd.to_datetime([])})
        parts.append(daily_sports_flags(cubs, bulls_df))

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)
