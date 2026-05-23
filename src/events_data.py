"""Sports schedules and festival flags for Chicago demand modeling."""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from src.config import (
    CHICAGO_SPECIAL_EVENTS_API,
    ESPN_BULLS_TEAM_ID,
    ESPN_CUBS_TEAM_ID,
    MAJOR_EVENTS_FILE,
    MLB_CUBS_TEAM_ID,
    MLB_STATS_API,
)

FESTIVAL_KEYWORDS = (
    "lollapalooza",
    "lolla",
    "taste of chicago",
    "chicago marathon",
    "air and water show",
    "pride parade",
    "st. patrick",
    "christkindlmarket",
)


def _fetch_mlb_cubs_home(start: date, end: date) -> pd.DataFrame:
    """Cubs home games via MLB Stats API (free, unofficial)."""
    rows: list[dict] = []
    chunk_start = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    while chunk_start <= end_ts:
        chunk_end = min(chunk_start + pd.DateOffset(months=2), end_ts)
        resp = requests.get(
            MLB_STATS_API,
            params={
                "sportId": 1,
                "teamId": MLB_CUBS_TEAM_ID,
                "startDate": chunk_start.strftime("%Y-%m-%d"),
                "endDate": chunk_end.strftime("%Y-%m-%d"),
            },
            timeout=60,
        )
        resp.raise_for_status()
        for day in resp.json().get("dates", []):
            game_date = day.get("date")
            for game in day.get("games", []):
                home_id = game.get("teams", {}).get("home", {}).get("team", {}).get("id")
                if home_id != MLB_CUBS_TEAM_ID:
                    continue
                rows.append(
                    {
                        "date": pd.Timestamp(game_date),
                        "game_id": game.get("gamePk"),
                        "opponent": game.get("teams", {})
                        .get("away", {})
                        .get("team", {})
                        .get("name"),
                        "venue": game.get("venue", {}).get("name"),
                        "game_datetime": game.get("gameDate"),
                    }
                )
        chunk_start = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.2)

    return pd.DataFrame(rows).drop_duplicates(subset=["date", "game_id"])


def _fetch_espn_home_games(sport_path: str, team_id: int, season: int) -> pd.DataFrame:
    """Fallback / supplement: ESPN public schedule API."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_path}/teams/{team_id}/schedule"
    resp = requests.get(url, params={"season": season}, timeout=60)
    resp.raise_for_status()
    rows: list[dict] = []
    for event in resp.json().get("events", []):
        comp = event["competitions"][0]
        ts = pd.Timestamp(event["date"])
        if ts.tzinfo is not None:
            ts = ts.tz_convert("America/Chicago").tz_localize(None)
        rows.append(
            {
                "date": ts.normalize(),
                "game_id": event.get("id"),
                "opponent": next(
                    c["team"]["displayName"]
                    for c in comp["competitors"]
                    if c.get("homeAway") == "away"
                ),
                "venue": comp.get("venue", {}).get("fullName"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["date", "game_id"])


def fetch_cubs_home_games(start: date, end: date) -> pd.DataFrame:
    df = _fetch_mlb_cubs_home(start, end)
    if df.empty:
        season = pd.Timestamp(start).year
        df = _fetch_espn_home_games("baseball/mlb", ESPN_CUBS_TEAM_ID, season)
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df


def fetch_bulls_home_games(start: date, end: date) -> pd.DataFrame:
    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    seasons = range(y0, y1 + 1)
    frames = [
        _fetch_espn_home_games("basketball/nba", ESPN_BULLS_TEAM_ID, s) for s in seasons
    ]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "game_id"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df


def load_major_events_calendar() -> pd.DataFrame:
    """Curated major Chicago festivals 2019+ (portal has no historical special events)."""
    return pd.read_csv(MAJOR_EVENTS_FILE, parse_dates=["date"])


def fetch_chicago_special_events_portal(
    start: date | str | None = None,
    end: date | str | None = None,
) -> pd.DataFrame:
    """
    Pull Special Events from Chicago Data Portal API (dataset xgse-8eg7).

    Note: the city only publishes *scheduled upcoming* events (typically ~3–9 months
  ahead). There are **no 2019 rows** in this API — use ``major_chicago_events.csv``.
    """
    params: dict[str, str | int] = {"$limit": 50000}
    clauses: list[str] = []
    if start is not None:
        clauses.append(f"date >= '{pd.Timestamp(start).strftime('%Y-%m-%d')}T00:00:00'")
    if end is not None:
        clauses.append(f"date <= '{pd.Timestamp(end).strftime('%Y-%m-%d')}T23:59:59'")
    if clauses:
        params["$where"] = " AND ".join(clauses)

    resp = requests.get(CHICAGO_SPECIAL_EVENTS_API, params=params, timeout=120)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        return df

    drop_cols = [c for c in df.columns if str(c).startswith(":")]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.rename(
        columns={
            "date": "Date",
            "venue": "Venue",
            "venue_address": "Venue Address",
            "event_type": "Event Type",
            "event_details": "Event Details",
            "start_time": "Start Time",
            "ward": "Ward",
            "location": "Location",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def festival_flags_from_special_events(events: pd.DataFrame) -> pd.DataFrame:
    """Daily festival / major event flags from city special-events CSV."""
    df = events.copy()
    date_col = "Date" if "Date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    details = df.get("Event Details", df.get("event_details", ""))
    etype = df.get("Event Type", df.get("event_type", ""))
    text = details.fillna("").astype(str) + " " + etype.fillna("").astype(str)
    text = text.str.lower()
    df["is_major_festival"] = False
    for kw in FESTIVAL_KEYWORDS:
        df.loc[text.str.contains(kw, na=False), "is_major_festival"] = True

    daily = (
        df.groupby(df[date_col].dt.normalize())
        .agg(
            city_special_events=(date_col, "count"),
            is_major_festival=("is_major_festival", "max"),
        )
        .reset_index()
        .rename(columns={date_col: "date"})
    )
    return daily


def daily_city_event_features(
    special_events: pd.DataFrame | None = None,
    *,
    include_major_calendar: bool = True,
) -> pd.DataFrame:
    """
    Combine portal special events + curated major festival calendar (2019+).
    """
    portal = pd.DataFrame(columns=["date", "city_special_events", "is_major_festival"])
    if special_events is not None and not special_events.empty:
        portal = festival_flags_from_special_events(special_events)

    major = pd.DataFrame(columns=["date", "major_event_count", "is_major_festival_major"])
    if include_major_calendar and MAJOR_EVENTS_FILE.exists():
        m = load_major_events_calendar()
        major = (
            m.groupby(m["date"].dt.normalize())
            .agg(
                major_event_count=("event_name", "count"),
                is_major_festival_major=("is_major_festival", "max"),
            )
            .reset_index()
        )

    if portal.empty and major.empty:
        return pd.DataFrame(columns=["date", "city_special_events", "is_major_festival"])

    out = portal.merge(major, on="date", how="outer")
    out["city_special_events"] = out.get("city_special_events", 0).fillna(0) + out.get(
        "major_event_count", 0
    ).fillna(0)
    out["is_major_festival"] = np.maximum(
        out.get("is_major_festival", 0).fillna(0).astype(int),
        out.get("is_major_festival_major", 0).fillna(0).astype(int),
    ).astype(bool)
    out = out.drop(columns=["major_event_count", "is_major_festival_major"], errors="ignore")
    return out.sort_values("date").reset_index(drop=True)
