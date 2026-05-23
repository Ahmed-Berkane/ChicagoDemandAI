"""CTA daily ridership as a public demand/traffic proxy."""

from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests

from src.config import CHICAGO_CTA_RIDERSHIP_API


def fetch_cta_ridership(
    start: date | str,
    end: date | str,
    *,
    page_size: int = 10_000,
) -> pd.DataFrame:
    """Systemwide CTA daily boarding totals (Chicago Data Portal 6iiy-9s97)."""
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    where = f"service_date >= '{start_s}T00:00:00' AND service_date <= '{end_s}T23:59:59'"

    rows: list[dict] = []
    offset = 0
    while True:
        params = {"$limit": page_size, "$offset": offset, "$where": where, "$order": "service_date"}
        for attempt in range(5):
            try:
                resp = requests.get(CHICAGO_CTA_RIDERSHIP_API, params=params, timeout=120)
                resp.raise_for_status()
                page = resp.json()
                break
            except requests.RequestException:
                time.sleep(2**attempt)
        else:
            raise RuntimeError("CTA API failed after retries")

        if not page:
            break
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["service_date"], errors="coerce").dt.normalize()
    for col in ("bus", "rail_boardings", "total_rides"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"total_rides": "cta_total_rides", "rail_boardings": "cta_rail_rides"})
    return df[["date", "day_type", "bus", "cta_rail_rides", "cta_total_rides"]].drop_duplicates("date")
