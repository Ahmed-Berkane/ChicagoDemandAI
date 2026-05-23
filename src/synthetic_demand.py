"""
Synthetic daily customer traffic (portfolio MVP target).

Transparent proxy sales based on real external signals — not claimed as real POS data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SYNTHETIC_BASE_TRAFFIC, SYNTHETIC_RANDOM_SEED


def _weather_multiplier(row: pd.Series) -> float:
    """Effect of weather on café foot traffic."""
    effect = 0.0
    precip = row.get("precipitation_in", 0) or 0
    snow = row.get("snowfall_in", 0) or 0
    temp = row.get("temperature_f", 60)

    if precip >= 0.25:
        effect -= 0.18
    elif precip > 0.05:
        effect -= 0.10
    if snow > 0:
        effect -= 0.12
    if temp < 25:
        effect -= 0.12
    elif temp < 40:
        effect -= 0.06
    elif 60 <= temp <= 78:
        effect += 0.05
    if pd.notna(row.get("wind_chill_f")) and row["wind_chill_f"] < 10:
        effect -= 0.08
    return effect


def _event_boost(row: pd.Series) -> float:
    boost = 0.0
    if row.get("is_weekend"):
        boost += 0.12
    if row.get("is_holiday"):
        boost -= 0.08
    if row.get("cubs_home_game"):
        boost += 0.10
    if row.get("bulls_home_game"):
        boost += 0.07
    if row.get("is_major_festival"):
        boost += 0.15
    elif (row.get("city_special_events") or 0) > 0:
        boost += 0.05
    if row.get("is_payweek"):
        boost += 0.04
    return boost


def _cta_boost(row: pd.Series, cta_mean: float) -> float:
    rides = row.get("cta_total_rides")
    if pd.isna(rides) or cta_mean <= 0:
        return 0.0
    z = (rides - cta_mean) / cta_mean
    return float(np.clip(z * 0.08, -0.06, 0.10))


def generate_synthetic_traffic(
    features: pd.DataFrame,
    *,
    base_traffic: float = SYNTHETIC_BASE_TRAFFIC,
    random_seed: int = SYNTHETIC_RANDOM_SEED,
) -> pd.DataFrame:
    """
    customer_traffic = base * (1 + weather + events + cta) + noise

    Returns a copy of `features` with `customer_traffic` and component columns.
    """
    rng = np.random.default_rng(random_seed)
    out = features.copy()

    cta_mean = out["cta_total_rides"].mean() if "cta_total_rides" in out.columns else 0.0

    weather_fx = out.apply(_weather_multiplier, axis=1)
    event_fx = out.apply(_event_boost, axis=1)
    cta_fx = out.apply(lambda r: _cta_boost(r, cta_mean), axis=1)
    noise = rng.normal(0, 8, size=len(out))

    out["weather_effect"] = weather_fx
    out["event_effect"] = event_fx
    out["cta_effect"] = cta_fx
    out["traffic_noise"] = noise

    multiplier = 1.0 + weather_fx + event_fx + cta_fx
    out["customer_traffic"] = np.maximum(
        20,
        base_traffic * multiplier + noise,
    ).round(1)

    out["is_synthetic_target"] = True
    return out
