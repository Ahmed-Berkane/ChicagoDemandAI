"""Feature engineering for demand forecasting (no target leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns derived from the synthetic target generator — never use as model inputs
LEAKAGE_COLUMNS = frozenset(
    {
        "customer_traffic",
        "is_synthetic_target",
        "weather_effect",
        "event_effect",
        "cta_effect",
        "traffic_noise",
    }
)

DROP_COLUMNS = LEAKAGE_COLUMNS | frozenset(
    {
        "date",
        "holiday_name",
        "business_name",
        "business_address",
        "license_description",
        "neighborhood",
        "day_type",
    }
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, weather, event, and lag features (past-only lags)."""
    out = df.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"])

    doy = out["date"].dt.dayofyear
    out["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month"] = out["date"].dt.month
    out["day_of_month"] = out["date"].dt.day

    temp = out["temperature_f"].fillna(out["temperature_f"].median())
    out["temp_deviation_comfort"] = (temp - 65).abs()
    out["is_rainy"] = (out["precipitation_in"].fillna(0) >= 0.1).astype(int)
    out["is_snowy"] = (out["snowfall_in"].fillna(0) > 0).astype(int)
    out["is_extreme_cold"] = (temp < 20).astype(int)

    def _b(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(0, index=out.index, dtype=int)
        return out[col].fillna(0).astype(int)

    out["event_score"] = (
        _b("cubs_home_game") * 1.0
        + _b("bulls_home_game") * 0.7
        + _b("is_major_festival") * 1.5
        + np.minimum(out.get("city_special_events", 0).fillna(0), 3) * 0.2
    )

    out["weekend_payweek"] = _b("is_weekend") * _b("is_payweek")
    out["rainy_weekend"] = out["is_rainy"] * _b("is_weekend")

    if "crime_count" in out.columns and "permit_count" in out.columns:
        out["crime_permit_ratio"] = out["crime_count"] / (out["permit_count"] + 1)

    # Past-only rolling / lags (no future information)
    if "cta_total_rides" in out.columns:
        out["cta_roll7"] = out["cta_total_rides"].shift(1).rolling(7, min_periods=1).mean()
    if "crime_count" in out.columns:
        out["crime_roll7"] = out["crime_count"].shift(1).rolling(7, min_periods=1).mean()
    if "customer_traffic" in out.columns:
        out["traffic_lag1"] = out["customer_traffic"].shift(1)
        out["traffic_lag7"] = out["customer_traffic"].shift(7)
        out["traffic_roll7"] = out["customer_traffic"].shift(1).rolling(7, min_periods=1).mean()

    return out


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered train / validation / test (no random shuffle)."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def feature_target_split(df: pd.DataFrame, target: str = "customer_traffic"):
    """Return X, y with leakage and ID columns removed."""
    engineered = engineer_features(df)
    y = engineered[target].astype(float)
    X = engineered.drop(columns=[c for c in engineered.columns if c in DROP_COLUMNS], errors="ignore")
    X = X.select_dtypes(include=[np.number, bool])
    X = X.astype(float)
    return X, y, engineered
