"""Build model inputs from minimal UI fields and score saved artifacts."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import (
    BOOL_COLS,
    CAT_COLS,
    FUTURE_MAX_HORIZON_DAYS,
    NUM_COLS,
    PAST_LOOKBACK_DAYS,
    TARGET,
    UNIFIED_PARQUET,
)
from src.future_features import fetch_city_features_for_date
from src.modeling.features import create_date_features
from src.modeling.persist import load_artifacts
from src.regions import REGION_MAP, REGION_OPTIONS

CITY_WIDE_COLS = [
    "temperature_f",
    "precipitation_in",
    "snowfall_in",
    "humidity_pct",
    "wind_speed_mph",
    "cta_total_rides",
    "is_weekend",
    "is_holiday",
    "is_payweek",
    "is_semimonthly_payday",
    "cubs_home_game",
    "bulls_home_game",
    "is_major_festival",
    "city_special_events",
]

DEMAND_TYPE_LABELS = {
    "baseline": "Normal day — typical customer traffic for your business type",
    "morning_peak": "Morning rush — higher traffic at opening / commute hours",
    "lunch_spike": "Lunch rush — busy midday period (11am–2pm pattern)",
    "evening_peak": "Evening rush — dinner and after-work traffic",
    "food_peak": "Food-focused peak — festivals or dining-heavy event days",
    "beverage_spike": "Beverage-heavy day — drinks-led traffic pattern",
    "retail_spike": "Retail spike — shopping or holiday-driven foot traffic",
    "delivery_spike": "Delivery-heavy day — more orders, less in-store traffic",
    "all_day_spike": "All-day surge — sustained high traffic from a major event",
}

DEMAND_TYPE_FLAGS = (
    "is_weekend",
    "is_holiday",
    "is_major_festival",
    "cubs_home_game",
    "bulls_home_game",
)

WEATHER_OVERRIDE_COLS = (
    "temperature_f",
    "precipitation_in",
    "snowfall_in",
    "humidity_pct",
    "wind_speed_mph",
)

EVENT_OVERRIDE_COLS = (
    "city_special_events",
    "is_major_festival",
    "cubs_home_game",
    "bulls_home_game",
)

WEATHER_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_f": (-10.0, 105.0),
    "precipitation_in": (0.0, 5.0),
    "snowfall_in": (0.0, 12.0),
    "humidity_pct": (0.0, 100.0),
    "wind_speed_mph": (0.0, 60.0),
}

EVENT_BOUNDS: dict[str, tuple[float, float]] = {
    "city_special_events": (0.0, 25.0),
}

SCENARIO_PRESETS: dict[str, dict[str, float | int]] = {
    "Auto (from date)": {},
    "Clear summer day": {
        "temperature_f": 82.0,
        "precipitation_in": 0.0,
        "snowfall_in": 0.0,
        "humidity_pct": 55.0,
        "wind_speed_mph": 8.0,
        "city_special_events": 0.0,
        "is_major_festival": 0,
        "cubs_home_game": 0,
        "bulls_home_game": 0,
    },
    "Heat wave": {
        "temperature_f": 98.0,
        "precipitation_in": 0.0,
        "snowfall_in": 0.0,
        "humidity_pct": 78.0,
        "wind_speed_mph": 6.0,
        "city_special_events": 0.0,
        "is_major_festival": 0,
        "cubs_home_game": 0,
        "bulls_home_game": 0,
    },
    "Heavy rainstorm": {
        "temperature_f": 58.0,
        "precipitation_in": 2.8,
        "snowfall_in": 0.0,
        "humidity_pct": 92.0,
        "wind_speed_mph": 28.0,
        "city_special_events": 0.0,
        "is_major_festival": 0,
        "cubs_home_game": 0,
        "bulls_home_game": 0,
    },
    "Blizzard / polar vortex": {
        "temperature_f": 5.0,
        "precipitation_in": 0.2,
        "snowfall_in": 9.0,
        "humidity_pct": 85.0,
        "wind_speed_mph": 35.0,
        "city_special_events": 0.0,
        "is_major_festival": 0,
        "cubs_home_game": 0,
        "bulls_home_game": 0,
    },
    "Major festival day": {
        "temperature_f": 75.0,
        "precipitation_in": 0.0,
        "snowfall_in": 0.0,
        "humidity_pct": 60.0,
        "wind_speed_mph": 10.0,
        "city_special_events": 12.0,
        "is_major_festival": 1,
        "cubs_home_game": 0,
        "bulls_home_game": 0,
    },
    "Cubs home game + street events": {
        "temperature_f": 72.0,
        "precipitation_in": 0.0,
        "snowfall_in": 0.0,
        "humidity_pct": 58.0,
        "wind_speed_mph": 12.0,
        "city_special_events": 4.0,
        "is_major_festival": 0,
        "cubs_home_game": 1,
        "bulls_home_game": 0,
    },
    "Bulls home game (United Center)": {
        "temperature_f": 38.0,
        "precipitation_in": 0.0,
        "snowfall_in": 0.0,
        "humidity_pct": 50.0,
        "wind_speed_mph": 14.0,
        "city_special_events": 2.0,
        "is_major_festival": 0,
        "cubs_home_game": 0,
        "bulls_home_game": 1,
    },
    "Crisp fall day": {
        "temperature_f": 58.0,
        "precipitation_in": 0.0,
        "snowfall_in": 0.0,
        "humidity_pct": 52.0,
        "wind_speed_mph": 14.0,
        "city_special_events": 1.0,
        "is_major_festival": 0,
        "cubs_home_game": 0,
        "bulls_home_game": 0,
    },
}

SEASON_PRESETS: dict[str, tuple[str, ...]] = {
    "Winter": (
        "Blizzard / polar vortex",
        "Bulls home game (United Center)",
        "Heavy rainstorm",
    ),
    "Spring": (
        "Heavy rainstorm",
        "Cubs home game + street events",
        "Major festival day",
    ),
    "Summer": (
        "Clear summer day",
        "Heat wave",
        "Major festival day",
    ),
    "Fall": (
        "Crisp fall day",
        "Heavy rainstorm",
        "Cubs home game + street events",
    ),
}


@lru_cache(maxsize=1)
def load_unified_data() -> pd.DataFrame:
    df = pd.read_parquet(UNIFIED_PARQUET, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    df["region"] = (
        pd.to_numeric(df["COMMUNITY AREA"], errors="coerce")
        .astype("Int64")
        .map(REGION_MAP)
        .fillna("OTHER")
    )
    return df.sort_values(["region", "demand_type", "business_category", "date"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def load_city_features_by_date() -> pd.DataFrame:
    unified = load_unified_data()
    return (
        unified.groupby("date", as_index=False)[CITY_WIDE_COLS]
        .first()
        .sort_values("date")
        .reset_index(drop=True)
    )


def historical_max_date() -> pd.Timestamp:
    return load_city_features_by_date()["date"].max().normalize()


def prediction_date_bounds() -> tuple[date, date]:
    today = date.today()
    return today - timedelta(days=PAST_LOOKBACK_DAYS), today + timedelta(days=FUTURE_MAX_HORIZON_DAYS)


def demand_type_label(demand_type: str) -> str:
    return DEMAND_TYPE_LABELS.get(demand_type, demand_type.replace("_", " ").title())


def _lookup_historical_city_features(
    target_date: pd.Timestamp,
    city_features: pd.DataFrame,
) -> dict[str, float | int] | None:
    row = city_features.loc[city_features["date"] == target_date]
    if row.empty:
        return None
    return row.iloc[0][CITY_WIDE_COLS].to_dict()


def _resolve_city_features(
    target_date: pd.Timestamp,
    city_features: pd.DataFrame,
) -> tuple[dict[str, float | int], str, dict[str, str]]:
    today = pd.Timestamp(date.today()).normalize()
    historical = _lookup_historical_city_features(target_date, city_features)
    if historical is not None:
        return historical, "historical_lookup", {"weather": "historical_lookup", "mode": "unified_dataset"}

    if target_date < today:
        values, sources = fetch_city_features_for_date(target_date, city_history=city_features)
        return values, "past_fetch", sources

    values, sources = fetch_city_features_for_date(target_date, city_history=city_features)
    return values, "live_forecast", sources


def infer_demand_type(
    region: str,
    business_category: str,
    city_values: dict[str, float | int],
) -> tuple[str, str]:
    unified = load_unified_data()
    subset = unified.loc[
        (unified["region"] == region) & (unified["business_category"] == business_category)
    ]
    if subset.empty:
        return "baseline", "default_baseline"

    matched = subset.copy()
    for flag in DEMAND_TYPE_FLAGS:
        if flag in city_values:
            matched = matched[matched[flag] == city_values[flag]]

    if len(matched) >= 30:
        demand_type = matched["demand_type"].mode().iloc[0]
        return str(demand_type), "matched_day_pattern"

    if city_values.get("is_major_festival"):
        fest = subset.loc[subset["is_major_festival"] == 1, "demand_type"]
        if not fest.empty:
            return str(fest.mode().iloc[0]), "festival_pattern"

    if float(city_values.get("city_special_events", 0) or 0) >= 3:
        events = subset.loc[subset["city_special_events"] >= 3, "demand_type"]
        if not events.empty:
            return str(events.mode().iloc[0]), "event_day_pattern"

    if float(city_values.get("snowfall_in", 0) or 0) >= 1:
        snowy = subset.loc[subset["snowfall_in"] >= 1, "demand_type"]
        if not snowy.empty:
            return str(snowy.mode().iloc[0]), "snow_day_pattern"

    return str(subset["demand_type"].mode().iloc[0]), "default_baseline"


def _coerce_override_value(key: str, value: float | int | bool) -> float | int:
    if key in {"is_major_festival", "cubs_home_game", "bulls_home_game"}:
        return int(bool(value))
    if key == "city_special_events":
        lo, hi = EVENT_BOUNDS[key]
        return float(max(lo, min(hi, float(value))))
    lo, hi = WEATHER_BOUNDS[key]
    return float(max(lo, min(hi, float(value))))


def _apply_feature_overrides(
    city_values: dict[str, float | int],
    feature_overrides: dict[str, float | int | bool] | None,
) -> tuple[dict[str, float | int], list[str]]:
    if not feature_overrides:
        return city_values, []

    updated = city_values.copy()
    overridden: list[str] = []
    allowed = set(WEATHER_OVERRIDE_COLS) | set(EVENT_OVERRIDE_COLS)
    for key, value in feature_overrides.items():
        if key not in allowed or value is None:
            continue
        updated[key] = _coerce_override_value(key, value)
        overridden.append(key)
    return updated, overridden


def fetch_auto_weather_events(
    target_date: pd.Timestamp | str,
) -> tuple[dict[str, float | int], dict[str, str]]:
    target_date = pd.Timestamp(target_date).normalize()
    city_features = load_city_features_by_date()
    city_values, _, feature_sources = _resolve_city_features(target_date, city_features)
    subset = {key: city_values[key] for key in WEATHER_OVERRIDE_COLS + EVENT_OVERRIDE_COLS}
    return subset, feature_sources


def scenario_preset_names() -> list[str]:
    return list(SCENARIO_PRESETS.keys())


def season_for_date(target_date: date | pd.Timestamp) -> str:
    month = pd.Timestamp(target_date).month
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def scenario_presets_for_date(target_date: date | pd.Timestamp) -> list[str]:
    season = season_for_date(target_date)
    presets = ["Auto (from date)"]
    for name in SEASON_PRESETS[season]:
        if name in SCENARIO_PRESETS:
            presets.append(name)
    return presets


def region_options() -> list[str]:
    available = sorted(load_unified_data()["region"].dropna().astype(str).unique())
    return [r for r in REGION_OPTIONS if r in available] or available


def business_category_options() -> list[str]:
    return sorted(load_unified_data()["business_category"].dropna().astype(str).unique())


def demand_type_options() -> list[str]:
    return sorted(load_unified_data()["demand_type"].dropna().astype(str).unique())


def _traffic_at(history: pd.Series, day: pd.Timestamp) -> float:
    if day not in history.index:
        return np.nan
    value = history.loc[day]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return float(value)


def _compute_traffic_lags(
    series_history: pd.DataFrame,
    target_date: pd.Timestamp,
) -> tuple[dict[str, float], str]:
    history = series_history.loc[series_history["date"] < target_date, ["date", TARGET]].copy()
    if history.empty:
        return (
            {"traffic_lag1": np.nan, "traffic_lag7": np.nan, "traffic_roll7": np.nan},
            "lags_unavailable",
        )

    history = (
        history.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")[TARGET]
    )

    lag1_date = target_date - pd.Timedelta(days=1)
    lag7_date = target_date - pd.Timedelta(days=7)
    roll_window = history.loc[
        (history.index >= target_date - pd.Timedelta(days=7))
        & (history.index <= target_date - pd.Timedelta(days=1))
    ]

    lag1 = _traffic_at(history, lag1_date)
    lag7 = _traffic_at(history, lag7_date)
    roll7 = float(roll_window.mean()) if not roll_window.empty else np.nan
    source = "historical_actuals"

    if pd.isna(lag1) or pd.isna(lag7) or pd.isna(roll7):
        yoy_shift = pd.DateOffset(years=1)
        yoy_lag1 = _traffic_at(history, lag1_date - yoy_shift)
        yoy_lag7 = _traffic_at(history, lag7_date - yoy_shift)
        yoy_roll = history.loc[
            (history.index >= lag7_date - yoy_shift) & (history.index <= lag1_date - yoy_shift)
        ]
        lag1 = lag1 if not pd.isna(lag1) else yoy_lag1
        lag7 = lag7 if not pd.isna(lag7) else yoy_lag7
        roll7 = roll7 if not pd.isna(roll7) else (float(yoy_roll.mean()) if not yoy_roll.empty else np.nan)
        if not pd.isna(yoy_lag1) or not pd.isna(yoy_lag7) or not yoy_roll.empty:
            source = "lags_yoy_proxy"

    if pd.isna(lag1) or pd.isna(lag7) or pd.isna(roll7):
        recent = history.tail(28)
        if not recent.empty:
            same_dow = recent[recent.index.dayofweek == target_date.dayofweek]
            proxy = same_dow if not same_dow.empty else recent
            lag1 = lag1 if not pd.isna(lag1) else float(proxy.iloc[-1])
            lag7 = lag7 if not pd.isna(lag7) else float(proxy.iloc[max(0, len(proxy) - 7)])
            roll7 = roll7 if not pd.isna(roll7) else float(proxy.tail(7).mean())
            source = "lags_recent_profile"

    return {
        "traffic_lag1": lag1,
        "traffic_lag7": lag7,
        "traffic_roll7": roll7,
    }, source


def _region_series_history(
    region: str,
    demand_type: str,
    business_category: str,
) -> pd.DataFrame:
    unified = load_unified_data()
    subset = unified.loc[
        (unified["region"] == region)
        & (unified["demand_type"] == demand_type)
        & (unified["business_category"] == business_category)
    ]
    if subset.empty:
        return pd.DataFrame(columns=["date", TARGET])
    return (
        subset.groupby("date", as_index=False)[TARGET]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_feature_row(
    target_date: pd.Timestamp | str,
    region: str,
    business_category: str,
    demand_type: str | None = None,
    feature_overrides: dict[str, float | int | bool] | None = None,
) -> tuple[pd.DataFrame, dict]:
    target_date = pd.Timestamp(target_date).normalize()
    min_date, max_date = prediction_date_bounds()

    if target_date.date() < min_date or target_date.date() > max_date:
        raise ValueError(
            f"Date must be between {min_date} and {max_date} "
            f"({PAST_LOOKBACK_DAYS} days back through {FUTURE_MAX_HORIZON_DAYS} days ahead)."
        )

    city_features = load_city_features_by_date()
    city_values, data_source, feature_sources = _resolve_city_features(target_date, city_features)
    city_values, overridden_fields = _apply_feature_overrides(city_values, feature_overrides)
    if overridden_fields:
        feature_sources["overrides"] = ", ".join(overridden_fields)

    if demand_type is None:
        demand_type, demand_type_source = infer_demand_type(region, business_category, city_values)
    else:
        demand_type_source = "user_selected"

    series_history = _region_series_history(region, demand_type, business_category)
    lag_values, lag_source = _compute_traffic_lags(series_history, target_date)

    row = {
        "date": target_date,
        "region": str(region),
        "demand_type": str(demand_type),
        "business_category": str(business_category),
        **city_values,
        **lag_values,
    }
    feature_df = pd.DataFrame([row])
    details = {
        "data_source": data_source,
        "region": region,
        "business_category": business_category,
        "demand_type": demand_type,
        "demand_type_label": demand_type_label(demand_type),
        "demand_type_source": demand_type_source,
        "overridden_fields": overridden_fields,
        "history_rows": int(len(series_history)),
        "feature_sources": {**feature_sources, "traffic_lags": lag_source, "demand_type": demand_type_source},
        "raw_row": row,
    }
    return feature_df, details


def predict_demand(
    target_date: pd.Timestamp | str,
    region: str,
    business_category: str,
    demand_type: str | None = None,
    feature_overrides: dict[str, float | int | bool] | None = None,
) -> tuple[int, pd.DataFrame, dict]:
    preprocessor, model, metadata = load_artifacts()
    raw_row, details = build_feature_row(
        target_date,
        region,
        business_category,
        demand_type,
        feature_overrides=feature_overrides,
    )

    X = create_date_features(raw_row)[NUM_COLS + BOOL_COLS + CAT_COLS]
    X_processed = preprocessor.transform(X)
    prediction = int(round(float(model.predict(X_processed)[0])))

    model_input = X.copy()
    model_input["prediction"] = prediction
    details["model_name"] = metadata["model_name"]
    details["target"] = metadata["target"]
    return prediction, model_input, details
