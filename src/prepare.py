import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_MANIFEST,
    RANDOM_STATE,
    TARGET,
    TEST_PARQUET,
    TRAIN_PARQUET,
    UNIFIED_PARQUET,
    VAL_PARQUET,
)
from src.regions import save_region_map

EDA_CAT_COLS = [
    "Risk",
    "is_holiday",
    "Results",
    "month",
    "primary_type_mode",
    "is_biweekly_payday",
    "demand_type",
    "is_payweek",
    "arrest",
    "is_weekend",
    "city_special_events",
    "had_inspection",
    "COMMUNITY AREA",
    "has_crime",
    "day_of_week",
    "is_semimonthly_payday",
    "cubs_home_game",
    "is_major_festival",
    "bulls_home_game",
    "business_category",
]

SELECTED_CAT_FEATURES = [
    "month",
    "cubs_home_game",
    "is_weekend",
    "bulls_home_game",
    "day_of_week",
    "city_special_events",
    "is_holiday",
    "demand_type",
    "is_payweek",
    "is_major_festival",
    "is_semimonthly_payday",
    "COMMUNITY AREA",
]


def _engineer_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    out["year"] = (out["date"].dt.year - out["date"].dt.year.min()).astype(float)
    out["day_of_year_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365)
    out["day_of_year_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365)
    out = out.drop(columns=["day_of_year", "days_since_biweekly_payday"], errors="ignore")
    return out


def _select_numeric_features(df: pd.DataFrame) -> list[str]:
    return [
        "temperature_f",
        "precipitation_in",
        "snowfall_in",
        "humidity_pct",
        "wind_speed_mph",
        "cta_total_rides",
        "year",
        "day_of_year_sin",
        "day_of_year_cos",
    ]


def _select_categorical_features(df: pd.DataFrame, sample_size: int = 100_000) -> tuple[list[str], list[str]]:
    available = [c for c in EDA_CAT_COLS if c in df.columns]
    encoded = pd.get_dummies(df[available], drop_first=True, dtype=float)
    y = df[TARGET].astype(float)

    if len(encoded) > sample_size:
        sample_idx = encoded.sample(sample_size, random_state=RANDOM_STATE).index
        encoded_fit = encoded.loc[sample_idx]
        y_fit = y.loc[sample_idx]
    else:
        encoded_fit = encoded
        y_fit = y

    f_vals, _ = f_regression(encoded, y)
    f_scores = pd.Series(f_vals, index=encoded.columns).sort_values(ascending=False)
    threshold = f_scores.quantile(0.30)
    dropped = f_scores[f_scores <= threshold].index.tolist()

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(encoded_fit, y_fit)

    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(encoded_fit)
    importance = pd.DataFrame(
        {"feature": encoded_fit.columns, "importance": np.abs(shap_values).mean(axis=0)}
    ).sort_values("importance", ascending=False)
    importance["cum_share"] = (importance["importance"] / importance["importance"].sum()).cumsum()
    shap_selected = importance.loc[importance["cum_share"] <= 0.98, "feature"].tolist()

    return shap_selected, dropped


def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date")
    train_df, temp_df = train_test_split(df, test_size=0.30, shuffle=False)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, shuffle=False)
    return train_df, val_df, test_df


def _save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for path, frame in (
        (TRAIN_PARQUET, train_df),
        (VAL_PARQUET, val_df),
        (TEST_PARQUET, test_df),
    ):
        frame.to_parquet(path, engine="pyarrow", compression="snappy")


def prepare_datasets() -> dict:
    if not UNIFIED_PARQUET.exists():
        raise FileNotFoundError(f"Missing unified dataset: {UNIFIED_PARQUET}")

    unified = pd.read_parquet(UNIFIED_PARQUET, engine="pyarrow")
    train_df, val_df, test_df = _chronological_split(unified)
    _save_splits(train_df, val_df, test_df)

    region_map = save_region_map()
    eda_df = _engineer_eda_features(train_df)
    num_features = _select_numeric_features(eda_df)
    shap_selected, dropped = _select_categorical_features(eda_df)

    manifest = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "selected_numeric_features": num_features,
        "shap_selected_onehot_features": shap_selected,
        "dropped_low_fscore_onehot_features": dropped[:20],
        "selected_categorical_columns": SELECTED_CAT_FEATURES,
        "region_count": len(region_map),
    }
    FEATURE_MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"Train: {train_df.shape}  Val: {val_df.shape}  Test: {test_df.shape}", flush=True)
    print(f"Selected numeric features: {len(num_features)}", flush=True)
    print(f"SHAP-selected one-hot features: {len(shap_selected)}", flush=True)
    print(f"Saved region map ({len(region_map)} areas) and feature manifest", flush=True)

    return manifest
