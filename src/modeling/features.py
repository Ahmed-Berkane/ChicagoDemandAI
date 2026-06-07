import numpy as np
import pandas as pd

from src.config import (
    BOOL_COLS,
    CAT_COLS,
    NUM_COLS,
    SERIES_COLS,
    TARGET,
)
from src.regions import REGION_MAP


def add_traffic_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(SERIES_COLS + ["date"])
    grouped = out.groupby(SERIES_COLS, sort=False)[TARGET]
    out["traffic_lag1"] = grouped.shift(1)
    out["traffic_lag7"] = grouped.shift(7)
    out["traffic_roll7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    return out


def apply_traffic_lags(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [
            train.assign(_split="train"),
            val.assign(_split="val"),
            test.assign(_split="test"),
        ],
        ignore_index=True,
    )
    combined = add_traffic_lags(combined)
    train_out = combined.loc[combined["_split"] == "train"].drop(columns="_split")
    val_out = combined.loc[combined["_split"] == "val"].drop(columns="_split")
    test_out = combined.loc[combined["_split"] == "test"].drop(columns="_split")
    return train_out, val_out, test_out


def create_date_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["date"] = pd.to_datetime(X["date"], errors="coerce")

    month = X["date"].dt.month
    day_of_week = X["date"].dt.dayofweek
    day_of_year = X["date"].dt.dayofyear
    min_year = X["date"].dt.year.min()
    X["year"] = X["date"].dt.year - min_year

    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    X["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    X["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    X["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    X = X.drop(columns=["date"])

    if "COMMUNITY AREA" in X.columns:
        X["region"] = (
            pd.to_numeric(X["COMMUNITY AREA"], errors="coerce")
            .astype("Int64")
            .map(REGION_MAP)
            .fillna("OTHER")
        )
        X = X.drop(columns=["COMMUNITY AREA"])
    elif "region" in X.columns:
        X["region"] = X["region"].astype(str).fillna("OTHER")

    for col in NUM_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in BOOL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("int64")

    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype(str)

    return X


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = create_date_features(df.drop(columns=[TARGET]))
    y = df[TARGET].astype(float)
    feature_cols = NUM_COLS + BOOL_COLS + CAT_COLS
    return X[feature_cols], y
