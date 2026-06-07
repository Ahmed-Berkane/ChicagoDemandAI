"""Compare feature sets before changing the deployed model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from xgboost import XGBRegressor

from src.config import BOOL_COLS, CAT_COLS, NUM_COLS, RANDOM_STATE, TEST_PARQUET, TRAIN_PARQUET, VAL_PARQUET
from src.modeling.evaluate import regression_metrics
from src.modeling.features import apply_traffic_lags, create_date_features
from src.modeling.preprocessor import build_preprocessor

WEATHER_COLS = [
    "temperature_f",
    "precipitation_in",
    "snowfall_in",
    "humidity_pct",
    "wind_speed_mph",
]


@dataclass(frozen=True)
class FeatureSet:
    name: str
    num_cols: tuple[str, ...]
    bool_cols: tuple[str, ...]
    cat_cols: tuple[str, ...]


FEATURE_SETS = [
    FeatureSet(
        "current_with_business_category",
        tuple(NUM_COLS),
        tuple(BOOL_COLS),
        tuple(CAT_COLS),
    ),
    FeatureSet(
        "minus_weather",
        tuple(c for c in NUM_COLS if c not in WEATHER_COLS),
        tuple(BOOL_COLS),
        tuple(CAT_COLS),
    ),
    FeatureSet(
        "plus_business_minus_weather",
        tuple(c for c in NUM_COLS if c not in WEATHER_COLS),
        tuple(BOOL_COLS),
        tuple(CAT_COLS) + ("business_category",),
    ),
]


def _build_matrix(df: pd.DataFrame, feature_set: FeatureSet) -> tuple[pd.DataFrame, pd.Series]:
    X = create_date_features(df.drop(columns=["customer_traffic"]))
    y = df["customer_traffic"].astype(float)
    cols = list(feature_set.num_cols) + list(feature_set.bool_cols) + list(feature_set.cat_cols)
    return X[cols], y


def run_ablation() -> pd.DataFrame:
    train = pd.read_parquet(TRAIN_PARQUET, engine="pyarrow")
    val = pd.read_parquet(VAL_PARQUET, engine="pyarrow")
    test = pd.read_parquet(TEST_PARQUET, engine="pyarrow")
    train, val, test = apply_traffic_lags(train, val, test)

    rows: list[dict] = []
    for feature_set in FEATURE_SETS:
        X_train, y_train = _build_matrix(train, feature_set)
        X_val, y_val = _build_matrix(val, feature_set)
        X_test, y_test = _build_matrix(test, feature_set)

        preprocessor = build_preprocessor(
            num_cols=list(feature_set.num_cols),
            bool_cols=list(feature_set.bool_cols),
            cat_cols=list(feature_set.cat_cols),
        )
        X_train_p = preprocessor.fit_transform(X_train)
        X_val_p = preprocessor.transform(X_val)
        X_test_p = preprocessor.transform(X_test)
        n_features = X_train_p.shape[1]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train_p, y_train)

        for split_name, X_s, y_s in [
            ("val", X_val_p, y_val),
            ("test", X_test_p, y_test),
        ]:
            metrics = regression_metrics(y_s, model.predict(X_s), n_features)
            rows.append(
                {
                    "feature_set": feature_set.name,
                    "split": split_name,
                    "n_input_features": len(feature_set.num_cols)
                    + len(feature_set.bool_cols)
                    + len(feature_set.cat_cols),
                    "n_encoded_features": n_features,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def print_ablation_report(results: pd.DataFrame) -> None:
    baseline = results[
        (results["feature_set"] == "current_with_business_category") & (results["split"] == "test")
    ].iloc[0]

    print("\nFeature ablation (XGBoost, test split vs current model):\n")
    for feature_set in results["feature_set"].unique():
        test_row = results[(results["feature_set"] == feature_set) & (results["split"] == "test")].iloc[0]
        val_row = results[(results["feature_set"] == feature_set) & (results["split"] == "val")].iloc[0]
        delta_adj_r2 = test_row["adj_r2"] - baseline["adj_r2"]
        delta_mae = test_row["mae"] - baseline["mae"]
        print(f"{feature_set}")
        print(
            f"  val  MAE={val_row['mae']:.3f}  AdjR2={val_row['adj_r2']:.4f}  "
            f"test MAE={test_row['mae']:.3f}  AdjR2={test_row['adj_r2']:.4f}  "
            f"(delta test AdjR2={delta_adj_r2:+.4f}, delta test MAE={delta_mae:+.3f})"
        )


if __name__ == "__main__":
    report = run_ablation()
    print_ablation_report(report)
