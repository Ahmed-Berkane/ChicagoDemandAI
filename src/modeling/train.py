import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.config import BOOL_COLS, CAT_COLS, MODELS_DIR, NUM_COLS, RANDOM_STATE, TEST_PARQUET, TRAIN_PARQUET, VAL_PARQUET
from src.modeling.evaluate import regression_metrics
from src.modeling.features import apply_traffic_lags, build_feature_matrix
from src.modeling.persist import save_artifacts
from src.modeling.preprocessor import build_preprocessor


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(TRAIN_PARQUET, engine="pyarrow")
    val = pd.read_parquet(VAL_PARQUET, engine="pyarrow")
    test = pd.read_parquet(TEST_PARQUET, engine="pyarrow")
    return train, val, test


def _create_model(name: str):
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            max_samples=50_000,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return models[name]


def _encoded_feature_names(preprocessor) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def _model_feature_ranking(model, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        label = "importance"
    elif hasattr(model, "coef_"):
        values = abs(model.coef_.ravel())
        label = "abs_coefficient"
    else:
        return pd.DataFrame()

    return (
        pd.DataFrame({"feature": feature_names, label: values})
        .sort_values(label, ascending=False)
        .reset_index(drop=True)
    )


def _print_best_model_summary(
    best_name: str,
    best_model,
    preprocessor,
    results_df: pd.DataFrame,
) -> None:
    val_metrics = results_df[(results_df["model"] == best_name) & (results_df["split"] == "val")].iloc[0]
    test_metrics = results_df[(results_df["model"] == best_name) & (results_df["split"] == "test")].iloc[0]
    input_features = NUM_COLS + BOOL_COLS + CAT_COLS
    encoded_features = _encoded_feature_names(preprocessor)

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'=' * 60}")
    print(
        f"Validation - MAE: {val_metrics['mae']:.3f}  "
        f"RMSE: {val_metrics['rmse']:.3f}  "
        f"Adj R2: {val_metrics['adj_r2']:.4f}"
    )
    print(
        f"Test       - MAE: {test_metrics['mae']:.3f}  "
        f"RMSE: {test_metrics['rmse']:.3f}  "
        f"Adj R2: {test_metrics['adj_r2']:.4f}"
    )

    print(f"\nInput features ({len(input_features)}):")
    for feature in input_features:
        print(f"  - {feature}")

    print(f"\nEncoded features after preprocessing ({len(encoded_features)}):")
    for feature in encoded_features:
        print(f"  - {feature}")

    ranking = _model_feature_ranking(best_model, encoded_features)
    if not ranking.empty:
        score_col = ranking.columns[1]
        print(f"\nTop {min(10, len(ranking))} features by {score_col}:")
        for _, row in ranking.head(10).iterrows():
            print(f"  - {row['feature']}: {row[score_col]:.4f}")


def run_training() -> pd.DataFrame:
    train, val, test = _load_splits()
    train, val, test = apply_traffic_lags(train, val, test)

    X_train, y_train = build_feature_matrix(train)
    X_val, y_val = build_feature_matrix(val)
    X_test, y_test = build_feature_matrix(test)

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    n_features = X_train_processed.shape[1]

    results = []
    best_name = None
    best_adj_r2 = float("-inf")
    best_model = None

    for name in ("linear_regression", "random_forest", "gradient_boosting", "xgboost"):
        model = _create_model(name)
        print(f"Training {name}...", flush=True)
        model.fit(X_train_processed, y_train)
        for split_name, X_s, y_s in [
            ("val", X_val_processed, y_val),
            ("test", X_test_processed, y_test),
        ]:
            pred = model.predict(X_s)
            metrics = regression_metrics(y_s, pred, n_features)
            metrics.update({"model": name, "split": split_name})
            results.append(metrics)

        val_adj_r2 = regression_metrics(y_val, model.predict(X_val_processed), n_features)["adj_r2"]
        if val_adj_r2 > best_adj_r2:
            best_adj_r2 = val_adj_r2
            best_name = name
            best_model = model

    results_df = pd.DataFrame(results)
    pivot = results_df.pivot(index="model", columns="split", values=["mae", "rmse", "adj_r2"])

    print("\nModel comparison (selection by val Adj R2):")
    print(pivot.to_string())

    _print_best_model_summary(best_name, best_model, preprocessor, results_df)

    print(f"\nRetraining {best_name} on full dataset...", flush=True)
    full_df = pd.concat([train, val, test], ignore_index=True)
    X_full, y_full = build_feature_matrix(full_df)
    final_preprocessor = build_preprocessor()
    X_full_processed = final_preprocessor.fit_transform(X_full)
    final_model = _create_model(best_name)
    final_model.fit(X_full_processed, y_full)

    metadata = save_artifacts(final_preprocessor, final_model, best_name)
    print(f"Saved artifacts to {MODELS_DIR}/")
    print(f"  - preprocessor.pkl")
    print(f"  - best_model.pkl ({best_name})")
    print(f"  - model_metadata.pkl ({len(metadata['encoded_features'])} encoded features)")

    return results_df
