"""Feature importance for tabular, hybrid, and sequence champions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression


def tabular_importance(
    model: Any,
    feature_names: list[str],
    *,
    kind: str,
) -> pd.DataFrame:
    """Native importance for sklearn / hybrid components."""
    if kind == "hybrid":
        imp = model.xgb.feature_importances_
        label = "hybrid (XGBoost component)"
    elif isinstance(model, LinearRegression):
        imp = np.abs(model.coef_)
        label = "linear (|coefficient|)"
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        label = kind
    else:
        raise TypeError(f"No native importance for {type(model)}")

    out = pd.DataFrame({"feature": feature_names, "importance": imp, "source": label})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def hybrid_component_importance(
    hybrid_model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Linear coefficients + XGBoost gains for the hybrid stack."""
    lin = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.abs(hybrid_model.linear.coef_),
            "source": "hybrid (linear |coef|)",
        }
    )
    xgb = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": hybrid_model.xgb.feature_importances_,
            "source": "hybrid (XGBoost gain)",
        }
    )
    return (
        pd.concat([lin, xgb], ignore_index=True)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_tabular(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_repeats: int = 8,
    random_state: int = 42,
    sample_rows: int | None = 400,
) -> pd.DataFrame:
    if sample_rows and len(X) > sample_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), sample_rows, replace=False)
        X, y = X[idx], y[idx]

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": result.importances_mean,
                "source": "permutation (validation MAE increase)",
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def sequence_permutation_importance(
    model: Any,
    processor: Any,
    context_df: pd.DataFrame,
    y_true: np.ndarray,
    feature_names: list[str],
    lookback: int,
    *,
    n_repeats: int = 2,
    top_n: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permute each input feature in the context frame and measure MAE increase
    on sequence predictions (for LSTM / GRU / transfer winners).
    """
    from src.modeling.transfer import _sequences_for_target

    X_base, _ = _sequences_for_target(
        processor, context_df, "customer_traffic", lookback
    )
    base_pred = model.predict(X_base, verbose=0).ravel()
    base_mae = np.mean(np.abs(y_true - base_pred))

    from src.modeling.features import engineer_features

    eng = engineer_features(context_df) if "traffic_lag1" not in context_df.columns else context_df.copy()
    if "customer_traffic" not in eng.columns:
        raise ValueError("context_df must include customer_traffic")

    rng = np.random.default_rng(random_state)
    rows: list[dict] = []

    for feat in feature_names:
        if feat not in eng.columns:
            continue
        deltas = []
        for _ in range(n_repeats):
            pert = eng.copy()
            pert[feat] = rng.permutation(pert[feat].values)
            X_p, y_p = _sequences_for_target(processor, pert, "customer_traffic", lookback)
            pred = model.predict(X_p, verbose=0).ravel()
            deltas.append(np.mean(np.abs(y_p - pred)) - base_mae)
        rows.append(
            {
                "feature": feat,
                "importance": float(np.mean(deltas)),
                "source": "sequence permutation (Δ MAE)",
            }
        )

    out = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    return out.head(top_n)


def importance_for_champion(
    *,
    best_kind: str,
    best_model: Any,
    feature_names: list[str],
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    processor: Any = None,
    val_context_df: pd.DataFrame | None = None,
    y_seq_val: np.ndarray | None = None,
    lookback: int | None = None,
) -> pd.DataFrame:
    """Single entry point used by the notebook."""
    if best_kind == "hybrid":
        return hybrid_component_importance(best_model, feature_names)
    if best_kind == "sklearn" and isinstance(best_model, LinearRegression):
        return tabular_importance(best_model, feature_names, kind="linear_regression")
    if best_kind == "sklearn" and hasattr(best_model, "feature_importances_"):
        return tabular_importance(best_model, feature_names, kind="sklearn")
    if best_kind in ("keras", "keras_transfer"):
        if processor is None or val_context_df is None or y_seq_val is None:
            raise ValueError("Sequence model needs processor, val_context_df, y_seq_val")
        lb = lookback or processor.lookback_
        return sequence_permutation_importance(
            best_model,
            processor,
            val_context_df,
            y_seq_val,
            feature_names,
            lb,
        )
    if X_val is not None and y_val is not None:
        return permutation_importance_tabular(best_model, X_val, y_val, feature_names)
    raise ValueError(f"Cannot compute importance for kind={best_kind}")
