"""Fit/transform pipeline with imputation and scaling for scoring."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.modeling.features import DROP_COLUMNS, engineer_features


class DemandDataProcessor:
    """
  Prepare daily demand data for sklearn / Keras.

    Fit on training data only; transform train/val/test or new rows the same way.
    """

    def __init__(self, target: str = "customer_traffic", *, apply_engineering: bool = False):
        self.target = target
        self.apply_engineering = apply_engineering
        self.feature_columns_: list[str] | None = None
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        self.lookback_: int = 14

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        eng = engineer_features(df) if self.apply_engineering else df.copy()
        drop = set(DROP_COLUMNS) | {self.target}
        X = eng.drop(columns=[c for c in eng.columns if c in drop], errors="ignore")
        X = X.select_dtypes(include=[np.number, bool]).astype(float)
        return X

    def fit(self, df: pd.DataFrame) -> DemandDataProcessor:
        X = self._prepare_frame(df)
        self.feature_columns_ = list(X.columns)
        self.imputer_.fit(X)
        self.scaler_.fit(self.imputer_.transform(X))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_columns_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = self._prepare_frame(df)
        for col in self.feature_columns_:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns_]
        X_imp = self.imputer_.transform(X)
        return self.scaler_.transform(X_imp)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform_with_target(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        eng = engineer_features(df) if self.apply_engineering else df
        y = eng[self.target].astype(float).values
        dates = pd.to_datetime(eng["date"])
        X = self.transform(df)
        return X, y, dates

    def build_lstm_sequences(
        self,
        df: pd.DataFrame,
        lookback: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shape X: (n_samples, lookback, n_features), y: (n_samples,)."""
        lb = lookback or self.lookback_
        X_flat, y, _ = self.transform_with_target(df)
        if len(X_flat) <= lb:
            raise ValueError(f"Need more than {lb} rows for LSTM sequences.")
        seq_X, seq_y = [], []
        for i in range(lb, len(X_flat)):
            seq_X.append(X_flat[i - lb : i])
            seq_y.append(y[i])
        return np.array(seq_X), np.array(seq_y)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> DemandDataProcessor:
        with Path(path).open("rb") as f:
            return pickle.load(f)
