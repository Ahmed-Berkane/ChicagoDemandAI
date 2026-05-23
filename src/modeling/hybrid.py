"""Hybrid demand model: Linear + XGBoost + small neural meta-learner."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.modeling.evaluate import regression_metrics


def _build_meta_mlp(n_inputs: int):
    import tensorflow as tf
    from tensorflow.keras import Sequential, regularizers
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    l2 = regularizers.l2(1e-4)
    model = Sequential(
        [
            Dense(32, activation="relu", kernel_regularizer=l2, input_shape=(n_inputs,)),
            Dropout(0.25),
            Dense(16, activation="relu", kernel_regularizer=l2),
            Dropout(0.15),
            Dense(1, kernel_regularizer=l2),
        ]
    )
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model, EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")


class HybridDemandModel:
    """
    Stacked ensemble:

    1. Linear regression — trend / linear structure
    2. XGBoost — nonlinear interactions
    3. Small MLP — combines base predictions (and optional extra signals)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.linear = LinearRegression()
        self.xgb = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        )
        self.meta_model = None
        self.use_extra_meta_features = True

    def _meta_features(self, X: np.ndarray, lin: np.ndarray, xgb: np.ndarray) -> np.ndarray:
        stack = np.column_stack([lin, xgb])
        if self.use_extra_meta_features and X.shape[1] >= 3:
            # Top signals for the meta-net (traffic lag + CTA if present in X)
            extras = X[:, : min(5, X.shape[1])]
            return np.hstack([stack, extras])
        return stack

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        meta_epochs: int = 80,
    ) -> HybridDemandModel:
        self.linear.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)

        p_lin_tr = self.linear.predict(X_train)
        p_xgb_tr = self.xgb.predict(X_train)
        meta_X_tr = self._meta_features(X_train, p_lin_tr, p_xgb_tr)

        if X_val is not None and y_val is not None:
            p_lin_va = self.linear.predict(X_val)
            p_xgb_va = self.xgb.predict(X_val)
            meta_X_va = self._meta_features(X_val, p_lin_va, p_xgb_va)
            val_data = (meta_X_va, y_val)
        else:
            val_data = None

        self.meta_model, es = _build_meta_mlp(meta_X_tr.shape[1])
        callbacks = [es] if val_data else []
        self.meta_model.fit(
            meta_X_tr,
            y_train,
            validation_data=val_data,
            epochs=meta_epochs,
            batch_size=32,
            verbose=0,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p_lin = self.linear.predict(X)
        p_xgb = self.xgb.predict(X)
        meta_X = self._meta_features(X, p_lin, p_xgb)
        return self.meta_model.predict(meta_X, verbose=0).ravel()

    def evaluate_splits(
        self,
        splits: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, dict[str, float]]:
        out = {}
        for name, (X_s, y_s) in splits.items():
            out[name] = regression_metrics(y_s, self.predict(X_s))
        return out

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "hybrid_linear.pkl", "wb") as f:
            pickle.dump(self.linear, f)
        with open(directory / "hybrid_xgb.pkl", "wb") as f:
            pickle.dump(self.xgb, f)
        self.meta_model.save(directory / "hybrid_meta.keras")
        with open(directory / "hybrid_config.pkl", "wb") as f:
            pickle.dump({"use_extra_meta_features": self.use_extra_meta_features}, f)

    @classmethod
    def load(cls, directory: str | Path) -> HybridDemandModel:
        import tensorflow as tf

        directory = Path(directory)
        obj = cls()
        with open(directory / "hybrid_linear.pkl", "rb") as f:
            obj.linear = pickle.load(f)
        with open(directory / "hybrid_xgb.pkl", "rb") as f:
            obj.xgb = pickle.load(f)
        obj.meta_model = tf.keras.models.load_model(directory / "hybrid_meta.keras")
        cfg_path = directory / "hybrid_config.pkl"
        if cfg_path.exists():
            with open(cfg_path, "rb") as f:
                cfg = pickle.load(f)
            obj.use_extra_meta_features = cfg.get("use_extra_meta_features", True)
        return obj
