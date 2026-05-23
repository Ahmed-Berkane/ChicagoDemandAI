"""Transfer learning: pretrain RNN on a proxy demand series, fine-tune on café traffic."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.optimizers import Adam

from src.modeling.evaluate import regression_metrics
from src.modeling.features import engineer_features
from src.modeling.keras_models import build_rnn_model
from src.modeling.preprocessor import DemandDataProcessor

RNNType = Literal["lstm", "gru"]


def _sequences_for_target(
    processor: DemandDataProcessor,
    df: pd.DataFrame,
    target_col: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n, lookback, n_feat) and y aligned to end of each window."""
    eng = engineer_features(df) if not processor.apply_engineering else df
    X_flat = processor.transform(df)
    y = eng[target_col].astype(float).values
    if len(X_flat) <= lookback:
        raise ValueError(f"Need more than {lookback} rows for sequences.")
    seq_x, seq_y = [], []
    for i in range(lookback, len(X_flat)):
        seq_x.append(X_flat[i - lookback : i])
        seq_y.append(y[i])
    return np.array(seq_x), np.array(seq_y)


def pretrain_and_finetune_rnn(
    processor: DemandDataProcessor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    rnn_type: RNNType = "gru",
    pretrain_target: str = "cta_total_rides",
    finetune_target: str = "customer_traffic",
    lookback: int | None = None,
    pretrain_epochs: int = 80,
    finetune_epochs: int = 60,
    finetune_lr: float = 1e-4,
    freeze_encoder: bool = True,
    random_state: int = 42,
):
    """
    Stage 1 — pretrain on citywide CTA ridership (similar demand dynamics).
    Stage 2 — fine-tune on café ``customer_traffic`` with a lower learning rate.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    lb = lookback or processor.lookback_
    RNNLayer = LSTM if rnn_type == "lstm" else GRU

    # Scale proxy target for stable MSE
    eng_train = engineer_features(train_df)
    cta_mean = eng_train[pretrain_target].mean()
    cta_std = eng_train[pretrain_target].std() or 1.0

    train_proxy = train_df.copy()
    train_proxy[pretrain_target] = (eng_train[pretrain_target] - cta_mean) / cta_std

    X_pre, y_pre = _sequences_for_target(processor, train_proxy, pretrain_target, lb)
    X_ft, y_ft = _sequences_for_target(processor, train_df, finetune_target, lb)

    val_ctx = pd.concat([train_df.tail(lb), val_df])
    X_val, y_val = _sequences_for_target(processor, val_ctx, finetune_target, lb)

    test_ctx = pd.concat([val_df.tail(lb), test_df])
    X_te, y_te = _sequences_for_target(processor, test_ctx, finetune_target, lb)

    n_feat = X_pre.shape[2]
    tf.random.set_seed(random_state)

    model = build_rnn_model(RNNLayer, lb, n_feat)
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])

    callbacks_pre = [EarlyStopping(patience=12, restore_best_weights=True, monitor="loss")]
    model.fit(X_pre, y_pre, epochs=pretrain_epochs, batch_size=32, verbose=0, callbacks=callbacks_pre)

    if freeze_encoder:
        for layer in model.layers[:-4]:
            layer.trainable = False

    model.compile(optimizer=Adam(finetune_lr), loss="mse", metrics=["mae"])
    callbacks_ft = [
        EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6, monitor="val_loss"),
    ]
    model.fit(
        X_ft,
        y_ft,
        validation_data=(X_val, y_val),
        epochs=finetune_epochs,
        batch_size=16,
        verbose=0,
        callbacks=callbacks_ft,
    )

    metrics = {}
    for split, X_s, y_s in [("val", X_val, y_val), ("test", X_te, y_te)]:
        pred = model.predict(X_s, verbose=0).ravel()
        metrics[split] = regression_metrics(y_s, pred)

    meta = {
        "model_kind": "keras_transfer",
        "rnn_type": rnn_type,
        "pretrain_target": pretrain_target,
        "finetune_target": finetune_target,
        "cta_mean": float(cta_mean),
        "cta_std": float(cta_std),
        "lookback": lb,
        "freeze_encoder": freeze_encoder,
    }
    return model, metrics, meta
