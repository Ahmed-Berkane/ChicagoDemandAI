"""Keras RNN builders with dropout and L2 regularization."""

from __future__ import annotations

from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout

# Tunable regularization (used by notebook + scoring)
L2_REG = regularizers.l2(1e-4)
RNN_DROPOUT = 0.2
RNN_RECURRENT_DROPOUT = 0.1
POST_RNN_DROPOUT = 0.3
PRE_OUTPUT_DROPOUT = 0.2


def build_rnn_model(RNNLayer, lookback: int, n_features: int) -> Sequential:
    """
    Two-layer LSTM or GRU with L2 penalties and dropout.

    - ``dropout`` / ``recurrent_dropout`` on recurrent layers (internal)
    - ``Dropout`` after each recurrent block (external)
    - L2 on kernels (recurrent + dense)
    """
    return Sequential(
        [
            RNNLayer(
                64,
                return_sequences=True,
                input_shape=(lookback, n_features),
                kernel_regularizer=L2_REG,
                recurrent_regularizer=L2_REG,
                dropout=RNN_DROPOUT,
                recurrent_dropout=RNN_RECURRENT_DROPOUT,
            ),
            Dropout(POST_RNN_DROPOUT),
            RNNLayer(
                32,
                kernel_regularizer=L2_REG,
                recurrent_regularizer=L2_REG,
                dropout=RNN_DROPOUT,
                recurrent_dropout=RNN_RECURRENT_DROPOUT,
            ),
            Dropout(POST_RNN_DROPOUT),
            Dense(16, activation="relu", kernel_regularizer=L2_REG),
            Dropout(PRE_OUTPUT_DROPOUT),
            Dense(1, kernel_regularizer=L2_REG),
        ]
    )
