import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def adjusted_r2(y_true, y_pred, n_features: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n <= n_features + 1:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def regression_metrics(y_true, y_pred, n_features: int) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(y_true, y_pred, n_features)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "adj_r2": adj_r2}
