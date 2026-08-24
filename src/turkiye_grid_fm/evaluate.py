from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(error**2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    denominator = np.maximum(np.abs(y) + np.abs(p), epsilon)
    return float(200 * np.mean(np.abs(y - p) / denominator))
