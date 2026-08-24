from __future__ import annotations

import numpy as np


def conformal_radius(y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.90) -> np.ndarray:
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1).")
    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    if residuals.ndim == 1:
        residuals = residuals[:, None]
    n = residuals.shape[0]
    q_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return np.quantile(residuals, q_level, axis=0, method="higher")


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true)
    return ((y >= lower) & (y <= upper)).mean(axis=0)
