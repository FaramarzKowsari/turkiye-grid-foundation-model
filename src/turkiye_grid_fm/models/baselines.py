from __future__ import annotations

import numpy as np


def persistence_forecast(last_observation: np.ndarray, horizons: int) -> np.ndarray:
    last = np.asarray(last_observation)
    if last.ndim != 1:
        raise ValueError("last_observation must be one-dimensional.")
    return np.tile(last, int(horizons))


def seasonal_naive(history: np.ndarray, period: int, horizons: list[int]) -> np.ndarray:
    history = np.asarray(history)
    if history.ndim != 2:
        raise ValueError("history must be [time, targets].")
    if len(history) < period:
        raise ValueError("history must contain at least one full seasonal period.")
    forecasts = []
    for h in horizons:
        idx = -period + ((h - 1) % period)
        forecasts.append(history[idx])
    return np.concatenate(forecasts, axis=0)
