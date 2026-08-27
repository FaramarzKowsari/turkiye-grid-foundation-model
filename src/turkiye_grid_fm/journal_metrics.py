from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricRow:
    model: str
    target: str
    horizon_hours: int
    n: int
    mae: float
    rmse: float
    smape: float


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _smape(y: np.ndarray, p: np.ndarray, epsilon: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y) + np.abs(p), epsilon)
    return float(200.0 * np.mean(np.abs(y - p) / denom))


def output_index(target_index: int, horizon_index: int, n_targets: int) -> int:
    """Index for arrays flattened horizon-major, matching GridWindowDataset."""
    return horizon_index * n_targets + target_index


def metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model: str,
    targets: list[str],
    horizons: list[int],
) -> list[MetricRow]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("y_true and y_pred must have the same [origins, outputs] shape.")
    expected = len(targets) * len(horizons)
    if y.shape[1] != expected:
        raise ValueError(f"Expected {expected} outputs, found {y.shape[1]}.")

    rows: list[MetricRow] = []
    for h_idx, horizon in enumerate(horizons):
        for t_idx, target in enumerate(targets):
            idx = output_index(t_idx, h_idx, len(targets))
            yt, yp = y[:, idx], p[:, idx]
            rows.append(
                MetricRow(
                    model=model,
                    target=target,
                    horizon_hours=int(horizon),
                    n=int(len(yt)),
                    mae=_mae(yt, yp),
                    rmse=_rmse(yt, yp),
                    smape=_smape(yt, yp),
                )
            )
    return rows


def interval_rows(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    targets: list[str],
    horizons: list[int],
) -> list[dict[str, float | int | str]]:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == lo.shape == hi.shape) or y.ndim != 2:
        raise ValueError("y_true, lower and upper must share [origins, outputs] shape.")
    rows: list[dict[str, float | int | str]] = []
    for h_idx, horizon in enumerate(horizons):
        for t_idx, target in enumerate(targets):
            idx = output_index(t_idx, h_idx, len(targets))
            covered = (y[:, idx] >= lo[:, idx]) & (y[:, idx] <= hi[:, idx])
            rows.append(
                {
                    "target": target,
                    "horizon_hours": int(horizon),
                    "n": int(len(covered)),
                    "coverage": float(np.mean(covered)),
                    "mean_interval_width": float(np.mean(hi[:, idx] - lo[:, idx])),
                }
            )
    return rows


def skill_against(reference_mae: float, model_mae: float) -> float:
    """Positive values indicate lower MAE than the reference."""
    if reference_mae <= 0:
        raise ValueError("reference_mae must be positive.")
    return float(1.0 - model_mae / reference_mae)
