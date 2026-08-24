from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from turkiye_grid_fm.data import chronological_split
from turkiye_grid_fm.evaluate import mae, rmse, smape
from turkiye_grid_fm.models.foundation import GridFoundationModel
from turkiye_grid_fm.train import train_model
from turkiye_grid_fm.uncertainty import conformal_radius, interval_coverage
from turkiye_grid_fm.windows import GridWindowDataset, Standardizer

FEATURES = [
    "consumption_mwh", "renewable_mwh", "total_generation_mwh", "renewable_share",
    "naturalGas", "importCoal", "lignite", "mcp_tl_mwh",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
]
TARGETS = ["consumption_mwh", "renewable_mwh", "mcp_tl_mwh"]
HORIZONS = (1, 6, 24)


def predict(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, means, scales = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            mean, scale = model(x)
            ys.append(y.numpy())
            means.append(mean.numpy())
            scales.append(scale.numpy())
    return np.concatenate(ys), np.concatenate(means), np.concatenate(scales)


def inverse_multihorizon(values: np.ndarray, scaler: Standardizer) -> np.ndarray:
    shaped = values.reshape(len(values), len(HORIZONS), len(TARGETS))
    restored = shaped * scaler.std[None, None, :] + scaler.mean[None, None, :]
    return restored.reshape(len(values), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exploratory real-data research baseline.")
    parser.add_argument("--data", default="data/processed/grid_hourly.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="artifacts/exploratory_metrics.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame = pd.read_csv(args.data, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    train_df, val_df, test_df = chronological_split(frame)

    train_ds = GridWindowDataset(train_df, FEATURES, TARGETS, context=168, horizons=HORIZONS)
    val_ds = GridWindowDataset(
        val_df, FEATURES, TARGETS, context=168, horizons=HORIZONS,
        feature_scaler=train_ds.feature_scaler, target_scaler=train_ds.target_scaler,
    )
    test_ds = GridWindowDataset(
        test_df, FEATURES, TARGETS, context=168, horizons=HORIZONS,
        feature_scaler=train_ds.feature_scaler, target_scaler=train_ds.target_scaler,
    )

    model = GridFoundationModel(input_dim=len(FEATURES), target_dim=train_ds.target_dim)
    result = train_model(
        model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        epochs=args.epochs, lr=3e-4,
    )

    val_y, val_mean, _ = predict(model, DataLoader(val_ds, batch_size=512))
    test_y, test_mean, _ = predict(model, DataLoader(test_ds, batch_size=512))
    radius = conformal_radius(val_y, val_mean, coverage=0.90)

    y_native = inverse_multihorizon(test_y, train_ds.target_scaler)
    pred_native = inverse_multihorizon(test_mean, train_ds.target_scaler)
    lower_native = inverse_multihorizon(test_mean - radius, train_ds.target_scaler)
    upper_native = inverse_multihorizon(test_mean + radius, train_ds.target_scaler)

    report = {
        "status": "exploratory_only",
        "seed": args.seed,
        "training_loss": result.losses,
        "rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "overall": {
            "mae": mae(y_native, pred_native),
            "rmse": rmse(y_native, pred_native),
            "smape": smape(y_native, pred_native),
            "conformal_90_coverage_mean": float(interval_coverage(y_native, lower_native, upper_native).mean()),
        },
        "warning": "This runner is exploratory. Do not treat its test split as a preregistered confirmatory result.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
