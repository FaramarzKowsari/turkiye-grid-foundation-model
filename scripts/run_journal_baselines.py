from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from turkiye_grid_fm.journal_metrics import metric_rows

TARGETS = ["consumption_mwh", "renewable_mwh", "mcp_tl_mwh"]
HORIZONS = [1, 6, 24]
CONTEXT = 168


def baseline_predictions(values: np.ndarray, origins: np.ndarray, kind: str) -> np.ndarray:
    out = np.empty((len(origins), len(TARGETS) * len(HORIZONS)), dtype=float)
    for i, origin in enumerate(origins):
        columns = []
        for h in HORIZONS:
            if kind == "persistence":
                pred = values[origin - 1]
            elif kind == "seasonal_naive_24":
                pred = values[origin + h - 1 - 24]
            elif kind == "seasonal_naive_168":
                pred = values[origin + h - 1 - 168]
            else:
                raise ValueError(kind)
            columns.append(pred)
        out[i] = np.concatenate(columns)
    return out


def truth_matrix(values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    rows = []
    for origin in origins:
        rows.append(np.concatenate([values[origin + h - 1] for h in HORIZONS]))
    return np.asarray(rows, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run journal-grade exploratory temporal baselines.")
    parser.add_argument("--data", default="data/processed/exploratory_2021_2025.csv")
    parser.add_argument("--out-dir", default="artifacts/journal_v03")
    args = parser.parse_args()

    frame = pd.read_csv(args.data, parse_dates=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    complete = frame[TARGETS].dropna()
    if len(complete) < CONTEXT + max(HORIZONS) + 1000:
        raise SystemExit("Not enough complete exploratory rows for journal baseline evaluation.")

    # Require a contiguous hourly grid for the baseline evidence table.
    idx = pd.DatetimeIndex(complete.index)
    gaps = idx.to_series().diff().dropna().dt.total_seconds().div(3600)
    if (gaps != 1.0).any():
        raise SystemExit(
            "Exploratory history is not fully contiguous. Review availability_audit.json before modeling."
        )

    values = complete[TARGETS].to_numpy(dtype=float)
    n = len(values)
    first_origin = CONTEXT
    last_origin_exclusive = n - max(HORIZONS) + 1
    all_origins = np.arange(first_origin, last_origin_exclusive)

    train_cut = int(len(all_origins) * 0.70)
    val_cut = int(len(all_origins) * 0.85)
    eval_origins = all_origins[val_cut:]
    if len(eval_origins) == 0:
        raise SystemExit("No exploratory evaluation origins.")

    y = truth_matrix(values, eval_origins)
    models = ["persistence", "seasonal_naive_24", "seasonal_naive_168"]
    rows = []
    predictions = pd.DataFrame({"forecast_origin": idx[eval_origins].astype(str)})
    timing = {}

    for model in models:
        start = time.perf_counter()
        pred = baseline_predictions(values, eval_origins, model)
        timing[model] = time.perf_counter() - start
        for row in metric_rows(y, pred, model=model, targets=TARGETS, horizons=HORIZONS):
            rows.append(row.__dict__)
        for h_idx, horizon in enumerate(HORIZONS):
            for t_idx, target in enumerate(TARGETS):
                col = h_idx * len(TARGETS) + t_idx
                predictions[f"truth__{target}__h{horizon}"] = y[:, col]
                predictions[f"{model}__{target}__h{horizon}"] = pred[:, col]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "baseline_metrics.csv", index=False)
    predictions.to_csv(out_dir / "baseline_predictions.csv", index=False)

    summary = {
        "status": "exploratory_only",
        "confirmatory_holdout": "UNDEFINED_AND_NOT_REQUESTED",
        "rows_complete": int(n),
        "forecast_origins_total": int(len(all_origins)),
        "exploratory_train_origins": int(train_cut),
        "exploratory_validation_origins": int(val_cut - train_cut),
        "exploratory_evaluation_origins": int(len(eval_origins)),
        "evaluation_start": idx[eval_origins[0]].isoformat(),
        "evaluation_end": idx[eval_origins[-1]].isoformat(),
        "baseline_inference_seconds": timing,
        "warning": "These are exploratory baseline results, not confirmatory evidence.",
    }
    (out_dir / "baseline_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
