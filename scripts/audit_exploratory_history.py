from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLUMNS = ["consumption_mwh", "renewable_mwh", "mcp_tl_mwh"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exploratory multi-year EPİAŞ history.")
    parser.add_argument("--data", default="data/processed/exploratory_2021_2025.csv")
    parser.add_argument("--out-dir", default="artifacts/journal_v03")
    args = parser.parse_args()

    frame = pd.read_csv(args.data, parse_dates=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        raise SystemExit("No exploratory rows found.")

    ts = pd.DatetimeIndex(frame["timestamp"])
    duplicates = int(ts.duplicated().sum())
    unique = pd.DatetimeIndex(ts.drop_duplicates()).sort_values()
    expected = pd.date_range(unique.min(), unique.max(), freq="h")
    missing = expected.difference(unique)

    work = frame.set_index("timestamp").sort_index()
    monthly = []
    for period, group in work.groupby(work.index.to_period("M")):
        p_start = group.index.min()
        p_end = group.index.max()
        expected_month = pd.date_range(p_start, p_end, freq="h")
        idx = pd.DatetimeIndex(group.index)
        row = {
            "month": str(period),
            "rows": int(len(group)),
            "unique_hours": int(idx.nunique()),
            "duplicates": int(idx.duplicated().sum()),
            "missing_between_first_last": int(len(expected_month.difference(idx))),
        }
        for col in KEY_COLUMNS:
            row[f"{col}_nulls"] = int(group[col].isna().sum()) if col in group else int(len(group))
        row["complete_three_target_rows"] = int(group[KEY_COLUMNS].dropna().shape[0])
        monthly.append(row)

    deltas = unique.to_series().diff().dropna().dt.total_seconds().div(3600)
    report = {
        "status": "exploratory_only",
        "confirmatory_holdout": "UNDEFINED_AND_NOT_REQUESTED",
        "rows": int(len(frame)),
        "unique_hours": int(len(unique)),
        "duplicate_timestamps": duplicates,
        "range_start": unique.min().isoformat(),
        "range_end": unique.max().isoformat(),
        "expected_hours_between_range_endpoints": int(len(expected)),
        "missing_expected_hours_between_range_endpoints": int(len(missing)),
        "max_gap_hours": float(deltas.max()) if len(deltas) else 0.0,
        "key_nulls": {col: int(work[col].isna().sum()) for col in KEY_COLUMNS},
        "complete_three_target_rows": int(work[KEY_COLUMNS].dropna().shape[0]),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "availability_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(monthly).to_csv(out_dir / "monthly_coverage.csv", index=False)

    print(json.dumps(report, indent=2))
    print(f"Monthly table: {out_dir / 'monthly_coverage.csv'}")
    if unique.max() >= pd.Timestamp("2026-01-01", tz=unique.tz):
        raise SystemExit("SAFETY STOP: exploratory data unexpectedly include 2026+ timestamps.")


if __name__ == "__main__":
    main()
