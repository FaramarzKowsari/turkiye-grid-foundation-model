from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from turkiye_grid_fm.data import build_hourly_frame
from turkiye_grid_fm.epias import EpiasClient

HARD_STOP = pd.Timestamp("2026-01-01T00:00:00+03:00")
DATASETS = ("consumption", "generation", "mcp")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def month_bounds(start: pd.Timestamp, end: pd.Timestamp):
    cursor = start
    while cursor <= end:
        next_month = (cursor + pd.offsets.MonthBegin(1)).normalize()
        chunk_end = min(end, next_month - pd.Timedelta(hours=1))
        yield cursor, chunk_end
        cursor = next_month


def expected_hours(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return len(pd.date_range(start, end, freq="h"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch monthly exploratory EPİAŞ history while hard-blocking 2026+ data."
    )
    parser.add_argument("--start", default="2021-01-01T00:00:00+03:00")
    parser.add_argument("--end", default="2025-12-31T23:00:00+03:00")
    parser.add_argument("--raw-dir", default="data/raw/journal_v03")
    parser.add_argument("--processed-dir", default="data/processed/journal_v03")
    parser.add_argument("--out", default="data/processed/exploratory_2021_2025.csv")
    parser.add_argument("--manifest", default="artifacts/journal_v03/acquisition_manifest.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None:
        raise SystemExit("Start/end must include an explicit timezone offset.")
    if start >= HARD_STOP or end >= HARD_STOP:
        raise SystemExit(
            "SAFETY STOP: v0.3 exploratory acquisition is forbidden from requesting 2026+ data. "
            "The confirmatory holdout remains undefined and untouched."
        )
    if end < start:
        raise SystemExit("End must be on or after start.")

    raw_root = Path(args.raw_dir)
    processed_root = Path(args.processed_dir)
    manifest_path = Path(args.manifest)
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    client = EpiasClient.from_env()
    tgt = client.get_tgt()

    manifest = {
        "status": "exploratory_only",
        "confirmatory_holdout": "UNDEFINED_AND_NOT_REQUESTED",
        "hard_stop_before": HARD_STOP.isoformat(),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "chunks": [],
    }
    monthly_frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end in month_bounds(start, end):
        label = chunk_start.strftime("%Y-%m")
        month_raw = raw_root / label
        month_raw.mkdir(parents=True, exist_ok=True)
        payloads: dict[str, list[dict]] = {}
        record = {
            "month": label,
            "start": chunk_start.isoformat(),
            "end": chunk_end.isoformat(),
            "expected_hours": expected_hours(chunk_start, chunk_end),
            "datasets": {},
        }

        for dataset in DATASETS:
            raw_path = month_raw / f"{dataset}.json"
            if args.resume and raw_path.exists():
                items = json.loads(raw_path.read_text(encoding="utf-8"))
                source = "local_resume"
            else:
                items = client.fetch(
                    dataset,
                    chunk_start.isoformat(),
                    chunk_end.isoformat(),
                    tgt=tgt,
                )
                raw_path.write_text(
                    json.dumps(items, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                source = "epias_api"
            payloads[dataset] = items
            record["datasets"][dataset] = {
                "rows": len(items),
                "sha256": sha256(raw_path),
                "source": source,
            }

        frame = build_hourly_frame(
            payloads["consumption"],
            payloads["generation"],
            payloads["mcp"],
        ).sort_index()
        month_csv = processed_root / f"{label}.csv"
        frame.to_csv(month_csv)
        record["aligned_rows_outer_join"] = int(len(frame))
        record["processed_sha256"] = sha256(month_csv)
        manifest["chunks"].append(record)
        monthly_frames.append(frame)
        print(f"{label}: aligned outer-join rows={len(frame):,}")

    combined = pd.concat(monthly_frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out)
    manifest["combined"] = {
        "rows": int(len(combined)),
        "start": combined.index.min().isoformat() if len(combined) else None,
        "end": combined.index.max().isoformat() if len(combined) else None,
        "sha256": sha256(out),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(combined):,} exploratory hourly rows to {out}")
    print(f"Manifest: {manifest_path}")
    print("CONFIRMATORY HOLDOUT: UNDEFINED_AND_NOT_REQUESTED")


if __name__ == "__main__":
    main()
