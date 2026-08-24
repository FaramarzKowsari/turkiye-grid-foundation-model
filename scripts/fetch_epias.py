from __future__ import annotations

import argparse
import json
from pathlib import Path

from turkiye_grid_fm.data import build_hourly_frame
from turkiye_grid_fm.epias import EpiasClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and align EPİAŞ Transparency Platform datasets.")
    parser.add_argument("--start", required=True, help="ISO-8601 start, e.g. 2025-01-01T00:00:00+03:00")
    parser.add_argument("--end", required=True, help="ISO-8601 end, e.g. 2025-01-31T23:00:00+03:00")
    parser.add_argument("--out", default="data/processed/grid_hourly.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    client = EpiasClient.from_env()
    tgt = client.get_tgt()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    payloads = {}
    for name in ("consumption", "generation", "mcp"):
        items = client.fetch(name, args.start, args.end, tgt=tgt)
        payloads[name] = items
        (raw_dir / f"{name}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    frame = build_hourly_frame(payloads["consumption"], payloads["generation"], payloads["mcp"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out)
    print(f"Wrote {len(frame):,} aligned hourly rows to {out}")


if __name__ == "__main__":
    main()
