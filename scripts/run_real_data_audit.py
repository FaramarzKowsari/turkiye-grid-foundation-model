from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turkiye_grid_fm.data import build_hourly_frame  # noqa: E402
from turkiye_grid_fm.epias import API_BASE, CAS_URL, ENDPOINTS, EpiasClient, EpiasError  # noqa: E402

AUDIT_START = "2024-01-01T00:00:00+03:00"
AUDIT_END = "2024-01-31T23:00:00+03:00"
AUDIT_ID = "2024-01-v0.2a"
EXPECTED_HOURS = 744

SOURCE_COLUMNS = (
    "fueloil", "blackCoal", "lignite", "geothermal", "naturalGas", "river",
    "dammedHydro", "lng", "biomass", "naphta", "importCoal", "asphaltiteCoal",
    "wind", "sun", "wasteheat",
)
CORE_COLUMNS = (
    "consumption_mwh", "renewable_mwh", "total_generation_mwh", "renewable_share", "mcp_tl_mwh"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_index(items: list[dict[str, Any]]) -> pd.DatetimeIndex:
    if not items:
        return pd.DatetimeIndex([], tz="UTC")
    df = pd.DataFrame(items)
    if "date" not in df.columns:
        return pd.DatetimeIndex([], tz="UTC")
    ts = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
    return pd.DatetimeIndex(ts)


def dataset_quality(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    idx = parse_index(items)
    start = pd.Timestamp(AUDIT_START).tz_convert("UTC")
    end = pd.Timestamp(AUDIT_END).tz_convert("UTC")
    expected = pd.date_range(start=start, end=end, freq="h")
    valid = idx[~idx.isna()]
    unique = pd.DatetimeIndex(valid.unique()).sort_values()
    duplicates = int(len(valid) - len(unique))
    missing = expected.difference(unique)
    outside = unique[(unique < start) | (unique > end)]
    in_scope = unique[(unique >= start) & (unique <= end)]
    gaps = in_scope.to_series().diff().dropna()
    max_gap_h = float(gaps.max().total_seconds() / 3600) if len(gaps) else 0.0
    df = pd.DataFrame(items)
    key = {"consumption": "consumption", "generation": "total", "mcp": "price"}[name]
    key_nulls = int(pd.to_numeric(df[key], errors="coerce").isna().sum()) if key in df else len(df)
    suffixes = []
    if "date" in df:
        suffixes = sorted({
            m.group(1)
            for x in df["date"].dropna().astype(str)
            for m in [re.search(r"(Z|[+-]\d{2}:\d{2})$", x)]
            if m
        })
    return {
        "dataset": name,
        "raw_rows": int(len(items)),
        "unique_timestamps": int(len(unique)),
        "duplicates": duplicates,
        "missing_expected_hours": int(len(missing)),
        "coverage_pct": round(100.0 * len(in_scope) / EXPECTED_HOURS, 3),
        "outside_scope_timestamps": int(len(outside)),
        "key_field": key,
        "key_numeric_nulls": key_nulls,
        "first_timestamp_utc": unique[0].isoformat() if len(unique) else None,
        "last_timestamp_utc": unique[-1].isoformat() if len(unique) else None,
        "max_gap_hours": round(max_gap_h, 3),
        "timezone_suffixes": suffixes,
        "schema_fields": sorted(map(str, df.columns.tolist())),
        "missing_examples_utc": [x.isoformat() for x in missing[:20]],
    }


def numeric_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for col in CORE_COLUMNS:
        s = pd.to_numeric(frame[col], errors="coerce").dropna() if col in frame else pd.Series(dtype=float)
        if s.empty:
            rows.append({"variable": col, "n": 0})
            continue
        rows.append({
            "variable": col, "n": int(s.size), "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.size > 1 else 0.0, "min": float(s.min()),
            "p05": float(s.quantile(0.05)), "median": float(s.median()),
            "p95": float(s.quantile(0.95)), "max": float(s.max()),
        })
    return rows


def iqr_outliers(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "low": None, "high": None, "flagged": 0}
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {"n": int(s.size), "low": float(low), "high": float(high), "flagged": int(((s < low) | (s > high)).sum())}


def generation_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(items)
    if df.empty:
        return {"available_source_columns": [], "closure_n": 0, "negative_source_counts": {}}
    available = [c for c in SOURCE_COLUMNS if c in df.columns]
    for c in available + (["total"] if "total" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out: dict[str, Any] = {"available_source_columns": available}
    if "total" in df and available:
        residual = (df["total"] - df[available].sum(axis=1, min_count=1)).dropna()
        out.update({
            "closure_n": int(residual.size),
            "closure_mean_residual_mwh": float(residual.mean()) if len(residual) else None,
            "closure_mean_abs_residual_mwh": float(residual.abs().mean()) if len(residual) else None,
            "closure_max_abs_residual_mwh": float(residual.abs().max()) if len(residual) else None,
        })
    else:
        out["closure_n"] = 0
    out["negative_source_counts"] = {c: int((df[c] < 0).sum()) for c in available}
    return out


def correlation_table(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    cols = [c for c in ("consumption_mwh", "renewable_mwh", "mcp_tl_mwh") if c in frame]
    corr = frame[cols].apply(pd.to_numeric, errors="coerce").corr()
    return {r: {c: (None if pd.isna(corr.loc[r, c]) else float(corr.loc[r, c])) for c in cols} for r in cols}


def fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if np.isnan(x) else f"{x:,.{digits}f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    esc = lambda v: str(v).replace("|", r"\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(esc(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def html_table(headers: list[str], rows: list[list[Any]]) -> str:
    th = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def write_reports(raw_paths, payloads, frame, retrieval_utc):
    reports_dir = ROOT / "reports"
    meta_dir = ROOT / "metadata" / "audits"
    docs_dir = ROOT / "docs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    qualities = {n: dataset_quality(n, x) for n, x in payloads.items()}
    summaries = numeric_summary(frame)
    gen = generation_diagnostics(payloads["generation"])
    corr = correlation_table(frame)
    complete = pd.Series(True, index=frame.index)
    for col in ("consumption_mwh", "renewable_mwh", "mcp_tl_mwh"):
        complete &= pd.to_numeric(frame[col], errors="coerce").notna()
    complete_rows = int(complete.sum())
    anomaly = {
        "consumption_nonpositive": int((pd.to_numeric(frame["consumption_mwh"], errors="coerce") <= 0).sum()),
        "renewable_negative": int((pd.to_numeric(frame["renewable_mwh"], errors="coerce") < 0).sum()),
        "renewable_share_outside_0_1": int((((pd.to_numeric(frame["renewable_share"], errors="coerce") < 0) | (pd.to_numeric(frame["renewable_share"], errors="coerce") > 1))).sum()),
        "mcp_negative": int((pd.to_numeric(frame["mcp_tl_mwh"], errors="coerce") < 0).sum()),
        "mcp_zero": int((pd.to_numeric(frame["mcp_tl_mwh"], errors="coerce") == 0).sum()),
        "iqr_flags": {c: iqr_outliers(frame[c]) for c in ("consumption_mwh", "renewable_mwh", "mcp_tl_mwh")},
    }
    processed = ROOT / "data" / "processed" / f"grid_hourly_{AUDIT_ID}.csv"
    prov = {
        "audit_id": AUDIT_ID, "phase": "exploratory-data-audit-only", "retrieval_utc": retrieval_utc,
        "audit_start": AUDIT_START, "audit_end": AUDIT_END, "expected_hours": EXPECTED_HOURS,
        "confirmatory_holdout": "UNDEFINED_AND_NOT_REQUESTED", "cas_url": CAS_URL,
        "api_base": API_BASE, "endpoints": ENDPOINTS,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__},
        "raw_sha256": {n: sha256_file(p) for n, p in raw_paths.items()},
        "processed_sha256": sha256_file(processed), "quality": qualities,
        "aligned_rows": int(len(frame)), "complete_core_target_rows": complete_rows,
        "numeric_summary": summaries, "generation_diagnostics": gen, "anomaly_diagnostics": anomaly,
        "descriptive_correlations": corr,
        "claims_boundary": "No model-comparison or confirmatory claim is made. Descriptive correlations and outlier flags are exploratory only.",
    }
    meta_path = meta_dir / f"{AUDIT_ID}-summary.json"
    meta_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")

    qrows = [[n,q["raw_rows"],q["unique_timestamps"],q["duplicates"],q["missing_expected_hours"],f'{q["coverage_pct"]:.3f}%',q["key_numeric_nulls"],q["outside_scope_timestamps"]] for n,q in qualities.items()]
    srows = [[r["variable"],r.get("n",0),fmt(r.get("mean")),fmt(r.get("std")),fmt(r.get("min")),fmt(r.get("p05")),fmt(r.get("median")),fmt(r.get("p95")),fmt(r.get("max"))] for r in summaries]
    cols = ["consumption_mwh","renewable_mwh","mcp_tl_mwh"]
    crows = [[r] + [fmt(corr.get(r,{}).get(c),4) for c in cols] for r in cols if r in corr]
    hrows = [[n, raw_paths[n].name, prov["raw_sha256"][n]] for n in raw_paths]
    irows = [[c,d["n"],fmt(d["low"]),fmt(d["high"]),d["flagged"]] for c,d in anomaly["iqr_flags"].items()]

    md = f'''# First Empirical EPİAŞ Data Audit — January 2024\n\n**Audit ID:** `{AUDIT_ID}`  \n**Scientific status:** Exploratory data-quality audit only; **not confirmatory**.  \n**Retrieval time (UTC):** `{retrieval_utc}`\n\n## 1. Evidence firewall\n\nThis run requested **only** `{AUDIT_START}` through `{AUDIT_END}` ({EXPECTED_HOURS} expected hourly timestamps). No final confirmatory holdout has been defined, and this runner did not request observations outside the locked January 2024 interval.\n\n## 2. Official source contract\n\n- CAS/TGT: `{CAS_URL}`\n- API base: `{API_BASE}`\n- Consumption: `{ENDPOINTS["consumption"]}`\n- Generation: `{ENDPOINTS["generation"]}`\n- MCP/PTF: `{ENDPOINTS["mcp"]}`\n\nRaw source records remain local and gitignored; aggregate audit outputs and hashes are preserved.\n\n## 3. Retrieval and hourly coverage\n\n{md_table(["Dataset","Raw rows","Unique hours","Duplicates","Missing expected","Coverage","Key nulls","Outside scope"], qrows)}\n\nAligned union rows: **{len(frame):,}**  \nComplete consumption + renewables + PTF rows: **{complete_rows:,}/{EXPECTED_HOURS:,} ({100*complete_rows/EXPECTED_HOURS:.3f}%)**\n\n## 4. Schema observations\n'''
    for n,q in qualities.items():
        md += f'- **{n}:** `{", ".join(q["schema_fields"])}`; timezone suffixes `{q["timezone_suffixes"]}`; maximum gap **{q["max_gap_hours"]} h**.\n'
        if q["missing_examples_utc"]:
            md += f'  - First missing examples (UTC): `{", ".join(q["missing_examples_utc"][:8])}`\n'
    md += f'''\n## 5. Descriptive numeric audit\n\n{md_table(["Variable","N","Mean","SD","Min","P05","Median","P95","Max"], srows)}\n\nThese are data-quality descriptions, not forecasting metrics.\n\n## 6. Generation-source consistency\n\nAvailable source fields: `{", ".join(gen.get("available_source_columns", []))}`\n\n- Rows checked: **{gen.get("closure_n",0):,}**\n- Mean residual (reported total − listed-source sum): **{fmt(gen.get("closure_mean_residual_mwh"))} MWh**\n- Mean absolute residual: **{fmt(gen.get("closure_mean_abs_residual_mwh"))} MWh**\n- Maximum absolute residual: **{fmt(gen.get("closure_max_abs_residual_mwh"))} MWh**\n\nResiduals are diagnostics, not automatic evidence of bad data; EPİAŞ definitions must be considered.\n\n## 7. Exploratory anomaly flags — no deletion performed\n\n- Non-positive consumption: **{anomaly["consumption_nonpositive"]}**\n- Negative renewable aggregate: **{anomaly["renewable_negative"]}**\n- Renewable share outside [0,1]: **{anomaly["renewable_share_outside_0_1"]}**\n- Negative PTF: **{anomaly["mcp_negative"]}**\n- Zero PTF: **{anomaly["mcp_zero"]}**\n\n{md_table(["Variable","N","IQR lower fence","IQR upper fence","Flagged"], irows)}\n\n## 8. Descriptive correlation matrix\n\n{md_table(["Variable"] + cols, crows)}\n\nExploratory only; not causal and not a confirmatory hypothesis.\n\n## 9. Cryptographic provenance\n\n{md_table(["Dataset","Local raw snapshot","SHA-256"], hrows)}\n\nProcessed aligned CSV SHA-256: `{prov["processed_sha256"]}`\n\nMachine-readable summary: `metadata/audits/{AUDIT_ID}-summary.json`\n\n## 10. First scientific interpretation\n\nThis checkpoint asks a narrower question than forecasting: **is the selected EPİAŞ data layer internally usable enough to justify a larger exploratory modelling stage?** The decision is based on timestamp coverage, key-field completeness, schema stability, cross-dataset alignment and unexplained structural inconsistencies—not on whether a Transformer appears accurate.\n\n### Next allowed actions\n\n1. Investigate any missing timestamps, duplicates, numeric nulls or source-total residuals.\n2. If structurally clean enough, expand only the exploratory/training-side availability audit under a separately committed scope extension.\n3. Benchmark strong baselines and runtime before choosing final seeds/model families.\n4. Keep the confirmatory holdout undefined and unseen until data and compute audits are complete.\n5. Freeze hypotheses, exclusions, metrics, seeds and eventual holdout before confirmatory evaluation.\n\n---\nGenerated automatically by Türkiye Grid Foundation Model v0.2a.\n'''
    report = reports_dir / "DATA_AUDIT_REPORT_2024-01.md"
    report.write_text(md, encoding="utf-8")

    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>First Empirical EPİAŞ Data Audit — Türkiye Grid Foundation Model</title><meta name="description" content="Exploratory January 2024 EPİAŞ real-data quality audit; no confirmatory holdout touched."><link rel="stylesheet" href="styles.css"></head><body><nav class="nav"><div class="wrap"><a class="brand" href="index.html">Türkiye Grid FM</a><div class="links"><a href="en/">English guide</a><a href="tr/">Türkçe</a><a href="es/">Español</a><a href="https://github.com/FaramarzKowsari/turkiye-grid-foundation-model">GitHub</a></div></div></nav><header class="hero"><div class="wrap"><div class="eyebrow">v0.2a · First empirical data audit</div><h1>January 2024 EPİAŞ real-data audit</h1><p class="lead">Availability, timestamps, missingness, schema, alignment, numeric ranges and provenance—without opening any confirmatory holdout.</p><div class="status"><strong>Evidence boundary:</strong> exploratory data audit only. No model-comparison or confirmatory claim is made.</div></div></header><main><section class="section"><div class="wrap"><h2>Locked scope</h2><p><strong>{AUDIT_START}</strong> through <strong>{AUDIT_END}</strong> · {EXPECTED_HOURS} expected hours.</p><p>Retrieval UTC: <code>{html.escape(retrieval_utc)}</code>.</p></div></section><section class="section"><div class="wrap"><h2>Coverage and integrity</h2>{html_table(["Dataset","Raw rows","Unique hours","Duplicates","Missing expected","Coverage","Key nulls","Outside scope"],qrows)}<p><strong>Complete three-target rows:</strong> {complete_rows:,}/{EXPECTED_HOURS:,} ({100*complete_rows/EXPECTED_HOURS:.3f}%).</p></div></section><section class="section"><div class="wrap"><h2>Descriptive numeric audit</h2>{html_table(["Variable","N","Mean","SD","Min","P05","Median","P95","Max"],srows)}</div></section><section class="section"><div class="wrap"><h2>Generation consistency</h2><div class="grid"><div class="card"><h3>Mean absolute residual</h3><p class="metric">{fmt(gen.get("closure_mean_abs_residual_mwh"))}</p><p>MWh</p></div><div class="card"><h3>Maximum absolute residual</h3><p class="metric">{fmt(gen.get("closure_max_abs_residual_mwh"))}</p><p>MWh</p></div><div class="card"><h3>Rows checked</h3><p class="metric">{gen.get("closure_n",0):,}</p></div></div></div></section><section class="section"><div class="wrap"><h2>Anomaly flags — no deletion</h2>{html_table(["Variable","N","Lower fence","Upper fence","Flagged"],irows)}</div></section><section class="section"><div class="wrap"><h2>Descriptive correlations</h2>{html_table(["Variable"]+cols,crows)}<p class="muted">Exploratory only.</p></div></section><section class="section"><div class="wrap"><h2>Provenance</h2>{html_table(["Dataset","Local raw snapshot","SHA-256"],hrows)}<p>Processed CSV SHA-256: <code>{prov["processed_sha256"]}</code></p><p><a href="https://github.com/FaramarzKowsari/turkiye-grid-foundation-model/blob/main/reports/DATA_AUDIT_REPORT_2024-01.md">Open full scientific report →</a></p></div></section></main><footer class="footer"><div class="wrap">Türkiye Grid Foundation Model · First empirical EPİAŞ audit · Faramarz Kowsari</div></footer></body></html>'''
    (docs_dir / "data-audit.html").write_text(page, encoding="utf-8")

    index = docs_dir / "index.html"
    if index.exists():
        text = index.read_text(encoding="utf-8")
        if "data-audit.html" not in text and "</main>" in text:
            section = '<section class="section"><div class="wrap"><div class="callout"><strong>New empirical checkpoint:</strong> The bounded January 2024 EPİAŞ real-data audit is available. <a href="data-audit.html">Read the first scientific data-audit report →</a></div></div></section>'
            index.write_text(text.replace("</main>", section + "\n</main>"), encoding="utf-8")
    print(f"Scientific report: {report}")
    print(f"Public page: {docs_dir / 'data-audit.html'}")
    print(f"Summary JSON: {meta_path}")


def run_real() -> int:
    if (AUDIT_START, AUDIT_END) != ("2024-01-01T00:00:00+03:00", "2024-01-31T23:00:00+03:00"):
        raise RuntimeError("Audit scope lock modified; refusing acquisition.")
    raw_dir = ROOT / "data" / "raw" / AUDIT_ID
    processed_dir = ROOT / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    client = EpiasClient.from_env()
    print("Authenticating with EPİAŞ live Transparency Platform...")
    tgt = client.get_tgt()
    print("TGT acquired. Fetching ONLY locked January 2024 audit window...")
    payloads, raw_paths = {}, {}
    retrieval_utc = datetime.now(timezone.utc).isoformat()
    for name in ("consumption", "generation", "mcp"):
        print(f"Fetching {name}...")
        items = client.fetch(name, AUDIT_START, AUDIT_END, tgt=tgt)
        payloads[name] = items
        path = raw_dir / f"{name}.json"
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        raw_paths[name] = path
        print(f"  {len(items):,} rows")
    frame = build_hourly_frame(payloads["consumption"], payloads["generation"], payloads["mcp"])
    processed = processed_dir / f"grid_hourly_{AUDIT_ID}.csv"
    frame.to_csv(processed)
    print(f"Aligned frame: {len(frame):,} rows")
    write_reports(raw_paths, payloads, frame, retrieval_utc)
    return 0


def self_test() -> int:
    idx = pd.date_range("2024-01-01T00:00:00+03:00", periods=48, freq="h")
    c = [{"date": x.isoformat(), "consumption": 30000 + i * 10} for i,x in enumerate(idx)]
    g=[]
    for i,x in enumerate(idx):
        row={"date":x.isoformat(),"sun":float(max(0,1000*np.sin((i%24)/24*np.pi))),"wind":2000.0,"river":3000.0,"dammedHydro":4000.0,"geothermal":1000.0,"biomass":500.0,"naturalGas":6000.0,"importCoal":5000.0,"lignite":3000.0,"fueloil":100.0,"blackCoal":100.0,"lng":0.0,"naphta":0.0,"asphaltiteCoal":0.0,"wasteheat":100.0}
        row["total"] = sum(row[k] for k in SOURCE_COLUMNS)
        g.append(row)
    p=[{"date":x.isoformat(),"price":2000+i} for i,x in enumerate(idx)]
    frame=build_hourly_frame(c,g,p)
    assert len(frame)==48
    assert dataset_quality("consumption",c)["raw_rows"]==48
    assert generation_diagnostics(g)["closure_max_abs_residual_mwh"] < 1e-9
    assert "consumption_mwh" in correlation_table(frame)
    print("SELF-TEST PASS: parsing, alignment, diagnostics.")
    return 0


def main() -> int:
    p=argparse.ArgumentParser(description="Run locked v0.2a January 2024 EPİAŞ real-data audit.")
    p.add_argument("--self-test", action="store_true")
    args=p.parse_args()
    if args.self_test:
        return self_test()
    try:
        return run_real()
    except EpiasError as exc:
        print(f"EPİAŞ ERROR: {exc}", file=sys.stderr)
        return 20


if __name__ == "__main__":
    raise SystemExit(main())
