from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

RENEWABLE_COLUMNS = ("sun", "wind", "river", "dammedHydro", "geothermal", "biomass")


def _frame(items: Iterable[dict], timestamp_fields: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame(list(items))
    if df.empty:
        return df
    timestamp = None
    for field in timestamp_fields:
        if field in df.columns:
            candidate = pd.to_datetime(df[field], utc=True, errors="coerce")
            if candidate.notna().any():
                timestamp = candidate
                break
    if timestamp is None:
        raise ValueError(f"None of timestamp fields {timestamp_fields} were present and parseable.")
    df = df.assign(timestamp=timestamp).dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.set_index("timestamp")


def build_hourly_frame(
    consumption_items: Iterable[dict],
    generation_items: Iterable[dict],
    mcp_items: Iterable[dict],
) -> pd.DataFrame:
    consumption = _frame(consumption_items, ("date", "time"))
    generation = _frame(generation_items, ("date",))
    mcp = _frame(mcp_items, ("date",))

    c = pd.DataFrame(index=consumption.index)
    c["consumption_mwh"] = pd.to_numeric(consumption.get("consumption"), errors="coerce")

    g = pd.DataFrame(index=generation.index)
    for col in (*RENEWABLE_COLUMNS, "total", "naturalGas", "importCoal", "lignite"):
        if col in generation.columns:
            g[col] = pd.to_numeric(generation[col], errors="coerce")
        else:
            g[col] = np.nan
    renewable_available = [col for col in RENEWABLE_COLUMNS if g[col].notna().any()]
    g["renewable_mwh"] = g[renewable_available].sum(axis=1, min_count=1)
    g["total_generation_mwh"] = g["total"]
    g["renewable_share"] = g["renewable_mwh"] / g["total_generation_mwh"].replace(0, np.nan)

    p = pd.DataFrame(index=mcp.index)
    p["mcp_tl_mwh"] = pd.to_numeric(mcp.get("price"), errors="coerce")

    merged = c.join(
        g[["renewable_mwh", "total_generation_mwh", "renewable_share", "naturalGas", "importCoal", "lignite"]],
        how="outer",
    ).join(p, how="outer")
    merged = merged.sort_index()
    merged.index.name = "timestamp"
    return add_calendar_features(merged)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    local = out.index.tz_convert("Europe/Istanbul") if out.index.tz is not None else out.index
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    doy = local.dayofyear.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def chronological_split(
    df: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Fractions must be within [0, 1].")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1.")
    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
