from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_grid(hours: int = 24 * 120, seed: int = 7) -> pd.DataFrame:
    if hours < 240:
        raise ValueError("Use at least 240 hours for a meaningful smoke dataset.")
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    index = pd.date_range("2024-01-01", periods=hours, freq="h", tz="Europe/Istanbul").tz_convert("UTC")
    daily = np.sin(2 * np.pi * t / 24)
    weekly = np.sin(2 * np.pi * t / (24 * 7))
    solar = np.maximum(0, np.sin(2 * np.pi * (t % 24 - 6) / 24))
    wind = 2200 + 450 * np.sin(2 * np.pi * t / (24 * 5)) + rng.normal(0, 120, hours)
    renewable = 5000 * solar + wind + 1600 + rng.normal(0, 100, hours)
    consumption = 33000 + 4500 * daily + 1800 * weekly + rng.normal(0, 350, hours)
    price = 1750 + 0.035 * (consumption - 33000) - 0.04 * (renewable - renewable.mean()) + rng.normal(0, 80, hours)
    total_generation = consumption + rng.normal(0, 250, hours)
    frame = pd.DataFrame(
        {
            "consumption_mwh": consumption,
            "renewable_mwh": renewable,
            "total_generation_mwh": total_generation,
            "renewable_share": renewable / total_generation,
            "naturalGas": np.maximum(0, 9000 + 1800 * daily - 0.35 * (renewable - renewable.mean())),
            "importCoal": 7000 + rng.normal(0, 150, hours),
            "lignite": 5000 + rng.normal(0, 120, hours),
            "mcp_tl_mwh": price,
        },
        index=index,
    )
    from .data import add_calendar_features

    frame.index.name = "timestamp"
    return add_calendar_features(frame)
