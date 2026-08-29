import numpy as np
import pytest

from turkiye_grid_fm.journal_metrics import (
    interval_rows,
    metric_rows,
    output_index,
    skill_against,
)


def test_output_index_is_horizon_major():
    assert output_index(0, 0, 3) == 0
    assert output_index(2, 0, 3) == 2
    assert output_index(0, 1, 3) == 3
    assert output_index(2, 2, 3) == 8


def test_metric_rows_are_per_target_per_horizon():
    y = np.zeros((4, 6))
    p = np.ones((4, 6))
    rows = metric_rows(
        y,
        p,
        model="x",
        targets=["a", "b"],
        horizons=[1, 6, 24],
    )
    assert len(rows) == 6
    assert all(row.mae == 1.0 for row in rows)


def test_interval_rows():
    y = np.zeros((5, 2))
    lo = -np.ones((5, 2))
    hi = np.ones((5, 2))
    rows = interval_rows(y, lo, hi, targets=["a"], horizons=[1, 6])
    assert len(rows) == 2
    assert all(row["coverage"] == 1.0 for row in rows)
    assert all(row["mean_interval_width"] == 2.0 for row in rows)


def test_skill_against():
    assert skill_against(10.0, 8.0) == pytest.approx(0.2)
