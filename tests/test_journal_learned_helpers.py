from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_journal_learned_evidence.py"
)
SPEC = spec_from_file_location(
    "tgfm_run_journal_learned_evidence",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load learned-evidence script for helper tests.")

MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CONTEXT = MODULE.CONTEXT
HORIZONS = MODULE.HORIZONS
build_split_origins = MODULE.build_split_origins
daily_block_bootstrap = MODULE.daily_block_bootstrap
output_column = MODULE.output_column


def test_output_column_is_horizon_major():
    assert output_column(0, 0, 3) == 0
    assert output_column(2, 0, 3) == 2
    assert output_column(0, 1, 3) == 3
    assert output_column(2, 2, 3) == 8


def test_split_origins_keep_targets_inside_partitions():
    n_rows = 10000
    train, validation, evaluation = build_split_origins(n_rows)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    max_h = max(HORIZONS)

    assert train[0] == CONTEXT
    assert train[-1] + max_h - 1 < train_end
    assert validation[0] == train_end
    assert validation[-1] + max_h - 1 < val_end
    assert evaluation[0] == val_end
    assert evaluation[-1] + max_h - 1 < n_rows


def test_daily_block_bootstrap_has_ordered_interval():
    values = np.linspace(-1.0, 1.0, 240)
    low, high = daily_block_bootstrap(
        values,
        reps=100,
        block=24,
        seed=7,
    )
    assert low <= high
