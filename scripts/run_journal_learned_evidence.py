from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from turkiye_grid_fm.journal_metrics import metric_rows
from turkiye_grid_fm.losses import gaussian_nll
from turkiye_grid_fm.models.foundation import GridFoundationModel
from turkiye_grid_fm.uncertainty import conformal_radius

FEATURES = [
    "consumption_mwh",
    "renewable_mwh",
    "total_generation_mwh",
    "renewable_share",
    "naturalGas",
    "importCoal",
    "lignite",
    "mcp_tl_mwh",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]
TARGETS = ["consumption_mwh", "renewable_mwh", "mcp_tl_mwh"]
HORIZONS = [1, 6, 24]
CONTEXT = 168
SEEDS = [2026, 2027, 2028]
COVERAGE = 0.90
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260828
TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85


@dataclass(frozen=True)
class Profile:
    name: str
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    lr: float


CPU_PROFILE = Profile(
    name="cpu_research",
    d_model=48,
    nhead=4,
    num_layers=2,
    dim_feedforward=96,
    dropout=0.10,
    batch_size=256,
    max_epochs=8,
    patience=2,
    lr=3e-4,
)

GPU_PROFILE = Profile(
    name="gpu_research",
    d_model=96,
    nhead=4,
    num_layers=3,
    dim_feedforward=192,
    dropout=0.10,
    batch_size=512,
    max_epochs=15,
    patience=3,
    lr=3e-4,
)


class OriginDataset(Dataset):
    def __init__(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        origins: np.ndarray,
        target_indices: list[int],
    ) -> None:
        self.feature_values = feature_values
        self.target_values = target_values
        self.origins = np.asarray(origins, dtype=np.int64)
        self.target_indices = list(target_indices)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        origin = int(self.origins[idx])
        x = self.feature_values[origin - CONTEXT : origin]
        blocks = [
            self.target_values[
                origin + horizon - 1,
                self.target_indices,
            ]
            for horizon in HORIZONS
        ]
        y = np.concatenate(blocks).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_std(values: np.ndarray) -> np.ndarray:
    std = np.nanstd(values, axis=0)
    return np.where(std < 1e-8, 1.0, std)


def build_split_origins(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n_rows * TRAIN_FRACTION)
    val_end = int(n_rows * VALIDATION_END_FRACTION)
    max_h = max(HORIZONS)

    train = np.arange(
        CONTEXT,
        train_end - max_h + 1,
        dtype=np.int64,
    )
    validation = np.arange(
        train_end,
        val_end - max_h + 1,
        dtype=np.int64,
    )
    evaluation = np.arange(
        val_end,
        n_rows - max_h + 1,
        dtype=np.int64,
    )
    if min(len(train), len(validation), len(evaluation)) <= 0:
        raise ValueError("Chronological split produced an empty origin partition.")
    return train, validation, evaluation


def output_stats(
    target_mean: np.ndarray,
    target_std: np.ndarray,
    target_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for _ in HORIZONS:
        means.extend(target_mean[target_indices])
        stds.extend(target_std[target_indices])
    return np.asarray(means), np.asarray(stds)


def inverse_outputs(
    values: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    target_indices: list[int],
) -> np.ndarray:
    mean_out, std_out = output_stats(
        target_mean,
        target_std,
        target_indices,
    )
    return values * std_out[None, :] + mean_out[None, :]


def inverse_scales(
    values: np.ndarray,
    target_std: np.ndarray,
    target_indices: list[int],
) -> np.ndarray:
    _, std_out = output_stats(
        np.zeros_like(target_std),
        target_std,
        target_indices,
    )
    return values * std_out[None, :]


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            mean, scale = model(x)
            loss = gaussian_nll(mean, scale, y)
            total += float(loss.detach().cpu()) * len(x)
            count += len(x)
    return total / max(count, 1)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    profile: Profile,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]], int]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=profile.lr,
    )
    history: list[dict[str, float]] = []
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0

    for epoch in range(1, profile.max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, scale = model(x)
            loss = gaussian_nll(mean, scale, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()
            total += float(loss.detach().cpu()) * len(x)
            count += len(x)

        train_loss = total / max(count, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_nll": float(train_loss),
                "validation_nll": float(val_loss),
            }
        )
        print(
            f"      epoch={epoch:02d} "
            f"train_nll={train_loss:.6f} "
            f"val_nll={val_loss:.6f}",
            flush=True,
        )

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= profile.patience:
            break

    if best_state is None:
        raise RuntimeError("Training never produced a valid best state.")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    means = []
    scales = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mean, scale = model(x)
            ys.append(y.numpy())
            means.append(mean.detach().cpu().numpy())
            scales.append(scale.detach().cpu().numpy())
    return (
        np.concatenate(ys),
        np.concatenate(means),
        np.concatenate(scales),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def smape(y: np.ndarray, p: np.ndarray) -> float:
    denom = np.maximum(np.abs(y) + np.abs(p), 1e-6)
    return float(200.0 * np.mean(np.abs(y - p) / denom))


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    error = y - p
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "smape": smape(y, p),
    }


def output_column(
    target_index: int,
    horizon_index: int,
    n_targets: int,
) -> int:
    return horizon_index * n_targets + target_index


def daily_block_bootstrap(
    delta_absolute_error: np.ndarray,
    *,
    reps: int = BOOTSTRAP_REPS,
    block: int = 24,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    values = np.asarray(delta_absolute_error, dtype=float)
    n = len(values)
    if n < block:
        raise ValueError("Not enough observations for daily-block bootstrap.")
    rng = np.random.default_rng(seed)
    max_start = n - block
    estimates = np.empty(reps, dtype=float)
    blocks_needed = math.ceil(n / block)

    for rep in range(reps):
        starts = rng.integers(
            0,
            max_start + 1,
            size=blocks_needed,
        )
        sampled = np.concatenate(
            [values[start : start + block] for start in starts]
        )[:n]
        estimates[rep] = sampled.mean()

    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def markdown_table(
    headers: list[str],
    rows: list[list[object]],
) -> str:
    def esc(value: object) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(esc(value) for value in row) + " |"
        )
    return "\n".join(lines)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    repo = Path(os.environ["TGFM_REPO"])
    data_path = repo / "data" / "processed" / "exploratory_2021_2025.csv"
    if not data_path.exists():
        raise SystemExit(
            "Missing exploratory_2021_2025.csv. Run Journal Phase A+B+C first."
        )

    out = repo / "artifacts" / "journal_v03_learned"
    runs_dir = out / "runs"
    out.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(
        data_path,
        parse_dates=["timestamp"],
    ).sort_values("timestamp")
    frame = frame.drop_duplicates("timestamp", keep="last")
    frame = frame.set_index("timestamp")

    required = list(dict.fromkeys(FEATURES + TARGETS))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    frame = frame[required].replace([np.inf, -np.inf], np.nan)
    if frame.isna().any().any():
        counts = frame.isna().sum()
        bad = counts[counts > 0].to_dict()
        raise SystemExit(
            "Learned-model gate requires complete rows. "
            f"Null counts: {bad}"
        )

    index = pd.DatetimeIndex(frame.index)
    gaps = index.to_series().diff().dropna()
    if not (gaps == pd.Timedelta(hours=1)).all():
        raise SystemExit("Hourly continuity check failed.")

    local_index = index.tz_convert("Europe/Istanbul")
    if local_index.max() >= pd.Timestamp(
        "2026-01-01T00:00:00",
        tz="Europe/Istanbul",
    ):
        raise SystemExit(
            "SAFETY STOP: learned-model data include 2026+ timestamps."
        )

    train_origins, val_origins, eval_origins = build_split_origins(len(frame))
    train_end = int(len(frame) * TRAIN_FRACTION)

    feature_native = frame[FEATURES].to_numpy(dtype=np.float32)
    target_native = frame[TARGETS].to_numpy(dtype=np.float32)

    feature_mean = np.mean(feature_native[:train_end], axis=0)
    feature_std = safe_std(feature_native[:train_end])
    target_mean = np.mean(target_native[:train_end], axis=0)
    target_std = safe_std(target_native[:train_end])

    feature_z = (
        (feature_native - feature_mean[None, :])
        / feature_std[None, :]
    ).astype(np.float32)
    target_z = (
        (target_native - target_mean[None, :])
        / target_std[None, :]
    ).astype(np.float32)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    profile = GPU_PROFILE if device.type == "cuda" else CPU_PROFILE

    if device.type == "cpu":
        threads = max(1, (os.cpu_count() or 4) - 1)
        torch.set_num_threads(threads)

    config = {
        "status": "exploratory_only",
        "confirmatory_holdout": "UNDEFINED_AND_NOT_REQUESTED",
        "data_sha256": file_sha256(data_path),
        "rows": int(len(frame)),
        "features": FEATURES,
        "targets": TARGETS,
        "horizons": HORIZONS,
        "context": CONTEXT,
        "seeds": SEEDS,
        "coverage": COVERAGE,
        "train_fraction": TRAIN_FRACTION,
        "validation_end_fraction": VALIDATION_END_FRACTION,
        "profile": asdict(profile),
        "device": str(device),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "train_origins": int(len(train_origins)),
        "validation_origins": int(len(val_origins)),
        "evaluation_origins": int(len(eval_origins)),
    }
    config_text = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(
        config_text.encode("utf-8")
    ).hexdigest()
    config["config_sha256"] = config_hash
    (out / "RUN_CONFIG.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("TGFM LEARNED-MODEL EXPLORATORY AUTOPILOT")
    print(f"device={device}")
    print(f"profile={profile.name}")
    print(f"rows={len(frame):,}")
    print(
        "origins="
        f"{len(train_origins):,}/"
        f"{len(val_origins):,}/"
        f"{len(eval_origins):,}"
    )
    print("confirmatory holdout=UNDEFINED_AND_NOT_REQUESTED")
    print("=" * 72)

    model_specs = [
        ("isolated_consumption", [0]),
        ("isolated_renewable", [1]),
        ("isolated_mcp", [2]),
        ("shared", [0, 1, 2]),
    ]

    metric_records = []
    compute_records = []
    uncertainty_records = []
    run_files: dict[tuple[str, int], Path] = {}

    for seed in SEEDS:
        for model_name, target_indices in model_specs:
            stem = f"{model_name}__seed_{seed}"
            npz_path = runs_dir / f"{stem}.npz"
            json_path = runs_dir / f"{stem}.json"
            run_files[(model_name, seed)] = npz_path

            if npz_path.exists() and json_path.exists():
                meta = json.loads(
                    json_path.read_text(encoding="utf-8")
                )
                if meta.get("config_sha256") == config_hash:
                    print(f"[resume] {stem}")
                    for record in meta["metrics"]:
                        metric_records.append(record)
                    compute_records.append(meta["compute"])
                    for record in meta["uncertainty"]:
                        uncertainty_records.append(record)
                    continue

            print("")
            print(f"[train] {stem}")
            set_seed(seed)

            train_ds = OriginDataset(
                feature_z,
                target_z,
                train_origins,
                target_indices,
            )
            val_ds = OriginDataset(
                feature_z,
                target_z,
                val_origins,
                target_indices,
            )
            eval_ds = OriginDataset(
                feature_z,
                target_z,
                eval_origins,
                target_indices,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=profile.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=1024,
                shuffle=False,
                num_workers=0,
            )
            eval_loader = DataLoader(
                eval_ds,
                batch_size=1024,
                shuffle=False,
                num_workers=0,
            )

            target_dim = len(target_indices) * len(HORIZONS)
            model = GridFoundationModel(
                input_dim=len(FEATURES),
                target_dim=target_dim,
                d_model=profile.d_model,
                nhead=profile.nhead,
                num_layers=profile.num_layers,
                dim_feedforward=profile.dim_feedforward,
                dropout=profile.dropout,
            )
            parameters = count_parameters(model)

            train_start = time.perf_counter()
            model, history, best_epoch = train_with_early_stopping(
                model,
                train_loader,
                val_loader,
                profile,
                device,
            )
            training_seconds = time.perf_counter() - train_start

            pred_start = time.perf_counter()
            val_y_z, val_p_z, val_scale_z = predict(
                model,
                val_loader,
                device,
            )
            eval_y_z, eval_p_z, eval_scale_z = predict(
                model,
                eval_loader,
                device,
            )
            inference_seconds = time.perf_counter() - pred_start

            val_y = inverse_outputs(
                val_y_z,
                target_mean,
                target_std,
                target_indices,
            )
            val_p = inverse_outputs(
                val_p_z,
                target_mean,
                target_std,
                target_indices,
            )
            eval_y = inverse_outputs(
                eval_y_z,
                target_mean,
                target_std,
                target_indices,
            )
            eval_p = inverse_outputs(
                eval_p_z,
                target_mean,
                target_std,
                target_indices,
            )
            eval_scale = inverse_scales(
                eval_scale_z,
                target_std,
                target_indices,
            )

            radius = conformal_radius(
                val_y,
                val_p,
                coverage=COVERAGE,
            )
            lower = eval_p - radius[None, :]
            upper = eval_p + radius[None, :]

            target_names = [TARGETS[i] for i in target_indices]
            run_metrics = []
            for row in metric_rows(
                eval_y,
                eval_p,
                model=model_name,
                targets=target_names,
                horizons=HORIZONS,
            ):
                record = {
                    **asdict(row),
                    "seed": seed,
                }
                run_metrics.append(record)
                metric_records.append(record)

            run_uncertainty = []
            for h_idx, horizon in enumerate(HORIZONS):
                for local_t_idx, target_name in enumerate(target_names):
                    col = output_column(
                        local_t_idx,
                        h_idx,
                        len(target_names),
                    )
                    covered = (
                        (eval_y[:, col] >= lower[:, col])
                        & (eval_y[:, col] <= upper[:, col])
                    )
                    record = {
                        "model": model_name,
                        "seed": seed,
                        "target": target_name,
                        "horizon_hours": horizon,
                        "coverage": float(np.mean(covered)),
                        "mean_interval_width": float(
                            np.mean(upper[:, col] - lower[:, col])
                        ),
                        "mean_native_gaussian_scale": float(
                            np.mean(eval_scale[:, col])
                        ),
                    }
                    run_uncertainty.append(record)
                    uncertainty_records.append(record)

            checkpoint_path = runs_dir / f"{stem}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_bytes = checkpoint_path.stat().st_size

            compute = {
                "model": model_name,
                "seed": seed,
                "device": str(device),
                "profile": profile.name,
                "trainable_parameters": parameters,
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "training_seconds": training_seconds,
                "inference_seconds_validation_plus_evaluation": inference_seconds,
                "checkpoint_bytes": checkpoint_bytes,
            }
            compute_records.append(compute)

            np.savez_compressed(
                npz_path,
                eval_origins=eval_origins,
                y=eval_y,
                pred=eval_p,
                lower=lower,
                upper=upper,
                scale=eval_scale,
            )
            metadata = {
                "config_sha256": config_hash,
                "model": model_name,
                "seed": seed,
                "target_indices": target_indices,
                "history": history,
                "metrics": run_metrics,
                "uncertainty": run_uncertainty,
                "compute": compute,
            }
            json_path.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            print(
                f"      done: {training_seconds / 60:.2f} min, "
                f"best_epoch={best_epoch}, params={parameters:,}"
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(metric_records)
    metrics_df.to_csv(out / "MODEL_METRICS_ALL_SEEDS.csv", index=False)

    summary_df = (
        metrics_df.groupby(
            ["model", "target", "horizon_hours"],
            as_index=False,
        )
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            smape_mean=("smape", "mean"),
            smape_std=("smape", "std"),
        )
    )
    summary_df.to_csv(
        out / "MODEL_METRICS_SUMMARY.csv",
        index=False,
    )

    pd.DataFrame(uncertainty_records).to_csv(
        out / "UNCERTAINTY_ALL_SEEDS.csv",
        index=False,
    )
    pd.DataFrame(compute_records).to_csv(
        out / "COMPUTE_AUDIT.csv",
        index=False,
    )

    transfer_seed_rows = []
    for seed in SEEDS:
        shared = metrics_df[
            (metrics_df["model"] == "shared")
            & (metrics_df["seed"] == seed)
        ]
        for target in TARGETS:
            isolated_name = (
                "isolated_consumption"
                if target == "consumption_mwh"
                else "isolated_renewable"
                if target == "renewable_mwh"
                else "isolated_mcp"
            )
            isolated = metrics_df[
                (metrics_df["model"] == isolated_name)
                & (metrics_df["seed"] == seed)
            ]
            for horizon in HORIZONS:
                s_row = shared[
                    (shared["target"] == target)
                    & (shared["horizon_hours"] == horizon)
                ].iloc[0]
                i_row = isolated[
                    (isolated["target"] == target)
                    & (isolated["horizon_hours"] == horizon)
                ].iloc[0]
                delta = float(s_row["mae"] - i_row["mae"])
                relative = float(
                    -delta / i_row["mae"]
                    if i_row["mae"] > 0
                    else np.nan
                )
                transfer_seed_rows.append(
                    {
                        "seed": seed,
                        "target": target,
                        "horizon_hours": horizon,
                        "shared_mae": float(s_row["mae"]),
                        "isolated_mae": float(i_row["mae"]),
                        "delta_mae_shared_minus_isolated": delta,
                        "relative_improvement_if_positive": relative,
                    }
                )

    transfer_seed_df = pd.DataFrame(transfer_seed_rows)
    transfer_seed_df.to_csv(
        out / "TRANSFER_BY_SEED.csv",
        index=False,
    )

    transfer_summary = (
        transfer_seed_df.groupby(
            ["target", "horizon_hours"],
            as_index=False,
        )
        .agg(
            delta_mae_mean=(
                "delta_mae_shared_minus_isolated",
                "mean",
            ),
            delta_mae_std=(
                "delta_mae_shared_minus_isolated",
                "std",
            ),
            relative_improvement_mean=(
                "relative_improvement_if_positive",
                "mean",
            ),
        )
    )

    ensemble_cache = {}
    bootstrap_rows = []
    regime_rows = []
    mcp_infoset_rows = []

    train_state = frame.iloc[:train_end]
    renewable_q = train_state["renewable_share"].quantile([0.33, 0.67])
    demand_q = train_state["consumption_mwh"].quantile([0.33, 0.67])
    ramp_train = train_state["consumption_mwh"].diff().abs().dropna()
    ramp_q90 = float(ramp_train.quantile(0.90))
    mcp_q = train_state["mcp_tl_mwh"].quantile([0.05, 0.95])

    issue_rows = eval_origins - 1
    issue_local = local_index[issue_rows]
    issue_state = frame.iloc[issue_rows].copy()
    issue_state["abs_demand_ramp"] = (
        frame["consumption_mwh"].diff().abs().iloc[issue_rows].to_numpy()
    )

    issue_state["renewable_regime"] = np.select(
        [
            issue_state["renewable_share"] < renewable_q.loc[0.33],
            issue_state["renewable_share"] > renewable_q.loc[0.67],
        ],
        ["low", "high"],
        default="middle",
    )
    issue_state["demand_regime"] = np.select(
        [
            issue_state["consumption_mwh"] < demand_q.loc[0.33],
            issue_state["consumption_mwh"] > demand_q.loc[0.67],
        ],
        ["low", "high"],
        default="middle",
    )
    issue_state["ramp_regime"] = np.where(
        issue_state["abs_demand_ramp"] > ramp_q90,
        "high",
        "normal",
    )
    issue_state["mcp_regime"] = np.where(
        (issue_state["mcp_tl_mwh"] < mcp_q.loc[0.05])
        | (issue_state["mcp_tl_mwh"] > mcp_q.loc[0.95]),
        "extreme",
        "normal",
    )

    for target_idx, target in enumerate(TARGETS):
        isolated_name = [
            "isolated_consumption",
            "isolated_renewable",
            "isolated_mcp",
        ][target_idx]

        shared_preds = []
        isolated_preds = []
        shared_truth = None
        for seed in SEEDS:
            shared_data = np.load(run_files[("shared", seed)])
            isolated_data = np.load(run_files[(isolated_name, seed)])

            if shared_truth is None:
                shared_truth = shared_data["y"]

            shared_preds.append(shared_data["pred"])
            isolated_preds.append(isolated_data["pred"])

        shared_ensemble = np.mean(shared_preds, axis=0)
        isolated_ensemble = np.mean(isolated_preds, axis=0)
        ensemble_cache[(target, "shared")] = shared_ensemble
        ensemble_cache[(target, "isolated")] = isolated_ensemble

        for h_idx, horizon in enumerate(HORIZONS):
            shared_col = output_column(target_idx, h_idx, len(TARGETS))
            isolated_col = output_column(0, h_idx, 1)

            truth = shared_truth[:, shared_col]
            shared_pred = shared_ensemble[:, shared_col]
            isolated_pred = isolated_ensemble[:, isolated_col]

            delta_error = (
                np.abs(truth - shared_pred)
                - np.abs(truth - isolated_pred)
            )
            ci_low, ci_high = daily_block_bootstrap(delta_error)
            bootstrap_rows.append(
                {
                    "target": target,
                    "horizon_hours": horizon,
                    "ensemble_delta_mae": float(delta_error.mean()),
                    "bootstrap_95_low": ci_low,
                    "bootstrap_95_high": ci_high,
                    "transfer_interpretation": (
                        "positive"
                        if ci_high < 0
                        else "negative"
                        if ci_low > 0
                        else "uncertain"
                    ),
                }
            )

            for regime_column in [
                "renewable_regime",
                "demand_regime",
                "ramp_regime",
                "mcp_regime",
            ]:
                values = issue_state[regime_column].to_numpy()
                for regime_value in sorted(set(values)):
                    mask = values == regime_value
                    regime_rows.append(
                        {
                            "target": target,
                            "horizon_hours": horizon,
                            "regime_dimension": regime_column,
                            "regime": regime_value,
                            "n": int(mask.sum()),
                            "delta_mae_shared_minus_isolated": float(
                                delta_error[mask].mean()
                            ),
                        }
                    )

            if target == "mcp_tl_mwh":
                if horizon in (1, 6):
                    mcp_infoset_rows.append(
                        {
                            "horizon_hours": horizon,
                            "subset": "all_origins",
                            "operational_status": (
                                "retrospective_representation_benchmark"
                            ),
                            "n": int(len(truth)),
                            **{
                                f"shared_{key}": value
                                for key, value in point_metrics(
                                    truth,
                                    shared_pred,
                                ).items()
                            },
                            **{
                                f"isolated_{key}": value
                                for key, value in point_metrics(
                                    truth,
                                    isolated_pred,
                                ).items()
                            },
                        }
                    )
                if horizon == 24:
                    masks = {
                        "all_origins": np.ones(
                            len(truth),
                            dtype=bool,
                        ),
                        "pre_publication_hour_lt_14": (
                            issue_local.hour < 14
                        ),
                        "pre_gate_proxy_hour_le_12": (
                            issue_local.hour <= 12
                        ),
                    }
                    for subset_name, mask in masks.items():
                        mcp_infoset_rows.append(
                            {
                                "horizon_hours": horizon,
                                "subset": subset_name,
                                "operational_status": (
                                    "information_set_aware"
                                    if subset_name != "all_origins"
                                    else "representation_benchmark"
                                ),
                                "n": int(mask.sum()),
                                **{
                                    f"shared_{key}": value
                                    for key, value in point_metrics(
                                        truth[mask],
                                        shared_pred[mask],
                                    ).items()
                                },
                                **{
                                    f"isolated_{key}": value
                                    for key, value in point_metrics(
                                        truth[mask],
                                        isolated_pred[mask],
                                    ).items()
                                },
                            }
                        )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(
        out / "TRANSFER_BLOCK_BOOTSTRAP.csv",
        index=False,
    )
    pd.DataFrame(regime_rows).to_csv(
        out / "REGIME_TRANSFER.csv",
        index=False,
    )
    pd.DataFrame(mcp_infoset_rows).to_csv(
        out / "MCP_INFORMATION_SET_METRICS.csv",
        index=False,
    )

    transfer_summary = transfer_summary.merge(
        bootstrap_df,
        on=["target", "horizon_hours"],
        how="left",
    )
    transfer_summary.to_csv(
        out / "TRANSFER_SUMMARY.csv",
        index=False,
    )

    table3_rows = []
    for row in transfer_summary.itertuples(index=False):
        table3_rows.append(
            [
                row.target,
                f"+{row.horizon_hours} h",
                f"{row.delta_mae_mean:.3f}",
                f"{row.delta_mae_std:.3f}",
                f"{100 * row.relative_improvement_mean:.2f}%",
                (
                    f"[{row.bootstrap_95_low:.3f}, "
                    f"{row.bootstrap_95_high:.3f}]"
                ),
                row.transfer_interpretation,
            ]
        )

    table3 = (
        "# Table 3. Exploratory shared-vs-isolated transfer\n\n"
        "Negative Delta MAE favors the shared model.\n\n"
        + markdown_table(
            [
                "Target",
                "Horizon",
                "Mean Delta MAE",
                "Seed SD",
                "Relative improvement",
                "24h-block bootstrap 95% interval",
                "Exploratory interpretation",
            ],
            table3_rows,
        )
        + "\n"
    )
    (out / "TABLE_3_TRANSFER.md").write_text(
        table3,
        encoding="utf-8",
    )

    uncertainty_df = pd.DataFrame(uncertainty_records)
    uncertainty_summary = (
        uncertainty_df.groupby(
            ["model", "target", "horizon_hours"],
            as_index=False,
        )
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            interval_width_mean=("mean_interval_width", "mean"),
            gaussian_scale_mean=("mean_native_gaussian_scale", "mean"),
        )
    )
    uncertainty_summary.to_csv(
        out / "UNCERTAINTY_SUMMARY.csv",
        index=False,
    )

    stable = transfer_summary[
        transfer_summary["bootstrap_95_high"] < 0
    ].copy()
    stable = stable.sort_values(
        "relative_improvement_mean",
        ascending=False,
    )

    freeze_lines = [
        "# Freeze-Ready Exploratory Evidence Report",
        "",
        "**This is NOT the confirmatory protocol freeze.**",
        "",
        "Confirmatory holdout: `UNDEFINED_AND_NOT_REQUESTED`.",
        "",
        "## Automated gates completed",
        "",
        "- multi-year data gate;",
        "- operational information-set audit;",
        "- three isolated task-specific Transformers;",
        "- one shared multi-task Transformer;",
        "- three exploratory seeds;",
        "- target × horizon point metrics;",
        "- 90% split-conformal calibration;",
        "- paired 24-hour block-bootstrap transfer analysis;",
        "- training-derived regime analysis;",
        "- compute audit.",
        "",
        "## Stable exploratory positive-transfer candidates",
        "",
    ]

    if stable.empty:
        freeze_lines.append(
            "No target × horizon has a bootstrap interval wholly below zero."
        )
    else:
        for row in stable.itertuples(index=False):
            freeze_lines.append(
                f"- {row.target} +{row.horizon_hours} h: "
                f"mean Delta MAE={row.delta_mae_mean:.3f}, "
                f"relative improvement="
                f"{100 * row.relative_improvement_mean:.2f}%, "
                f"bootstrap 95% interval="
                f"[{row.bootstrap_95_low:.3f}, "
                f"{row.bootstrap_95_high:.3f}]"
            )

    freeze_lines.extend(
        [
            "",
            "## Mandatory work before confirmatory preregistration",
            "",
            "1. Review modern architecture comparators "
            "(at minimum DLinear and PatchTST-class evidence).",
            "2. Review the compute audit and decide whether full-capacity "
            "retraining is feasible at zero cost.",
            "3. Verify historical EPİAŞ DAM timing changes, if any, "
            "across 2021–2025.",
            "4. Select the primary target and primary horizon.",
            "5. Freeze seeds, model hyperparameters, preprocessing, "
            "metrics, inference and exclusions.",
            "6. Define the confirmatory holdout only after items 1–5.",
            "7. Register a separate OSF confirmatory preregistration.",
            "8. Only then open and run the confirmatory holdout.",
            "",
            "The automation intentionally stops here.",
        ]
    )
    (out / "FREEZE_READY_REPORT.md").write_text(
        "\n".join(freeze_lines) + "\n",
        encoding="utf-8",
    )

    prereg = f"""# OSF Confirmatory Preregistration — DRAFT ONLY

**DO NOT REGISTER YET.**

This draft was generated after exploratory learned-model analysis.
It does not define or expose the confirmatory holdout.

## Scientific question

Under a frozen chronological evaluation protocol, does a shared temporal
Transformer improve forecasting relative to a task-specific Transformer,
and in which electricity-system regimes does transfer help or hurt?

## Exploratory evidence package

Config SHA-256: `{config_hash}`

Exploratory data SHA-256: `{config['data_sha256']}`

Exploratory period: 2021–2025.

## Fields still requiring freeze

- Primary target: NOT FROZEN
- Primary horizon: NOT FROZEN
- Confirmatory holdout: UNDEFINED_AND_NOT_REQUESTED
- Final architecture comparator set: NOT FROZEN
- Final seed policy: NOT FROZEN
- Final paired inference procedure: NOT FROZEN
- Final exclusion/missing-data policy: NOT FROZEN

## Already established candidate framework

- Context: 168 hours
- Candidate horizons: +1 h, +6 h, +24 h
- Candidate targets: consumption, renewable generation, MCP/PTF
- Point metrics: MAE, RMSE, sMAPE
- Uncertainty: split conformal, nominal 90% coverage
- Transfer estimand: MAE(shared) - MAE(isolated)
- Chronological evaluation only
- No random split

This file is a preparation aid, not a preregistration.
"""
    (out / "OSF_CONFIRMATORY_PREREG_DRAFT.md").write_text(
        prereg,
        encoding="utf-8",
    )

    results_draft = (
        "# Exploratory Journal Results Draft\n\n"
        "## Data integrity\n\n"
        "The preceding data gate established 43,824 aligned hourly "
        "observations across 2021–2025 with complete three-target coverage.\n\n"
        "## Learned-model comparison\n\n"
        "See `TABLE_3_TRANSFER.md` and `TRANSFER_SUMMARY.csv`.\n\n"
        "## Uncertainty\n\n"
        "See `UNCERTAINTY_SUMMARY.csv`.\n\n"
        "## Regime dependence\n\n"
        "See `REGIME_TRANSFER.csv`.\n\n"
        "## MCP information-set qualification\n\n"
        "MCP +1 h and +6 h results are treated as retrospective "
        "representation benchmarks. MCP +24 h information-set-aware "
        "subsets are reported separately in "
        "`MCP_INFORMATION_SET_METRICS.csv`.\n\n"
        "## Evidence boundary\n\n"
        "All results in this package are exploratory. The confirmatory "
        "holdout remains undefined and untouched.\n"
    )
    (out / "JOURNAL_RESULTS_EXPLORATORY_DRAFT.md").write_text(
        results_draft,
        encoding="utf-8",
    )

    result_zip = (
        Path.home()
        / "Desktop"
        / "TGFM_AUTOPILOT_FREEZE_READY_RESULTS.zip"
    )
    include_names = [
        "RUN_CONFIG.json",
        "MODEL_METRICS_ALL_SEEDS.csv",
        "MODEL_METRICS_SUMMARY.csv",
        "TRANSFER_BY_SEED.csv",
        "TRANSFER_BLOCK_BOOTSTRAP.csv",
        "TRANSFER_SUMMARY.csv",
        "UNCERTAINTY_ALL_SEEDS.csv",
        "UNCERTAINTY_SUMMARY.csv",
        "REGIME_TRANSFER.csv",
        "MCP_INFORMATION_SET_METRICS.csv",
        "COMPUTE_AUDIT.csv",
        "TABLE_3_TRANSFER.md",
        "FREEZE_READY_REPORT.md",
        "OSF_CONFIRMATORY_PREREG_DRAFT.md",
        "JOURNAL_RESULTS_EXPLORATORY_DRAFT.md",
    ]
    with zipfile.ZipFile(
        result_zip,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for name in include_names:
            path = out / name
            if path.exists():
                archive.write(path, name)

    print("")
    print("=" * 72)
    print("SUCCESS - AUTOPILOT REACHED THE FREEZE-READY GATE.")
    print(f"Results: {result_zip}")
    print("Confirmatory holdout: UNDEFINED_AND_NOT_REQUESTED")
    print("No 2026+ data were used.")
    print("=" * 72)


if __name__ == "__main__":
    main()
