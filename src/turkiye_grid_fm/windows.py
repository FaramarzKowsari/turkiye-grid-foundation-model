from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class GridWindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        target_columns: Sequence[str],
        context: int = 168,
        horizons: Sequence[int] = (1, 6, 24),
        feature_scaler: Standardizer | None = None,
        target_scaler: Standardizer | None = None,
    ) -> None:
        if context < 1 or not horizons or min(horizons) < 1:
            raise ValueError("context and horizons must be positive.")
        needed = list(dict.fromkeys([*feature_columns, *target_columns]))
        clean = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(clean) <= context + max(horizons):
            raise ValueError("Not enough complete rows for the requested context and horizons.")

        features = clean[list(feature_columns)].to_numpy(dtype=np.float32)
        targets = clean[list(target_columns)].to_numpy(dtype=np.float32)
        self.feature_scaler = feature_scaler or Standardizer.fit(features)
        self.target_scaler = target_scaler or Standardizer.fit(targets)
        self.features = self.feature_scaler.transform(features).astype(np.float32)
        self.targets = self.target_scaler.transform(targets).astype(np.float32)
        self.context = context
        self.horizons = tuple(int(h) for h in horizons)
        self.max_horizon = max(self.horizons)
        self.target_dim = len(target_columns) * len(self.horizons)

    def __len__(self) -> int:
        return len(self.features) - self.context - self.max_horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = idx + self.context
        x = self.features[idx:end]
        y = np.concatenate([self.targets[end + h - 1] for h in self.horizons], axis=0)
        return torch.from_numpy(x), torch.from_numpy(y)
