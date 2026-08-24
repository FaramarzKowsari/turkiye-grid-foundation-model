from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .losses import gaussian_nll


@dataclass
class TrainResult:
    losses: list[float]


def train_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    epochs: int = 2,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TrainResult:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, scale = model(x)
            loss = gaussian_nll(mean, scale, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        history.append(total / max(count, 1))
    return TrainResult(losses=history)
