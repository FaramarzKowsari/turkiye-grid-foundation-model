from __future__ import annotations

import torch


def gaussian_nll(mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    variance = scale.square()
    return (0.5 * (torch.log(variance) + (target - mean).square() / variance)).mean()
