from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    target_dim: int
    context_length: int = 168
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class WindowConfig:
    context: int = 168
    horizons: tuple[int, ...] = field(default_factory=lambda: (1, 6, 24))
    targets: tuple[str, ...] = field(
        default_factory=lambda: ("consumption_mwh", "renewable_mwh", "mcp_tl_mwh")
    )
