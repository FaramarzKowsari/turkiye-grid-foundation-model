from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from turkiye_grid_fm.models.foundation import GridFoundationModel
from turkiye_grid_fm.synthetic import make_synthetic_grid
from turkiye_grid_fm.train import train_model
from turkiye_grid_fm.windows import GridWindowDataset

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


def main() -> None:
    torch.manual_seed(7)
    frame = make_synthetic_grid(hours=24 * 30)
    dataset = GridWindowDataset(frame, FEATURES, TARGETS, context=48, horizons=(1, 6, 24))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = GridFoundationModel(
        input_dim=len(FEATURES), target_dim=dataset.target_dim, d_model=32, nhead=4, num_layers=1
    )
    result = train_model(model, loader, epochs=1, lr=1e-3)
    print({"status": "ok", "loss": result.losses[-1], "samples": len(dataset)})


if __name__ == "__main__":
    main()
