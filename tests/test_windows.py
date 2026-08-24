from turkiye_grid_fm.synthetic import make_synthetic_grid
from turkiye_grid_fm.windows import GridWindowDataset


def test_window_dataset_shape():
    frame = make_synthetic_grid(400)
    features = ["consumption_mwh", "renewable_mwh", "mcp_tl_mwh", "hour_sin", "hour_cos"]
    targets = ["consumption_mwh", "renewable_mwh", "mcp_tl_mwh"]
    ds = GridWindowDataset(frame, features, targets, context=48, horizons=(1, 6, 24))
    x, y = ds[0]
    assert x.shape == (48, len(features))
    assert y.shape == (9,)
