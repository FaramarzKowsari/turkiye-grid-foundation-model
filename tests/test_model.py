import torch

from turkiye_grid_fm.models.foundation import GridFoundationModel


def test_model_shapes_and_positive_scale():
    model = GridFoundationModel(input_dim=14, target_dim=9, d_model=32, nhead=4, num_layers=1)
    x = torch.randn(4, 48, 14)
    mean, scale = model(x)
    assert mean.shape == (4, 9)
    assert scale.shape == (4, 9)
    assert torch.all(scale > 0)
