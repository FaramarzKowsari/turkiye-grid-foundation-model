import numpy as np

from turkiye_grid_fm.uncertainty import conformal_radius, interval_coverage


def test_conformal_radius_and_coverage_shape():
    y = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    p = np.array([[0.1], [0.8], [2.2], [2.7], [4.1]])
    r = conformal_radius(y, p, coverage=0.8)
    assert r.shape == (1,)
    cov = interval_coverage(y, p - r, p + r)
    assert cov.shape == (1,)
