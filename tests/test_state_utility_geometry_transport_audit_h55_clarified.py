from __future__ import annotations

import numpy as np

from cb16_local_opt.state_utility_geometry_transport_audit_h55_clarified import (
    nearest_decile_ratio_clarified_h55,
    normalized_rank_rows_clarified_h55,
)


def test_constant_rank_row_maps_to_zero_information():
    x = np.asarray([[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    ranks, deg = normalized_rank_rows_clarified_h55(x)
    assert deg.tolist() == [True, False]
    assert np.array_equal(ranks[0], np.zeros(3, dtype=np.float64))
    assert np.isclose(np.linalg.norm(ranks[1]), 1.0)


def test_nearest_ratio_is_one_when_state_distance_is_degenerate():
    d = np.asarray([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
    u = np.asarray([[9.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    ratio = nearest_decile_ratio_clarified_h55(d, u, np.asarray([True, False]))
    assert ratio[0] == 1.0
    assert np.isfinite(ratio[1])


def test_nonconstant_rank_rows_match_standard_average_rank_geometry():
    x = np.asarray([[4.0, 1.0, 3.0, 2.0]], dtype=np.float64)
    ranks, deg = normalized_rank_rows_clarified_h55(x)
    assert deg.tolist() == [False]
    expected = np.asarray([4.0, 1.0, 3.0, 2.0], dtype=np.float64)
    expected -= expected.mean()
    expected /= np.linalg.norm(expected)
    assert np.allclose(ranks[0], expected)
