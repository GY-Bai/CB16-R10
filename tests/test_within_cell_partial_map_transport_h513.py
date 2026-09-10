from __future__ import annotations

import numpy as np

from cb16_local_opt.common_support_rank_geometry_decomposition_h511 import weighted_partial_rank_h511
from cb16_local_opt.within_cell_partial_map_transport_h513 import (
    H513_TOL,
    _map_from_contribution_h513,
    classify_h513,
    evaluate_pair_transport_h513,
    partial_cell_contributions_h513,
    q_weighted_corr_h513,
)


def test_partial_cell_contributions_exactly_reconstruct_weighted_partial() -> None:
    rng = np.random.default_rng(13)
    n = 81
    o = rng.normal(size=n)
    m = 0.25 * o + rng.normal(size=n)
    u = -0.15 * o + 0.4 * m + rng.normal(size=n)
    cells = np.arange(n, dtype=np.int32) % 27
    w = 0.2 + rng.random(n)
    row = partial_cell_contributions_h513(o, m, u, cells, w)
    frozen, _ = weighted_partial_rank_h511(o, m, u, w)
    assert abs(row["partial_rho"] - frozen) <= H513_TOL
    assert abs(sum(row["cell_contribution"]) - frozen) <= H513_TOL
    assert row["reconstruction_error"] <= H513_TOL


def test_cell_rate_map_has_q_weighted_mean_rho_and_centered_zero() -> None:
    q = np.arange(1, 28, dtype=np.float64); q /= q.sum()
    rho = 0.17
    h = np.linspace(-1.0, 1.0, 27)
    h -= np.sum(q * h)
    g = rho + h
    c = q * g
    out = _map_from_contribution_h513(c, rho, q)
    assert abs(out["q_weighted_mean_g"] - rho) <= H513_TOL
    assert abs(out["q_weighted_mean_h"]) <= H513_TOL
    np.testing.assert_allclose(out["h"], h, atol=1e-12, rtol=0.0)


def test_q_weighted_corr_detects_identical_and_opposite_patterns() -> None:
    q = np.ones(27, dtype=np.float64) / 27.0
    h = np.linspace(-2.0, 2.0, 27)
    a, da = q_weighted_corr_h513(h, h, q)
    b, db = q_weighted_corr_h513(h, -h, q)
    assert not da and not db
    assert abs(a - 1.0) <= 1e-12
    assert abs(b + 1.0) <= 1e-12


def _fake_side(rho: float, h: np.ndarray, q: np.ndarray, null_sign: float = -1.0):
    g = rho + h
    amap = {"g": g.tolist(), "h": h.tolist()}
    nulls = {}
    for s in (1, 7, 13, 23, 31):
        hn = null_sign * np.roll(h, s % len(h))
        nulls[s] = {"g": hn.tolist(), "h": hn.tolist()}
    return {"aligned_partial_rho": rho, "aligned_map": amap, "null_maps_by_shift": nulls}


def test_pair_transport_orthogonal_level_pattern_identity() -> None:
    q = np.ones(27, dtype=np.float64) / 27.0
    h = np.linspace(-1.0, 1.0, 27); h -= np.sum(q * h)
    a = _fake_side(-0.2, h, q)
    b = _fake_side(0.2, h * 1.01, q)
    out = evaluate_pair_transport_h513(a, b, q)
    assert out["orthogonal_identity_error"] <= H513_TOL
    assert out["level_fraction"] > 0.5
    assert out["aligned_centered_pattern_similarity"] > 0.99


def test_classification_global_level_and_mixed() -> None:
    def row(pair, ppass=True, level=True, pattern=False):
        return {"pair": list(pair), "transport": {"pattern_transport_pass": ppass, "level_dominant": level, "pattern_dominant": pattern}}
    strong = [row((1,2)), row((2,3)), row((3,4)), row((4,5), ppass=True, level=False)]
    assert classify_h513(strong, True)["classification"] == "CENTERED_CELL_PATTERN_TRANSPORTS__GLOBAL_LEVEL_SHIFT_DOMINANT"
    mixed = [row((1,2)), row((2,3), ppass=False, level=False, pattern=True), row((3,4)), row((4,5), ppass=True, level=False)]
    assert classify_h513(mixed, True)["classification"] == "MIXED_LEVEL_AND_WITHIN_CELL_MAPPING_DRIFT"
    assert classify_h513(strong, False)["classification"] == "EXECUTION_BLOCKED__H5_12R_PARENT_REPRODUCTION_FAILED"
