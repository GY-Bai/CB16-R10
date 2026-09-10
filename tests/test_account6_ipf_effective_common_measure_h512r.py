from __future__ import annotations

import numpy as np

from cb16_local_opt.account6_ipf_effective_common_measure_h512r import (
    H512R_FORMAL_TOL,
    _effective_mass_h512r,
    calibrate_ipf_h512r,
    cell_ids_27_h512r,
    classify_h512r,
)
from cb16_local_opt.common_support_rank_geometry_decomposition_h511 import rank_percentile_h511


def test_h512r_average_tie_degenerate_account_rank_maps_middle_bin():
    d = np.zeros(8, dtype=np.float64)
    r, pct = rank_percentile_h511(d)
    assert np.all(r == 4.5)
    assert np.all(pct == 0.5)
    o = np.linspace(0.0, 1.0, 8)
    m = np.linspace(1.0, 0.0, 8)
    cells = cell_ids_27_h512r(o, m, pct)
    assert np.all((cells % 3) == 1)


def _synthetic_rows() -> list[np.ndarray]:
    return [
        np.asarray([0,0,1,2,3,4,5,6], dtype=np.int32),
        np.asarray([0,1,1,2,3,4,5,6], dtype=np.int32),
        np.asarray([0,1,2,2,3,4,5,6], dtype=np.int32),
        np.asarray([0,1,2,3,3,4,5,6], dtype=np.int32),
        np.asarray([0,1,2,3,4,4,5,6], dtype=np.int32),
    ]


def test_h512r_ipf_supports_zero_q_cells_and_closes_effective_margin():
    rows = _synthetic_rows()
    p = np.zeros(27, dtype=np.float64)
    for row in rows:
        p += np.bincount(row, minlength=27) / float(len(row))
    p /= float(len(rows))
    q = np.zeros(27, dtype=np.float64)
    q[:7] = np.asarray([0.08,0.12,0.16,0.20,0.18,0.14,0.12])
    assert abs(float(q.sum()) - 1.0) < 1e-15
    out = calibrate_ipf_h512r(cell_rows=rows, p=p, q=q)
    assert out["converged"] is True
    assert out["blocked_zero_support_rows"] is False
    m = np.asarray(out["multiplier"], dtype=np.float64)
    assert np.all(m[7:] == 0.0)
    receipt = _effective_mass_h512r(rows, m)
    eff = np.asarray(receipt["effective_macro_cell_mass"], dtype=np.float64)
    assert receipt["zero_positive_support_macro_row_count"] == 0
    assert np.max(np.abs(eff - q)) <= H512R_FORMAL_TOL


def test_h512r_ipf_fails_closed_when_macro_row_has_no_positive_support_candidate():
    rows = [np.asarray([0,0,1]), np.asarray([8,8,8])]
    p = np.zeros(27, dtype=np.float64)
    p[0] = 0.25; p[1] = 0.25; p[8] = 0.5
    q = np.zeros(27, dtype=np.float64)
    q[0] = 0.5; q[1] = 0.5
    out = calibrate_ipf_h512r(cell_rows=rows, p=p, q=q)
    assert out["converged"] is False
    assert out["blocked_zero_support_rows"] is True
    assert out["zero_positive_support_macro_row_count"] == 1


def _pair(pair: tuple[int,int], oa: str, ob: str) -> dict:
    return {
        "pair": list(pair),
        "blocked": None,
        "exact_effective_margin_closure": True,
        "side_a": {"orientation": oa},
        "side_b": {"orientation": ob},
    }


def test_h512r_persistent_classification_requires_all_three_native_flips_and_control():
    rows = [
        _pair((1,2), "ANTI_ALIGNMENT", "POSITIVE_ALIGNMENT"),
        _pair((2,3), "POSITIVE_ALIGNMENT", "ANTI_ALIGNMENT"),
        _pair((3,4), "ANTI_ALIGNMENT", "POSITIVE_ALIGNMENT"),
        _pair((4,5), "POSITIVE_ALIGNMENT", "POSITIVE_ALIGNMENT"),
    ]
    out = classify_h512r(rows)
    assert out["classification"] == "ACCOUNT6_EXACT_EFFECTIVE_MARGIN_STANDARDIZATION__NATIVE_TRANSITIONS_PERSIST"
    assert out["native_flip_persistence_count_of_3"] == 3
    assert out["native_flip_clean_removal_count_of_3"] == 0
    assert out["positive_control_pair_stable"] is True
