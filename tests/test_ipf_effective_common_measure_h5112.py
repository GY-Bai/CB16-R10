from __future__ import annotations

import numpy as np

from cb16_local_opt.common_support_rank_geometry_decomposition_h511 import evaluate_fold_overlap_h511
from cb16_local_opt.ipf_effective_common_measure_h5112 import (
    H5112_FORMAL_TOL,
    calibrate_ipf_h5112,
    classify_h5112,
    effective_macro_cell_mass_h5112,
    evaluate_fold_multiplier_h5112,
)
from cb16_local_opt.operator_conditional_medium_geometry_h58 import H58_SHIFTS


def _rows_all_cells() -> list[np.ndarray]:
    return [
        np.asarray([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        np.asarray([0, 1, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        np.asarray([0, 1, 2, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        np.asarray([0, 1, 2, 3, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        np.asarray([0, 1, 2, 3, 4, 4, 5, 6, 7, 8], dtype=np.int32),
        np.asarray([0, 1, 2, 3, 4, 5, 5, 6, 7, 8], dtype=np.int32),
    ]


def test_h5112_ipf_closes_effective_macro_margin_to_q():
    rows = _rows_all_cells()
    q = np.asarray([0.07, 0.09, 0.11, 0.13, 0.15, 0.12, 0.10, 0.11, 0.12], dtype=np.float64)
    assert abs(float(q.sum()) - 1.0) < 1e-15
    out = calibrate_ipf_h5112(cell_rows=rows, q=q, initial_multiplier=np.ones(9))
    assert out["converged"] is True
    m = np.asarray(out["multiplier"], dtype=np.float64)
    eff = np.asarray(effective_macro_cell_mass_h5112(rows, m)["effective_macro_cell_mass"], dtype=np.float64)
    assert np.max(np.abs(eff - q)) <= H5112_FORMAL_TOL
    assert out["damping_used"] is False
    assert out["regularization_used"] is False
    assert out["clipping_used"] is False


def test_h5112_effective_margin_is_invariant_to_global_multiplier_gauge():
    rows = _rows_all_cells()
    m = np.asarray([0.4, 0.8, 1.2, 1.6, 2.0, 0.7, 1.1, 1.5, 1.9], dtype=np.float64)
    a = np.asarray(effective_macro_cell_mass_h5112(rows, m)["effective_macro_cell_mass"])
    b = np.asarray(effective_macro_cell_mass_h5112(rows, 37.0 * m)["effective_macro_cell_mass"])
    assert np.max(np.abs(a - b)) <= 1e-15


def _synthetic_payload() -> dict:
    operator = np.asarray([1, 4, 2, 8, 5, 9, 3, 7, 6], dtype=np.float64)
    medium = np.asarray([5, 1, 7, 3, 9, 2, 8, 4, 6], dtype=np.float64)
    utility = np.asarray([2, 8, 4, 1, 7, 5, 9, 3, 6], dtype=np.float64)
    nulls = {int(s): np.roll(medium, -(int(s) % len(medium))) for s in H58_SHIFTS}
    scenario = {
        "scenario": "SYNTHETIC",
        "operator_rank": operator,
        "medium_rank": medium,
        "utility_rank": utility,
        "cell_ids": np.arange(9, dtype=np.int32),
        "null_medium_rank_by_shift": nulls,
    }
    return {
        "fold": 1,
        "groups": [{"future_group_id": "SYNTHETIC", "timestamp_ms": 0, "scenarios": [scenario]}],
        "cell_mass": (np.ones(9, dtype=np.float64) / 9.0).tolist(),
        "eval_dependence_groups": 1,
    }


def test_h5112_sibling_evaluator_matches_h511_when_old_nominal_q_receipt_is_valid():
    payload = _synthetic_payload()
    m = np.ones(9, dtype=np.float64)
    q = np.ones(9, dtype=np.float64) / 9.0
    old = evaluate_fold_overlap_h511(payload, m, q)
    new = evaluate_fold_multiplier_h5112(payload, m, q)
    assert abs(float(old["aligned_partial_rho"]) - float(new["aligned_partial_rho"])) <= 1e-15
    assert abs(float(old["null_median_partial_rho"]) - float(new["null_median_partial_rho"])) <= 1e-15
    assert old["orientation"] == new["orientation"]
    for shift in H58_SHIFTS:
        assert abs(float(old["null_partial_rho_by_shift"][int(shift)]) - float(new["null_partial_rho_by_shift"][int(shift)])) <= 1e-15


def _pair(pair: tuple[int, int], oa: str, ob: str) -> dict:
    return {
        "pair": list(pair),
        "calibration": {"side_a": {"converged": True}, "side_b": {"converged": True}},
        "exact_effective_margin_closure": True,
        "side_a": {"orientation": oa},
        "side_b": {"orientation": ob},
    }


def test_h5112_persistent_classification_requires_all_three_native_transitions_and_control():
    rows = [
        _pair((1, 2), "ANTI_ALIGNMENT", "POSITIVE_ALIGNMENT"),
        _pair((2, 3), "POSITIVE_ALIGNMENT", "ANTI_ALIGNMENT"),
        _pair((3, 4), "ANTI_ALIGNMENT", "POSITIVE_ALIGNMENT"),
        _pair((4, 5), "POSITIVE_ALIGNMENT", "POSITIVE_ALIGNMENT"),
    ]
    out = classify_h5112(rows, reproduction_passed=True)
    assert out["classification"] == "EXACT_MACRO_CELL_OCCUPANCY_STANDARDIZATION__NATIVE_TRANSITIONS_PERSIST"
    assert out["native_flip_persistence_count_of_3"] == 3
    assert out["positive_control_pair_stable"] is True
