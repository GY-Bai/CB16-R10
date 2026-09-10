from __future__ import annotations

import inspect

import numpy as np

from cb16_local_opt.common_support_rank_geometry_decomposition_h511 import _post_weight_cell_mass_h511
from cb16_local_opt.h511_effective_measure_closure_h5111 import (
    H5111_TOL,
    _distance_receipt_h5111,
    effective_macro_mass_from_cell_rows_h5111,
    fold_cell_rows_h5111,
)


def test_h5111_local_normalization_can_break_nominal_global_q_identity():
    # Two equal macro rows occupy disjoint cells.  Nominal global weighting changes
    # their relative mass, but H5.11 renormalizes each row to total mass one, so the
    # effective macro mass stays 1/2 : 1/2.
    p = np.zeros(9, dtype=np.float64)
    p[0] = 0.5
    p[1] = 0.5
    multiplier = np.zeros(9, dtype=np.float64)
    multiplier[0] = 2.0
    multiplier[1] = 1.0
    nominal = _post_weight_cell_mass_h511(p, multiplier)
    got = effective_macro_mass_from_cell_rows_h5111(
        [np.asarray([0, 0]), np.asarray([1, 1])], multiplier
    )
    effective = np.asarray(got["effective_macro_candidate_mass"], dtype=np.float64)
    assert np.allclose(nominal[:2], [2.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(effective[:2], [0.5, 0.5])
    assert _distance_receipt_h5111(effective, nominal)["max_abs"] > H5111_TOL


def test_h5111_exact_closure_when_every_macro_row_has_same_cell_composition():
    p = np.full(9, 1.0 / 9.0, dtype=np.float64)
    q = np.asarray([0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.16, 0.12, 0.12], dtype=np.float64)
    assert abs(float(q.sum()) - 1.0) < 1e-15
    multiplier = q / p
    rows = [np.arange(9, dtype=np.int32) for _ in range(7)]
    got = effective_macro_mass_from_cell_rows_h5111(rows, multiplier)
    effective = np.asarray(got["effective_macro_candidate_mass"], dtype=np.float64)
    assert np.max(np.abs(effective - q)) <= 1e-15
    assert got["target_scenario_raw_multiplier_sum_min"] == got["target_scenario_raw_multiplier_sum_max"]


def test_h5111_effective_mass_is_equal_macro_not_pair_row_weighted():
    multiplier = np.ones(9, dtype=np.float64)
    rows = [np.asarray([0]), np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1])]
    got = effective_macro_mass_from_cell_rows_h5111(rows, multiplier)
    effective = np.asarray(got["effective_macro_candidate_mass"], dtype=np.float64)
    # Each target-scenario macro row has equal authority despite unequal candidate count.
    assert np.allclose(effective[:2], [0.5, 0.5])
    assert got["candidate_count_min"] == 1
    assert got["candidate_count_max"] == 9


def test_h5111_distance_receipt_total_variation_is_half_l1():
    a = np.asarray([0.6, 0.4] + [0.0] * 7)
    b = np.asarray([0.5, 0.5] + [0.0] * 7)
    d = _distance_receipt_h5111(a, b)
    assert abs(d["l1"] - 0.2) < 1e-15
    assert abs(d["total_variation"] - 0.1) < 1e-15
    assert abs(d["max_abs"] - 0.1) < 1e-15


def test_h5111_audit_module_never_references_index_utilities():
    src = inspect.getsource(fold_cell_rows_h5111)
    assert "index.utilities" not in src
    assert "utility_mse" not in src
