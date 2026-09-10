from __future__ import annotations

import inspect

import numpy as np

from cb16_local_opt.operator_conditional_medium_geometry_h58 import (
    H58_H57_FAILURE_FOLDS,
    H58_SHIFTS,
    adjudicate_h58,
    partial_spearman_rank_h58,
    run_fold_h58,
)


def test_h58_frozen_shifts_and_failure_folds():
    assert H58_SHIFTS == (1, 7, 13, 23, 31)
    assert H58_H57_FAILURE_FOLDS == (1, 3)


def test_run_fold_signature_has_no_teacher_student_or_fusion_weight_inputs():
    params = set(inspect.signature(run_fold_h58).parameters)
    forbidden = {
        "teacher",
        "teacher_config",
        "kernel",
        "student",
        "organ_weight",
        "fusion_weight",
        "distance_weight",
        "future_utility_selector",
    }
    assert not (params & forbidden)


def test_partial_spearman_is_one_when_medium_and_utility_rank_residuals_match():
    operator = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    medium = np.asarray([0.0, 2.0, 1.0, 5.0, 3.0, 4.0])
    utility = medium.copy()
    rho, receipt = partial_spearman_rank_h58(operator, medium, utility)
    assert abs(rho - 1.0) <= 1e-15
    assert receipt["zero_information"] is False


def test_partial_spearman_matches_closed_form_rank_partial_correlation():
    operator = np.asarray([0.0, 2.0, 1.0, 4.0, 3.0, 5.0])
    medium = np.asarray([1.0, 0.0, 3.0, 2.0, 5.0, 4.0])
    utility = np.asarray([0.0, 3.0, 1.0, 5.0, 2.0, 4.0])
    rho, receipt = partial_spearman_rank_h58(operator, medium, utility)
    r_mu = float(receipt["rho_medium_utility"])
    r_mo = float(receipt["rho_medium_operator"])
    r_uo = float(receipt["rho_operator_utility"])
    expected = (r_mu - r_mo * r_uo) / np.sqrt((1.0 - r_mo * r_mo) * (1.0 - r_uo * r_uo))
    assert abs(rho - float(expected)) <= 1e-14


def test_partial_spearman_collinear_medium_operator_is_zero_information():
    operator = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    medium = operator.copy()
    utility = np.asarray([0.0, 2.0, 1.0, 4.0, 3.0])
    rho, receipt = partial_spearman_rank_h58(operator, medium, utility)
    assert rho == 0.0
    assert receipt["zero_information"] is True


def _row(fold: int, aligned: float, null_median: float, *, all_below: bool = True) -> dict:
    if all_below:
        nulls = {
            1: null_median - 0.04,
            7: null_median - 0.03,
            13: null_median,
            23: null_median - 0.02,
            31: null_median - 0.01,
        }
    else:
        nulls = {
            1: null_median + 0.04,
            7: null_median + 0.03,
            13: null_median,
            23: null_median + 0.02,
            31: null_median + 0.01,
        }
    return {
        "fold": fold,
        "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": nulls,
        "median_null_partial_rho": null_median,
        "aligned_minus_null_median": aligned - null_median,
        "aligned_positive": aligned > 0.0,
        "aligned_gt_null_median": aligned > null_median,
        "aligned_gt_each_null_count": sum(aligned > x for x in nulls.values()),
        "all_null_medium_distance_multisets_preserved": True,
    }


def test_adjudication_can_support_composition_extraction_bottleneck():
    rows = [_row(i, 0.05, 0.0) for i in range(1, 6)]
    out = adjudicate_h58(rows)
    assert out["classification"] == "OPERATOR_CONDITIONAL_MEDIUM_VALUE_SUPPORTED__COMPOSITION_EXTRACTION_BOTTLENECK"
    assert out["global_gate"]["passed"] is True
    assert out["h5_7_failure_fold_gate"]["conditional_value_survives"] is True


def test_adjudication_can_support_global_value_but_failure_fold_mixed():
    rows = [_row(i, 0.05, 0.0) for i in range(1, 6)]
    rows[0] = _row(1, -0.01, -0.02)
    out = adjudicate_h58(rows)
    assert out["global_gate"]["passed"] is True
    assert out["h5_7_failure_fold_gate"]["conditional_value_survives"] is False
    assert out["classification"] == "OPERATOR_CONDITIONAL_MEDIUM_VALUE_SUPPORTED__H5_7_FAILURE_FOLDS_MIXED"


def test_adjudication_can_identify_failure_fold_conditional_anti_alignment():
    rows = [_row(i, 0.01, 0.0) for i in range(1, 6)]
    rows[0] = _row(1, -0.05, -0.01, all_below=False)
    rows[2] = _row(3, -0.04, -0.01, all_below=False)
    out = adjudicate_h58(rows)
    assert out["global_gate"]["passed"] is False
    assert out["h5_7_failure_fold_gate"]["conditional_anti_alignment"] is True
    assert out["classification"] == "H5_7_FAILURE_FOLDS_SHOW_MEDIUM_CONDITIONAL_ANTI_ALIGNMENT"


def test_adjudication_can_leave_operator_conditional_geometry_unsupported():
    rows = [
        _row(1, -0.01, 0.0),
        _row(2, -0.01, 0.0),
        _row(3, 0.01, 0.0),
        _row(4, 0.01, 0.0),
        _row(5, 0.01, 0.0),
    ]
    out = adjudicate_h58(rows)
    assert out["global_gate"]["passed"] is False
    assert out["h5_7_failure_fold_gate"]["conditional_value_survives"] is False
    assert out["h5_7_failure_fold_gate"]["conditional_anti_alignment"] is False
    assert out["classification"] == "OPERATOR_CONDITIONAL_MEDIUM_GEOMETRY_NOT_SUPPORTED"
