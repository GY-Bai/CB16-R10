from __future__ import annotations

import inspect

import numpy as np

from cb16_local_opt.medium48_relational_composition_falsification_h57 import (
    H57_SHIFTS,
    adjudicate_h57,
    run_fold_h57,
    tau_gap_h57,
)


def test_h57_frozen_shifts_match_h5_family():
    assert H57_SHIFTS == (1, 7, 13, 23, 31)


def test_run_fold_signature_has_no_teacher_student_or_tuning_inputs():
    params = set(inspect.signature(run_fold_h57).parameters)
    forbidden = {
        "teacher",
        "teacher_config",
        "kernel",
        "student",
        "organ_weight",
        "distance_weight",
        "future_utility_selector",
    }
    assert not (params & forbidden)


def test_tau_gap_perfect_reverse_and_positive_affine_reference_scale_invariance():
    ref = np.asarray([0.0, 0.2, 0.8, 1.7, 4.0], dtype=np.float64)
    pred_good = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    pred_bad = pred_good[::-1].copy()
    good, _ = tau_gap_h57(ref, pred_good)
    bad, _ = tau_gap_h57(ref, pred_bad)
    scaled, _ = tau_gap_h57(7.25 * ref + 13.0, pred_good)
    assert good == 1.0
    assert bad == -1.0
    assert abs(scaled - good) <= 1e-15


def test_tau_gap_prediction_tie_gets_half_credit():
    ref = np.asarray([0.0, 1.0], dtype=np.float64)
    pred = np.asarray([2.0, 2.0], dtype=np.float64)
    tau, receipt = tau_gap_h57(ref, pred)
    assert tau == 0.0
    assert receipt["prediction_tie_pair_count"] == 1


def test_tau_gap_all_reference_ties_are_zero_information():
    ref = np.asarray([3.0, 3.0, 3.0, 3.0], dtype=np.float64)
    pred = np.asarray([9.0, 1.0, 5.0, 2.0], dtype=np.float64)
    tau, receipt = tau_gap_h57(ref, pred)
    assert tau == 0.0
    assert receipt["degenerate_reference_gap_rank_count"] == 3


def test_frozen_rms_distance_decomposition_and_operator_duplicate_identity():
    # H5.5/H5.6 divide squared Euclidean distance by active dimension count.
    # Therefore 96D Market = sqrt(0.5*(dO^2+dM^2)), and [O,O] at 96D = dO.
    o_sq_sum = np.asarray([48.0, 12.0, 3.0], dtype=np.float64)
    m_sq_sum = np.asarray([0.0, 48.0, 12.0], dtype=np.float64)
    d_o = np.sqrt(o_sq_sum / 48.0)
    d_m = np.sqrt(m_sq_sum / 48.0)
    d_market_direct = np.sqrt((o_sq_sum + m_sq_sum) / 96.0)
    d_market_decomposed = np.sqrt(0.5 * (d_o * d_o + d_m * d_m))
    d_oo = np.sqrt(0.5 * (d_o * d_o + d_o * d_o))
    assert np.allclose(d_market_direct, d_market_decomposed, rtol=0.0, atol=1e-15)
    assert np.allclose(d_oo, d_o, rtol=0.0, atol=1e-15)


def _row(fold: int, delta_operator: float, aligned_minus_null: float) -> dict:
    tau_operator = 0.10
    tau_true = tau_operator + delta_operator
    median_null = tau_true - aligned_minus_null
    if aligned_minus_null > 0:
        nulls = {1: median_null - 0.03, 7: median_null - 0.02, 13: median_null, 23: median_null - 0.01, 31: median_null - 0.04}
    elif aligned_minus_null < 0:
        nulls = {1: median_null + 0.03, 7: median_null + 0.02, 13: median_null, 23: median_null + 0.01, 31: median_null + 0.04}
    else:
        nulls = {1: tau_true - 0.02, 7: tau_true - 0.01, 13: tau_true, 23: tau_true + 0.01, 31: tau_true + 0.02}
    return {
        "fold": fold,
        "tau_operator": tau_operator,
        "tau_true_market": tau_true,
        "delta_true_market_minus_operator": delta_operator,
        "tau_null_by_shift": nulls,
        "median_tau_null": median_null if aligned_minus_null != 0 else tau_true,
        "aligned_minus_null_median": aligned_minus_null,
        "aligned_gt_each_null_count": sum(tau_true > v for v in nulls.values()),
        "aligned_lt_each_null_count": sum(tau_true < v for v in nulls.values()),
        "all_null_medium_distance_multisets_preserved": True,
        "market_distance_decomposition_max_abs_error": 0.0,
        "operator_duplicate_96d_max_abs_error": 0.0,
    }


def test_adjudication_aligned_medium_benefits_and_market_beats_operator():
    rows = [_row(i, 0.04, 0.03) for i in range(1, 6)]
    out = adjudicate_h57(rows)
    assert out["classification"] == "ALIGNED_MEDIUM_BENEFITS_AND_MARKET_BEATS_OPERATOR"
    assert out["aligned_medium_beats_null_gate"]["passed"] is True
    assert out["true_market_beats_operator_gate"]["passed"] is True


def test_adjudication_aligned_medium_beneficial_but_market_still_below_operator():
    rows = [_row(i, -0.04, 0.03) for i in range(1, 6)]
    out = adjudicate_h57(rows)
    assert out["classification"] == "ALIGNED_MEDIUM_BENEFICIAL_BUT_INSUFFICIENT_TO_BEAT_OPERATOR"


def test_adjudication_can_localize_alignment_specific_harm():
    rows = [_row(i, -0.04, -0.03) for i in range(1, 6)]
    out = adjudicate_h57(rows)
    assert out["classification"] == "ALIGNED_MEDIUM_RELATIONAL_STRUCTURE_SPECIFICALLY_HARMS_COMPOSITE_GEOMETRY"
    assert out["aligned_medium_worse_than_null_gate"]["passed"] is True
    assert out["true_market_worse_than_operator_gate"]["passed"] is True


def test_adjudication_generic_augmentation_compatible_when_null_direction_is_unstable():
    rows = [_row(i, -0.04, 0.0) for i in range(1, 6)]
    out = adjudicate_h57(rows)
    assert out["classification"] == "NO_STABLE_ALIGNMENT_SPECIFIC_EFFECT__GENERIC_AUGMENTATION_COMPATIBLE"
    assert out["aligned_medium_beats_null_gate"]["passed"] is False
    assert out["aligned_medium_worse_than_null_gate"]["passed"] is False
