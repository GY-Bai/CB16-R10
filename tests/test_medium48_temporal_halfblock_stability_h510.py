from __future__ import annotations

import inspect

from cb16_local_opt.medium48_temporal_halfblock_stability_h510 import (
    H510_EXPECTED,
    H510_FOLDS,
    _aggregate_group_rows_h510,
    adjudicate_h510,
    orientation_h510,
    run_fold_h510,
)


def test_h510_frozen_folds_and_parent_orientations():
    assert H510_FOLDS == (1, 2, 3, 4, 5)
    assert [H510_EXPECTED[i]["orientation"] for i in H510_FOLDS] == [
        "ANTI_ALIGNMENT",
        "POSITIVE_ALIGNMENT",
        "ANTI_ALIGNMENT",
        "POSITIVE_ALIGNMENT",
        "POSITIVE_ALIGNMENT",
    ]


def test_run_fold_signature_has_no_teacher_student_weight_or_boundary_inputs():
    params = set(inspect.signature(run_fold_h510).parameters)
    forbidden = {
        "teacher",
        "teacher_config",
        "kernel",
        "student",
        "organ_weight",
        "fusion_weight",
        "distance_weight",
        "change_point",
        "boundary",
        "regime_selector",
    }
    assert not (params & forbidden)


def test_orientation_is_symmetric_and_mixed_when_sign_and_null_disagree():
    assert orientation_h510(0.2, 0.1) == "POSITIVE_ALIGNMENT"
    assert orientation_h510(-0.2, -0.1) == "ANTI_ALIGNMENT"
    assert orientation_h510(0.1, 0.2) == "MIXED"
    assert orientation_h510(-0.1, -0.2) == "MIXED"
    assert orientation_h510(0.0, 0.0) == "MIXED"


def test_group_aggregation_preserves_future_group_macro_weighting():
    rows = [
        {
            "future_group_id": "FUT:BTCUSDT:1000",
            "aligned_partial_rho": -0.2,
            "null_partial_rho_by_shift": {1: 0.1, 7: 0.1, 13: 0.1, 23: 0.1, 31: 0.1},
        },
        {
            "future_group_id": "FUT:ETHUSDT:2000",
            "aligned_partial_rho": 0.0,
            "null_partial_rho_by_shift": {1: 0.1, 7: 0.1, 13: 0.1, 23: 0.1, 31: 0.1},
        },
    ]
    out = _aggregate_group_rows_h510(rows)
    assert abs(out["aligned_partial_rho"] + 0.1) <= 1e-15
    assert abs(out["null_median_partial_rho"] - 0.1) <= 1e-15
    assert out["orientation"] == "ANTI_ALIGNMENT"
    assert out["first_timestamp_ms"] == 1000
    assert out["last_timestamp_ms"] == 2000


def _fold_result(fold: int, half_orientations: tuple[str, str], *, reproduce: bool = True) -> dict:
    expected = H510_EXPECTED[fold]
    halves = []
    for i, orient in enumerate(half_orientations, start=1):
        if orient == "ANTI_ALIGNMENT":
            aligned, null = -0.2, -0.1
        elif orient == "POSITIVE_ALIGNMENT":
            aligned, null = 0.2, 0.1
        else:
            aligned, null = 0.05, 0.1
        halves.append({
            "half": i,
            "orientation": orient,
            "aligned_partial_rho": aligned,
            "null_median_partial_rho": null,
            "aligned_minus_null_median": aligned - null,
            "future_group_count": 40,
            "first_timestamp_ms": fold * 100000 + i * 1000,
            "last_timestamp_ms": fold * 100000 + i * 1000 + 500,
        })
    return {
        "fold": fold,
        "full_fold": {
            "reproduced": reproduce,
            "orientation": expected["orientation"],
        },
        "halves": halves,
    }


def _stable_rows():
    out = []
    for fold in H510_FOLDS:
        orient = H510_EXPECTED[fold]["orientation"]
        out.append(_fold_result(fold, (orient, orient)))
    return out


def test_adjudication_can_identify_within_fold_stable_cross_fold_reversal():
    out = adjudicate_h510(_stable_rows())
    assert out["baseline_reproduction_passed"] is True
    assert out["failure_fold_persistence_4_of_4"] is True
    assert out["positive_fold_persistence_6_of_6"] is True
    assert out["classification"] == "WITHIN_FOLD_RELATION_STABLE__CROSS_FOLD_SIGN_REVERSAL"


def test_adjudication_can_identify_failure_fold_internal_mixture_only():
    rows = _stable_rows()
    rows[0] = _fold_result(1, ("ANTI_ALIGNMENT", "MIXED"))
    out = adjudicate_h510(rows)
    assert out["failure_fold_persistence_4_of_4"] is False
    assert out["positive_fold_persistence_6_of_6"] is True
    assert out["classification"] == "FAILURE_FOLDS_INTERNALLY_MIXED__POSITIVE_FOLDS_STABLE"


def test_adjudication_can_identify_positive_fold_internal_mixture_only():
    rows = _stable_rows()
    rows[1] = _fold_result(2, ("POSITIVE_ALIGNMENT", "MIXED"))
    out = adjudicate_h510(rows)
    assert out["failure_fold_persistence_4_of_4"] is True
    assert out["positive_fold_persistence_6_of_6"] is False
    assert out["classification"] == "FAILURE_FOLDS_STABLE__POSITIVE_FOLDS_INTERNALLY_MIXED"


def test_adjudication_can_identify_mixed_within_and_across_folds():
    rows = _stable_rows()
    rows[0] = _fold_result(1, ("ANTI_ALIGNMENT", "MIXED"))
    rows[1] = _fold_result(2, ("POSITIVE_ALIGNMENT", "MIXED"))
    out = adjudicate_h510(rows)
    assert out["classification"] == "TEMPORAL_RELATION_MIXED_WITHIN_AND_ACROSS_FOLDS"


def test_adjudication_fails_closed_on_baseline_reproduction_failure():
    rows = _stable_rows()
    rows[0] = _fold_result(1, ("ANTI_ALIGNMENT", "ANTI_ALIGNMENT"), reproduce=False)
    out = adjudicate_h510(rows)
    assert out["baseline_reproduction_passed"] is False
    assert out["classification"] == "H5_10_BASELINE_REPRODUCTION_FAILED"
