from __future__ import annotations

import numpy as np

from cb16_local_opt.m_series_m0_policy_drift_attribution import (
    adjudicate_m0,
    attribute_direction_migration_m0,
    direction_advantage_from_teacher_probs,
)


def _softmax_means(means, tau=0.002):
    x = np.asarray(means, dtype=np.float64) / tau
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _probs_for_classes(classes):
    out = []
    for c in classes:
        p = np.full(3, 0.05, dtype=np.float64)
        p[int(c)] = 0.90
        out.append(p)
    return np.asarray(out)


def test_direction_advantage_exactly_recovers_teacher_best_mean_gap():
    means = np.asarray(
        [
            [0.010, 0.014, 0.012],
            [0.021, 0.020, 0.027],
            [-0.003, -0.005, -0.004],
        ],
        dtype=np.float64,
    )
    p = _softmax_means(means)
    g0_dir = np.asarray([0, 1, 2], dtype=np.int64)
    got = direction_advantage_from_teacher_probs(p, g0_dir)
    expected = np.asarray(
        [0.014 - 0.010, 0.027 - 0.020, -0.003 - (-0.004)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_m0_row_classification_and_majority_attribution():
    teacher_dir = [0, 0, 1, 2, 2]
    g0_dir = [0, 0, 0, 0, 1]
    challenger_dir = [0, 1, 1, 0, 0]
    rows, summary = attribute_direction_migration_m0(
        parent_ids=[f"p{i}" for i in range(5)],
        dependence_group_ids=[f"g{i}" for i in range(5)],
        teacher_probs=_probs_for_classes(teacher_dir),
        g0_probs=_probs_for_classes(g0_dir),
        challenger_probs=_probs_for_classes(challenger_dir),
        g0_requested_risk_raw=[0.1, 0.2, 0.3, 0.4, 0.5],
        challenger_requested_risk_raw=[0.1, 0.3, 0.4, 0.4, 0.6],
    )
    assert [r["class"] for r in rows] == [
        "PRESERVED_TEACHER_AGREEMENT",
        "DRIFT_ON_TEACHER_AGREEMENT",
        "DIRECT_TEACHER_CORRECTION",
        "STAYED_G0_DESPITE_TEACHER_DISAGREEMENT",
        "THIRD_DIRECTION_DRIFT",
    ]
    assert summary["direction_change_count"] == 3
    assert summary["direct_teacher_correction_count"] == 1
    assert summary["non_teacher_attributable_change_count"] == 2
    assert abs(summary["direct_teacher_correction_fraction_of_all_direction_changes"] - 1 / 3) < 1e-12
    assert abs(summary["non_teacher_attributable_fraction_of_all_direction_changes"] - 2 / 3) < 1e-12
    assert abs(summary["teacher_g0_agreement_direction_change_rate"] - 0.5) < 1e-12
    assert abs(summary["teacher_g0_disagreement_move_to_teacher_rate"] - 1 / 3) < 1e-12


def _fold(fold, direct_fraction, changes=10):
    return {
        "fold": fold,
        "direction_change_count": changes,
        "direction_change_rate": 0.2,
        "direct_teacher_correction_fraction_of_all_direction_changes": direct_fraction
        if changes
        else None,
        "non_teacher_attributable_fraction_of_all_direction_changes": (1.0 - direct_fraction)
        if changes
        else None,
        "teacher_g0_agreement_direction_change_rate": 0.1,
        "teacher_g0_disagreement_move_to_teacher_rate": 0.6,
    }


def test_m0_adjudication_uses_fold_majorities_not_pooled_rows():
    direct = adjudicate_m0(
        [_fold(1, 0.8), _fold(2, 0.7), _fold(3, 0.6), _fold(4, 0.9), _fold(5, 0.4)]
    )
    assert direct["direct_teacher_correction_majority_fold_count"] == 4
    assert direct["conclusion"] == (
        "DIRECTION_UPDATE_MAJORITY_TEACHER_ATTRIBUTABLE_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
    )

    non_teacher = adjudicate_m0(
        [_fold(1, 0.2), _fold(2, 0.3), _fold(3, 0.4), _fold(4, 0.1), _fold(5, 0.8)]
    )
    assert non_teacher["non_teacher_attributable_majority_fold_count"] == 4
    assert non_teacher["conclusion"] == (
        "DIRECTION_UPDATE_MAJORITY_NOT_TEACHER_ATTRIBUTABLE_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
    )


def test_zero_change_fold_counts_toward_neither_majority():
    x = adjudicate_m0(
        [_fold(1, 0.8), _fold(2, 0.8), _fold(3, 0.8), _fold(4, 0.2), _fold(5, 0.5, changes=0)]
    )
    assert x["zero_direction_change_fold_count"] == 1
    assert x["direct_teacher_correction_majority_fold_count"] == 3
    assert x["non_teacher_attributable_majority_fold_count"] == 1
    assert x["conclusion"] == "MIXED_POLICY_DRIFT_ATTRIBUTION_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
