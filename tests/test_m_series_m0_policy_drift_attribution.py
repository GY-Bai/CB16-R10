from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cb16_local_opt.m_series_m0_policy_drift_attribution import (
    adjudicate_m0,
    attribute_direction_migration_m0,
    teacher_direction_geometry_from_action_laws_m0,
)


def _softmax_means(means, tau=0.002):
    x = np.asarray(means, dtype=np.float64) / tau
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(np.clip(x, -60.0, 60.0))
    return e / e.sum(axis=1, keepdims=True)


def _probs_for_classes(classes):
    out = []
    for c in classes:
        p = np.full(3, 0.05, dtype=np.float64)
        p[int(c)] = 0.90
        out.append(p)
    return np.asarray(out)


def _evidence_from_best_means(means, risks=None):
    means = np.asarray(means, dtype=np.float64)
    if risks is None:
        risks = np.asarray([0.25, 0.0, 0.75], dtype=np.float64)
    out = []
    for row in means:
        laws = []
        for cls, direction in enumerate((-1, 0, 1)):
            laws.append(
                SimpleNamespace(
                    direction=direction,
                    requested_risk=float(risks[cls]),
                    mean_utility=float(row[cls]),
                )
            )
            # Add an inferior same-Direction candidate so the M0 helper must reproduce
            # the Teacher compiler's within-Direction maximization rather than assuming
            # one law per Direction.
            if direction != 0:
                laws.append(
                    SimpleNamespace(
                        direction=direction,
                        requested_risk=float(min(1.0, risks[cls] + 0.2)),
                        mean_utility=float(row[cls] - 0.01),
                    )
                )
        out.append(SimpleNamespace(action_laws=tuple(laws)))
    return out


def test_direction_advantage_uses_exact_rich_teacher_mean_gap():
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
    geometry = teacher_direction_geometry_from_action_laws_m0(
        evidence=_evidence_from_best_means(means),
        g0_direction=g0_dir,
        reduced_teacher_probs=p,
    )
    expected = np.asarray(
        [0.014 - 0.010, 0.027 - 0.020, -0.003 - (-0.004)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        geometry["direction_advantage_over_g0"], expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(geometry["best_mean_by_direction"], means, rtol=0.0, atol=0.0)


def test_m0_row_classification_and_majority_attribution():
    teacher_dir = [0, 0, 1, 2, 2]
    g0_dir = [0, 0, 0, 0, 1]
    challenger_dir = [0, 1, 1, 0, 0]
    teacher_probs = _probs_for_classes(teacher_dir)
    means = np.zeros((5, 3), dtype=np.float64)
    for i, cls in enumerate(teacher_dir):
        means[i, cls] = 0.02
    geometry = teacher_direction_geometry_from_action_laws_m0(
        evidence=_evidence_from_best_means(means),
        g0_direction=np.asarray(g0_dir),
        reduced_teacher_probs=teacher_probs,
    )
    rows, summary = attribute_direction_migration_m0(
        parent_ids=[f"p{i}" for i in range(5)],
        dependence_group_ids=[f"g{i}" for i in range(5)],
        teacher_probs=teacher_probs,
        teacher_best_mean_by_direction=geometry["best_mean_by_direction"],
        teacher_best_risk_by_direction=geometry["best_risk_by_direction"],
        teacher_direction_advantage_over_g0=geometry["direction_advantage_over_g0"],
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
    assert summary["softmax_log_ratio_advantage_used"] is False


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
