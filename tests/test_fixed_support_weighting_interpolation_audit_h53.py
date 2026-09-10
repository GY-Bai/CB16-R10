from __future__ import annotations

import inspect

import numpy as np

from cb16_local_opt.fixed_support_weighting_interpolation_audit_h53 import (
    H53_ARMS,
    _shadow_distance_h53,
    _weights_from_distance_h53,
    adjudicate_h53,
)


def test_weight_arms_are_frozen_and_complete():
    assert H53_ARMS == (
        "FULL102_CANONICAL",
        "DROP_MEDIUM48_WEIGHT_ONLY",
        "DROP_OPERATOR48_WEIGHT_ONLY",
        "UNIFORM_WEIGHT_FIXED_SUPPORT",
    )


def test_weight_function_is_positive_normalized_and_distance_sensitive():
    d = np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=np.float64)
    w = _weights_from_distance_h53(d)
    assert w.shape == d.shape
    assert np.all(w > 0.0)
    assert np.allclose(w.sum(axis=1), 1.0, rtol=0.0, atol=1e-14)
    assert w[0, 0] > w[0, 1] > w[0, 2]
    assert w[1, 2] > w[1, 1] > w[1, 0]


def test_shadow_distance_signature_has_no_utility_argument():
    params = set(inspect.signature(_shadow_distance_h53).parameters)
    forbidden = {"utility", "utilities", "outcome", "future_utility", "target_utility"}
    assert not (params & forbidden)


def _fold(fold: int, full: float, dm: float, do: float, uni: float):
    return {
        "fold": fold,
        "full102_frozen_teacher_qscore_identity_guard": "PASS",
        "full102_support_hash_identity_guard": "PASS",
        "selected_parent_identity_across_all_arms_guard": "PASS",
        "arms": {
            "FULL102_CANONICAL": {"qscore": full},
            "DROP_MEDIUM48_WEIGHT_ONLY": {"qscore": dm},
            "DROP_OPERATOR48_WEIGHT_ONLY": {"qscore": do},
            "UNIFORM_WEIGHT_FIXED_SUPPORT": {"qscore": uni},
        },
        "weight_total_variation_vs_full102": {
            "DROP_MEDIUM48_WEIGHT_ONLY": 0.2,
            "DROP_OPERATOR48_WEIGHT_ONLY": 0.2,
            "UNIFORM_WEIGHT_FIXED_SUPPORT": 0.3,
        },
    }


def test_adjudication_medium48_primary_requires_specific_late_replication():
    rows = [
        _fold(1, 1.0, 1.1, 1.1, 1.1),
        _fold(2, 1.0, 1.1, 1.1, 1.1),
        _fold(3, 1.0, 1.1, 1.1, 1.1),
        _fold(4, 1.0, 0.9, 1.1, 1.1),
        _fold(5, 1.0, 0.8, 1.2, 1.1),
    ]
    out = adjudicate_h53(rows)
    assert out["classification"] == "MEDIUM48_WEIGHTING_PRIMARY"
    assert out["drop_medium48_weight_beats_full102_both_late_folds"] is True
    assert out["drop_operator48_weight_beats_full102_both_late_folds"] is False
    assert out["uniform_weight_beats_full102_both_late_folds"] is False
    assert out["all_identity_guards_pass"] is True


def test_adjudication_broad_kernel_takes_precedence_when_uniform_wins_late():
    rows = [
        _fold(1, 1.0, 1.1, 1.1, 1.1),
        _fold(2, 1.0, 1.1, 1.1, 1.1),
        _fold(3, 1.0, 1.1, 1.1, 1.1),
        _fold(4, 1.0, 0.9, 1.1, 0.8),
        _fold(5, 1.0, 0.8, 1.2, 0.7),
    ]
    out = adjudicate_h53(rows)
    assert out["classification"] == "BROAD_KERNEL_WEIGHTING_FAILURE"
    assert out["uniform_weight_beats_full102_both_late_folds"] is True


def test_adjudication_operator_primary_is_separate_from_medium():
    rows = [
        _fold(1, 1.0, 1.1, 1.1, 1.1),
        _fold(2, 1.0, 1.1, 1.1, 1.1),
        _fold(3, 1.0, 1.1, 1.1, 1.1),
        _fold(4, 1.0, 1.1, 0.8, 1.1),
        _fold(5, 1.0, 1.2, 0.9, 1.1),
    ]
    out = adjudicate_h53(rows)
    assert out["classification"] == "OPERATOR48_WEIGHTING_PRIMARY"


def test_adjudication_unresolved_when_no_mechanism_replicates():
    rows = [
        _fold(1, 1.0, 1.1, 1.1, 1.1),
        _fold(2, 1.0, 1.1, 1.1, 1.1),
        _fold(3, 1.0, 1.1, 1.1, 1.1),
        _fold(4, 1.0, 0.9, 1.1, 1.1),
        _fold(5, 1.0, 1.1, 0.9, 1.1),
    ]
    out = adjudicate_h53(rows)
    assert out["classification"] == "H5_3_WEIGHTING_INTERPOLATION_UNRESOLVED"
