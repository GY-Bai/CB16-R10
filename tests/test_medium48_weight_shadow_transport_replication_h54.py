from __future__ import annotations

import inspect

from cb16_local_opt.medium48_weight_shadow_transport_replication_h54 import (
    H54_SHADOW_ARM,
    adjudicate_h54,
    compile_shadow_shuffle_h54,
)


def test_shadow_rule_is_exact_h53_winner():
    assert H54_SHADOW_ARM == "DROP_MEDIUM48_WEIGHT_ONLY"


def test_shuffle_compiler_signature_does_not_accept_future_utility():
    params = set(inspect.signature(compile_shadow_shuffle_h54).parameters)
    forbidden = {"utility", "utilities", "outcome", "future_utility", "target_utility"}
    assert not (params & forbidden)


def _receipt(ok: bool = True):
    return {
        "rotation": {
            "target_feature_multiset_preserved": ok,
            "train_features_byte_identical": ok,
            "scenario_identity_preserved": ok,
        },
        "fixed_support_weighting": {
            "all_shadow_arms_selected_parent_rows_identical": ok,
            "full102_reference_law_identity_guard": "PASS" if ok else "FAIL",
        },
    }


def _fold(
    fold: int,
    *,
    shadow: float,
    frozen: float,
    climate: float,
    shuffles: tuple[float, float, float, float, float],
    identity: bool = True,
):
    median = sorted(shuffles)[2]
    return {
        "fold": fold,
        "shadow_aligned": {"qscore": shadow},
        "frozen_canonical_aligned": {"qscore": frozen},
        "climatology": {"qscore": climate},
        "shadow_shuffles": {s: {"qscore": q} for s, q in zip((1, 7, 13, 23, 31), shuffles)},
        "median_shadow_shuffle_qscore": median,
        "shadow_minus_frozen_qscore": shadow - frozen,
        "shadow_minus_climatology_qscore": shadow - climate,
        "shadow_minus_median_shuffle_qscore": shadow - median,
        "shadow_lt_frozen": shadow < frozen,
        "shadow_lt_climatology": shadow < climate,
        "shadow_lt_median_shuffle": shadow < median,
        "shadow_lt_each_shuffle_count": sum(shadow < q for q in shuffles),
        "h53_shadow_aligned_qscore_identity_guard": "PASS" if identity else "FAIL",
        "h5_frozen_canonical_qscore_identity_guard": "PASS" if identity else "FAIL",
        "aligned_fixed_support_weighting_guard": "PASS" if identity else "FAIL",
        "shuffle_receipts": {s: _receipt(identity) for s in (1, 7, 13, 23, 31)},
    }


def test_adjudication_pass_requires_full_original_h5_style_gate_plus_frozen_improvement():
    rows = [
        _fold(i, shadow=0.8, frozen=1.0, climate=1.1, shuffles=(1.2, 1.3, 1.4, 1.5, 1.6))
        for i in range(1, 6)
    ]
    out = adjudicate_h54(rows)
    assert out["overall_pass"] is True
    assert out["shadow_lt_frozen_fold_count"] == 5
    assert out["shadow_lt_frozen_both_late_folds"] is True
    assert out["shadow_lt_climatology_fold_count"] == 5
    assert out["shadow_lt_median_shuffle_fold_count"] == 5
    assert out["shadow_lt_each_shuffle_pair_count_of_25"] == 25
    assert out["folds_4_and_5_both_pass_climatology_and_shuffle"] is True
    assert out["all_identity_guards_pass"] is True


def test_adjudication_fails_when_shadow_gain_does_not_hold_in_four_folds():
    rows = [
        _fold(1, shadow=1.1, frozen=1.0, climate=1.2, shuffles=(1.3, 1.3, 1.3, 1.3, 1.3)),
        _fold(2, shadow=1.1, frozen=1.0, climate=1.2, shuffles=(1.3, 1.3, 1.3, 1.3, 1.3)),
        _fold(3, shadow=0.9, frozen=1.0, climate=1.2, shuffles=(1.3, 1.3, 1.3, 1.3, 1.3)),
        _fold(4, shadow=0.9, frozen=1.0, climate=1.2, shuffles=(1.3, 1.3, 1.3, 1.3, 1.3)),
        _fold(5, shadow=0.9, frozen=1.0, climate=1.2, shuffles=(1.3, 1.3, 1.3, 1.3, 1.3)),
    ]
    out = adjudicate_h54(rows)
    assert out["shadow_lt_frozen_fold_count"] == 3
    assert out["overall_pass"] is False


def test_adjudication_fails_if_late_transport_not_restored_even_when_shadow_improves_frozen():
    rows = [
        _fold(i, shadow=0.8, frozen=1.0, climate=1.1, shuffles=(1.2, 1.3, 1.4, 1.5, 1.6))
        for i in range(1, 5)
    ]
    rows.append(_fold(5, shadow=0.8, frozen=1.0, climate=1.1, shuffles=(0.7, 0.7, 0.7, 0.7, 0.7)))
    out = adjudicate_h54(rows)
    assert out["shadow_lt_frozen_fold_count"] == 5
    assert out["folds_4_and_5_both_pass_climatology_and_shuffle"] is False
    assert out["overall_pass"] is False


def test_adjudication_fail_closed_on_identity_guard():
    rows = [
        _fold(i, shadow=0.8, frozen=1.0, climate=1.1, shuffles=(1.2, 1.3, 1.4, 1.5, 1.6))
        for i in range(1, 6)
    ]
    rows[2]["h53_shadow_aligned_qscore_identity_guard"] = "FAIL"
    out = adjudicate_h54(rows)
    assert out["all_identity_guards_pass"] is False
    assert out["overall_pass"] is False
