from __future__ import annotations

"""H5.4 replication of the single H5.3-localized shadow Teacher rule.

Support selection remains FULL102.  Only post-selection Gaussian weighting drops
Medium48 (Operator48 + Account6 remain active).  Whole-future-group target-feature
rotations are exactly the H5 transform.  Student code is never loaded.
"""

import statistics
from typing import Any, Mapping, Sequence

from .fixed_support_weighting_interpolation_audit_h53 import (
    compile_fixed_support_weight_arms_h53,
    score_weight_arm_h53,
)
from .r11_teacher_authority_candidate import R11_VALIDATION_TEACHER_CONFIG
from .teacher_temporal_transport_audit_h5 import (
    H5_SHIFTS,
    _fold_material_h5,
    _support_and_aligned_h5,
    climatology_laws_h5,
    require,
    score_climatology_h5,
    score_teacher_evidence_h5,
    shuffled_target_feature_index_h5,
)
from .teacher_vectorized_r11 import _compile_block_r11

H54_RUNTIME = "CB16_R11_H5_4_MEDIUM48_WEIGHT_SHADOW_TRANSPORT_REPLICATION_R0_V1"
H54_SHADOW_ARM = "DROP_MEDIUM48_WEIGHT_ONLY"


def _shadow_rows_h54(
    *,
    index,
    regime,
    eval_parent_ids: Sequence[str],
    evidence: Sequence[Any],
    block_targets: int,
) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Any]]:
    arms, receipt = compile_fixed_support_weight_arms_h53(
        index=index,
        regime=regime,
        eval_parent_ids=eval_parent_ids,
        aligned_evidence=evidence,
        block_targets=block_targets,
    )
    require(H54_SHADOW_ARM in arms, "H54_SHADOW_ARM_MISSING")
    require(receipt["all_shadow_arms_selected_parent_rows_identical"] is True, "H54_SUPPORT_IDENTITY_GUARD_FAIL")
    require(receipt["full102_reference_law_identity_guard"] == "PASS", "H54_FULL102_REFERENCE_IDENTITY_FAIL")
    return arms[H54_SHADOW_ARM], receipt


def compile_shadow_shuffle_h54(
    *,
    index,
    regime,
    eval_parents: Mapping[str, Any],
    eval_parent_ids: Sequence[str],
    shift: int,
    block_targets: int = 32,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the exact H5 feature rotation, then FULL102-select support and shadow-weight it."""
    shuffled_index, rotation_receipt = shuffled_target_feature_index_h5(
        index=index,
        eval_parents=eval_parents,
        eval_parent_ids=eval_parent_ids,
        shift=int(shift),
    )
    evidence = []
    ids = tuple(str(x) for x in eval_parent_ids)
    for start in range(0, len(ids), int(block_targets)):
        evidence.extend(
            _compile_block_r11(
                target_parent_ids=ids[start : start + int(block_targets)],
                index=shuffled_index,
                regime=regime,
                config=R11_VALIDATION_TEACHER_CONFIG,
            )
        )
    require(len(evidence) == len(ids), f"H54_SHUFFLE_EVIDENCE_COUNT:{shift}")
    require(all(x.admission.admitted for x in evidence), f"H54_SHUFFLE_UNADMITTED:{shift}")
    shadow_rows, fixed_receipt = _shadow_rows_h54(
        index=shuffled_index,
        regime=regime,
        eval_parent_ids=ids,
        evidence=evidence,
        block_targets=block_targets,
    )
    score = score_weight_arm_h53(arm_rows=shadow_rows, index=shuffled_index, eval_parents=eval_parents)
    return score, {
        "rotation": rotation_receipt,
        "fixed_support_weighting": {
            "target_count": int(fixed_receipt["target_count"]),
            "all_shadow_arms_selected_parent_rows_identical": bool(
                fixed_receipt["all_shadow_arms_selected_parent_rows_identical"]
            ),
            "full102_reference_law_identity_guard": fixed_receipt["full102_reference_law_identity_guard"],
        },
    }


def run_fold_h54(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any],
    expected_h53_shadow_qscore: float,
    expected_h53_full102_qscore: float,
    block_targets: int = 32,
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    train_parents, eval_parents, parents, train_samples, eval_samples, samples = _fold_material_h5(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    eval_parent_ids = tuple(sorted(eval_parents))
    index, regime, aligned, execution = _support_and_aligned_h5(
        parents=parents,
        samples=samples,
        eval_parent_ids=eval_parent_ids,
        block_targets=block_targets,
    )

    frozen = score_teacher_evidence_h5(evidence=aligned, index=index, eval_parents=eval_parents)
    shadow_rows, aligned_receipt = _shadow_rows_h54(
        index=index,
        regime=regime,
        eval_parent_ids=eval_parent_ids,
        evidence=aligned,
        block_targets=block_targets,
    )
    shadow = score_weight_arm_h53(arm_rows=shadow_rows, index=index, eval_parents=eval_parents)

    require(
        abs(float(shadow["qscore"]) - float(expected_h53_shadow_qscore)) <= 1e-15,
        f"H54_H53_SHADOW_QSCORE_IDENTITY:{fold}",
    )
    require(
        abs(float(frozen["qscore"]) - float(expected_h53_full102_qscore)) <= 1e-15,
        f"H54_H5_FROZEN_QSCORE_IDENTITY:{fold}",
    )

    climate_law = climatology_laws_h5(index=index, regime=regime)
    climate = score_climatology_h5(
        climatology=climate_law,
        index=index,
        eval_parent_ids=eval_parent_ids,
        eval_parents=eval_parents,
    )

    shuffles: dict[int, dict[str, Any]] = {}
    shuffle_receipts: dict[int, dict[str, Any]] = {}
    for shift in H5_SHIFTS:
        score, receipt = compile_shadow_shuffle_h54(
            index=index,
            regime=regime,
            eval_parents=eval_parents,
            eval_parent_ids=eval_parent_ids,
            shift=int(shift),
            block_targets=block_targets,
        )
        shuffles[int(shift)] = score
        shuffle_receipts[int(shift)] = receipt

    median_shuffle = float(statistics.median(float(shuffles[s]["qscore"]) for s in H5_SHIFTS))
    return {
        "fold": fold,
        "train_parent_contexts": len(train_parents),
        "eval_parent_contexts": len(eval_parents),
        "train_dependence_groups": int(regime.group_count),
        "eval_dependence_groups": int(shadow["future_dependence_groups"]),
        "train_clock_count": int(fold_spec["train_clock_count"]),
        "eval_clock_count": int(fold_spec["eval_clock_count"]),
        "train_to_eval_gap_hours": float(fold_spec["train_to_eval_gap_hours"]),
        "teacher_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
        "shadow_aligned": shadow,
        "frozen_canonical_aligned": frozen,
        "climatology": climate,
        "shadow_shuffles": shuffles,
        "median_shadow_shuffle_qscore": median_shuffle,
        "shadow_minus_frozen_qscore": float(shadow["qscore"] - frozen["qscore"]),
        "shadow_minus_climatology_qscore": float(shadow["qscore"] - climate["qscore"]),
        "shadow_minus_median_shuffle_qscore": float(shadow["qscore"] - median_shuffle),
        "shadow_lt_frozen": bool(float(shadow["qscore"]) < float(frozen["qscore"])),
        "shadow_lt_climatology": bool(float(shadow["qscore"]) < float(climate["qscore"])),
        "shadow_lt_median_shuffle": bool(float(shadow["qscore"]) < median_shuffle),
        "shadow_lt_each_shuffle_count": int(
            sum(float(shadow["qscore"]) < float(shuffles[s]["qscore"]) for s in H5_SHIFTS)
        ),
        "h53_shadow_aligned_qscore_identity_guard": "PASS",
        "h5_frozen_canonical_qscore_identity_guard": "PASS",
        "aligned_fixed_support_weighting_guard": aligned_receipt["full102_reference_law_identity_guard"],
        "teacher_execution": execution,
        "shuffle_receipts": shuffle_receipts,
    }


def adjudicate_h54(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == (1, 2, 3, 4, 5), "H54_FOLD_SET_DRIFT")
    improve = sum(bool(x["shadow_lt_frozen"]) for x in rows)
    clim = sum(bool(x["shadow_lt_climatology"]) for x in rows)
    shuf = sum(bool(x["shadow_lt_median_shuffle"]) for x in rows)
    pair = sum(int(x["shadow_lt_each_shuffle_count"]) for x in rows)
    improve_late = all(bool(rows[f - 1]["shadow_lt_frozen"]) for f in (4, 5))
    late_transport = all(
        bool(rows[f - 1]["shadow_lt_climatology"]) and bool(rows[f - 1]["shadow_lt_median_shuffle"])
        for f in (4, 5)
    )
    identities = all(
        x["h53_shadow_aligned_qscore_identity_guard"] == "PASS"
        and x["h5_frozen_canonical_qscore_identity_guard"] == "PASS"
        and x["aligned_fixed_support_weighting_guard"] == "PASS"
        and all(
            r["rotation"]["target_feature_multiset_preserved"] is True
            and r["rotation"]["train_features_byte_identical"] is True
            and r["rotation"]["scenario_identity_preserved"] is True
            and r["fixed_support_weighting"]["all_shadow_arms_selected_parent_rows_identical"] is True
            and r["fixed_support_weighting"]["full102_reference_law_identity_guard"] == "PASS"
            for r in x["shuffle_receipts"].values()
        )
        for x in rows
    )
    passed = bool(
        improve >= 4
        and improve_late
        and clim >= 4
        and shuf >= 4
        and late_transport
        and pair >= 20
        and identities
    )
    return {
        "schema": "CB16_R11_M_SERIES_H5_4_MEDIUM48_WEIGHT_SHADOW_TRANSPORT_REPLICATION_R0_ADJUDICATION_V1",
        "overall_pass": passed,
        "shadow_lt_frozen_fold_count": int(improve),
        "shadow_lt_frozen_both_late_folds": bool(improve_late),
        "shadow_lt_climatology_fold_count": int(clim),
        "shadow_lt_median_shuffle_fold_count": int(shuf),
        "shadow_lt_each_shuffle_pair_count_of_25": int(pair),
        "folds_4_and_5_both_pass_climatology_and_shuffle": bool(late_transport),
        "all_identity_guards_pass": bool(identities),
        "conclusion": (
            "H5_4_SINGLE_MEDIUM48_WEIGHTING_INTERVENTION_RESTORES_H5_STYLE_STATE_ALIGNMENT_TRANSPORT_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
            if passed
            else "H5_4_MEDIUM48_WEIGHTING_GAIN_DOES_NOT_RESTORE_H5_STYLE_STATE_ALIGNMENT_TRANSPORT__DO_NOT_CHANGE_CANONICAL_TEACHER"
        ),
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "promotion_authorized": False,
        "r7_evaluation_authorized": False,
        "final_opening_authorized": False,
        "per_fold": [
            {
                "fold": int(x["fold"]),
                "shadow_qscore": float(x["shadow_aligned"]["qscore"]),
                "frozen_qscore": float(x["frozen_canonical_aligned"]["qscore"]),
                "climatology_qscore": float(x["climatology"]["qscore"]),
                "median_shadow_shuffle_qscore": float(x["median_shadow_shuffle_qscore"]),
                "shadow_minus_frozen_qscore": float(x["shadow_minus_frozen_qscore"]),
                "shadow_minus_climatology_qscore": float(x["shadow_minus_climatology_qscore"]),
                "shadow_minus_median_shuffle_qscore": float(x["shadow_minus_median_shuffle_qscore"]),
                "shadow_lt_frozen": bool(x["shadow_lt_frozen"]),
                "shadow_lt_climatology": bool(x["shadow_lt_climatology"]),
                "shadow_lt_median_shuffle": bool(x["shadow_lt_median_shuffle"]),
                "shadow_lt_each_shuffle_count": int(x["shadow_lt_each_shuffle_count"]),
            }
            for x in rows
        ],
    }
