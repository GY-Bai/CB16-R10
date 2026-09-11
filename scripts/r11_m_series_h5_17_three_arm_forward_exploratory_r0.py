#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from cb16_local_opt.h517_three_arm_forward_exploratory import (
    BOTTLENECK_DIM,
    PURGE_HOURS,
    RANDOM_PROJECTION_SEED,
    RIDGE_ALPHA,
    _arm_losses,
    build_fold_pair_h517,
    known_answer_diagnostic_check,
    strip_internal,
    summarize_three_arm,
    support_diagnostic,
    synthetic_pathway_check,
)
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_17_THREE_ARM_FORWARD_EXPLORATORY_R0_RESULT_V1"
STATUS = "H5_17_THREE_ARM_FORWARD_EXPLORATORY_EXECUTION_COMPLETE"
BASE_COMMIT = "bb072a1eb8d6dbf1b6228a69539c5f4ca143cc5b"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", "3c401a0a350984381912f7860181e3e96eb8d7cf"),
    "h55_gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_GATE_V1.json", "54b9492e74d5017711b671837ef402abd604c7c6"),
    "h516_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_16_FULL_STATE_NONLINEAR_UTILITY_INVARIANCE_R0_ADJUDICATION_V1.json", None),
    "h516_helper": ("cb16_local_opt/full_state_nonlinear_utility_invariance_h516.py", "bfa28be3ad6676f26676404bb170a73b474b6192"),
    "h5_helper": ("cb16_local_opt/teacher_temporal_transport_audit_h5.py", "09b05de23c659fe6f47a43117ec70b5cafcf7d21"),
    "columnar_index": ("cb16_local_opt/teacher_vectorized_r11.py", "083ca6541b45577cfeb6d9a376b25e0eaf8d060a"),
    "r6_helper": ("cb16_local_opt/reduced_teacher_target_information_audit_r6.py", "bd7a8780e8dcab05cfae92744fde523ce62cc9ae"),
    "r6_executor": ("scripts/r11_science_g0_reduced_teacher_target_information_audit_r6.py", "72bede8dd09015c702ec26ead639a81f48efe38a"),
}


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def verify_repo() -> dict[str, Any]:
    require(subprocess.call(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"]) == 0, "H517_BASE_NOT_ANCESTOR")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        if expected is not None:
            require(got == expected, f"H517_PIN_DRIFT:{name}:{got}")
        observed[name] = got
    changed = git("diff", "--name-only", f"{BASE_COMMIT}..HEAD").splitlines()
    require(not any(p.startswith("authority/") for p in changed), "H517_AUTHORITY_MUTATION_FORBIDDEN")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "base_commit": BASE_COMMIT,
        "immutable_blobs": observed,
        "changed_paths_since_base": changed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require(tuple(int(f["fold"]) for f in folds) == (1, 2, 3, 4, 5), "H517_OUTER_FOLD_SET_DRIFT")

    known = known_answer_diagnostic_check()
    synth = synthetic_pathway_check()
    require(synth["positive_pathway_detected"] is True, "H517_FORWARD_POSITIVE_CONTROL_FAILED")
    require(synth["null_rejected"] is True, "H517_FORWARD_NULL_CONTROL_FAILED")

    fold_results: list[dict[str, Any]] = []
    support_results: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    internal: list[dict[str, Any]] = []
    for spec in folds:
        train, evald = build_fold_pair_h517(
            fold_spec=spec,
            all_train_parents=parents,
            all_train_samples=samples,
        )
        require(float(train["train_to_eval_gap_hours"]) >= PURGE_HOURS, "H517_PURGE_GAP_DRIFT")
        res = _arm_losses(train, evald)
        internal.append(res)
        fold_results.append(strip_internal(res))
        support_results.append(support_diagnostic(train, evald))
        receipts.append({
            "fold": int(train["fold"]),
            "train_groups": int(train["future_group_count"]),
            "eval_groups": int(evald["future_group_count"]),
            "train_clocks": int(train["unique_decision_clock_count"]),
            "eval_clocks": int(evald["unique_decision_clock_count"]),
            "train_to_eval_gap_hours": float(train["train_to_eval_gap_hours"]),
        })

    summary = summarize_three_arm(internal)
    if summary["task_supervised_representation_improves_pipeline"]:
        classification = "EXPLORATORY_C_IMPROVES_A_AND_B_AND_PASSES_BINDING_CONTROLS"
    elif summary["blind_compression_improves_pipeline"]:
        classification = "EXPLORATORY_B_IMPROVES_A_WITHOUT_QUALIFIED_C_SUPERVISION_INCREMENT"
    else:
        classification = "EXPLORATORY_PREDEFINED_THREE_ARM_NO_QUALIFIED_IMPROVEMENT"

    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "classification": classification,
        "framing": "EXPLORATORY_ONLY__ALREADY_CONSUMED_TRAIN__NOT_AUTHORITY__NO_FINAL__NO_FRESH_DATA",
        "repo": repo,
        "support_receipt": support_receipt,
        "fold_receipts": receipts,
        "known_answer_diagnostics": known,
        "synthetic_forward_pathway_controls": synth,
        "market_fold_results": fold_results,
        "market_support_diagnostics": support_results,
        "summary": summary,
        "configuration": {
            "arm_A": "RAW_FULL102",
            "arm_B": f"TRAIN_SIDE_STANDARDISED_MARKET96_TO_FIXED_RANDOM_{BOTTLENECK_DIM}D_PLUS_RAW_ACCOUNT6",
            "arm_C": f"TRAIN_SIDE_STANDARDISED_MARKET96_TO_DETERMINISTIC_SUPERVISED_LINEAR_{BOTTLENECK_DIM}D_PLUS_RAW_ACCOUNT6",
            "supervised_encoder": "WEIGHTED_JOINT_RIDGE_MARKET_PLUS_ACCOUNT__SVD_TOP8_MARKET_COEFFICIENT_SUBSPACE__SECOND_LINEAR_AUXILIARY_HEAD__ENCODER_FROZEN_BEFORE_RF",
            "ridge_alpha": RIDGE_ALPHA,
            "random_projection_seed": RANDOM_PROJECTION_SEED,
            "rf": "EXACT_H5_16_RANDOM_FOREST_REGRESSOR_CONFIGURATION",
            "target": "EXACT_H5_16_CENTERED_9_ACTION_REALIZED_UTILITY_PROFILE",
            "primary_loss": "CLOCK_EQUAL_CENTERED_PROFILE_MSE",
            "evaluation_blocks": "FIVE_FROZEN_R6_OUTER_FOLD_TRAIN_TO_EVAL_PAIRS",
            "minimum_train_to_eval_purge_hours": PURGE_HOURS,
            "frozen_utility_horizon_hours": 72,
            "negative_controls": "FIVE_H5_16_WHOLE_TRAIN_FUTURE_GROUP_STATE_SHIFTS__C_ENCODER_AND_RF_RETRAINED_FROM_SCRATCH",
            "support_diagnostic_is_gate": False,
        },
        "interpretation_limits": [
            "A_FAILURE_ONLY_CONSTRAINS_THIS_PREDEFINED_LINEAR_8D_SUPERVISED_REPRESENTATION_PLUS_FIXED_RF_PIPELINE",
            "A_FAILURE_DOES_NOT_PROVE_COMPACT_REPRESENTATIONS_ARE_USELESS",
            "A_FAILURE_DOES_NOT_PROVE_CAPACITY_IS_NOT_A_BOTTLENECK",
            "A_FAILURE_DOES_NOT_PROVE_SINGLE_TIME_STATE_IS_INSUFFICIENT",
            "A_SUCCESS_IS_EXPLORATORY_ON_ALREADY_CONSUMED_TRAIN_AND_DOES_NOT_REOPEN_THE_FROZEN_MARKET_INFORMATION_VERDICT",
            "RF_MAX_FEATURES_10_HAS_DIFFERENT_EFFECTIVE_FRACTION_IN_102D_VS_14D__INTERPRET_AS_PIPELINE_COMPARISON_NOT_PURE_REPRESENTATION_CAUSAL_EFFECT",
            "SUPPORT_COVERAGE_IS_MECHANISTIC_DIAGNOSTIC_ONLY_AND_CANNOT_VETO_PREDICTIVE_IMPROVEMENT",
        ],
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "semantic_guards": {
            "already_consumed_train_only": True,
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "canonical_teacher_changed": False,
            "canonical_student_changed": False,
            "physics_or_supervisor_changed": False,
            "r7_candidate_evaluated": False,
            "champion_promoted": False,
            "authority_written": False,
        },
        "runtime": {"python_version": platform.python_version()},
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "classification": classification,
        "summary": summary,
        "fold_receipts": receipts,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
