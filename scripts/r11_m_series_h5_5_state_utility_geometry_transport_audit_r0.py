#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from cb16_local_opt.state_utility_geometry_transport_audit_h55 import H55_METRICS, adjudicate_h55, run_fold_h55
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_RESULT_V1"
PREREG_COMMIT = "e4a3b661869cd51b3a234fc9e027d15cf2e7289e"
GATE_BLOB = "c31821757f0b2ae058846e4243eaac116a7ce5fa"
H54_ADJ_BLOB = "5fd2450fbda34cfb7f2815d2e33107623f36aa8a"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "gate": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_GATE_V1.json",
        GATE_BLOB,
    ),
    "h54_adjudication": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_4_MEDIUM48_WEIGHT_SHADOW_TRANSPORT_REPLICATION_R0_ADJUDICATION_V1.json",
        H54_ADJ_BLOB,
    ),
    "h55_helper": (
        "cb16_local_opt/state_utility_geometry_transport_audit_h55.py",
        "1dd00ea9b40b8da4061c0983fc44fb2307c72587",
    ),
    "h5_helper": (
        "cb16_local_opt/teacher_temporal_transport_audit_h5.py",
        "09b05de23c659fe6f47a43117ec70b5cafcf7d21",
    ),
    "teacher_vectorized_data_index": (
        "cb16_local_opt/teacher_vectorized_r11.py",
        "083ca6541b45577cfeb6d9a376b25e0eaf8d060a",
    ),
    "evidence_cache_loader": (
        "cb16_local_opt/r102_evidence_cache.py",
        "91fb73565ec60f66bf4232957f9b9b2f88cc3cfe",
    ),
    "r6_helper": (
        "cb16_local_opt/reduced_teacher_target_information_audit_r6.py",
        "bd7a8780e8dcab05cfae92744fde523ce62cc9ae",
    ),
    "r6_executor": (
        "scripts/r11_science_g0_reduced_teacher_target_information_audit_r6.py",
        "72bede8dd09015c702ec26ead639a81f48efe38a",
    ),
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
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0,
        "H55_NOT_DESCENDED_FROM_PREREGISTRATION",
    )
    observed = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H55_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "immutable_blobs": observed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require(len(folds) == 5, "H55_OUTER_FOLD_COUNT_DRIFT")
    results = [
        run_fold_h55(fold_spec=spec, all_train_parents=parents, all_train_samples=samples)
        for spec in folds
    ]
    summary = adjudicate_h55(results)

    result = {
        "schema": SCHEMA,
        "status": "H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_COMPLETE",
        "repo": repo,
        "runtime_identity": {
            "python": platform.python_version(),
            "student_training": False,
            "student_inference": False,
            "teacher_compilation": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "teacher_hyperparameter_tuned": False,
            "columnar_index_used_as_data_container_only": True,
        },
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "support": support_receipt,
        "protocol": {
            "support_classification": "CONSUMED_TRAIN_ONLY_MECHANISTIC_SUPPORT__NOT_INDEPENDENT_MARKET_SUPPORT",
            "adaptive_method_selection_acknowledged": True,
            "independent_qualification": False,
            "outer_folds": 5,
            "clock_blocks": 6,
            "train_to_eval_gap_hours_each_fold": 256,
            "all_ten_symbols": True,
            "state_metrics": list(H55_METRICS),
            "state_normalization": "OUTER_TRAIN_MEAN_STD_ONLY",
            "support_parent_rule": "UNIQUE_SAME_SCENARIO_PARENT_PER_TRAIN_FUTURE_GROUP",
            "primary_profile": "CENTERED_9_ACTION_REALIZED_UTILITY_PROFILE",
            "primary_statistic": "SPEARMAN_STATE_DISTANCE_VS_CENTERED_PROFILE_MSE",
            "shifts": [1, 7, 13, 23, 31],
            "shuffle_transform": "EXACT_H5_WHOLE_FUTURE_GROUP_TARGET_FEATURE_ROTATION_SCENARIO_PRESERVED",
            "utility_profiles_rotated": False,
            "teacher_compilation": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "teacher_config_tuning": False,
            "student_training": False,
            "student_inference": False,
            "raw_market_payload_read": False,
            "legacy_validation_rows_used": False,
            "r5_purge_support_used": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_opened": False,
            "r7_candidate_evaluated": False,
        },
        "folds": results,
        "h5_5_summary": summary,
        "semantic_guards": {
            "student_trained": False,
            "student_inference_used": False,
            "teacher_compiled_for_scientific_score": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "teacher_hyperparameter_tuned": False,
            "future_utility_used_to_select_support_parent": False,
            "utility_outcomes_rotated_with_features": False,
            "new_direction_loss_created": False,
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "production_cutover": False,
            "market_information_verdict_reopened": False,
            "canonical_change_authorized": False,
            "fresh_market_data_downloaded": False,
            "raw_market_payload_used": False,
            "legacy_validation_rows_used": False,
            "r5_purge_support_reopened": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False,
        },
        "next_legal_step": (
            "ADJUDICATE_H5_5_THEN_RETURN_TO_H2_H3_MECHANISTIC_AUDITS_WITH_TEACHER_TRANSPORT_CAVEAT__NO_CANONICAL_CHANGE"
            if summary["classification"] == "FULL_STATE_GEOMETRY_TRANSPORT_SUPPORTED"
            else "ADJUDICATE_H5_5_THEN_REASSESS_H1_TEMPORAL_NONTRANSFER_WEIGHT_AND_H2_H3__DO_NOT_TUNE_ORGAN_OR_TEACHER_GEOMETRY"
        ),
    }
    atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": summary["classification"],
                "conclusion": summary["conclusion"],
                "all_rotation_identity_guards_pass": summary["all_rotation_identity_guards_pass"],
                "metric_gates": summary["metric_gates"],
                "next_legal_step": result["next_legal_step"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
