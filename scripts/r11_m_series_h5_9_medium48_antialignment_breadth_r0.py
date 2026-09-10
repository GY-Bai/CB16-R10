#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from cb16_local_opt.medium48_antialignment_breadth_h59 import H59_FOLDS, adjudicate_h59, run_fold_h59
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_9_MEDIUM48_ANTIALIGNMENT_BREADTH_R0_RESULT_V1"
PREREG_COMMIT = "ea234bd381ffded3149704d3df46bc7fc704a102"
GATE_BLOB = "91902d36806c70108de8eca2472e717acb5f3a20"
H58_ADJ_COMMIT = "67168af364aeadb6b28e78371a4d92daa2962ab4"
H58_ADJ_BLOB = "f2c8a075b9fd67afccc66198a9efc64ec0b1ceed"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": (
        "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json",
        "3c401a0a350984381912f7860181e3e96eb8d7cf",
    ),
    "gate": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_9_MEDIUM48_ANTIALIGNMENT_BREADTH_R0_GATE_V1.json",
        GATE_BLOB,
    ),
    "h58_adjudication": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_8_OPERATOR_CONDITIONAL_MEDIUM48_GEOMETRY_R0_ADJUDICATION_V1.json",
        H58_ADJ_BLOB,
    ),
    "h59_helper": (
        "cb16_local_opt/medium48_antialignment_breadth_h59.py",
        "f85fcd609a7e34b4f42cfa5873ba0a752927b58c",
    ),
    "h59_test": (
        "tests/test_medium48_antialignment_breadth_h59.py",
        "e8f5fbf6834a5c822495c7cae431ee0df9dcff6d",
    ),
    "h58_helper": (
        "cb16_local_opt/operator_conditional_medium_geometry_h58.py",
        "aaddbad059c7262a8137db87bee25d38c07afd83",
    ),
    "h56_helper": (
        "cb16_local_opt/time_local_vs_forward_geometry_contrast_h56.py",
        "a264ff21d0447527c4dda06ee8fd63cf20207daf",
    ),
    "h55_helper": (
        "cb16_local_opt/state_utility_geometry_transport_audit_h55.py",
        "1dd00ea9b40b8da4061c0983fc44fb2307c72587",
    ),
    "h55_clarified_helper": (
        "cb16_local_opt/state_utility_geometry_transport_audit_h55_clarified.py",
        "faa5233170aa421c3bf44f023b1d8e2f7c187568",
    ),
    "h5_helper": (
        "cb16_local_opt/teacher_temporal_transport_audit_h5.py",
        "09b05de23c659fe6f47a43117ec70b5cafcf7d21",
    ),
    "columnar_index": (
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
        "H59_NOT_DESCENDED_FROM_PREREGISTRATION",
    )
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", H58_ADJ_COMMIT, PREREG_COMMIT]) == 0,
        "H59_PREREG_NOT_DESCENDED_FROM_H58_ADJUDICATION",
    )
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H59_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H59_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "h58_adjudication_commit": H58_ADJ_COMMIT,
        "immutable_blobs": observed,
    }


def load_h58_authority() -> dict[str, Any]:
    path = Path(PINNED["h58_adjudication"][0])
    x = json.loads(path.read_text(encoding="utf-8"))
    require(
        x["adjudication"]["classification"] == "H5_7_FAILURE_FOLDS_SHOW_MEDIUM_CONDITIONAL_ANTI_ALIGNMENT",
        "H59_H58_CLASS_DRIFT",
    )
    require(x["adjudication"]["h5_7_failure_fold_conditional_anti_alignment"] is True, "H59_H58_ANTI_DRIFT")
    require(x["preregistered_gates"]["h5_7_failure_fold_gate"]["frozen_folds"] == [1, 3], "H59_H58_FOLD_DRIFT")
    require(x["authority"]["canonical_change_authorized"] is False, "H59_H58_CANONICAL_AUTHORITY_DRIFT")
    require(x["authority"]["final_opening_authorized"] is False, "H59_H58_FINAL_AUTHORITY_DRIFT")
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    h58 = load_h58_authority()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    all_folds = build_outer_folds_r6(parents)
    fold_by_id = {int(x["fold"]): x for x in all_folds}
    require(tuple(sorted(fold_by_id)) == (1, 2, 3, 4, 5), "H59_OUTER_FOLD_SET_DRIFT")
    results = [
        run_fold_h59(
            fold_spec=fold_by_id[fold],
            all_train_parents=parents,
            all_train_samples=samples,
        )
        for fold in H59_FOLDS
    ]
    summary = adjudicate_h59(results)

    result = {
        "schema": SCHEMA,
        "status": "H5_9_MEDIUM48_ANTIALIGNMENT_BREADTH_COMPLETE",
        "repo": repo,
        "runtime_identity": {
            "python": platform.python_version(),
            "student_training": False,
            "student_inference": False,
            "teacher_law_compilation": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "columnar_index_builder_used_for_state_and_utility_arrays_only": True,
        },
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "parent_h5_8": {
            "classification": h58["adjudication"]["classification"],
            "conclusion": h58["adjudication"]["conclusion"],
            "failure_folds": [1, 3],
        },
        "support": support_receipt,
        "protocol": {
            "support_classification": "CONSUMED_TRAIN_ONLY_MECHANISTIC_SUPPORT__NOT_INDEPENDENT_MARKET_SUPPORT",
            "scientific_folds": [1, 3],
            "time_local_support": "EXACT_H5_8_SAME_EVAL_BLOCK_OTHER_FUTURE_GROUPS_SAME_SCENARIO_TARGET_GROUP_EXCLUDED",
            "primary_measure": "EXACT_H5_8_OPERATOR_CONDITIONAL_PARTIAL_SPEARMAN",
            "symbol_leave_one_out": "REMOVE_ONLY_TARGET_FUTURE_GROUPS_OF_ONE_FROZEN_SYMBOL_FROM_FINAL_AGGREGATION",
            "scenario_leave_one_out": "REMOVE_ONLY_ONE_FROZEN_SCENARIO_FROM_FINAL_WITHIN_FUTURE_GROUP_AGGREGATION",
            "support_recomputed_after_omission": False,
            "normalization_recomputed_after_omission": False,
            "null_mapping_recomputed_after_omission": False,
            "new_representation_used": False,
            "medium_dimensions_subselected": False,
            "fusion_weight_search": False,
            "teacher_compilation": False,
            "teacher_kernel": False,
            "teacher_support_selection": False,
            "student_training": False,
            "student_inference": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_opened": False,
            "r7_candidate_evaluated": False,
        },
        "folds": results,
        "h5_9_summary": summary,
        "semantic_guards": {
            "student_trained": False,
            "student_inference_used": False,
            "teacher_compiled_for_scientific_score": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "future_utility_used_to_select_omissions": False,
            "support_recomputed_after_omission": False,
            "normalization_recomputed_after_omission": False,
            "null_mapping_recomputed_after_omission": False,
            "medium_dimensions_split_post_hoc": False,
            "timesfm_1280d_pre_adapter_latent_accessed": False,
            "fusion_weights_searched": False,
            "organ_weights_tuned": False,
            "distance_metric_tuned": False,
            "handcrafted_regime_or_cycle_activation_created": False,
            "canonical_teacher_changed": False,
            "canonical_student_changed": False,
            "new_direction_loss_created": False,
            "market_information_verdict_reopened": False,
            "canonical_change_authorized": False,
            "organ_change_authorized": False,
            "champion_promoted": False,
            "production_cutover": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False,
        },
        "next_legal_step": "ADJUDICATE_H5_9_FROM_THIS_FROZEN_RESULT__DO_NOT_CREATE_HANDCRAFTED_REGIME_SWITCHES_OR_TUNE_FUSION_WEIGHTS__ANY_NEXT_EXPERIMENT_REQUIRES_SEPARATE_PREREGISTRATION",
    }
    atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": summary["classification"],
                "conclusion": summary["conclusion"],
                "baseline_reproduction_passed": summary["baseline_reproduction_passed"],
                "symbol_loo_robust_both_failure_folds": summary["symbol_loo_robust_both_failure_folds"],
                "scenario_loo_robust_both_failure_folds": summary["scenario_loo_robust_both_failure_folds"],
                "per_fold": summary["per_fold"],
                "next_legal_step": result["next_legal_step"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
