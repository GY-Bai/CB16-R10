#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from cb16_local_opt.medium48_temporal_halfblock_stability_h510 import H510_FOLDS, adjudicate_h510, run_fold_h510
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_10_MEDIUM48_TEMPORAL_HALFBLOCK_STABILITY_R0_RESULT_V1"
PREREG_COMMIT = "1de6d3d10b6698f4fe2b8800c02f2d18f05ffaab"
GATE_BLOB = "PLACEHOLDER_GATE"
H59_ADJ_COMMIT = "2585cfde44a71c5400d1881382983deaacf495d1"
H59_ADJ_BLOB = "99ca12bdfa3edf458a82820f76dbfeb3ab9a2444"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": (
        "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json",
        "3c401a0a350984381912f7860181e3e96eb8d7cf",
    ),
    "gate": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_10_MEDIUM48_TEMPORAL_HALFBLOCK_STABILITY_R0_GATE_V1.json",
        GATE_BLOB,
    ),
    "h59_adjudication": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_9_MEDIUM48_ANTIALIGNMENT_BREADTH_R0_ADJUDICATION_V1.json",
        H59_ADJ_BLOB,
    ),
    "h510_helper": (
        "cb16_local_opt/medium48_temporal_halfblock_stability_h510.py",
        "176983eb0f7beb6396823126f4bc1504c4418054",
    ),
    "h510_test": (
        "tests/test_medium48_temporal_halfblock_stability_h510.py",
        "672668cb4b207eae8ccb9ef7d2cfdf3bf27e4118",
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0, "H510_NOT_DESCENDED_FROM_PREREGISTRATION")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", H59_ADJ_COMMIT, PREREG_COMMIT]) == 0, "H510_PREREG_NOT_DESCENDED_FROM_H59_ADJUDICATION")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H510_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H510_PREREG_GATE_BLOB_DRIFT")
    return {"execution_head": git("rev-parse", "HEAD"), "preregistration_commit": PREREG_COMMIT, "h59_adjudication_commit": H59_ADJ_COMMIT, "immutable_blobs": observed}


def load_h59_authority() -> dict[str, Any]:
    x = json.loads(Path(PINNED["h59_adjudication"][0]).read_text(encoding="utf-8"))
    require(x["adjudication"]["classification"] == "MEDIUM48_CONDITIONAL_ANTI_ALIGNMENT_BROAD_ACROSS_SYMBOLS_AND_SCENARIOS", "H510_H59_CLASS_DRIFT")
    require(x["authority"]["time_gating_authorized"] is False if "time_gating_authorized" in x["authority"] else True, "H510_H59_TIME_GATING_AUTHORITY_DRIFT")
    require(x["authority"]["handcrafted_regime_activation_authorized"] is False, "H510_H59_REGIME_AUTHORITY_DRIFT")
    require(x["authority"]["canonical_change_authorized"] is False, "H510_H59_CANONICAL_AUTHORITY_DRIFT")
    require(x["authority"]["final_opening_authorized"] is False, "H510_H59_FINAL_AUTHORITY_DRIFT")
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    h59 = load_h59_authority()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    all_folds = build_outer_folds_r6(parents)
    fold_by_id = {int(x["fold"]): x for x in all_folds}
    require(tuple(sorted(fold_by_id)) == H510_FOLDS, "H510_OUTER_FOLD_SET_DRIFT")
    results = [run_fold_h510(fold_spec=fold_by_id[f], all_train_parents=parents, all_train_samples=samples) for f in H510_FOLDS]
    summary = adjudicate_h510(results)

    result = {
        "schema": SCHEMA,
        "status": "H5_10_MEDIUM48_TEMPORAL_HALFBLOCK_STABILITY_COMPLETE",
        "repo": repo,
        "runtime_identity": {"python": platform.python_version(), "student_training": False, "student_inference": False, "teacher_law_compilation": False, "teacher_kernel_used": False, "teacher_support_selection_used": False, "columnar_index_builder_used_for_state_and_utility_arrays_only": True},
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "parent_h5_9": {"classification": h59["adjudication"]["classification"], "conclusion": h59["adjudication"]["conclusion"]},
        "support": support_receipt,
        "protocol": {
            "support_classification": "CONSUMED_TRAIN_ONLY_MECHANISTIC_SUPPORT__NOT_INDEPENDENT_MARKET_SUPPORT",
            "scientific_folds": [1,2,3,4,5],
            "outer_clock_partition": "EXACT_R6_SIX_EQUAL_ORDERED_CLOCK_BLOCKS",
            "half_partition": "EXISTING_CHRONOLOGICAL_EVAL_FUTURE_GROUP_ORDER_SPLIT_BY_NUMPY_ARRAY_SPLIT_INTO_TWO_CONTIGUOUS_NEAR_EQUAL_COUNT_HALVES",
            "primary_measure": "EXACT_H5_8_OPERATOR_CONDITIONAL_PARTIAL_SPEARMAN",
            "partition_boundary_uses_utility_or_features": False,
            "support_normalization_distance_or_null_recomputed_after_partition": False,
            "time_gating_or_regime_classifier_created": False,
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
            "r7_candidate_evaluated": False
        },
        "folds": results,
        "h5_10_summary": summary,
        "semantic_guards": {
            "student_trained": False,
            "student_inference_used": False,
            "teacher_compiled_for_scientific_score": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "future_utility_used_to_select_temporal_boundaries": False,
            "features_used_to_select_temporal_boundaries": False,
            "change_point_search_used": False,
            "support_normalization_distance_or_null_recomputed_after_partition": False,
            "medium_dimensions_split_post_hoc": False,
            "timesfm_1280d_pre_adapter_latent_accessed": False,
            "fusion_weights_searched": False,
            "organ_weights_tuned": False,
            "distance_metric_tuned": False,
            "handcrafted_regime_or_cycle_activation_created": False,
            "fold_or_half_id_used_as_decision_feature_or_permission_rule": False,
            "chronological_belief_expiry_created": False,
            "canonical_teacher_changed": False,
            "canonical_student_changed": False,
            "new_direction_loss_created": False,
            "market_information_verdict_reopened": False,
            "canonical_change_authorized": False,
            "organ_change_authorized": False,
            "time_gating_authorized": False,
            "champion_promoted": False,
            "production_cutover": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False
        },
        "next_legal_step": "ADJUDICATE_H5_10_FROM_THIS_FROZEN_RESULT__DO_NOT_CREATE_TIME_GATING_OR_HANDCRAFTED_REGIME_ACTIVATION__ANY_NEXT_EXPERIMENT_REQUIRES_SEPARATE_PREREGISTRATION"
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "classification": summary["classification"], "conclusion": summary["conclusion"], "baseline_reproduction_passed": summary["baseline_reproduction_passed"], "failure_fold_persistence_4_of_4": summary["failure_fold_persistence_4_of_4"], "positive_fold_persistence_6_of_6": summary["positive_fold_persistence_6_of_6"], "chronological_half_sequence": summary["chronological_half_sequence"], "next_legal_step": result["next_legal_step"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
