#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from cb16_local_opt.medium48_strict_clock_halfblock_replication_h5101 import (
    H5101_FOLDS,
    adjudicate_h5101,
    run_fold_h5101,
)
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_10_1_STRICT_CLOCK_HALFBLOCK_REPLICATION_R0_RESULT_V1"
PREREG_COMMIT = "d917980113d2893d0d3ac4248d7c409898dc3602"
GATE_BLOB = "0cf5460a704d0713fb4c45fc335151e134335c8a"
H510_ADJ_COMMIT = "8564ea932dadc0d2b1e1a661f0563755a1566e2c"
H510_ADJ_BLOB = "00166dfa2c6692b68d0fcbf65b6fbb332edfece5"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": (
        "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json",
        "3c401a0a350984381912f7860181e3e96eb8d7cf",
    ),
    "gate": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_10_1_STRICT_CLOCK_HALFBLOCK_REPLICATION_R0_GATE_V1.json",
        GATE_BLOB,
    ),
    "h510_adjudication": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_10_MEDIUM48_TEMPORAL_HALFBLOCK_STABILITY_R0_ADJUDICATION_V1.json",
        H510_ADJ_BLOB,
    ),
    "h5101_helper": (
        "cb16_local_opt/medium48_strict_clock_halfblock_replication_h5101.py",
        "f63e68673a8763265d7ff1b8009a3cb04f23fbd3",
    ),
    "h5101_test": (
        "tests/test_medium48_strict_clock_halfblock_replication_h5101.py",
        "4243c9278015a27cceb921cc135a48c9dcf381a4",
    ),
    "h510_helper": (
        "cb16_local_opt/medium48_temporal_halfblock_stability_h510.py",
        "176983eb0f7beb6396823126f4bc1504c4418054",
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", H510_ADJ_COMMIT, PREREG_COMMIT]) == 0, "H5101_PREREG_NOT_DESCENDED_FROM_H510_ADJUDICATION")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0, "H5101_NOT_DESCENDED_FROM_PREREGISTRATION")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H5101_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H5101_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "h510_adjudication_commit": H510_ADJ_COMMIT,
        "immutable_blobs": observed,
    }


def load_h510_authority() -> dict[str, Any]:
    x = json.loads(Path(PINNED["h510_adjudication"][0]).read_text(encoding="utf-8"))
    require(x["status"] == "ADJUDICATED", "H5101_H510_NOT_ADJUDICATED")
    require(x["adjudication"]["classification"] == "FAILURE_FOLDS_STABLE__POSITIVE_FOLDS_INTERNALLY_MIXED", "H5101_H510_CLASS_DRIFT")
    require(x["authority"]["time_gating_authorized"] is False, "H5101_H510_TIME_GATING_AUTHORITY_DRIFT")
    require(x["authority"]["handcrafted_regime_activation_authorized"] is False, "H5101_H510_REGIME_AUTHORITY_DRIFT")
    require(x["authority"]["canonical_change_authorized"] is False, "H5101_H510_CANONICAL_AUTHORITY_DRIFT")
    require(x["authority"]["final_opening_authorized"] is False, "H5101_H510_FINAL_AUTHORITY_DRIFT")
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    h510 = load_h510_authority()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    all_folds = build_outer_folds_r6(parents)
    fold_by_id = {int(x["fold"]): x for x in all_folds}
    require(tuple(sorted(fold_by_id)) == H5101_FOLDS, "H5101_OUTER_FOLD_SET_DRIFT")
    results = [
        run_fold_h5101(
            fold_spec=fold_by_id[f],
            all_train_parents=parents,
            all_train_samples=samples,
        )
        for f in H5101_FOLDS
    ]
    summary = adjudicate_h5101(results)

    result = {
        "schema": SCHEMA,
        "status": "H5_10_1_STRICT_CLOCK_HALFBLOCK_REPLICATION_COMPLETE",
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
        "parent_h5_10": {
            "classification": h510["adjudication"]["classification"],
            "conclusion": h510["adjudication"]["conclusion"],
            "group_order_half_limitation": h510["adjudication"]["limitations"][1],
        },
        "support": support_receipt,
        "protocol": {
            "support_classification": "CONSUMED_TRAIN_ONLY_MECHANISTIC_SUPPORT__NOT_INDEPENDENT_MARKET_SUPPORT",
            "scientific_folds": [1, 2, 3, 4, 5],
            "outer_clock_partition": "EXACT_R6_SIX_EQUAL_ORDERED_CLOCK_BLOCKS",
            "strict_half_partition": "SORT_UNIQUE_EVAL_DECISION_CLOCKS_ASCENDING__NUMPY_ARRAY_SPLIT_CLOCKS_INTO_TWO_CONTIGUOUS_NEAR_EQUAL_COUNT_HALVES__ASSIGN_ALL_FUTURE_GROUPS_SHARING_A_CLOCK_TO_THE_SAME_HALF",
            "single_manipulated_variable": "HALF_PARTITION_UNIT__UNIQUE_DECISION_CLOCK_INSTEAD_OF_ORDERED_FUTURE_GROUP_COUNT",
            "primary_measure": "EXACT_H5_8_OPERATOR_CONDITIONAL_PARTIAL_SPEARMAN",
            "partition_boundary_uses_utility_or_features": False,
            "change_point_search": False,
            "calendar_or_market_regime_labels_used": False,
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
            "r7_candidate_evaluated": False,
        },
        "folds": results,
        "h5_10_1_summary": summary,
        "semantic_guards": {
            "student_trained": False,
            "student_inference_used": False,
            "teacher_compiled_for_scientific_score": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "future_utility_used_to_select_temporal_boundaries": False,
            "features_used_to_select_temporal_boundaries": False,
            "change_point_search_used": False,
            "calendar_or_market_regime_labels_used": False,
            "support_normalization_distance_or_null_recomputed_after_partition": False,
            "same_decision_clock_split_across_halves": False,
            "strict_clock_half_sets_disjoint": True,
            "strict_clock_order_required": True,
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
            "r7_candidate_evaluated": False,
        },
        "next_legal_step": "ADJUDICATE_H5_10_1_FROM_THIS_FROZEN_RESULT__DO_NOT_CREATE_TIME_GATING_OR_HANDCRAFTED_REGIME_ACTIVATION__DO_NOT_OPEN_A_NEW_MECHANISTIC_EXPERIMENT_AUTOMATICALLY",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "classification": summary["classification"],
        "conclusion": summary["conclusion"],
        "baseline_reproduction_passed": summary["baseline_reproduction_passed"],
        "failure_fold_persistence_4_of_4": summary["failure_fold_persistence_4_of_4"],
        "positive_fold_persistence_6_of_6": summary["positive_fold_persistence_6_of_6"],
        "strict_clock_half_sequence": summary["strict_clock_half_sequence"],
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
