#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from cb16_local_opt.operator_conditional_medium_geometry_h58 import adjudicate_h58, run_fold_h58
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_8_OPERATOR_CONDITIONAL_MEDIUM48_GEOMETRY_R0_RESULT_V1"
PREREG_COMMIT = "a043e41921090280062e4a09c7b22e55b3917e5b"
GATE_BLOB = "85142af8fba96b41c8e305b1e5e9cab35fc2593c"
H57_ADJ_COMMIT = "d6699197758fb408a2c64c0e9260b971b37d2fcc"
H57_ADJ_BLOB = "a999a766b6dd534c51ce088345bfb7c77576b19c"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": (
        "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json",
        "3c401a0a350984381912f7860181e3e96eb8d7cf",
    ),
    "gate": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_8_OPERATOR_CONDITIONAL_MEDIUM48_GEOMETRY_R0_GATE_V1.json",
        GATE_BLOB,
    ),
    "h57_adjudication": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_7_MEDIUM48_RELATIONAL_COMPOSITION_FALSIFICATION_R0_ADJUDICATION_V1.json",
        H57_ADJ_BLOB,
    ),
    "h58_helper": (
        "cb16_local_opt/operator_conditional_medium_geometry_h58.py",
        "aaddbad059c7262a8137db87bee25d38c07afd83",
    ),
    "h58_test": (
        "tests/test_operator_conditional_medium_geometry_h58.py",
        "aa08db14057f5ffc523d8db9ddecd0541f5ac4a8",
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
        "H58_NOT_DESCENDED_FROM_PREREGISTRATION",
    )
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", H57_ADJ_COMMIT, PREREG_COMMIT]) == 0,
        "H58_PREREG_NOT_DESCENDED_FROM_H57_ADJUDICATION",
    )
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H58_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H58_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "h57_adjudication_commit": H57_ADJ_COMMIT,
        "immutable_blobs": observed,
    }


def load_h57_authority() -> dict[str, Any]:
    path = Path(PINNED["h57_adjudication"][0])
    x = json.loads(path.read_text(encoding="utf-8"))
    require(x["adjudication"]["classification"] == "MEDIUM48_RELATIONAL_COMPOSITION_MIXED_UNRESOLVED", "H58_H57_CLASS_DRIFT")
    require(x["adjudication"]["aligned_medium_beats_structured_null"] is True, "H58_H57_ALIGNMENT_GATE_DRIFT")
    require(x["adjudication"]["true_market_beats_operator"] is False, "H58_H57_MARKET_BEAT_DRIFT")
    require(x["adjudication"]["true_market_worse_than_operator"] is False, "H58_H57_MARKET_WORSE_DRIFT")
    require(float(x["fold_receipts"]["fold_1"]["true_market_minus_operator"]) < 0.0, "H58_H57_FOLD1_CONTEXT_DRIFT")
    require(float(x["fold_receipts"]["fold_3"]["true_market_minus_operator"]) < 0.0, "H58_H57_FOLD3_CONTEXT_DRIFT")
    require(x["authority"]["canonical_change_authorized"] is False, "H58_H57_CANONICAL_AUTHORITY_DRIFT")
    require(x["authority"]["final_opening_authorized"] is False, "H58_H57_FINAL_AUTHORITY_DRIFT")
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    h57 = load_h57_authority()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require(len(folds) == 5, "H58_OUTER_FOLD_COUNT_DRIFT")

    results = [
        run_fold_h58(
            fold_spec=spec,
            all_train_parents=parents,
            all_train_samples=samples,
        )
        for spec in folds
    ]
    summary = adjudicate_h58(results)

    result = {
        "schema": SCHEMA,
        "status": "H5_8_OPERATOR_CONDITIONAL_MEDIUM48_GEOMETRY_COMPLETE",
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
        "parent_h5_7": {
            "classification": h57["adjudication"]["classification"],
            "conclusion": h57["adjudication"]["conclusion"],
            "market_below_operator_folds": [1, 3],
            "fold_1_true_market_minus_operator": float(h57["fold_receipts"]["fold_1"]["true_market_minus_operator"]),
            "fold_3_true_market_minus_operator": float(h57["fold_receipts"]["fold_3"]["true_market_minus_operator"]),
        },
        "support": support_receipt,
        "protocol": {
            "support_classification": "CONSUMED_TRAIN_ONLY_MECHANISTIC_SUPPORT__NOT_INDEPENDENT_MARKET_SUPPORT",
            "outer_folds": 5,
            "time_local_support": "EXACT_H5_6_H5_7_SAME_EVAL_BLOCK_OTHER_FUTURE_GROUPS_SAME_SCENARIO_TARGET_GROUP_EXCLUDED",
            "local_normalization": "EXACT_H5_6_SAME_SCENARIO_LOCAL_SUPPORT_ROWS_TARGET_GROUP_EXCLUDED",
            "primary_measure": "PARTIAL_PEARSON_OF_SPEARMAN_RANKS__MEDIUM_AND_UTILITY_PROJECTED_OFF_OPERATOR_RANK",
            "interpretation_limit": "INCREMENTAL_MONOTONIC_LINEAR_ASSOCIATION_IN_RANK_SPACE__NOT_GENERAL_CONDITIONAL_INDEPENDENCE",
            "shifts": [1, 7, 13, 23, 31],
            "medium_null": "ROTATE_ONLY_MEDIUM48_DISTANCE_VECTOR_ACROSS_SELECTED_SUPPORT_IDENTITIES",
            "medium_distance_multiset_preserved_per_anchor": True,
            "synthetic_feature_vectors_created": False,
            "fusion_weight_search": False,
            "formal_permutation_p_value_claim": False,
            "utility_profiles_rotated": False,
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
        "h5_8_summary": summary,
        "semantic_guards": {
            "student_trained": False,
            "student_inference_used": False,
            "teacher_compiled_for_scientific_score": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "future_utility_used_to_select_support_or_null_mapping": False,
            "utility_outcomes_rotated": False,
            "synthetic_operator_medium_feature_vectors_created": False,
            "fusion_weights_searched": False,
            "organ_weights_tuned": False,
            "distance_metric_tuned": False,
            "partial_spearman_claimed_as_general_conditional_independence": False,
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
        "next_legal_step": "ADJUDICATE_H5_8_FROM_THIS_FROZEN_RESULT__DO_NOT_TUNE_FUSION_WEIGHTS_OR_CHANGE_CANONICAL_ARCHITECTURE__ANY_NEXT_EXPERIMENT_REQUIRES_SEPARATE_PREREGISTRATION",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "classification": summary["classification"],
        "conclusion": summary["conclusion"],
        "global_gate": summary["global_gate"],
        "h5_7_failure_fold_gate": summary["h5_7_failure_fold_gate"],
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
