#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np

from cb16_local_opt.medium48_relational_composition_falsification_h57 import (
    H57_SHIFTS,
    adjudicate_h57,
    run_fold_h57,
    tau_gap_h57,
)
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_7_MEDIUM48_RELATIONAL_COMPOSITION_FALSIFICATION_R0_RESULT_V1"
PREREG_COMMIT = "d5232fef2ff2fd557bb9527704c30e294f7b4aea"
GATE_BLOB = "e1e69586da3c0cce8a2dda8270348bb558d750b3"
H56_ADJ_COMMIT = "995fc53389646900e2033d74c4b4d6c370e92780"
H56_ADJ_BLOB = "de6b2c53ff08b98f2232989dfd368dfe7ffa1998"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": (
        "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json",
        "3c401a0a350984381912f7860181e3e96eb8d7cf",
    ),
    "gate": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_7_MEDIUM48_RELATIONAL_COMPOSITION_FALSIFICATION_R0_GATE_V1.json",
        GATE_BLOB,
    ),
    "h56_adjudication": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_6_TIME_LOCAL_VS_FORWARD_STATE_UTILITY_GEOMETRY_CONTRAST_R0_ADJUDICATION_V1.json",
        H56_ADJ_BLOB,
    ),
    "h57_helper": (
        "cb16_local_opt/medium48_relational_composition_falsification_h57.py",
        "32c7f67960f0ea2a111bd60a19796bb797cf9ac0",
    ),
    "h57_test": (
        "tests/test_medium48_relational_composition_falsification_h57.py",
        "eca494719631f43f2c8fddd1dc15fd186a4d3ba0",
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
        "H57_NOT_DESCENDED_FROM_PREREGISTRATION",
    )
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", H56_ADJ_COMMIT, PREREG_COMMIT]) == 0,
        "H57_PREREG_NOT_DESCENDED_FROM_H56_ADJUDICATION",
    )
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H57_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H57_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "h56_adjudication_commit": H56_ADJ_COMMIT,
        "immutable_blobs": observed,
    }


def load_h56_authority() -> dict[str, Any]:
    path = Path(PINNED["h56_adjudication"][0])
    x = json.loads(path.read_text(encoding="utf-8"))
    require(
        x["adjudication"]["classification"] == "TIME_LOCAL_OPERATOR_GEOMETRY_EXISTS__MEDIUM48_DEGRADES_CROSS_TIME_COMPOSITION",
        "H57_H56_CLASS_DRIFT",
    )
    require(
        x["adjudication"]["conclusion"] == "H5_6_TIME_LOCAL_OPERATOR_GEOMETRY_EXISTS__MEDIUM48_DEGRADES_CROSS_TIME_COMPOSITION",
        "H57_H56_CONCLUSION_DRIFT",
    )
    require(x["authority"]["canonical_change_authorized"] is False, "H57_H56_CANONICAL_AUTHORITY_DRIFT")
    require(x["authority"]["final_opening_authorized"] is False, "H57_H56_FINAL_AUTHORITY_DRIFT")
    return x


def tau_gap_self_check() -> dict[str, Any]:
    ref = np.asarray([0.0, 0.25, 0.75, 2.0, 4.0], dtype=np.float64)
    pred = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    base, _ = tau_gap_h57(ref, pred)
    transformed, _ = tau_gap_h57(11.0 * ref + 17.0, pred)
    require(abs(base - transformed) <= 1e-15, "H57_TAUGAP_AFFINE_SCALE_IDENTITY_FAIL")
    require(base == 1.0, "H57_TAUGAP_PERFECT_ORDER_FAIL")
    return {
        "positive_affine_reference_scale_identity": True,
        "perfect_order_tau_gap": float(base),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    h56 = load_h56_authority()
    self_check = tau_gap_self_check()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require(len(folds) == 5, "H57_OUTER_FOLD_COUNT_DRIFT")

    results = [
        run_fold_h57(
            fold_spec=spec,
            all_train_parents=parents,
            all_train_samples=samples,
        )
        for spec in folds
    ]
    summary = adjudicate_h57(results)

    result = {
        "schema": SCHEMA,
        "status": "H5_7_MEDIUM48_RELATIONAL_COMPOSITION_FALSIFICATION_COMPLETE",
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
        "parent_h5_6": {
            "classification": h56["adjudication"]["classification"],
            "conclusion": h56["adjudication"]["conclusion"],
        },
        "support": support_receipt,
        "protocol": {
            "support_classification": "CONSUMED_TRAIN_ONLY_MECHANISTIC_SUPPORT__NOT_INDEPENDENT_MARKET_SUPPORT",
            "outer_folds": 5,
            "time_local_support": "EXACT_H5_6_SAME_EVAL_BLOCK_OTHER_FUTURE_GROUPS_SAME_SCENARIO_TARGET_GROUP_EXCLUDED",
            "local_normalization": "EXACT_H5_6_SAME_SCENARIO_LOCAL_SUPPORT_ROWS_TARGET_GROUP_EXCLUDED",
            "primary_measure": "TAU_GAP_PER_ANCHOR_THEN_SCENARIO_MEAN_WITHIN_FUTURE_GROUP_THEN_EQUAL_WEIGHT_FUTURE_GROUP_MEAN",
            "shifts": list(H57_SHIFTS),
            "medium_null": "ROTATE_ONLY_MEDIUM48_SQUARED_DISTANCE_CONTRIBUTION_ACROSS_SELECTED_SUPPORT_IDENTITIES",
            "medium_distance_multiset_preserved_per_anchor": True,
            "synthetic_feature_vectors_created": False,
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
        "tau_gap_self_check": self_check,
        "folds": results,
        "h5_7_summary": summary,
        "semantic_guards": {
            "student_trained": False,
            "student_inference_used": False,
            "teacher_compiled_for_scientific_score": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "future_utility_used_to_select_support_or_null_mapping": False,
            "utility_outcomes_rotated": False,
            "synthetic_operator_medium_feature_vectors_created": False,
            "organ_weights_tuned": False,
            "distance_metric_tuned": False,
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
        "next_legal_step": "ADJUDICATE_H5_7_FROM_THIS_FROZEN_RESULT__DO_NOT_AUTOMATICALLY_CHANGE_OR_REMOVE_MEDIUM48__DO_NOT_TUNE_ORGAN_WEIGHTS_OR_DISTANCE_METRICS__ANY_NEXT_EXPERIMENT_REQUIRES_SEPARATE_PREREGISTRATION",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "classification": summary["classification"],
        "conclusion": summary["conclusion"],
        "true_market_beats_operator_gate": summary["true_market_beats_operator_gate"],
        "true_market_worse_than_operator_gate": summary["true_market_worse_than_operator_gate"],
        "aligned_medium_beats_null_gate": summary["aligned_medium_beats_null_gate"],
        "aligned_medium_worse_than_null_gate": summary["aligned_medium_worse_than_null_gate"],
        "all_geometry_identity_guards_pass": summary["all_geometry_identity_guards_pass"],
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
