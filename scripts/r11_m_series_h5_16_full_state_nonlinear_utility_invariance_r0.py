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

from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import (
    H516_DIM,
    H516_PAIRS,
    H516_SEED,
    H516_SHIFTS,
    H516_UTILITY_DIM,
    SKLEARN_VERSION,
    build_fold_dataset_h516,
    classify_h516,
    run_environment_arm_h516,
    run_forward_arm_h516,
    run_local_arm_h516,
)
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_16_FULL_STATE_NONLINEAR_UTILITY_INVARIANCE_R0_RESULT_V1"
STATUS = "H5_16_FULL_STATE_NONLINEAR_UTILITY_INVARIANCE_EXECUTION_COMPLETE"
PREREG_COMMIT = "68f2f7ea43c333983297a67a7253f963db19eb7f"
CLARIFICATION_COMMIT = "1fe97c634f0d7850dc638b51f8e068e5c5957e84"
REASSESSMENT_COMMIT = "efd6427a861c836185919d8192f6f74c66cb440e"
H515_ADJUDICATION_COMMIT = "1d5c68706cbc227a014951e8df12eada6440e084"
GATE_BLOB = "b4847abda958f0c3e00b6fb23ccd4463c4e41084"
CLARIFICATION_BLOB = "c92f8dcbed19175e4781a6095105bd1bd49d20ed"
H515_ADJUDICATION_BLOB = "22fd3a62b9695e3246eb95b457bfb36ad0b4ce29"
H55_GATE_BLOB = "54b9492e74d5017711b671837ef402abd604c7c6"
H55_ADJUDICATION_BLOB = "c336e2789ba41e2131679abcd1e6d51c1b545a1c"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", "3c401a0a350984381912f7860181e3e96eb8d7cf"),
    "gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_16_FULL_STATE_NONLINEAR_UTILITY_INVARIANCE_R0_GATE_V1.json", GATE_BLOB),
    "clarification": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_16_FULL_STATE_NONLINEAR_UTILITY_INVARIANCE_R0_PRE_EXECUTION_CLARIFICATION_V1.json", CLARIFICATION_BLOB),
    "h515_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_15_FULL_STATE_LINEAR_OFFSET_TRANSPORT_R0_ADJUDICATION_V1.json", H515_ADJUDICATION_BLOB),
    "h515_reassessment": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_15_POSTMORTEM_COMPETING_HYPOTHESES_REASSESSMENT_V1.json", "41a1b0e70c612ea268d84c15a08752edc84d0836"),
    "h55_gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_GATE_V1.json", H55_GATE_BLOB),
    "h55_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_ADJUDICATION_V1.json", H55_ADJUDICATION_BLOB),
    "h516_helper": ("cb16_local_opt/full_state_nonlinear_utility_invariance_h516.py", "bfa28be3ad6676f26676404bb170a73b474b6192"),
    "h516_test": ("tests/test_full_state_nonlinear_utility_invariance_h516.py", "7937c23facc6305b960c0ec19ae976ebb53df3ae"),
    "h55_helper": ("cb16_local_opt/state_utility_geometry_transport_audit_h55.py", "1dd00ea9b40b8da4061c0983fc44fb2307c72587"),
    "h5_helper": ("cb16_local_opt/teacher_temporal_transport_audit_h5.py", "09b05de23c659fe6f47a43117ec70b5cafcf7d21"),
    "columnar_index": ("cb16_local_opt/teacher_vectorized_r11.py", "083ca6541b45577cfeb6d9a376b25e0eaf8d060a"),
    "r6_helper": ("cb16_local_opt/reduced_teacher_target_information_audit_r6.py", "bd7a8780e8dcab05cfae92744fde523ce62cc9ae"),
    "r6_executor": ("scripts/r11_science_g0_reduced_teacher_target_information_audit_r6.py", "72bede8dd09015c702ec26ead639a81f48efe38a"),
    "shanxi_requirements": ("requirements-shanxi-pascal.txt", "da3813ec5bfd79973e425ce49eff473db6f07c4b"),
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", REASSESSMENT_COMMIT, PREREG_COMMIT]) == 0, "H516_PARENT_AUTHORITY_DRIFT:REASSESSMENT_NOT_ANCESTOR")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, CLARIFICATION_COMMIT]) == 0, "H516_PARENT_AUTHORITY_DRIFT:CLARIFICATION_NOT_DESCENDED")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", CLARIFICATION_COMMIT, "HEAD"]) == 0, "H516_PARENT_AUTHORITY_DRIFT:HEAD_NOT_DESCENDED")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", H515_ADJUDICATION_COMMIT, REASSESSMENT_COMMIT]) == 0, "H516_PARENT_AUTHORITY_DRIFT:H515_NOT_ANCESTOR")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H516_PARENT_AUTHORITY_DRIFT:BLOB:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H516_PARENT_AUTHORITY_DRIFT:PREREG_GATE")
    require(git("rev-parse", f"{CLARIFICATION_COMMIT}:{PINNED['clarification'][0]}") == CLARIFICATION_BLOB, "H516_PARENT_AUTHORITY_DRIFT:CLARIFICATION_BLOB")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "clarification_commit": CLARIFICATION_COMMIT,
        "post_h515_reassessment_commit": REASSESSMENT_COMMIT,
        "h515_adjudication_commit": H515_ADJUDICATION_COMMIT,
        "immutable_blobs": observed,
    }


def blocked_classification(exc: RuntimeError) -> str | None:
    code = str(exc)
    if code.startswith("H516_PARENT_AUTHORITY_DRIFT"):
        return "EXECUTION_BLOCKED__PARENT_AUTHORITY_DRIFT"
    if code.startswith(("H516_FEATURE_DIM", "H516_STATE_ROW_DRIFT", "H516_UTILITY_ROW_DRIFT", "H516_CENTERING_DRIFT", "H516_X_SHAPE", "H516_Y_SHAPE", "H516_DATA_NONFINITE")):
        return "EXECUTION_BLOCKED__STATE_UTILITY_OR_ACTION_GRID_DRIFT"
    if code.startswith(("H516_GROUP_CLOCK_DRIFT", "H516_SCENARIO_COUNT_DRIFT", "H516_GROUP_ORDER_DRIFT", "H516_DEPENDENCE_FOLD_SET_DRIFT", "H516_TOO_FEW_CLOCKS", "H516_EMPTY_CLOCK_BLOCK", "H516_CLOCK_PARTITION", "H516_LOCAL_EMPTY_SPLIT", "H516_POOLED_CLOCK_MULTI_ENV", "H516_CLOCK_ENV_DRIFT", "H516_SUPPORT_CLASSIFIER_TRAIN_CLASS_DRIFT")):
        return "EXECUTION_BLOCKED__DEPENDENCE_CLOCK_OR_SCENARIO_DRIFT"
    if code.startswith(("H516_NEGATIVE_CONTROL_IDENTITY", "H516_NEGATIVE_CONTROL_INDEX_IDENTITY", "H516_FAKE_ENV_SHIFT_IDENTITY_INDEX", "H516_FAKE_ENV_LABEL_IDENTITY", "H516_FAKE_ENV_MARGINAL_DRIFT")):
        return "EXECUTION_BLOCKED__NEGATIVE_CONTROL_IDENTITY"
    if code.startswith(("H516_RF_", "H516_LOSS_", "H516_SCALAR_NONFINITE", "H516_SUPPORT_CLASSIFIER_CLASS_ORDER", "H516_WEIGHT_DRIFT", "H516_TARGET_MEAN_NONFINITE", "H516_ENV_AUGMENT")):
        return "EXECUTION_BLOCKED__NONLINEAR_RUNTIME_OR_NUMERICAL_FAILURE"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo: dict[str, Any] | None = None
    support_receipt: dict[str, Any] | None = None
    dataset_receipts: list[dict[str, Any]] = []
    local_results: list[dict[str, Any]] = []
    forward_results: list[dict[str, Any]] = []
    environment_results: list[dict[str, Any]] = []
    summary: dict[str, Any]
    execution_error: str | None = None

    try:
        repo = verify_repo()
        gate = json.loads(Path(PINNED["gate"][0]).read_text(encoding="utf-8"))
        clarification = json.loads(Path(PINNED["clarification"][0]).read_text(encoding="utf-8"))
        h515 = json.loads(Path(PINNED["h515_adjudication"][0]).read_text(encoding="utf-8"))
        h55 = json.loads(Path(PINNED["h55_adjudication"][0]).read_text(encoding="utf-8"))
        require(gate["status"] == "PREREGISTERED__UNEVALUATED", "H516_PARENT_AUTHORITY_DRIFT:GATE_STATUS")
        require(clarification["status"] == "PRE_EXECUTION_CLARIFICATION__NO_SCIENTIFIC_RESULT_OBSERVED", "H516_PARENT_AUTHORITY_DRIFT:CLARIFICATION_STATUS")
        require(clarification["gate_blob_unchanged"] == GATE_BLOB, "H516_PARENT_AUTHORITY_DRIFT:CLARIFICATION_GATE")
        require(h515["classification"] == gate["parent"]["required_h5_15_classification"], "H516_PARENT_AUTHORITY_DRIFT:H515_CLASS")
        require(h55["adjudication"]["classification"] == gate["parent"]["required_h5_5_classification"], "H516_PARENT_AUTHORITY_DRIFT:H55_CLASS")

        parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
        folds = build_outer_folds_r6(parents)
        require(tuple(int(f["fold"]) for f in folds) == (1, 2, 3, 4, 5), "H516_DEPENDENCE_FOLD_SET_DRIFT")
        datasets = [
            build_fold_dataset_h516(fold_spec=f, all_train_parents=parents, all_train_samples=samples)
            for f in folds
        ]
        by_fold = {int(d["fold"]): d for d in datasets}
        dataset_receipts = [
            {
                "fold": int(d["fold"]),
                "future_group_count": int(d["future_group_count"]),
                "unique_decision_clock_count": int(d["unique_decision_clock_count"]),
                "scenario_count_per_group": int(d["scenario_count_per_group"]),
                "state_dimension": int(d["state_dimension"]),
                "utility_dimension": int(d["utility_dimension"]),
                "first_timestamp_ms": int(d["timestamps_ms"][0]),
                "last_timestamp_ms": int(d["timestamps_ms"][-1]),
                "timestamps_nondecreasing": bool(np.all(np.diff(np.asarray(d["timestamps_ms"], dtype=np.int64)) >= 0)),
            }
            for d in datasets
        ]
        require(all(r["timestamps_nondecreasing"] for r in dataset_receipts), "H516_GROUP_ORDER_DRIFT:RECEIPT")

        local_results = [run_local_arm_h516(d) for d in datasets]
        for a, b in H516_PAIRS:
            forward_results.append(run_forward_arm_h516(by_fold[a], by_fold[b]))
            environment_results.append(run_environment_arm_h516(by_fold[a], by_fold[b]))
        summary = classify_h516(local_results, forward_results, environment_results)
    except RuntimeError as exc:
        blocked = blocked_classification(exc)
        if blocked is None:
            raise
        execution_error = str(exc)
        summary = {
            "classification": blocked,
            "local_map_supported": None,
            "native_forward_pair_pass_count": None,
            "native_environment_increment_pair_pass_count": None,
        }

    result = {
        "schema": SCHEMA,
        "status": STATUS if not str(summary["classification"]).startswith("EXECUTION_BLOCKED__") else "H5_16_EXECUTION_BLOCKED",
        "repo": repo,
        "support": support_receipt,
        "dataset_receipts": dataset_receipts,
        "runtime": {
            "python_version": platform.python_version(),
            "sklearn_version": SKLEARN_VERSION,
            "fixed_regressor": {
                "n_estimators": 100,
                "criterion": "squared_error",
                "max_features": 10,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 5,
                "bootstrap": True,
                "max_samples": None,
                "ccp_alpha": 0.0,
                "random_state": H516_SEED,
                "n_jobs": 1,
                "multi_output": "DIRECT_NATIVE_9D_REGRESSION",
            },
            "secondary_classifier_is_gate": False,
        },
        "local_results": local_results,
        "forward_results": forward_results,
        "environment_results": environment_results,
        "summary": summary,
        "execution_error": execution_error,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "semantic_guards": {
            "h5_12r_pair_side_ipf_target_used": False,
            "target_is_common_centered_9_action_realized_utility_profile": True,
            "rf_hyperparameter_or_seed_search_used": False,
            "feature_subselection_ablation_or_reweighting_search_used": False,
            "calendar_or_fold_id_runtime_input_used": False,
            "environment_index_runtime_feature_used": False,
            "environment_index_science_diagnostic_only": True,
            "teacher_changed": False,
            "student_changed": False,
            "canonical_architecture_changed": False,
            "learned_runtime_router_or_regime_gate_created": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False,
            "champion_promoted": False,
        },
        "authority_effect": {
            "market_information_verdict_change": False,
            "canonical_change_authorized": False,
            "teacher_change_authorized": False,
            "student_change_authorized": False,
            "runtime_environment_gate_authorized": False,
            "promotion_authorized": False,
            "r7_evaluation_authorized": False,
            "final_opening_authorized": False,
        },
        "next_legal_step": "FREEZE_H5_16_RESULT__FORMALLY_ADJUDICATE__DO_NOT_OPEN_H5_17_OR_CHANGE_CANONICAL_ARCHITECTURE_WITHOUT_SEPARATE_PREREGISTRATION",
    }
    atomic_json(args.output, result)
    print(json.dumps({"classification": summary["classification"], "summary": summary, "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
