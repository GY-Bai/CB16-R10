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

from cb16_local_opt.common_support_rank_geometry_decomposition_h511 import (
    H511_PAIRS,
    adjudicate_h511,
    build_pair_overlap_design_h511,
    evaluate_fold_overlap_h511,
    native_fold_summary_h511,
    prepare_fold_payload_h511,
    run_fold_design_h511,
)
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_11_COMMON_SUPPORT_RANK_GEOMETRY_MECHANISM_DECOMPOSITION_R0_RESULT_V1"
PREREG_COMMIT = "e23534ec1b835dbc741ab02b074625cb3e8b6146"
GATE_BLOB = "bec91b527d66113d28681b55cd7a432133f8ee2c"
H5101_ADJ_COMMIT = "336f2617219c7783c6de4599cc86972355df5e0e"
H5101_ADJ_BLOB = "b6b0b0af59a899df15438c93550795e799269c65"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", "3c401a0a350984381912f7860181e3e96eb8d7cf"),
    "gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_11_COMMON_SUPPORT_RANK_GEOMETRY_MECHANISM_DECOMPOSITION_R0_GATE_V1.json", GATE_BLOB),
    "h5101_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_10_1_STRICT_CLOCK_HALFBLOCK_REPLICATION_R0_ADJUDICATION_V1.json", H5101_ADJ_BLOB),
    "h511_helper": ("cb16_local_opt/common_support_rank_geometry_decomposition_h511.py", "918f98d499ff5ed2f36e7d6a6bf3268fca37c27c"),
    "h511_test": ("tests/test_common_support_rank_geometry_decomposition_h511.py", "36e417581a8db1f7c0864461b5f64f89ba2abfcb"),
    "h5101_helper": ("cb16_local_opt/medium48_strict_clock_halfblock_replication_h5101.py", "f63e68673a8763265d7ff1b8009a3cb04f23fbd3"),
    "h510_helper": ("cb16_local_opt/medium48_temporal_halfblock_stability_h510.py", "176983eb0f7beb6396823126f4bc1504c4418054"),
    "h58_helper": ("cb16_local_opt/operator_conditional_medium_geometry_h58.py", "aaddbad059c7262a8137db87bee25d38c07afd83"),
    "h56_helper": ("cb16_local_opt/time_local_vs_forward_geometry_contrast_h56.py", "a264ff21d0447527c4dda06ee8fd63cf20207daf"),
    "h55_helper": ("cb16_local_opt/state_utility_geometry_transport_audit_h55.py", "1dd00ea9b40b8da4061c0983fc44fb2307c72587"),
    "h55_clarified_helper": ("cb16_local_opt/state_utility_geometry_transport_audit_h55_clarified.py", "faa5233170aa421c3bf44f023b1d8e2f7c187568"),
    "h5_helper": ("cb16_local_opt/teacher_temporal_transport_audit_h5.py", "09b05de23c659fe6f47a43117ec70b5cafcf7d21"),
    "columnar_index": ("cb16_local_opt/teacher_vectorized_r11.py", "083ca6541b45577cfeb6d9a376b25e0eaf8d060a"),
    "evidence_cache_loader": ("cb16_local_opt/r102_evidence_cache.py", "91fb73565ec60f66bf4232957f9b9b2f88cc3cfe"),
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0, "H511_NOT_DESCENDED_FROM_PREREGISTRATION")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", H5101_ADJ_COMMIT, PREREG_COMMIT]) == 0, "H511_PREREG_NOT_DESCENDED_FROM_H5101_ADJUDICATION")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H511_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H511_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "h5101_adjudication_commit": H5101_ADJ_COMMIT,
        "immutable_blobs": observed,
    }


def load_h5101_authority() -> dict[str, Any]:
    x = json.loads(Path(PINNED["h5101_adjudication"][0]).read_text(encoding="utf-8"))
    require(x["adjudication"]["classification"] == "STRICT_CLOCK_FAILURE_FOLDS_STABLE__POSITIVE_FOLDS_INTERNALLY_MIXED", "H511_H5101_CLASS_DRIFT")
    require(x["authority"]["canonical_change_authorized"] is False, "H511_PARENT_CANONICAL_AUTHORITY_DRIFT")
    require(x["authority"]["time_gating_authorized"] is False, "H511_PARENT_TIME_GATE_AUTHORITY_DRIFT")
    require(x["authority"]["final_opening_authorized"] is False, "H511_PARENT_FINAL_AUTHORITY_DRIFT")
    require(x["frozen_scientific_status"]["after"] == FROZEN_STATUS, "H511_PARENT_MARKET_STATUS_DRIFT")
    return x


def _blocked_summary(classification: str) -> dict[str, Any]:
    return {
        "classification": classification,
        "conclusion": f"H5_11_{classification}",
        "native_h5_8_baseline_reproduction_passed": False,
        "all_pair_common_support_gates_passed": False,
        "native_flip_persistence_count_of_3": 0,
        "native_flip_clean_removal_count_of_3": 0,
        "positive_control_pair_stable": False,
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "organ_change_authorized": False,
        "learned_gate_or_router_authorized": False,
        "time_or_regime_gate_authorized": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    parent = load_h5101_authority()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require([int(x["fold"]) for x in folds] == [1, 2, 3, 4, 5], "H511_OUTER_FOLD_SET_DRIFT")

    # Pass 1 is intentionally outcome-blind: run_fold_design_h511 never reads index.utilities.
    designs = [run_fold_design_h511(fold_spec=f, all_train_parents=parents, all_train_samples=samples) for f in folds]
    by_design = {int(x["fold"]): x for x in designs}
    pair_designs = [build_pair_overlap_design_h511(by_design[a], by_design[b]) for a, b in H511_PAIRS]
    support_ok = bool(all(bool(x["common_support_valid"]) for x in pair_designs))

    common = {
        "schema": SCHEMA,
        "repo": repo,
        "runtime_identity": {
            "python": platform.python_version(),
            "student_training": False,
            "student_inference": False,
            "teacher_law_compilation": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
        },
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "parent_h5_10_1": {
            "classification": parent["adjudication"]["classification"],
            "conclusion": parent["adjudication"]["conclusion"],
        },
        "support": support_receipt,
        "protocol": {
            "single_manipulated_variable": "EMPIRICAL_MEASURE_OVER_FROZEN_H5_8_ALIGNED_OPERATOR_MEDIUM_RANK_GEOMETRY",
            "design_pass_utility_access": False,
            "grid": "FIXED_3X3_ON_WITHIN_ANCHOR_OPERATOR_AND_MEDIUM_RANK_PERCENTILES",
            "base_measure": "TARGET_FUTURE_GROUP_EQUAL__SIX_SCENARIOS_EQUAL__SUPPORT_CANDIDATES_EQUAL_WITHIN_ANCHOR_SCENARIO",
            "overlap_target": "Q_C_PROPORTIONAL_TO_P_A_C_TIMES_P_B_C_DIVIDED_BY_P_A_C_PLUS_P_B_C",
            "q_computed_from_aligned_geometry_once": True,
            "same_candidate_weights_reused_for_all_nulls": True,
            "primary_measure": "OVERLAP_STANDARDIZED_H5_8_RANK_PARTIAL_CORRELATION",
            "weighted_spearman_canonical_claim": False,
            "formal_conditional_distribution_equality_test_claim": False,
            "formal_permutation_p_value_claim": False,
            "pair_count_is_statistical_n": False,
            "same_future_group_is_dependence_unit": True,
            "fresh_market_data_downloaded": False,
            "final_holdout_opened": False,
            "r7_candidate_evaluated": False,
        },
        "outcome_blind_fold_designs": designs,
        "outcome_blind_pair_designs": pair_designs,
    }

    if not support_ok:
        result = dict(common)
        result.update({
            "status": "H5_11_EXECUTION_BLOCKED__INSUFFICIENT_COMMON_SUPPORT",
            "utility_pass_executed": False,
            "native_h5_8_baselines": [],
            "pair_results": [],
            "h5_11_summary": _blocked_summary("EXECUTION_BLOCKED__INSUFFICIENT_COMMON_SUPPORT"),
            "semantic_guards": {
                "utility_accessed_during_overlap_design": False,
                "utility_pass_executed_after_failed_support_gate": False,
                "propensity_model_fit": False,
                "new_representation_trained": False,
                "learned_gate_or_router_created": False,
                "time_or_regime_gate_created": False,
                "canonical_architecture_changed": False,
                "teacher_changed": False,
                "student_changed": False,
                "organ_changed": False,
                "final_holdout_payload_opened": False,
                "r7_candidate_evaluated": False,
            },
            "next_legal_step": "ADJUDICATE_EXECUTION_BLOCKED_H5_11__DO_NOT_RESCUE_BY_CHANGING_GRID_OR_SUPPORT_RULE_IN_THIS_EXPERIMENT",
        })
        atomic_json(args.output.resolve(), result)
        print(json.dumps({"status": result["status"], "classification": result["h5_11_summary"]["classification"]}, indent=2, sort_keys=True))
        return 0

    # Pass 2 may now read utilities.  It must reproduce H5.8 before any weighted interpretation.
    payloads = [prepare_fold_payload_h511(fold_spec=f, all_train_parents=parents, all_train_samples=samples) for f in folds]
    by_payload = {int(x["fold"]): x for x in payloads}
    for d in designs:
        f = int(d["fold"])
        maxerr = float(np.max(np.abs(np.asarray(d["cell_mass"], dtype=np.float64) - np.asarray(by_payload[f]["cell_mass"], dtype=np.float64))))
        require(maxerr <= 1e-12, f"H511_DESIGN_PAYLOAD_CELL_MASS_DRIFT:{f}:{maxerr}")
    native = [native_fold_summary_h511(by_payload[f]) for f in range(1, 6)]
    baseline_ok = bool(all(bool(x["reproduced"]) for x in native))

    if baseline_ok:
        pair_results = []
        for d in pair_designs:
            a, b = map(int, d["pair"])
            side_a = evaluate_fold_overlap_h511(by_payload[a], d["multiplier_a"], d["q"])
            side_b = evaluate_fold_overlap_h511(by_payload[b], d["multiplier_b"], d["q"])
            pair_results.append({**d, "side_a": side_a, "side_b": side_b})
    else:
        pair_results = list(pair_designs)

    summary = adjudicate_h511(native, pair_results)
    result = dict(common)
    result.update({
        "status": "H5_11_COMMON_SUPPORT_RANK_GEOMETRY_COMPLETE" if baseline_ok else "H5_11_EXECUTION_BLOCKED__H5_8_BASELINE_REPRODUCTION_FAILED",
        "utility_pass_executed": True,
        "native_h5_8_baselines": native,
        "pair_results": pair_results if baseline_ok else [],
        "h5_11_summary": summary,
        "semantic_guards": {
            "utility_accessed_during_overlap_design": False,
            "overlap_q_or_cell_selection_used_utility": False,
            "q_recomputed_for_structured_null": False,
            "same_aligned_candidate_weights_used_for_all_structured_nulls": True,
            "support_or_normalization_reselected_after_weighting": False,
            "distance_metric_changed": False,
            "candidate_rank_order_changed_for_cell_standardization": False,
            "grid_used_as_new_representation": False,
            "propensity_model_fit": False,
            "new_representation_trained": False,
            "teacher_compiled_for_scientific_score": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "student_training_used": False,
            "student_inference_used": False,
            "fusion_weights_searched": False,
            "organ_weights_tuned": False,
            "learned_gate_or_router_created": False,
            "handcrafted_time_or_regime_gate_created": False,
            "calendar_or_fold_id_used_as_runtime_permission_signal": False,
            "change_point_search_used": False,
            "canonical_architecture_changed": False,
            "market_information_verdict_reopened": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False,
        },
        "next_legal_step": "ADJUDICATE_H5_11_FROM_THIS_FROZEN_RESULT__DO_NOT_CREATE_A_GATE_ROUTER_TIME_RULE_OR_CANONICAL_CHANGE__ANY_NEXT_EXPERIMENT_REQUIRES_SEPARATE_PREREGISTRATION",
    })
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "classification": summary["classification"],
        "flip_persistence_count_of_3": summary["native_flip_persistence_count_of_3"],
        "flip_clean_removal_count_of_3": summary["native_flip_clean_removal_count_of_3"],
        "positive_control_pair_stable": summary["positive_control_pair_stable"],
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
