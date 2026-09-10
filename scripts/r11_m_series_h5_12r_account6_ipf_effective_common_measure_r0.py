#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import numpy as np

from cb16_local_opt.account6_ipf_effective_common_measure_h512r import (
    H512R_FORMAL_TOL,
    H512R_SIDE_TOL,
    build_pair_design_h512r,
    classify_h512r,
    prepare_fold_payload_h512r,
    run_pair_h512r,
)
from cb16_local_opt.common_support_rank_geometry_decomposition_h511 import (
    H511_PAIRS,
    build_pair_overlap_design_h511,
    prepare_fold_payload_h511,
    run_fold_design_h511,
)
from cb16_local_opt.ipf_effective_common_measure_h5112 import run_pair_h5112
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_12R_ACCOUNT6_IPF_EFFECTIVE_COMMON_MEASURE_R0_RESULT_V1"
STATUS = "H5_12R_ACCOUNT6_IPF_EFFECTIVE_COMMON_MEASURE_EXECUTION_COMPLETE"
PREREG_COMMIT = "ec1b6b3c43813bca7bdfab29178cf3be94563eab"
GATE_BLOB = "e5458dd6c47f5cbe91b923e1180bd800fc5f7a4a"
PARENT_ADJ_COMMIT = "54937bedc0dfc19c79ddf3b1fe452bfbd73d5732"
PARENT_ADJ_BLOB = "c19bf407480fbaf783ebe8a98c24d2e0ff59cb78"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", "3c401a0a350984381912f7860181e3e96eb8d7cf"),
    "gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_12R_ACCOUNT6_IPF_EFFECTIVE_COMMON_MEASURE_R0_GATE_V1.json", GATE_BLOB),
    "parent_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_11_2_IPF_EFFECTIVE_COMMON_MEASURE_R0_ADJUDICATION_V1.json", PARENT_ADJ_BLOB),
    "h512r_helper": ("cb16_local_opt/account6_ipf_effective_common_measure_h512r.py", "a2950648617ad5327658e0fa1379444b5cb51015"),
    "h512r_test": ("tests/test_account6_ipf_effective_common_measure_h512r.py", "3d3a0fdea1b7344a3d3392dabc7c4915a7cd58d7"),
    "h5112_helper": ("cb16_local_opt/ipf_effective_common_measure_h5112.py", "e23e939fa43208a948ecf9a3a108fea1831777ce"),
    "h511_helper": ("cb16_local_opt/common_support_rank_geometry_decomposition_h511.py", "918f98d499ff5ed2f36e7d6a6bf3268fca37c27c"),
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PARENT_ADJ_COMMIT, PREREG_COMMIT]) == 0, "H512R_PREREG_NOT_DESCENDED_FROM_PARENT")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0, "H512R_HEAD_NOT_DESCENDED_FROM_PREREG")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H512R_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H512R_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "parent_adjudication_commit": PARENT_ADJ_COMMIT,
        "immutable_blobs": observed,
    }


def load_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    gate = json.loads(Path(PINNED["gate"][0]).read_text(encoding="utf-8"))
    parent = json.loads(Path(PINNED["parent_adjudication"][0]).read_text(encoding="utf-8"))
    require(gate["status"] == "PREREGISTERED__UNEVALUATED", "H512R_GATE_STATUS_DRIFT")
    require(parent["adjudication"]["classification"] == "EXACT_MACRO_CELL_OCCUPANCY_STANDARDIZATION__NATIVE_TRANSITIONS_PERSIST", "H512R_PARENT_CLASS_DRIFT")
    require(parent["frozen_scientific_status"]["after"] == FROZEN_STATUS, "H512R_PARENT_MARKET_STATUS_DRIFT")
    require(parent["authority"]["old_h5_12_preregistration_execution_authorized"] is False, "H512R_STALE_H512_AUTHORITY_DRIFT")
    return gate, parent


def _pair_key(pair: tuple[int, int], fold: int) -> str:
    return f"{pair[0]}_{pair[1]}_side_{fold}"


def reproduce_parent_h5112(
    *, folds: list[Mapping[str, Any]], parents: Mapping[str, Any], samples: list[Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    designs = [run_fold_design_h511(fold_spec=f, all_train_parents=parents, all_train_samples=samples) for f in folds]
    by_design = {int(x["fold"]): x for x in designs}
    pair_designs = [build_pair_overlap_design_h511(by_design[a], by_design[b]) for a, b in H511_PAIRS]
    require(all(bool(x["common_support_valid"]) for x in pair_designs), "H512R_PARENT_9CELL_SUPPORT_DRIFT")
    payloads = [prepare_fold_payload_h511(fold_spec=f, all_train_parents=parents, all_train_samples=samples) for f in folds]
    by_payload = {int(x["fold"]): x for x in payloads}
    pair_results = [run_pair_h5112(pair_design=d, payload_a=by_payload[int(d["pair"][0])], payload_b=by_payload[int(d["pair"][1])]) for d in pair_designs]
    expected_a = gate["parent_reproduction_gate"]["required_pair_side_aligned_partial_rho"]
    expected_n = gate["parent_reproduction_gate"]["required_pair_side_null_median"]
    tol = float(gate["parent_reproduction_gate"]["tolerance"])
    max_a = 0.0; max_n = 0.0
    rows: dict[str, Any] = {}
    for p in pair_results:
        pair = tuple(int(x) for x in p["pair"])
        for side, fold in (("side_a", pair[0]), ("side_b", pair[1])):
            r = p[side]
            key = _pair_key(pair, fold)
            ae = abs(float(r["aligned_partial_rho"]) - float(expected_a[key]))
            ne = abs(float(r["null_median_partial_rho"]) - float(expected_n[key]))
            max_a = max(max_a, ae); max_n = max(max_n, ne)
            rows[key] = {"pair": list(pair), "fold": fold, "aligned_abs_error": ae, "null_median_abs_error": ne, "orientation": r["orientation"]}
    return {
        "passed": bool(max_a <= tol and max_n <= tol),
        "tolerance": tol, "max_aligned_abs_error": max_a, "max_null_median_abs_error": max_n,
        "rows": rows, "pair_results": pair_results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    gate, parent = load_authority()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require([int(x["fold"]) for x in folds] == [1,2,3,4,5], "H512R_FOLD_SET_DRIFT")

    parent_rep = reproduce_parent_h5112(folds=folds, parents=parents, samples=samples, gate=gate)
    pair_designs_3d: list[dict[str, Any]] = []
    pair_results_3d: list[dict[str, Any]] = []
    fold_payloads_3d: list[dict[str, Any]] = []
    if parent_rep["passed"]:
        fold_payloads_3d = [prepare_fold_payload_h512r(fold_spec=f, all_train_parents=parents, all_train_samples=samples) for f in folds]
        by3 = {int(x["fold"]): x for x in fold_payloads_3d}
        pair_designs_3d = [build_pair_design_h512r(by3[a], by3[b]) for a, b in H511_PAIRS]
        pair_results_3d = [run_pair_h512r(design=d, payload_a=by3[int(d["pair"][0])], payload_b=by3[int(d["pair"][1])]) for d in pair_designs_3d]
        summary = classify_h512r(pair_results_3d)
    else:
        summary = {"classification": "EXECUTION_BLOCKED__H5_11_2_PARENT_REPRODUCTION_FAILED", "conclusion": "H5_12R_EXECUTION_BLOCKED__H5_11_2_PARENT_REPRODUCTION_FAILED", "native_flip_persistence_count_of_3": 0, "native_flip_clean_removal_count_of_3": 0, "positive_control_pair_stable": False}

    valid_pairs = [p for p in pair_results_3d if not p.get("blocked")]
    eval_rows = [p[s] for p in valid_pairs for s in ("side_a", "side_b")]
    cal_rows = [p["calibration"][s] for p in valid_pairs for s in ("side_a", "side_b")]
    global_max_vs_q = max([float(x["effective_max_abs_error_vs_q"]) for x in eval_rows], default=None)
    global_max_side = max([float(x["effective_side_to_side_max_abs"]) for x in valid_pairs], default=None)
    max_iters = max([int(x["iterations"]) for x in cal_rows], default=0)
    max_ratio = max([float(x["multiplier_positive_max_to_min_ratio"]) for x in cal_rows], default=None)

    result = {
        "schema": SCHEMA, "status": STATUS, "repo": repo,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "github_run_id": os.environ.get("GITHUB_RUN_ID"), "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")},
        "support_receipt": support_receipt,
        "protocol": {
            "single_manipulated_variable": gate["single_manipulated_variable"],
            "stale_h5_12_executed": False,
            "account6_features": [96,97,98,99,100,101],
            "account6_feature_subselection_or_reweighting": False,
            "grid": "3x3x3", "cell_count": 27,
            "actual_local_effective_margin_ipf": True,
            "support_unchanged": True, "normalization_unchanged": True, "utility_unchanged": True,
            "primary_statistic_unchanged": True, "structured_null_unchanged": True,
            "same_future_group_is_dependence_unit": True,
        },
        "h5_11_2_parent_reproduction": parent_rep,
        "fold_payload_receipts_3d": [{k:v for k,v in x.items() if k != "groups"} for x in fold_payloads_3d],
        "pair_designs_3d": pair_designs_3d,
        "pair_results_3d": pair_results_3d,
        "summary": {**summary, "global_max_abs_effective_side_vs_q": global_max_vs_q, "global_max_abs_effective_side_to_side": global_max_side, "max_ipf_iterations": max_iters, "max_positive_multiplier_max_to_min_ratio": max_ratio, "market_information_verdict_changed": False, "canonical_change_authorized": False, "runtime_gate_authorized": False},
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "semantic_guards": {
            "stale_h5_12_executed": False, "result_adaptive_grid_change": False, "support_trimmed": False, "cells_merged": False,
            "account6_feature_selected_or_reweighted": False, "primary_statistic_changed": False, "structured_null_changed": False,
            "support_changed": False, "normalization_changed": False, "distance_metric_changed": False,
            "teacher_changed": False, "student_changed": False, "canonical_architecture_changed": False,
            "medium48_deleted_or_reweighted": False, "fusion_weight_search_executed": False,
            "learned_router_or_gate_created": False, "time_or_regime_gate_created": False, "calendar_or_fold_id_runtime_feature_used": False,
            "fresh_market_data_downloaded": False, "final_holdout_payload_opened": False, "r7_candidate_evaluated": False, "champion_promoted": False,
        },
        "next_legal_step": "ADJUDICATE_H5_12R_RESULT_ONLY__NO_RUNTIME_GATE_OR_CANONICAL_CHANGE__IF_EXECUTION_BLOCKED_DO_NOT_RESCUE_WITH_GRID_COARSENING_OR_SUPPORT_TRIMMING",
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "status": STATUS, "classification": result["summary"]["classification"],
        "parent_reproduction_passed": parent_rep["passed"],
        "parent_max_aligned_error": parent_rep["max_aligned_abs_error"],
        "parent_max_null_median_error": parent_rep["max_null_median_abs_error"],
        "fold_account_degenerate_counts": {str(x["fold"]): int(x["account_degenerate_target_scenario_count"]) for x in fold_payloads_3d},
        "positive_q_cell_counts": {f"{d['pair'][0]}_{d['pair'][1]}": len(d["positive_q_cells"]) for d in pair_designs_3d},
        "global_max_abs_effective_side_vs_q": global_max_vs_q,
        "global_max_abs_effective_side_to_side": global_max_side,
        "max_ipf_iterations": max_iters, "max_positive_multiplier_ratio": max_ratio,
        "native_flip_persistence_count_of_3": result["summary"]["native_flip_persistence_count_of_3"],
        "native_flip_clean_removal_count_of_3": result["summary"]["native_flip_clean_removal_count_of_3"],
        "positive_control_pair_stable": result["summary"]["positive_control_pair_stable"],
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
