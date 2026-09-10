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

from cb16_local_opt.common_support_rank_geometry_decomposition_h511 import (
    H511_PAIRS,
    build_pair_overlap_design_h511,
    prepare_fold_payload_h511,
    run_fold_design_h511,
)
from cb16_local_opt.ipf_effective_common_measure_h5112 import (
    H5112_FORMAL_TOL,
    H5112_SIDE_TO_SIDE_TOL,
    classify_h5112,
    evaluate_fold_multiplier_h5112,
    run_pair_h5112,
)
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_11_2_IPF_EFFECTIVE_COMMON_MEASURE_R0_RESULT_V1"
STATUS = "H5_11_2_IPF_EFFECTIVE_COMMON_MEASURE_EXECUTION_COMPLETE"
PREREG_COMMIT = "7446e07238e8d2575194b50cf9e8890d08d59607"
GATE_BLOB = "92bb7001070e31669f6765dfb05e598926683452"
PARENT_ADJ_COMMIT = "dca61e693858d90af785dfb7697d399df4eb65c3"
PARENT_ADJ_BLOB = "d4be3239da97e89b99f9992068ce13490d86cabf"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", "3c401a0a350984381912f7860181e3e96eb8d7cf"),
    "gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_11_2_IPF_EFFECTIVE_COMMON_MEASURE_R0_GATE_V1.json", GATE_BLOB),
    "parent_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_11_1_EFFECTIVE_MEASURE_CLOSURE_R0_ADJUDICATION_V1.json", PARENT_ADJ_BLOB),
    "h511_helper": ("cb16_local_opt/common_support_rank_geometry_decomposition_h511.py", "918f98d499ff5ed2f36e7d6a6bf3268fca37c27c"),
    "h5112_helper": ("cb16_local_opt/ipf_effective_common_measure_h5112.py", "e23e939fa43208a948ecf9a3a108fea1831777ce"),
    "h5112_test": ("tests/test_ipf_effective_common_measure_h5112.py", "0c2abc3e15df5dca07766daa5efc7cfdc6f671fd"),
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PARENT_ADJ_COMMIT, PREREG_COMMIT]) == 0, "H5112_PREREG_NOT_DESCENDED_FROM_PARENT")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0, "H5112_HEAD_NOT_DESCENDED_FROM_PREREG")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H5112_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H5112_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "parent_adjudication_commit": PARENT_ADJ_COMMIT,
        "immutable_blobs": observed,
    }


def load_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    gate = json.loads(Path(PINNED["gate"][0]).read_text(encoding="utf-8"))
    parent = json.loads(Path(PINNED["parent_adjudication"][0]).read_text(encoding="utf-8"))
    require(gate["status"] == "PREREGISTERED__UNEVALUATED", "H5112_GATE_STATUS_DRIFT")
    require(parent["adjudication"]["classification"] == "H5_11_EXACT_EFFECTIVE_COMMON_MEASURE_CLAIM_FALSE__SHARED_CELL_MULTIPLIER_ROBUSTNESS_ONLY", "H5112_PARENT_CLASS_DRIFT")
    require(parent["frozen_scientific_status"]["after"] == FROZEN_STATUS, "H5112_PARENT_MARKET_STATUS_DRIFT")
    require(parent["authority"]["h5_12_execution_authorized"] is False, "H5112_PARENT_H512_AUTHORITY_DRIFT")
    return gate, parent


def _pair_key(pair: tuple[int, int], fold: int) -> str:
    return f"{pair[0]}_{pair[1]}_side_{fold}"


def reproduce_old_h511_sibling(
    *,
    pair_designs: list[Mapping[str, Any]],
    payload_by_fold: Mapping[int, Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    expected_a = gate["h5_11_sibling_evaluator_reproduction_gate"]["required_pair_side_aligned_partial_rho"]
    expected_n = gate["h5_11_sibling_evaluator_reproduction_gate"]["required_pair_side_null_median"]
    tol = float(gate["h5_11_sibling_evaluator_reproduction_gate"]["tolerance"])
    rows: dict[str, Any] = {}
    max_a = 0.0
    max_n = 0.0
    for design in pair_designs:
        pair = tuple(int(x) for x in design["pair"])
        q = np.asarray(design["q"], dtype=np.float64)
        for side, fold, field in (
            ("side_a", pair[0], "multiplier_a"),
            ("side_b", pair[1], "multiplier_b"),
        ):
            observed = evaluate_fold_multiplier_h5112(payload_by_fold[fold], design[field], q)
            key = _pair_key(pair, fold)
            ae = abs(float(observed["aligned_partial_rho"]) - float(expected_a[key]))
            ne = abs(float(observed["null_median_partial_rho"]) - float(expected_n[key]))
            max_a = max(max_a, ae)
            max_n = max(max_n, ne)
            rows[key] = {
                "pair": list(pair),
                "fold": fold,
                "side": side,
                "observed_aligned_partial_rho": float(observed["aligned_partial_rho"]),
                "expected_aligned_partial_rho": float(expected_a[key]),
                "aligned_abs_error": float(ae),
                "observed_null_median_partial_rho": float(observed["null_median_partial_rho"]),
                "expected_null_median_partial_rho": float(expected_n[key]),
                "null_median_abs_error": float(ne),
                "orientation": str(observed["orientation"]),
            }
    passed = bool(max_a <= tol and max_n <= tol)
    return {
        "passed": passed,
        "tolerance": tol,
        "max_aligned_abs_error": float(max_a),
        "max_null_median_abs_error": float(max_n),
        "rows": rows,
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
    require([int(x["fold"]) for x in folds] == [1, 2, 3, 4, 5], "H5112_OUTER_FOLD_SET_DRIFT")

    designs = [
        run_fold_design_h511(fold_spec=f, all_train_parents=parents, all_train_samples=samples)
        for f in folds
    ]
    by_design = {int(x["fold"]): x for x in designs}
    pair_designs = [build_pair_overlap_design_h511(by_design[a], by_design[b]) for a, b in H511_PAIRS]
    require(all(bool(x["common_support_valid"]) for x in pair_designs), "H5112_PARENT_COMMON_SUPPORT_DRIFT")
    require(all(np.all(np.asarray(x["q"], dtype=np.float64) > 0.0) for x in pair_designs), "H5112_Q_NOT_FULL_NINE_CELL_SUPPORT")

    payloads = [
        prepare_fold_payload_h511(fold_spec=f, all_train_parents=parents, all_train_samples=samples)
        for f in folds
    ]
    by_payload = {int(x["fold"]): x for x in payloads}

    reproduction = reproduce_old_h511_sibling(
        pair_designs=pair_designs, payload_by_fold=by_payload, gate=gate
    )

    pair_results: list[dict[str, Any]] = []
    if reproduction["passed"]:
        pair_results = [
            run_pair_h5112(
                pair_design=d,
                payload_a=by_payload[int(d["pair"][0])],
                payload_b=by_payload[int(d["pair"][1])],
            )
            for d in pair_designs
        ]
        summary = classify_h5112(pair_results, reproduction_passed=True)
    else:
        summary = {
            "classification": "EXECUTION_BLOCKED__H5_11_SIBLING_EVALUATOR_REPRODUCTION_FAILED",
            "conclusion": "H5_11_2_EXECUTION_BLOCKED__H5_11_SIBLING_EVALUATOR_REPRODUCTION_FAILED",
            "native_flip_persistence_count_of_3": 0,
            "native_flip_clean_removal_count_of_3": 0,
            "positive_control_pair_stable": False,
            "market_information_verdict_changed": False,
            "canonical_change_authorized": False,
            "h5_12_execution_authorized": False,
        }

    calibrated_rows = [x for p in pair_results for x in (p["side_a"], p["side_b"])]
    calibration_rows = [x for p in pair_results for x in (p["calibration"]["side_a"], p["calibration"]["side_b"])]
    global_max_effective_vs_q = max([float(x["effective_max_abs_error_vs_q"]) for x in calibrated_rows], default=float("nan"))
    global_max_side_diff = max([float(x["effective_side_to_side_max_abs"]) for x in pair_results], default=float("nan"))
    max_iterations = max([int(x["iterations"]) for x in calibration_rows], default=0)
    max_multiplier_ratio = max([float(x["multiplier_max_to_min_ratio"]) for x in calibration_rows], default=float("nan"))

    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "repo": repo,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        },
        "support_receipt": support_receipt,
        "protocol": {
            "single_manipulated_variable": gate["single_manipulated_variable"],
            "pairwise_q_unchanged_from_h5_11": True,
            "grid_unchanged": True,
            "support_unchanged": True,
            "normalization_unchanged": True,
            "distance_metric_unchanged": True,
            "primary_rank_statistic_unchanged": True,
            "structured_null_unchanged": True,
            "ipf_uses_utility_or_outcomes": False,
            "same_calibrated_weights_reused_for_all_nulls": True,
            "same_future_group_is_dependence_unit": True,
            "formal_effective_margin_tolerance": H5112_FORMAL_TOL,
            "formal_side_to_side_tolerance": H5112_SIDE_TO_SIDE_TOL,
        },
        "h5_11_sibling_reproduction": reproduction,
        "fold_designs": designs,
        "pair_designs": pair_designs,
        "pair_results": pair_results,
        "summary": {
            **summary,
            "global_max_abs_effective_side_vs_q": global_max_effective_vs_q,
            "global_max_abs_effective_side_to_side": global_max_side_diff,
            "max_ipf_iterations": int(max_iterations),
            "max_multiplier_max_to_min_ratio": max_multiplier_ratio,
        },
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "semantic_guards": {
            "h5_11_q_changed_or_tuned": False,
            "h5_11_grid_changed": False,
            "h5_11_support_changed": False,
            "h5_11_normalization_changed": False,
            "h5_11_distance_metric_changed": False,
            "h5_11_primary_statistic_changed": False,
            "h5_11_structured_null_changed": False,
            "account6_extension_executed": False,
            "h5_12_executed": False,
            "teacher_changed": False,
            "student_changed": False,
            "canonical_architecture_changed": False,
            "medium48_deleted_or_reweighted": False,
            "fusion_weight_search_executed": False,
            "learned_router_or_gate_created": False,
            "time_or_regime_gate_created": False,
            "calendar_or_fold_id_used_as_runtime_feature": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False,
            "champion_promoted": False,
        },
        "next_legal_step": "ADJUDICATE_H5_11_2_RESULT__KEEP_H5_12_HELD_UNTIL_FORMAL_ADJUDICATION__NO_CANONICAL_OR_GATE_CHANGE",
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "status": STATUS,
        "classification": result["summary"]["classification"],
        "reproduction_passed": reproduction["passed"],
        "max_reproduction_aligned_error": reproduction["max_aligned_abs_error"],
        "max_reproduction_null_median_error": reproduction["max_null_median_abs_error"],
        "global_max_abs_effective_side_vs_q": global_max_effective_vs_q,
        "global_max_abs_effective_side_to_side": global_max_side_diff,
        "max_ipf_iterations": max_iterations,
        "max_multiplier_max_to_min_ratio": max_multiplier_ratio,
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
