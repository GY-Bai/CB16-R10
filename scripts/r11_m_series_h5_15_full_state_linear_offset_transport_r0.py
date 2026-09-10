#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Any

from cb16_local_opt.account6_ipf_effective_common_measure_h512r import (
    build_pair_design_h512r,
    classify_h512r,
    prepare_fold_payload_h512r,
    run_pair_h512r,
)
from cb16_local_opt.common_support_rank_geometry_decomposition_h511 import H511_PAIRS
from cb16_local_opt.full_state_linear_offset_transport_h515 import (
    H515_DIM,
    H515_SHIFT_CONTROLS,
    H515_TOL,
    build_side_dataset_h515,
    classify_h515,
    evaluate_forward_pair_h515,
)
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_15_FULL_STATE_LINEAR_OFFSET_TRANSPORT_R0_RESULT_V1"
STATUS = "H5_15_FULL_STATE_LINEAR_OFFSET_TRANSPORT_EXECUTION_COMPLETE"
PREREG_COMMIT = "99509da77e0e09ac91a15f6940fbcb4e1dc69c4d"
GATE_BLOB = "1d73d20b6863629cb11e864bbaa3c09dc524c935"
PARENT_H514_COMMIT = "bfb87e9e1129f533939ba15c8bcba4de52d6d909"
PARENT_H514_BLOB = "e02d453de9a898f4f96d3047cb7d22cae8ff7e8f"
PARENT_H513_COMMIT = "e6301f78be3f7fe1013c1b433bb8ce8b244b3d6b"
PARENT_H513_BLOB = "0851985984839fa880e0014e85ab725f52c2b0f4"
PARENT_H512R_COMMIT = "f772da0663ba7c4b930d0a7fe4ac5ffe703992b2"
PARENT_H512R_BLOB = "4c9887a6649c0009d98507ffcd0f80c1ee0fd017"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", "3c401a0a350984381912f7860181e3e96eb8d7cf"),
    "gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_15_FULL_STATE_LINEAR_OFFSET_TRANSPORT_R0_GATE_V1.json", GATE_BLOB),
    "parent_h514_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_14_TARGET_OM_CONCORDANCE_OCCUPANCY_R0_ADJUDICATION_V1.json", PARENT_H514_BLOB),
    "parent_h513_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_13_WITHIN_CELL_PARTIAL_MAP_TRANSPORT_R0_ADJUDICATION_V1.json", PARENT_H513_BLOB),
    "parent_h512r_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_12R_ACCOUNT6_IPF_EFFECTIVE_COMMON_MEASURE_R0_ADJUDICATION_V1.json", PARENT_H512R_BLOB),
    "h515_helper": ("cb16_local_opt/full_state_linear_offset_transport_h515.py", "a03cca00fed8e366b27ff52e67189d7c70ab5d36"),
    "h515_test": ("tests/test_full_state_linear_offset_transport_h515.py", "97e5799e886a96149fbc203d842cf8be4ae5028e"),
    "h512r_helper": ("cb16_local_opt/account6_ipf_effective_common_measure_h512r.py", "a2950648617ad5327658e0fa1379444b5cb51015"),
    "h5112_helper": ("cb16_local_opt/ipf_effective_common_measure_h5112.py", "e23e939fa43208a948ecf9a3a108fea1831777ce"),
    "h511_helper": ("cb16_local_opt/common_support_rank_geometry_decomposition_h511.py", "918f98d499ff5ed2f36e7d6a6bf3268fca37c27c"),
    "h58_helper": ("cb16_local_opt/operator_conditional_medium_geometry_h58.py", "aaddbad059c7262a8137db87bee25d38c07afd83"),
    "h56_helper": ("cb16_local_opt/time_local_vs_forward_geometry_contrast_h56.py", "a264ff21d0447527c4dda06ee8fd63cf20207daf"),
    "h55_helper": ("cb16_local_opt/state_utility_geometry_transport_audit_h55.py", "1dd00ea9b40b8da4061c0983fc44fb2307c72587"),
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PARENT_H514_COMMIT, PREREG_COMMIT]) == 0, "H515_PREREG_NOT_DESCENDED_FROM_PARENT")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0, "H515_HEAD_NOT_DESCENDED_FROM_PREREG")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H515_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H515_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "parent_h514_adjudication_commit": PARENT_H514_COMMIT,
        "parent_h513_adjudication_commit": PARENT_H513_COMMIT,
        "parent_h512r_adjudication_commit": PARENT_H512R_COMMIT,
        "immutable_blobs": observed,
    }


def pair_key(pair: tuple[int, int], fold: int) -> str:
    return f"{pair[0]}_{pair[1]}_side_{fold}"


def _payload_order_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    groups = list(payload["groups"])
    timestamps = [int(g["timestamp_ms"]) for g in groups]
    unique_ids = [str(g["future_group_id"]) for g in groups]
    six = all(len(g["scenarios"]) == 6 for g in groups)
    chronological = all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))
    unique = len(set(unique_ids)) == len(unique_ids)
    return {
        "fold": int(payload["fold"]),
        "future_group_count": len(groups),
        "strictly_chronological": chronological,
        "unique_future_group_ids": unique,
        "all_groups_have_six_scenarios": six,
        "first_timestamp_ms": timestamps[0] if timestamps else None,
        "last_timestamp_ms": timestamps[-1] if timestamps else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    gate = json.loads(Path(PINNED["gate"][0]).read_text(encoding="utf-8"))
    h514_adj = json.loads(Path(PINNED["parent_h514_adjudication"][0]).read_text(encoding="utf-8"))
    h513_adj = json.loads(Path(PINNED["parent_h513_adjudication"][0]).read_text(encoding="utf-8"))
    h512r_adj = json.loads(Path(PINNED["parent_h512r_adjudication"][0]).read_text(encoding="utf-8"))
    require(gate["status"] == "PREREGISTERED__UNEVALUATED", "H515_GATE_STATUS_DRIFT")
    require(h514_adj["classification"] == gate["parent"]["required_h5_14_classification"], "H515_H514_PARENT_CLASS_DRIFT")
    require(h513_adj["classification"] == gate["parent"]["required_h5_13_classification"], "H515_H513_PARENT_CLASS_DRIFT")
    require(h512r_adj["classification"] == gate["parent_reproduction_gate"]["required_h5_12r_classification"], "H515_H512R_PARENT_CLASS_DRIFT")

    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require([int(x["fold"]) for x in folds] == [1, 2, 3, 4, 5], "H515_FOLD_SET_DRIFT")
    by_fold_spec = {int(f["fold"]): f for f in folds}

    payloads = [prepare_fold_payload_h512r(fold_spec=f, all_train_parents=parents, all_train_samples=samples) for f in folds]
    by_payload = {int(x["fold"]): x for x in payloads}
    order_receipts = [_payload_order_receipt(p) for p in payloads]
    bundle_ok = all(r["strictly_chronological"] and r["unique_future_group_ids"] and r["all_groups_have_six_scenarios"] for r in order_receipts)

    designs = [build_pair_design_h512r(by_payload[a], by_payload[b]) for a, b in H511_PAIRS]
    parent_results = [run_pair_h512r(design=d, payload_a=by_payload[int(d["pair"][0])], payload_b=by_payload[int(d["pair"][1])]) for d in designs]
    parent_summary = classify_h512r(parent_results)

    expected_a = gate["parent_reproduction_gate"]["required_h5_12r_pair_side_aligned_values"]
    expected_n = gate["parent_reproduction_gate"]["required_h5_12r_pair_side_null_medians"]
    tol = float(gate["parent_reproduction_gate"]["tolerance"])
    max_a = 0.0
    max_n = 0.0
    reproduction_rows: dict[str, Any] = {}
    for p in parent_results:
        require(p.get("blocked") is None, f"H515_PARENT_BLOCKED:{p.get('blocked')}")
        pair = tuple(int(x) for x in p["pair"])
        for side_name, fold in (("side_a", pair[0]), ("side_b", pair[1])):
            row = p[side_name]
            key = pair_key(pair, fold)
            ae = abs(float(row["aligned_partial_rho"]) - float(expected_a[key]))
            ne = abs(float(row["null_median_partial_rho"]) - float(expected_n[key]))
            max_a = max(max_a, ae)
            max_n = max(max_n, ne)
            reproduction_rows[key] = {
                "aligned_abs_error": ae,
                "null_median_abs_error": ne,
                "orientation": row["orientation"],
            }
    parent_ok = bool(
        parent_summary["classification"] == gate["parent_reproduction_gate"]["required_h5_12r_classification"]
        and h513_adj["classification"] == gate["parent_reproduction_gate"]["required_h5_13_classification"]
        and h514_adj["classification"] == gate["parent_reproduction_gate"]["required_h5_14_classification"]
        and max_a <= tol
        and max_n <= tol
        and bundle_ok
    )

    pair_results: list[dict[str, Any]] = []
    dataset_receipts: list[dict[str, Any]] = []
    blocked_classification: str | None = None
    if parent_ok:
        for parent in parent_results:
            pair = tuple(int(x) for x in parent["pair"])
            try:
                side_a = build_side_dataset_h515(
                    fold_spec=by_fold_spec[pair[0]],
                    all_train_parents=parents,
                    all_train_samples=samples,
                    payload=by_payload[pair[0]],
                    candidate_multiplier=parent["calibration"]["side_a"]["multiplier"],
                )
                side_b = build_side_dataset_h515(
                    fold_spec=by_fold_spec[pair[1]],
                    all_train_parents=parents,
                    all_train_samples=samples,
                    payload=by_payload[pair[1]],
                    candidate_multiplier=parent["calibration"]["side_b"]["multiplier"],
                )
                require(abs(float(side_a["aligned_mean"]) - float(parent["side_a"]["aligned_partial_rho"])) <= H515_TOL, "H515_SIDE_A_PARENT_ALIGNED_DRIFT")
                require(abs(float(side_b["aligned_mean"]) - float(parent["side_b"]["aligned_partial_rho"])) <= H515_TOL, "H515_SIDE_B_PARENT_ALIGNED_DRIFT")
                require(abs(float(side_a["null_median"]) - float(parent["side_a"]["null_median_partial_rho"])) <= H515_TOL, "H515_SIDE_A_PARENT_NULL_DRIFT")
                require(abs(float(side_b["null_median"]) - float(parent["side_b"]["null_median_partial_rho"])) <= H515_TOL, "H515_SIDE_B_PARENT_NULL_DRIFT")
                dataset_receipts.append({
                    "pair": list(pair),
                    "side_a_group_count": len(side_a["future_group_ids"]),
                    "side_b_group_count": len(side_b["future_group_ids"]),
                    "dimension": int(side_a["x"].shape[2]),
                    "side_a_aligned_reproduction_error": abs(float(side_a["aligned_mean"]) - float(parent["side_a"]["aligned_partial_rho"])),
                    "side_b_aligned_reproduction_error": abs(float(side_b["aligned_mean"]) - float(parent["side_b"]["aligned_partial_rho"])),
                    "side_a_null_median_reproduction_error": abs(float(side_a["null_median"]) - float(parent["side_a"]["null_median_partial_rho"])),
                    "side_b_null_median_reproduction_error": abs(float(side_b["null_median"]) - float(parent["side_b"]["null_median_partial_rho"])),
                    "side_a_zero_information_aligned_rows": side_a["zero_information_aligned_row_count"],
                    "side_b_zero_information_aligned_rows": side_b["zero_information_aligned_row_count"],
                })
                pair_results.append(evaluate_forward_pair_h515(pair=pair, side_a=side_a, side_b=side_b))
            except RuntimeError as exc:
                msg = str(exc)
                if "FULL_STATE_DIM" in msg or "INDEX_DIM" in msg or "FULL_STATE_NONFINITE" in msg:
                    blocked_classification = "EXECUTION_BLOCKED__FULL_STATE_DIMENSION_OR_LAYOUT_DRIFT"
                elif "RIDGE" in msg or "STANDARDIZE" in msg:
                    blocked_classification = "EXECUTION_BLOCKED__RIDGE_NUMERICAL_FAILURE"
                elif "SCENARIO" in msg or "ORDER" in msg or "FUTURE_GROUP" in msg or "SHIFT_TOO_LARGE" in msg:
                    blocked_classification = "EXECUTION_BLOCKED__DEPENDENCE_OR_SCENARIO_BUNDLE_DRIFT"
                elif "PARENT" in msg:
                    blocked_classification = "EXECUTION_BLOCKED__H5_14_H5_13_H5_12R_PARENT_REPRODUCTION_FAILED"
                else:
                    raise
                break

    if not bundle_ok:
        summary = {"classification": "EXECUTION_BLOCKED__DEPENDENCE_OR_SCENARIO_BUNDLE_DRIFT"}
    elif not parent_ok:
        summary = {"classification": "EXECUTION_BLOCKED__H5_14_H5_13_H5_12R_PARENT_REPRODUCTION_FAILED"}
    elif blocked_classification is not None:
        summary = {"classification": blocked_classification}
    else:
        summary = classify_h515(pair_results, True)

    valid_pair_count = len(pair_results)
    summary.update({
        "parent_reproduction_passed": parent_ok,
        "parent_max_aligned_abs_error": max_a,
        "parent_max_null_median_abs_error": max_n,
        "dependence_bundle_gate_passed": bundle_ok,
        "valid_pair_count": valid_pair_count,
        "full_state_dimension": H515_DIM,
        "ridge_alpha": 1.0,
        "negative_control_shifts": list(H515_SHIFT_CONTROLS),
        "market_information_verdict_changed": False,
        "canonical_change_authorized": False,
        "runtime_gate_authorized": False,
    })

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
            "single_scientific_manipulated_variable": gate["single_scientific_manipulated_variable"],
            "full_state_dimension": H515_DIM,
            "ridge_alpha": 1.0,
            "alpha_or_model_search": False,
            "feature_subselection_or_ablation_scan": False,
            "calendar_or_fold_id_input": False,
            "realized_future_or_utility_input": False,
            "teacher_or_student_output_input": False,
            "forward_pair_direction_only": True,
            "same_future_group_is_dependence_unit": True,
            "bundle_shift_controls": list(H515_SHIFT_CONTROLS),
        },
        "parent_reproduction": {
            "passed": parent_ok,
            "h5_12r_classification": parent_summary["classification"],
            "h5_13_classification": h513_adj["classification"],
            "h5_14_classification": h514_adj["classification"],
            "max_aligned_abs_error": max_a,
            "max_null_median_abs_error": max_n,
            "rows": reproduction_rows,
        },
        "dependence_bundle_receipts": order_receipts,
        "dataset_receipts": dataset_receipts,
        "pair_results": pair_results,
        "summary": summary,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "semantic_guards": {
            "alpha_or_model_family_searched": False,
            "feature_subselection_ablation_or_reweighting_searched": False,
            "nonlinear_or_interaction_model_used": False,
            "calendar_or_fold_id_input_used": False,
            "realized_future_or_utility_input_used": False,
            "teacher_or_student_output_input_used": False,
            "structured_null_changed": False,
            "h5_12r_candidate_measure_changed": False,
            "teacher_changed": False,
            "student_changed": False,
            "canonical_architecture_changed": False,
            "medium_or_account_reweighted_as_architecture": False,
            "learned_router_or_gate_created": False,
            "time_or_regime_gate_created": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False,
            "champion_promoted": False,
        },
        "next_legal_step": "ADJUDICATE_H5_15_RESULT_ONLY__NO_ALPHA_FEATURE_NONLINEAR_MODEL_SEARCH_OR_RUNTIME_GATE_BEFORE_FORMAL_ADJUDICATION",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": STATUS, **summary, "next_legal_step": result["next_legal_step"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
