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
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from cb16_local_opt.target_om_concordance_occupancy_h514 import (
    H514_TOL,
    build_target_pair_design_h514,
    classify_h514,
    run_pair_h514,
    target_kappa_payload_h514,
)
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_14_TARGET_OM_CONCORDANCE_OCCUPANCY_R0_RESULT_V1"
STATUS = "H5_14_TARGET_OM_CONCORDANCE_OCCUPANCY_EXECUTION_COMPLETE"
PREREG_COMMIT = "1b1c300cb0b8746dab4ee09fb92e84f63d8ec75e"
GATE_BLOB = "a23ecc6df8de327c218aeae58f98abe5eef03dd6"
PARENT_H513_COMMIT = "e6301f78be3f7fe1013c1b433bb8ce8b244b3d6b"
PARENT_H513_BLOB = "0851985984839fa880e0014e85ab725f52c2b0f4"
PARENT_H512R_COMMIT = "f772da0663ba7c4b930d0a7fe4ac5ffe703992b2"
PARENT_H512R_BLOB = "4c9887a6649c0009d98507ffcd0f80c1ee0fd017"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", "3c401a0a350984381912f7860181e3e96eb8d7cf"),
    "gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_14_TARGET_OM_CONCORDANCE_OCCUPANCY_R0_GATE_V1.json", GATE_BLOB),
    "parent_h513_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_13_WITHIN_CELL_PARTIAL_MAP_TRANSPORT_R0_ADJUDICATION_V1.json", PARENT_H513_BLOB),
    "parent_h512r_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_12R_ACCOUNT6_IPF_EFFECTIVE_COMMON_MEASURE_R0_ADJUDICATION_V1.json", PARENT_H512R_BLOB),
    "h514_helper": ("cb16_local_opt/target_om_concordance_occupancy_h514.py", "f075928fdc97e921e630928a794003338ce5bf3e"),
    "h514_test": ("tests/test_target_om_concordance_occupancy_h514.py", "fbbf9103f494e886ba9cde4195c24d9a1e56b42e"),
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PARENT_H513_COMMIT, PREREG_COMMIT]) == 0, "H514_PREREG_NOT_DESCENDED_FROM_PARENT")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0, "H514_HEAD_NOT_DESCENDED_FROM_PREREG")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H514_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H514_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "parent_h513_adjudication_commit": PARENT_H513_COMMIT,
        "parent_h512r_adjudication_commit": PARENT_H512R_COMMIT,
        "immutable_blobs": observed,
    }


def pair_key(pair: tuple[int, int], fold: int) -> str:
    return f"{pair[0]}_{pair[1]}_side_{fold}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    gate = json.loads(Path(PINNED["gate"][0]).read_text(encoding="utf-8"))
    h513_adj = json.loads(Path(PINNED["parent_h513_adjudication"][0]).read_text(encoding="utf-8"))
    h512r_adj = json.loads(Path(PINNED["parent_h512r_adjudication"][0]).read_text(encoding="utf-8"))
    require(gate["status"] == "PREREGISTERED__UNEVALUATED", "H514_GATE_STATUS_DRIFT")
    require(h513_adj["classification"] == gate["parent"]["required_parent_classification"], "H514_H513_PARENT_CLASS_DRIFT")
    require(h512r_adj["classification"] == gate["parent_reproduction_gate"]["required_parent_h5_12r_classification"], "H514_H512R_PARENT_CLASS_DRIFT")

    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require([int(x["fold"]) for x in folds] == [1,2,3,4,5], "H514_FOLD_SET_DRIFT")

    payloads = [prepare_fold_payload_h512r(fold_spec=f, all_train_parents=parents, all_train_samples=samples) for f in folds]
    by_payload = {int(x["fold"]): x for x in payloads}
    designs_27 = [build_pair_design_h512r(by_payload[a], by_payload[b]) for a, b in H511_PAIRS]
    parent_results = [run_pair_h512r(design=d, payload_a=by_payload[int(d["pair"][0])], payload_b=by_payload[int(d["pair"][1])]) for d in designs_27]
    parent_summary = classify_h512r(parent_results)

    expected_a = gate["parent_reproduction_gate"]["required_h5_12r_pair_side_aligned_values"]
    expected_n = gate["parent_reproduction_gate"]["required_h5_12r_pair_side_null_medians"]
    tol = float(gate["parent_reproduction_gate"]["tolerance"])
    max_a = 0.0; max_n = 0.0; reproduction_rows: dict[str, Any] = {}
    for p in parent_results:
        require(p.get("blocked") is None, f"H514_PARENT_BLOCKED:{p.get('blocked')}")
        pair = tuple(int(x) for x in p["pair"])
        for side, fold in (("side_a", pair[0]), ("side_b", pair[1])):
            row = p[side]; key = pair_key(pair, fold)
            ae = abs(float(row["aligned_partial_rho"]) - float(expected_a[key]))
            ne = abs(float(row["null_median_partial_rho"]) - float(expected_n[key]))
            max_a = max(max_a, ae); max_n = max(max_n, ne)
            reproduction_rows[key] = {"aligned_abs_error": ae, "null_median_abs_error": ne, "orientation": row["orientation"]}
    parent_rep_pass = bool(
        parent_summary["classification"] == gate["parent_reproduction_gate"]["required_parent_h5_12r_classification"]
        and h513_adj["classification"] == gate["parent_reproduction_gate"]["required_parent_h5_13_classification"]
        and max_a <= tol and max_n <= tol
    )

    kappa_payloads: list[dict[str, Any]] = []
    target_designs: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    if parent_rep_pass:
        kappa_payloads = [target_kappa_payload_h514(p) for p in payloads]
        by_kappa = {int(x["fold"]): x for x in kappa_payloads}
        target_designs = [build_target_pair_design_h514(by_kappa[a], by_kappa[b]) for a, b in H511_PAIRS]
        pair_results = [
            run_pair_h514(
                target_design=td,
                h512r_pair_result=pr,
                payload_a=by_payload[int(td["pair"][0])],
                payload_b=by_payload[int(td["pair"][1])],
            )
            for td, pr in zip(target_designs, parent_results)
        ]
        summary = classify_h514(pair_results)
    else:
        summary = {
            "classification": "EXECUTION_BLOCKED__H5_13_H5_12R_PARENT_REPRODUCTION_FAILED",
            "native_flip_persistence_count_of_3": 0,
            "native_flip_clean_removal_count_of_3": 0,
            "positive_control_pair_stable": False,
        }

    valid = [x for x in pair_results if x.get("blocked") is None]
    summary.update({
        "parent_reproduction_passed": parent_rep_pass,
        "parent_max_aligned_abs_error": max_a,
        "parent_max_null_median_abs_error": max_n,
        "max_target_scenario_kappa_spread": max([float(x["max_scenario_kappa_spread"]) for x in kappa_payloads], default=None),
        "all_target_pair_support_valid": bool(target_designs and all(bool(x["common_support_valid"]) for x in target_designs)),
        "max_candidate_effective_margin_error_vs_q27": max([float(p[s]["effective_candidate_max_abs_error_vs_q27"]) for p in valid for s in ("side_a","side_b")], default=None),
        "max_candidate_side_to_side_margin_difference": max([float(p["candidate_side_to_side_max_abs"]) for p in valid], default=None),
        "max_candidate_ipf_iterations": max([int(p["candidate_calibration"][s]["iterations"]) for p in valid for s in ("side_a","side_b")], default=0),
        "market_information_verdict_changed": False,
        "canonical_change_authorized": False,
        "runtime_gate_authorized": False,
    })

    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "repo": repo,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "github_run_id": os.environ.get("GITHUB_RUN_ID"), "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")},
        "support_receipt": support_receipt,
        "protocol": {
            "single_scientific_manipulated_variable": gate["single_scientific_manipulated_variable"],
            "target_context_scalar": "TARGET_OPERATOR_MEDIUM_NEIGHBORHOOD_RANK_CONCORDANCE_KAPPA",
            "utility_used_to_construct_kappa_or_target_measure": False,
            "calendar_or_fold_id_used_as_runtime_feature": False,
            "other_state_summary_scanned": False,
            "candidate_h5_12r_q27_preserved": True,
            "same_target_and_candidate_weights_used_for_all_nulls": True,
            "same_future_group_is_dependence_unit": True,
        },
        "parent_reproduction": {
            "passed": parent_rep_pass,
            "h5_12r_classification": parent_summary["classification"],
            "h5_13_classification": h513_adj["classification"],
            "max_aligned_abs_error": max_a,
            "max_null_median_abs_error": max_n,
            "rows": reproduction_rows,
        },
        "kappa_fold_receipts": kappa_payloads,
        "target_pair_designs": target_designs,
        "pair_results": pair_results,
        "summary": summary,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "semantic_guards": {
            "kappa_definition_changed": False,
            "multiple_state_summaries_scanned": False,
            "target_bins_changed_or_merged": False,
            "h5_12r_q27_changed": False,
            "support_or_normalization_changed": False,
            "structured_null_changed": False,
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
        "next_legal_step": "ADJUDICATE_H5_14_RESULT_ONLY__NO_RUNTIME_GATE_OR_ALTERNATE_STATE_SUMMARY_SEARCH_BEFORE_FORMAL_ADJUDICATION"
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": STATUS, **summary, "next_legal_step": result["next_legal_step"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
