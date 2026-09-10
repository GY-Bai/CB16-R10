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
from cb16_local_opt.within_cell_partial_map_transport_h513 import (
    H513_TOL,
    classify_h513,
    evaluate_pair_transport_h513,
    evaluate_payload_maps_h513,
)
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_13_WITHIN_CELL_PARTIAL_MAP_TRANSPORT_R0_RESULT_V1"
STATUS = "H5_13_WITHIN_CELL_PARTIAL_MAP_TRANSPORT_EXECUTION_COMPLETE"
PREREG_COMMIT = "93514d54b9cdaff0715f4ce1a499629c615a3b0c"
GATE_BLOB = "b7473f09aad04d111f30db107d0a94d20774585c"
PARENT_ADJ_COMMIT = "f772da0663ba7c4b930d0a7fe4ac5ffe703992b2"
PARENT_ADJ_BLOB = "4c9887a6649c0009d98507ffcd0f80c1ee0fd017"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", "3c401a0a350984381912f7860181e3e96eb8d7cf"),
    "gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_13_WITHIN_CELL_PARTIAL_MAP_TRANSPORT_R0_GATE_V1.json", GATE_BLOB),
    "parent_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_12R_ACCOUNT6_IPF_EFFECTIVE_COMMON_MEASURE_R0_ADJUDICATION_V1.json", PARENT_ADJ_BLOB),
    "h513_helper": ("cb16_local_opt/within_cell_partial_map_transport_h513.py", "1339c1fa6f8bb1ff37f34e7ce1106724ea17ec4a"),
    "h513_test": ("tests/test_within_cell_partial_map_transport_h513.py", "145b27d0f9bff3c70101570a9a572083da4bfef8"),
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PARENT_ADJ_COMMIT, PREREG_COMMIT]) == 0, "H513_PREREG_NOT_DESCENDED_FROM_PARENT")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0, "H513_HEAD_NOT_DESCENDED_FROM_PREREG")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H513_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H513_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "parent_adjudication_commit": PARENT_ADJ_COMMIT,
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
    parent_adj = json.loads(Path(PINNED["parent_adjudication"][0]).read_text(encoding="utf-8"))
    require(gate["status"] == "PREREGISTERED__UNEVALUATED", "H513_GATE_STATUS_DRIFT")
    require(parent_adj["classification"] == gate["parent"]["required_parent_classification"], "H513_PARENT_CLASS_DRIFT")

    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require([int(x["fold"]) for x in folds] == [1,2,3,4,5], "H513_FOLD_SET_DRIFT")

    payloads = [prepare_fold_payload_h512r(fold_spec=f, all_train_parents=parents, all_train_samples=samples) for f in folds]
    by_payload = {int(x["fold"]): x for x in payloads}
    designs = [build_pair_design_h512r(by_payload[a], by_payload[b]) for a, b in H511_PAIRS]
    parent_results = [run_pair_h512r(design=d, payload_a=by_payload[int(d["pair"][0])], payload_b=by_payload[int(d["pair"][1])]) for d in designs]
    parent_summary = classify_h512r(parent_results)

    expected_a = gate["parent_reproduction_gate"]["required_h5_12r_pair_side_aligned_values"]
    expected_n = gate["parent_reproduction_gate"]["required_h5_12r_pair_side_null_medians"]
    tol = float(gate["parent_reproduction_gate"]["tolerance"])
    max_a = 0.0; max_n = 0.0
    reproduction_rows: dict[str, Any] = {}
    for p in parent_results:
        require(p.get("blocked") is None, f"H513_PARENT_BLOCKED:{p.get('blocked')}")
        pair = tuple(int(x) for x in p["pair"])
        for side, fold in (("side_a", pair[0]), ("side_b", pair[1])):
            row = p[side]
            key = pair_key(pair, fold)
            ae = abs(float(row["aligned_partial_rho"]) - float(expected_a[key]))
            ne = abs(float(row["null_median_partial_rho"]) - float(expected_n[key]))
            max_a = max(max_a, ae); max_n = max(max_n, ne)
            reproduction_rows[key] = {"aligned_abs_error": ae, "null_median_abs_error": ne, "orientation": row["orientation"]}
    parent_reproduction_passed = bool(
        parent_summary["classification"] == gate["parent_reproduction_gate"]["required_parent_classification"]
        and max_a <= tol and max_n <= tol
    )

    pair_rows: list[dict[str, Any]] = []
    blocked_classification: str | None = None
    if parent_reproduction_passed:
        for design, parent_result in zip(designs, parent_results):
            pair = tuple(int(x) for x in design["pair"])
            q = parent_result["q"]
            try:
                side_a = evaluate_payload_maps_h513(by_payload[pair[0]], parent_result["calibration"]["side_a"]["multiplier"], q)
                side_b = evaluate_payload_maps_h513(by_payload[pair[1]], parent_result["calibration"]["side_b"]["multiplier"], q)
                require(abs(side_a["aligned_partial_rho"] - parent_result["side_a"]["aligned_partial_rho"]) <= H513_TOL, "H513_SIDE_A_PARENT_RHO_DRIFT")
                require(abs(side_b["aligned_partial_rho"] - parent_result["side_b"]["aligned_partial_rho"]) <= H513_TOL, "H513_SIDE_B_PARENT_RHO_DRIFT")
                require(abs(statistics.median(side_a["null_partial_rho_by_shift"].values()) - parent_result["side_a"]["null_median_partial_rho"]) <= H513_TOL, "H513_SIDE_A_PARENT_NULL_DRIFT")
                require(abs(statistics.median(side_b["null_partial_rho_by_shift"].values()) - parent_result["side_b"]["null_median_partial_rho"]) <= H513_TOL, "H513_SIDE_B_PARENT_NULL_DRIFT")
                transport = evaluate_pair_transport_h513(side_a, side_b, q)
                pair_rows.append({"pair": list(pair), "q": q, "side_a": side_a, "side_b": side_b, "transport": transport})
            except RuntimeError as e:
                msg = str(e)
                if "RECONSTRUCTION" in msg:
                    blocked_classification = "EXECUTION_BLOCKED__PARTIAL_MAP_DECOMPOSITION_RECONSTRUCTION_FAILED"
                elif "CELL_RATE_MAP_DEGENERATE" in msg:
                    blocked_classification = "EXECUTION_BLOCKED__CELL_RATE_MAP_DEGENERATE"
                elif "ORTHOGONAL_IDENTITY" in msg:
                    blocked_classification = "EXECUTION_BLOCKED__ORTHOGONAL_DECOMPOSITION_IDENTITY_FAILED"
                else:
                    raise
                break

    if not parent_reproduction_passed:
        summary = {"classification": "EXECUTION_BLOCKED__H5_12R_PARENT_REPRODUCTION_FAILED"}
    elif blocked_classification is not None:
        summary = {"classification": blocked_classification}
    else:
        summary = classify_h513(pair_rows, True)

    summary.update({
        "parent_reproduction_passed": parent_reproduction_passed,
        "parent_max_aligned_abs_error": max_a,
        "parent_max_null_median_abs_error": max_n,
        "max_row_reconstruction_error": max([float(s["max_row_reconstruction_error"]) for p in pair_rows for s in (p["side_a"], p["side_b"])], default=None),
        "max_fold_reconstruction_error": max([float(s["fold_reconstruction_error"]) for p in pair_rows for s in (p["side_a"], p["side_b"])], default=None),
        "max_orthogonal_identity_error": max([float(p["transport"]["orthogonal_identity_error"]) for p in pair_rows], default=None),
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
            "single_manipulated_variable": gate["single_manipulated_variable"],
            "new_representation_created": False,
            "new_estimator_fit": False,
            "h5_12r_q_or_ipf_changed": False,
            "support_or_normalization_changed": False,
            "structured_null_changed": False,
            "calendar_or_fold_id_runtime_feature_used": False,
            "same_future_group_is_dependence_unit": True
        },
        "h5_12r_parent_reproduction": {
            "passed": parent_reproduction_passed,
            "classification": parent_summary["classification"],
            "max_aligned_abs_error": max_a,
            "max_null_median_abs_error": max_n,
            "rows": reproduction_rows
        },
        "pair_results": pair_rows,
        "summary": summary,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "semantic_guards": {
            "teacher_changed": False,
            "student_changed": False,
            "canonical_architecture_changed": False,
            "medium_or_account_reweighted": False,
            "learned_router_or_gate_created": False,
            "time_or_regime_gate_created": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False,
            "champion_promoted": False
        },
        "next_legal_step": "ADJUDICATE_H5_13_RESULT_ONLY__NO_FULL_STATE_MATCHER_OR_RUNTIME_GATE_BEFORE_FORMAL_ADJUDICATION"
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": STATUS, **summary, "next_legal_step": result["next_legal_step"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
