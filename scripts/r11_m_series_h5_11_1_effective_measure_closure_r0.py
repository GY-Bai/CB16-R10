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

from cb16_local_opt.h511_effective_measure_closure_h5111 import H5111_TOL, run_h5111
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_11_1_EFFECTIVE_MEASURE_CLOSURE_R0_RESULT_V1"
PREREG_COMMIT = "0eaa3f705308f0a8247aaa60eb87e4c3f7be9f56"
GATE_BLOB = "734bee34eeecdbcaf649d3b07a6db05470f11579"
H511_ADJ_COMMIT = "ffa3c1c1c9277e6ed50c66dc66aff0079a0cf8ee"
H511_ADJ_BLOB = "58f9c604cab90f61853bf6bb8594371485b21caf"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", "3c401a0a350984381912f7860181e3e96eb8d7cf"),
    "gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_11_1_EFFECTIVE_MEASURE_CLOSURE_R0_GATE_V1.json", GATE_BLOB),
    "h511_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_11_COMMON_SUPPORT_RANK_GEOMETRY_MECHANISM_DECOMPOSITION_R0_ADJUDICATION_V1.json", H511_ADJ_BLOB),
    "h511_helper": ("cb16_local_opt/common_support_rank_geometry_decomposition_h511.py", "918f98d499ff5ed2f36e7d6a6bf3268fca37c27c"),
    "h5111_helper": ("cb16_local_opt/h511_effective_measure_closure_h5111.py", "7063d52784880bb24555e7b177d5d86cb892fc15"),
    "h5111_test": ("tests/test_h511_effective_measure_closure_h5111.py", "8ed1928e0dd329fc0817c186c88d959b0d5e18bc"),
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", H511_ADJ_COMMIT, PREREG_COMMIT]) == 0, "H5111_PREREG_NOT_DESCENDED_FROM_H511_ADJUDICATION")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0, "H5111_NOT_DESCENDED_FROM_PREREGISTRATION")
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H5111_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    require(git("rev-parse", f"{PREREG_COMMIT}:{PINNED['gate'][0]}") == GATE_BLOB, "H5111_PREREG_GATE_BLOB_DRIFT")
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "h511_adjudication_commit": H511_ADJ_COMMIT,
        "immutable_blobs": observed,
    }


def load_parent_authority() -> dict[str, Any]:
    x = json.loads(Path(PINNED["h511_adjudication"][0]).read_text(encoding="utf-8"))
    require(x["adjudication"]["classification"] == "COARSENED_COMMON_SUPPORT_CONDITIONAL_MAPPING_INSTABILITY_SUPPORTED", "H5111_PARENT_CLASS_DRIFT")
    require(x["source"]["execution_head"] == "209b3096fd563b51b8ba96609ac12e78768da5af", "H5111_PARENT_EXECUTION_HEAD_DRIFT")
    require(x["authority"]["canonical_change_authorized"] is False, "H5111_PARENT_CANONICAL_AUTHORITY_DRIFT")
    require(x["frozen_scientific_status"]["after"] == FROZEN_STATUS, "H5111_PARENT_MARKET_STATUS_DRIFT")
    return x


def expected_q_by_pair(parent: dict[str, Any]) -> dict[tuple[int, int], np.ndarray]:
    p = parent["pair_receipts"]
    return {
        (1, 2): np.asarray(p["fold_1_to_fold_2"]["q"], dtype=np.float64),
        (2, 3): np.asarray(p["fold_2_to_fold_3"]["q"], dtype=np.float64),
        (3, 4): np.asarray(p["fold_3_to_fold_4"]["q"], dtype=np.float64),
        (4, 5): np.asarray(p["fold_4_to_fold_5_positive_control"]["q"], dtype=np.float64),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    parent = load_parent_authority()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require([int(x["fold"]) for x in folds] == [1, 2, 3, 4, 5], "H5111_OUTER_FOLD_SET_DRIFT")

    audit = run_h5111(fold_specs=folds, all_train_parents=parents, all_train_samples=samples)
    expected = expected_q_by_pair(parent)
    q_errors: dict[str, float] = {}
    for d in audit["pair_designs"]:
        pair = tuple(int(x) for x in d["pair"])
        require(pair in expected, f"H5111_UNEXPECTED_PAIR:{pair}")
        err = float(np.max(np.abs(np.asarray(d["q"], dtype=np.float64) - expected[pair])))
        q_errors[f"{pair[0]}_{pair[1]}"] = err
        require(err <= H5111_TOL, f"H5111_FROZEN_Q_REPRODUCTION_FAIL:{pair}:{err}")

    max_side_q = max(
        max(float(p["side_a"]["distance_vs_q"]["max_abs"]), float(p["side_b"]["distance_vs_q"]["max_abs"]))
        for p in audit["pair_audits"]
    )
    max_side_to_side = max(float(p["effective_side_to_side_distance"]["max_abs"]) for p in audit["pair_audits"])
    max_tv_side_q = max(
        max(float(p["side_a"]["distance_vs_q"]["total_variation"]), float(p["side_b"]["distance_vs_q"]["total_variation"]))
        for p in audit["pair_audits"]
    )
    max_tv_side_to_side = max(float(p["effective_side_to_side_distance"]["total_variation"]) for p in audit["pair_audits"])

    result = {
        "schema": SCHEMA,
        "status": "H5_11_1_EFFECTIVE_MEASURE_CLOSURE_AUDIT_COMPLETE",
        "repo": repo,
        "runtime_identity": {
            "python": platform.python_version(),
            "teacher_compilation": False,
            "teacher_kernel": False,
            "teacher_support_selection": False,
            "student_training": False,
            "student_inference": False,
        },
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "parent_h5_11": {
            "classification": parent["adjudication"]["classification"],
            "conclusion": parent["adjudication"]["conclusion"],
            "execution_head": parent["source"]["execution_head"],
        },
        "support": support_receipt,
        "protocol": {
            "audit_type": "OUTCOME_UNUSED_AUTHORITY_CLOSURE_AUDIT__NO_NEW_MARKET_INFORMATION_TEST",
            "utility_values_used_by_audit_computation": False,
            "historical_support_loader_contains_existing_utility_records_but_audit_functions_never_reference_them": True,
            "nominal_receipt": "NORMALIZE_P_E_TIMES_CELL_MULTIPLIER",
            "effective_measure": "EQUAL_MACRO_AVERAGE_OF_LOCALLY_NORMALIZED_CANDIDATE_WEIGHTS",
            "exact_tolerance": H5111_TOL,
            "approximate_pass_threshold_defined": False,
            "same_future_group_is_dependence_unit": True,
            "fresh_market_data_downloaded": False,
            "final_holdout_opened": False,
            "r7_candidate_evaluated": False,
        },
        "frozen_q_reproduction_max_abs_error_by_pair": q_errors,
        "h5_11_1_audit": audit,
        "summary": {
            "classification": audit["classification"],
            "conclusion": audit["conclusion"],
            "exact_closure_all_pairs": bool(audit["exact_closure_all_pairs"]),
            "global_max_abs_effective_side_vs_q": float(max_side_q),
            "global_max_abs_effective_side_to_side": float(max_side_to_side),
            "global_max_total_variation_effective_side_vs_q": float(max_tv_side_q),
            "global_max_total_variation_effective_side_to_side": float(max_tv_side_to_side),
            "h5_11_orientation_numbers_invalidated": False,
            "h5_11_exact_common_measure_interpretation_confirmed": bool(audit["exact_closure_all_pairs"]),
            "market_information_verdict_changed": False,
            "canonical_change_authorized": False,
            "h5_12_execution_authorized": False,
        },
        "semantic_guards": {
            "utility_values_referenced_by_audit_computation": False,
            "h5_11_q_tuned_or_changed": False,
            "h5_11_grid_changed": False,
            "h5_11_support_changed": False,
            "h5_11_distance_metric_changed": False,
            "h5_11_orientation_statistic_recomputed_for_new_scientific_claim": False,
            "h5_12_executed": False,
            "teacher_changed": False,
            "student_changed": False,
            "canonical_architecture_changed": False,
            "learned_gate_created": False,
            "time_or_regime_gate_created": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False,
        },
        "next_legal_step": "ADJUDICATE_H5_11_1_AUTHORITY_EFFECT__KEEP_H5_12_HELD__IF_EXACT_CLOSURE_FAILS_DESIGN_A_SEPARATELY_PREREGISTERED_CORRECTED_STANDARDIZATION_TEST_BEFORE_ANY_ACCOUNT6_EXTENSION",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "classification": audit["classification"],
        "exact_closure_all_pairs": audit["exact_closure_all_pairs"],
        "global_max_abs_effective_side_vs_q": max_side_q,
        "global_max_abs_effective_side_to_side": max_side_to_side,
        "global_max_total_variation_effective_side_vs_q": max_tv_side_q,
        "global_max_total_variation_effective_side_to_side": max_tv_side_to_side,
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
