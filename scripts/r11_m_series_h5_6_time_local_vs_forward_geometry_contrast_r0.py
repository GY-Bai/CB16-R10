#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from cb16_local_opt.time_local_vs_forward_geometry_contrast_h56 import H56_METRICS, adjudicate_h56, run_fold_h56
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_6_TIME_LOCAL_VS_FORWARD_STATE_UTILITY_GEOMETRY_CONTRAST_R0_RESULT_V1"
PREREG_COMMIT = "8dc374cae5dc08d3e879779fcabb86e0b390eff6"
GATE_BLOB = "a85c3e1c2a60cb80f8c396c4259557a914166416"
H55_ADJ_BLOB = "c336e2789ba41e2131679abcd1e6d51c1b545a1c"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "gate": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_6_TIME_LOCAL_VS_FORWARD_STATE_UTILITY_GEOMETRY_CONTRAST_R0_GATE_V1.json",
        GATE_BLOB,
    ),
    "h55_adjudication": (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_ADJUDICATION_V1.json",
        H55_ADJ_BLOB,
    ),
    "h56_helper": (
        "cb16_local_opt/time_local_vs_forward_geometry_contrast_h56.py",
        "a264ff21d0447527c4dda06ee8fd63cf20207daf",
    ),
    "h56_test": (
        "tests/test_time_local_vs_forward_geometry_contrast_h56.py",
        "fe66adf84801f5e3de7a2cc1f39d6bb8f1036e56",
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
        "H56_NOT_DESCENDED_FROM_PREREGISTRATION",
    )
    observed: dict[str, str] = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H56_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "immutable_blobs": observed,
    }


def load_h55_authority() -> dict[str, Any]:
    path = Path(PINNED["h55_adjudication"][0])
    x = json.loads(path.read_text(encoding="utf-8"))
    require(x["adjudication"]["classification"] == "MARKET_STATE_UTILITY_GEOMETRY_TEMPORAL_NONTRANSPORT", "H56_H55_AUTHORITY_CLASS_DRIFT")
    require(set(x["primary_metric_gates"]) == set(H56_METRICS), "H56_H55_AUTHORITY_METRIC_SET_DRIFT")
    return x


def verify_forward_reproduction(summary: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    require(summary["classification"] == authority["adjudication"]["classification"], "H56_FORWARD_CLASS_DRIFT")
    fields = (
        "positive_fold_count",
        "positive_both_late_folds",
        "beats_shuffle_median_fold_count",
        "beats_shuffle_median_both_late_folds",
        "pairwise_shuffle_count_of_25",
        "metric_transport_supported",
    )
    receipt: dict[str, Any] = {}
    for metric in H56_METRICS:
        got = summary["metric_gates"][metric]
        exp = authority["primary_metric_gates"][metric]
        for field in fields:
            require(got[field] == exp[field], f"H56_FORWARD_GATE_DRIFT:{metric}:{field}:{got[field]}:{exp[field]}")
        late_expected = exp.get("late_fold_rho")
        if late_expected:
            by_fold = {int(r["fold"]): float(r["aligned_rho"]) for r in got["per_fold"]}
            require(abs(by_fold[4] - float(late_expected["fold_4"])) <= 1e-15, f"H56_FORWARD_LATE_RHO_DRIFT:{metric}:4")
            require(abs(by_fold[5] - float(late_expected["fold_5"])) <= 1e-15, f"H56_FORWARD_LATE_RHO_DRIFT:{metric}:5")
        receipt[metric] = {field: got[field] for field in fields}
    return {
        "status": "PASS",
        "classification": summary["classification"],
        "metric_gates": receipt,
        "late_rho_identity_checked_where_authoritative": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = verify_repo()
    h55_authority = load_h55_authority()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require(len(folds) == 5, "H56_OUTER_FOLD_COUNT_DRIFT")

    results = [
        run_fold_h56(
            fold_spec=spec,
            all_train_parents=parents,
            all_train_samples=samples,
        )
        for spec in folds
    ]
    summary = adjudicate_h56(results)
    forward_receipt = verify_forward_reproduction(summary["forward_h5_5_reproduction"], h55_authority)
    require(summary["all_rotation_identity_guards_pass"] is True, "H56_ROTATION_IDENTITY_FAIL")

    result = {
        "schema": SCHEMA,
        "status": "H5_6_TIME_LOCAL_VS_FORWARD_STATE_UTILITY_GEOMETRY_CONTRAST_COMPLETE",
        "repo": repo,
        "runtime_identity": {
            "python": platform.python_version(),
            "student_training": False,
            "student_inference": False,
            "teacher_law_compilation": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "columnar_index_builder_used_for_state_and_utility_arrays_only": True,
            "future_utility_used_to_select_support": False,
        },
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "support": support_receipt,
        "protocol": {
            "support_classification": "CONSUMED_TRAIN_ONLY_MECHANISTIC_SUPPORT__NOT_INDEPENDENT_MARKET_SUPPORT",
            "outer_folds": 5,
            "state_metrics": list(H56_METRICS),
            "time_forward": "EXACT_CLARIFIED_H5_5",
            "time_local": "SAME_EVAL_BLOCK_OTHER_FUTURE_GROUPS_SAME_SCENARIO_TARGET_GROUP_EXCLUDED",
            "local_normalization": "EXACT_SAME_SCENARIO_LOCAL_SUPPORT_ROWS_TARGET_GROUP_EXCLUDED",
            "shifts": [1, 7, 13, 23, 31],
            "utility_profiles_rotated": False,
            "teacher_compilation": False,
            "teacher_kernel": False,
            "teacher_support_selection": False,
            "student_training": False,
            "student_inference": False,
            "fresh_market_data_downloaded": False,
            "legacy_validation_rows_used": False,
            "r5_purge_support_used": False,
            "final_holdout_opened": False,
            "r7_candidate_evaluated": False,
        },
        "folds": results,
        "forward_h5_5_authority_reproduction": forward_receipt,
        "h5_6_summary": summary,
        "semantic_guards": {
            "student_trained": False,
            "student_inference_used": False,
            "teacher_compiled_for_scientific_score": False,
            "teacher_kernel_used": False,
            "teacher_support_selection_used": False,
            "future_utility_used_to_select_local_support": False,
            "utility_outcomes_rotated_with_features": False,
            "organ_weights_tuned": False,
            "distance_metric_tuned": False,
            "canonical_teacher_changed": False,
            "canonical_student_changed": False,
            "new_direction_loss_created": False,
            "market_information_verdict_reopened": False,
            "canonical_change_authorized": False,
            "champion_promoted": False,
            "production_cutover": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False,
        },
        "next_legal_step": "ADJUDICATE_H5_6__IF_LOCAL_MARKET_GEOMETRY_EXISTS_PRIORITIZE_H1_TEMPORAL_NONTRANSFER_MECHANISM__IF_LOCAL_GEOMETRY_WEAK_PRIORITIZE_REPRESENTATION_OBSERVATION_FALSIFICATION__NO_AUTOMATIC_CANONICAL_CHANGE",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "classification": summary["classification"],
        "conclusion": summary["conclusion"],
        "forward_reproduction": forward_receipt["status"],
        "local_metric_gates": summary["local_metric_gates"],
        "local_vs_forward_contrast": summary["local_vs_forward_contrast"],
        "all_rotation_identity_guards_pass": summary["all_rotation_identity_guards_pass"],
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
