#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from cb16_local_opt.fixed_support_weighting_interpolation_audit_h53 import adjudicate_h53, run_fold_h53
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA = "CB16_R11_M_SERIES_H5_3_FIXED_SUPPORT_WEIGHTING_INTERPOLATION_AUDIT_R0_RESULT_V1"
PREREG_COMMIT = "a01f1c3ceadff3926b9cdc58b675606dce48fc4f"
GATE_BLOB = "91943883d4a2f4511613d81861922536464b8b11"
H52_ADJ_BLOB = "d77d9aaf54b093d39baf745681118136cff6a45c"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED = {
    "gate": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_3_FIXED_SUPPORT_WEIGHTING_INTERPOLATION_AUDIT_R0_GATE_V1.json", GATE_BLOB),
    "h52_adjudication": ("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_2_MEDIUM48_SUPPORT_SELECTION_CAUSAL_AUDIT_R0_ADJUDICATION_V1.json", H52_ADJ_BLOB),
    "h53_helper": ("cb16_local_opt/fixed_support_weighting_interpolation_audit_h53.py", "338a5b29f90685467bc7afd139a0963fe4e51cb1"),
    "h52_helper": ("cb16_local_opt/medium48_support_selection_causal_audit_h52.py", "eadcfc071a58f5b3825d224a067bac15242d2a58"),
    "h51_batch_geometry": ("cb16_local_opt/teacher_failure_localization_h51_batch_geometry.py", "9b91ca1d2b8cae05cdf28a7e04ccbdf1698fadf7"),
    "h51_helper": ("cb16_local_opt/teacher_failure_localization_h51.py", "8dba35e5c54c8bc9d8eeb97ea6d3aec373a09ccc"),
    "h5_helper": ("cb16_local_opt/teacher_temporal_transport_audit_h5.py", "09b05de23c659fe6f47a43117ec70b5cafcf7d21"),
    "teacher_balanced_runtime": ("cb16_local_opt/teacher_balanced_runtime_r11.py", "e834facdbf2d08de25e3cab917f0faf587f3386e"),
    "teacher_vectorized": ("cb16_local_opt/teacher_vectorized_r11.py", "083ca6541b45577cfeb6d9a376b25e0eaf8d060a"),
    "teacher_authority_candidate": ("cb16_local_opt/r11_teacher_authority_candidate.py", "6e77e8f989a6d4358a7feac99c269db7b7d6549e"),
    "probabilistic_teacher": ("cb16_local_opt/probabilistic_teacher_r6.py", "3de3092d244c3fd315950c9f8b0bf4d78c50ee2b"),
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
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0, "H53_NOT_DESCENDED_FROM_PREREGISTRATION")
    observed = {}
    for name, (path, expected) in PINNED.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"H53_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    return {"execution_head": git("rev-parse", "HEAD"), "preregistration_commit": PREREG_COMMIT, "immutable_blobs": observed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    repo = verify_repo()
    parents, samples, support_receipt = load_train_only_support(args.r104_root.resolve())
    folds = build_outer_folds_r6(parents)
    require(len(folds) == 5, "H53_OUTER_FOLD_COUNT_DRIFT")
    results = [run_fold_h53(fold_spec=spec, all_train_parents=parents, all_train_samples=samples, block_targets=32) for spec in folds]
    summary = adjudicate_h53(results)
    result = {
        "schema": SCHEMA,
        "status": "H5_3_FIXED_SUPPORT_WEIGHTING_INTERPOLATION_AUDIT_COMPLETE",
        "repo": repo,
        "runtime_identity": {
            "python": platform.python_version(),
            "student_training": False,
            "student_inference": False,
            "production_teacher_changed": False,
            "support_reselection_in_shadow_arms": False,
            "selected_parent_identity_fixed_across_all_arms": True,
            "eval_future_utility_used_for_weight_computation": False,
        },
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "support": support_receipt,
        "protocol": {
            "support_classification": "CONSUMED_TRAIN_ONLY_MECHANISTIC_SUPPORT__NOT_INDEPENDENT_MARKET_SUPPORT",
            "outer_folds": 5,
            "primary_late_folds": [4, 5],
            "train_to_eval_gap_hours_each_fold": 256,
            "all_ten_symbols": True,
            "arms": ["FULL102_CANONICAL", "DROP_MEDIUM48_WEIGHT_ONLY", "DROP_OPERATOR48_WEIGHT_ONLY", "UNIFORM_WEIGHT_FIXED_SUPPORT"],
            "selected_support_membership_fixed": True,
            "selected_nearest_parent_rows_fixed": True,
            "frozen_distance_temperature_used": True,
            "teacher_config_tuning": False,
            "teacher_semantics_changed": False,
            "student_training": False,
            "student_inference": False,
            "raw_market_payload_read": False,
            "legacy_validation_rows_used": False,
            "r5_purge_support_used": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_opened": False,
            "r7_candidate_evaluated": False,
        },
        "folds": results,
        "h5_3_summary": summary,
        "semantic_guards": {
            "student_trained": False,
            "student_inference_used": False,
            "production_teacher_changed": False,
            "teacher_hyperparameter_tuned": False,
            "support_reselected_in_shadow_arms": False,
            "eval_future_utility_used_to_compute_weights": False,
            "new_direction_loss_created": False,
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "production_cutover": False,
            "market_information_verdict_reopened": False,
            "canonical_change_authorized": False,
            "fresh_market_data_downloaded": False,
            "raw_market_payload_used": False,
            "legacy_validation_rows_used": False,
            "r5_purge_support_reopened": False,
            "final_holdout_payload_opened": False,
            "r7_candidate_evaluated": False,
        },
        "next_legal_step": (
            "ADJUDICATE_H5_3_THEN_DESIGN_ONE_MINIMAL_SHADOW_INTERVENTION_MATCHING_THE_LOCALIZED_WEIGHTING_MECHANISM__NO_CANONICAL_CHANGE"
            if summary["classification"] in {"MEDIUM48_WEIGHTING_PRIMARY", "BROAD_KERNEL_WEIGHTING_FAILURE", "OPERATOR48_WEIGHTING_PRIMARY"}
            else
            "ADJUDICATE_H5_3_THEN_REASSESS_BROADER_TEACHER_TRANSPORT_OR_H1_H2__DO_NOT_TUNE_WEIGHTS_OR_ORGANS"
        ),
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "classification": summary["classification"],
        "conclusion": summary["conclusion"],
        "drop_medium48_both_late": summary["drop_medium48_weight_beats_full102_both_late_folds"],
        "drop_operator48_both_late": summary["drop_operator48_weight_beats_full102_both_late_folds"],
        "uniform_both_late": summary["uniform_weight_beats_full102_both_late_folds"],
        "all_identity_guards_pass": summary["all_identity_guards_pass"],
        "per_fold": summary["per_fold"],
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
