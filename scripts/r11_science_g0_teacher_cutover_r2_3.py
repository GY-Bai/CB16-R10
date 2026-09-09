#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 Teacher R2.3 — authority cutover qualification.

R2.2 qualified the dependence-balanced mechanics in shadow.  R2.3 gives those
mechanics a distinct R11 protocol identity and proves that the identity transition
is clean before any production caller is rebound.

This gate does not reopen the frozen historical market-information verdict.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from cb16_local_opt.r11_teacher_authority_candidate import (
    R11_TEACHER_AUTHORITY_VERSION,
    R11_TRAIN_TEACHER_CONFIG,
    R11_VALIDATION_TEACHER_CONFIG,
)

SCHEMA = "CB16_R11_SCIENCE_G0_TEACHER_CUTOVER_R2_3_RESULT_V1"
R2_2_PASS_SEED = "b79da74f311f7ea20bb772af0856b47ac5d4f514"


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def config_without_version(config) -> dict:
    x = asdict(config)
    x.pop("teacher_version")
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-root", type=Path, default=Path(os.environ.get("CB16_G0_ROOT", "/cb16/g0")))
    ap.add_argument("--package-root", type=Path, default=Path(os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package")))
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--train-groups", type=int, default=48)
    ap.add_argument("--validation-groups", type=int, default=12)
    ap.add_argument("--candidate-factor", type=int, default=2)
    ap.add_argument("--sensory-batch-size", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=32)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    args = ap.parse_args()

    work = args.work_root.resolve(); work.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R2_2_PASS_SEED, "HEAD"]) == 0,
            "R11_R2_3_NOT_DESCENDED_FROM_R2_2_PASS_SEED")

    # Identity must move, but no hyperparameter or Teacher objective may move with it.
    require(config_without_version(TRAIN_TEACHER_CONFIG_R102) == config_without_version(R11_TRAIN_TEACHER_CONFIG),
            "R11_R2_3_TRAIN_HYPERPARAMETER_DRIFT")
    require(config_without_version(VAL_TEACHER_CONFIG_R102) == config_without_version(R11_VALIDATION_TEACHER_CONFIG),
            "R11_R2_3_VALIDATION_HYPERPARAMETER_DRIFT")
    require(TRAIN_TEACHER_CONFIG_R102.content_hash != R11_TRAIN_TEACHER_CONFIG.content_hash,
            "R11_R2_3_TRAIN_PROTOCOL_HASH_NOT_ROTATED")
    require(VAL_TEACHER_CONFIG_R102.content_hash != R11_VALIDATION_TEACHER_CONFIG.content_hash,
            "R11_R2_3_VALIDATION_PROTOCOL_HASH_NOT_ROTATED")

    # Re-run the exact R2.2 formal mechanics gate on this descendant.  The R2.3
    # receipt consumes its result rather than inventing a new scoring protocol.
    nested = work / "r2_2_requalification"
    nested.mkdir(parents=True, exist_ok=True)
    nested_result = nested / "CB16_R11_SCIENCE_G0_TEACHER_REQUAL_R2_2_RESULT.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "r11_science_g0_teacher_requal_r2_2_entrypoint.py"),
        "--g0-root", str(args.g0_root),
        "--package-root", str(args.package_root),
        "--work-root", str(nested),
        "--output", str(nested_result),
        "--device", args.device,
        "--symbol", args.symbol,
        "--train-groups", str(args.train_groups),
        "--validation-groups", str(args.validation_groups),
        "--candidate-factor", str(args.candidate_factor),
        "--sensory-batch-size", str(args.sensory_batch_size),
        "--replicas", str(args.replicas),
        "--bootstrap-reps", str(args.bootstrap_reps),
    ]
    subprocess.check_call(cmd)
    r22 = json.loads(nested_result.read_text())
    require(r22["status"] == "TEACHER_GEOMETRY_MECHANICS_REQUALIFIED_SHADOW_ONLY",
            "R11_R2_3_R2_2_MECHANICS_NOT_REQUALIFIED")
    require(r22["baseline_current_vs_shadow"]["maximum_abs_qcrps_delta"] <= 1e-12,
            "R11_R2_3_R2_2_BASELINE_DRIFT")
    require(r22["coverage_current_vs_shadow"]["maximum_abs_coverage_delta"] <= 1e-12,
            "R11_R2_3_R2_2_COVERAGE_DRIFT")
    require(r22["shadow_replica_sensitivity"]["maximum_abs_qcrps_delta"] <= 1e-12,
            "R11_R2_3_R2_2_REPLICA_INVARIANCE_FAILED")
    require(r22["historical_scope"]["final_holdout_payload_opened"] is False,
            "R11_R2_3_FINAL_HOLDOUT_OPENED")

    result = {
        "schema": SCHEMA,
        "status": "TEACHER_GEOMETRY_AUTHORITY_CUTOVER_QUALIFIED_NOT_BOUND",
        "execution_head": git("rev-parse", "HEAD"),
        "r2_2_pass_seed": R2_2_PASS_SEED,
        "candidate_authority_version": R11_TEACHER_AUTHORITY_VERSION,
        "identity_rotation": {
            "legacy_train_teacher_version": TRAIN_TEACHER_CONFIG_R102.teacher_version,
            "candidate_train_teacher_version": R11_TRAIN_TEACHER_CONFIG.teacher_version,
            "legacy_train_protocol_hash": TRAIN_TEACHER_CONFIG_R102.content_hash,
            "candidate_train_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash,
            "legacy_validation_teacher_version": VAL_TEACHER_CONFIG_R102.teacher_version,
            "candidate_validation_teacher_version": R11_VALIDATION_TEACHER_CONFIG.teacher_version,
            "legacy_validation_protocol_hash": VAL_TEACHER_CONFIG_R102.content_hash,
            "candidate_validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
            "non_version_hyperparameters_identical": True,
        },
        "consumed_r2_2_mechanics": {
            "status": r22["status"],
            "baseline_max_abs_qcrps_delta": r22["baseline_current_vs_shadow"]["maximum_abs_qcrps_delta"],
            "baseline_max_abs_coverage_delta": r22["coverage_current_vs_shadow"]["maximum_abs_coverage_delta"],
            "current_replica_max_abs_qcrps_delta": r22["current_replica_sensitivity"]["maximum_abs_qcrps_delta"],
            "candidate_replica_max_abs_qcrps_delta": r22["shadow_replica_sensitivity"]["maximum_abs_qcrps_delta"],
            "bounded_true_vs_shuffle": r22["bounded_true_vs_shuffle_observation"],
            "historical_market_information_verdict_reopened": False,
        },
        "production_binding_plan": {
            "authorized": True,
            "completed": False,
            "required_bindings": [
                "cb16_local_opt.r102_learning sequential Teacher constructors and R11 Teacher configs",
                "cb16_local_opt.r102_teacher_parallel default and incremental worker classes",
                "cb16_local_opt.r102_teacher_incremental exact compiler class and compiler semantics identity",
                "cb16_local_opt.r102_controls R10.2 diagnostic Teacher binding",
            ],
            "forbid_same_protocol_hash_reuse": True,
            "require_single_vs_process_farm_semantic_equivalence": True,
            "require_compiled_cache_identity_rotation": True,
            "require_real_history_post_binding_receipt": True,
        },
        "semantic_guards": {
            "production_teacher_cutover_completed": False,
            "teacher_objective_changed": False,
            "teacher_hyperparameters_changed": False,
            "student_architecture_changed": False,
            "student_training_started": False,
            "generation_advanced": False,
            "champion_promoted": False,
            "historical_market_information_verdict_reopened": False,
            "scientific_verdict_created": False,
            "market_information_qualified": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
        },
        "next_legal_step": "R2_4_PRODUCTION_TEACHER_BINDING_WITH_POST_BINDING_REAL_HISTORY_RECEIPT",
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "legacy_validation_protocol_hash": VAL_TEACHER_CONFIG_R102.content_hash,
        "candidate_validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
        "baseline_max_abs_qcrps_delta": result["consumed_r2_2_mechanics"]["baseline_max_abs_qcrps_delta"],
        "candidate_replica_max_abs_qcrps_delta": result["consumed_r2_2_mechanics"]["candidate_replica_max_abs_qcrps_delta"],
        "production_binding_authorized": True,
        "production_binding_completed": False,
        "historical_market_information_verdict_reopened": False,
        "final_holdout_payload_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
