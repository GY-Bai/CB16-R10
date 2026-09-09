#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 Teacher R2.4 — production binding qualification.

This gate binds only the R11 Teacher runtime. It does not train the Student,
advance generation, reopen the historical market-information verdict, or touch
the final holdout. R10.2 legacy Teacher code remains available for reproduction.
"""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from cb16_local_opt.r102_market import load_anchor_frames
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
from cb16_local_opt.r11_teacher_authority_candidate import (
    R11_TRAIN_TEACHER_CONFIG,
    R11_VALIDATION_TEACHER_CONFIG,
    DependenceBalancedProbabilisticTeacherR11,
)
from cb16_local_opt.teacher_balanced_runtime_r11 import (
    R11_BALANCED_SCHEDULER,
    R11_BALANCED_TEACHER_ENGINE,
)
from cb16_local_opt.teacher_runtime_r11 import R11_TEACHER_RUNTIME, compile_teacher_evidence_r11
from scripts import r11_science_g0_geometry_r2_1 as r21
from scripts import r11_science_g0_historical_r1 as r1

SCHEMA = "CB16_R11_SCIENCE_G0_TEACHER_BINDING_R2_4_RESULT_V1"
R2_3_PASS_SEED = "a5c55654e89e348a7132f1c0a42baf64eaf9db98"


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


def evidence_map(rows):
    return {x.parent_id: x for x in rows}


def compare_evidence_numeric(a, b) -> dict:
    require(a.parent_id == b.parent_id, "R11_R2_4_PARENT_ID_DRIFT")
    require(a.student_context_object_id == b.student_context_object_id, "R11_R2_4_CONTEXT_ID_DRIFT")
    require(a.target_dependence_group_id == b.target_dependence_group_id, "R11_R2_4_DEPENDENCE_GROUP_DRIFT")
    require(a.teacher_protocol_hash == b.teacher_protocol_hash, "R11_R2_4_PROTOCOL_HASH_DRIFT")
    require(a.teacher_version == b.teacher_version, "R11_R2_4_TEACHER_VERSION_DRIFT")
    require(a.train_dependence_group_hash == b.train_dependence_group_hash, "R11_R2_4_SUPPORT_GROUP_HASH_DRIFT")
    require(a.admission.status == b.admission.status, "R11_R2_4_ADMISSION_STATUS_DRIFT")
    require(a.admission.reasons == b.admission.reasons, "R11_R2_4_ADMISSION_REASON_DRIFT")
    require(len(a.action_laws) == len(b.action_laws), "R11_R2_4_ACTION_LAW_COUNT_DRIFT")
    max_delta = 0.0
    for x, y in zip(a.action_laws, b.action_laws):
        require((x.direction, x.requested_risk) == (y.direction, y.requested_risk), "R11_R2_4_ACTION_GRID_DRIFT")
        require(x.quantile_levels == y.quantile_levels, "R11_R2_4_QUANTILE_LEVEL_DRIFT")
        require(x.support_dependence_group_hash == y.support_dependence_group_hash, "R11_R2_4_SELECTED_SUPPORT_DRIFT")
        for xv, yv in (
            (x.mean_utility, y.mean_utility),
            (x.std_utility, y.std_utility),
            (x.effective_dependence_n, y.effective_dependence_n),
            (x.nearest_distance, y.nearest_distance),
            (x.max_distance_used, y.max_distance_used),
        ):
            if math.isfinite(float(xv)) and math.isfinite(float(yv)):
                max_delta = max(max_delta, abs(float(xv) - float(yv)))
            else:
                require(xv == yv, "R11_R2_4_NONFINITE_LAW_DRIFT")
        max_delta = max(max_delta, float(np.max(np.abs(np.asarray(x.quantiles) - np.asarray(y.quantiles)))))
    max_delta = max(
        max_delta,
        float(np.max(np.abs(np.asarray(a.direction_target_probs) - np.asarray(b.direction_target_probs)))),
        abs(float(a.requested_risk_target) - float(b.requested_risk_target)),
    )
    return {"maximum_abs_numeric_delta": float(max_delta)}


def compare_maps_numeric(left, right, parent_ids) -> dict:
    maximum = 0.0
    for parent_id in parent_ids:
        require(parent_id in left and parent_id in right, f"R11_R2_4_MISSING_COMPARISON_PARENT:{parent_id}")
        maximum = max(maximum, compare_evidence_numeric(left[parent_id], right[parent_id])["maximum_abs_numeric_delta"])
    return {"parents_compared": len(parent_ids), "maximum_abs_numeric_delta": float(maximum)}


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
    args = ap.parse_args()

    work_root = args.work_root.resolve(); work_root.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve(); g0_root = args.g0_root.resolve(); package_root = args.package_root.resolve()
    require(args.symbol == "BTCUSDT", "R11_R2_4_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R2_3_PASS_SEED, "HEAD"]) == 0,
            "R11_R2_4_NOT_DESCENDED_FROM_R2_3_PASS_SEED")

    repo = r1.verify_repo_seed(); repo["r2_3_pass_seed"] = R2_3_PASS_SEED; repo["execution_head"] = git("rev-parse", "HEAD")
    runtime = r1.runtime_identity(args.device)
    lineage, g0_identity = r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_before = dict(g0_identity["authority_file_hashes"])

    market_cache = MarketRuntimeCacheR11(g0_root); market = market_cache.get(args.symbol)
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R2_4_ANCHOR_SHA_DRIFT")
    frames = load_anchor_frames(args.symbol, anchor_path)
    selected = r1.select_candidate_frames(
        frames,
        train_target=int(args.train_groups),
        validation_target=int(args.validation_groups),
        candidate_factor=int(args.candidate_factor),
    )
    encoded, sensory_receipt = r1.encode_selected_frames(
        package_root=package_root, device=args.device, selected=selected,
        batch_size=int(args.sensory_batch_size),
    )
    physics = FrozenPhysicsRuntimeR102.load(package_root)
    parents, samples, support = r21.build_multi_account_counterfactual_support(
        symbol=args.symbol, selected=selected, encoded=encoded, physics=physics,
        hourly_ts=market.open_time_ms, hourly_ohlcv=market.ohlcv, funding=market.funding_rate,
        train_target_groups=int(args.train_groups), validation_target_groups=int(args.validation_groups),
    )
    replica_parents, replica_samples, replica_meta = r21.exact_replica_variant(
        parents, samples, replicas=int(args.replicas)
    )

    base_train_1, base_val_1, stats_1 = compile_teacher_evidence_r11(
        samples=samples, parents=parents,
        train_config=TRAIN_TEACHER_CONFIG_R102, val_config=VAL_TEACHER_CONFIG_R102,
        workers=1, block_targets=32,
    )
    base_train_4, base_val_4, stats_4 = compile_teacher_evidence_r11(
        samples=samples, parents=parents,
        train_config=TRAIN_TEACHER_CONFIG_R102, val_config=VAL_TEACHER_CONFIG_R102,
        workers=4, block_targets=32,
    )
    replica_train, replica_val, replica_stats = compile_teacher_evidence_r11(
        samples=replica_samples, parents=replica_parents,
        train_config=TRAIN_TEACHER_CONFIG_R102, val_config=VAL_TEACHER_CONFIG_R102,
        workers=4, block_targets=32,
    )

    require(base_train_1 == base_train_4 and base_val_1 == base_val_4,
            "R11_R2_4_WORKER_TOPOLOGY_CHANGED_EVIDENCE")
    for stats in (stats_1, stats_4, replica_stats):
        require(stats.core.engine == R11_BALANCED_TEACHER_ENGINE, "R11_R2_4_WRONG_TEACHER_ENGINE")
        require(stats.scheduler == R11_BALANCED_SCHEDULER, "R11_R2_4_WRONG_TEACHER_SCHEDULER")
        require(stats.topology_in_scientific_identity is False, "R11_R2_4_TOPOLOGY_ENTERED_SCIENTIFIC_IDENTITY")

    require({e.teacher_protocol_hash for e in base_train_1} == {R11_TRAIN_TEACHER_CONFIG.content_hash},
            "R11_R2_4_TRAIN_PROTOCOL_IDENTITY_NOT_ROTATED")
    require({e.teacher_protocol_hash for e in base_val_1} == {R11_VALIDATION_TEACHER_CONFIG.content_hash},
            "R11_R2_4_VALIDATION_PROTOCOL_IDENTITY_NOT_ROTATED")
    require({e.teacher_version for e in base_train_1} == {R11_TRAIN_TEACHER_CONFIG.teacher_version},
            "R11_R2_4_TRAIN_TEACHER_VERSION_NOT_ROTATED")
    require({e.teacher_version for e in base_val_1} == {R11_VALIDATION_TEACHER_CONFIG.teacher_version},
            "R11_R2_4_VALIDATION_TEACHER_VERSION_NOT_ROTATED")

    base_map = evidence_map(base_train_1 + base_val_1)
    replica_map = evidence_map(replica_train + replica_val)
    original_ids = sorted(base_map)
    require(set(original_ids).issubset(replica_map), "R11_R2_4_REPLICA_WORLD_LOST_ORIGINAL_TARGET")
    for parent_id in original_ids:
        require(base_map[parent_id] == replica_map[parent_id], f"R11_R2_4_EXACT_REPLICA_CHANGED_EVIDENCE:{parent_id}")

    # Direct candidate is the semantic oracle qualified in R2.3; vectorized production
    # execution may differ only by floating-point reduction noise, never support identity.
    direct_val_teacher = DependenceBalancedProbabilisticTeacherR11(R11_VALIDATION_TEACHER_CONFIG)
    direct_index = direct_val_teacher.index(samples)
    train_deps = {p.dependence_group_id for p in parents.values() if p.split == "TRAIN"}
    direct_val = {}
    for evidence in base_val_1:
        direct_val[evidence.parent_id] = direct_val_teacher.compile_one(
            target_parent=evidence.parent_id,
            index=direct_index,
            eligible_train_dependence_groups=train_deps,
        )
    direct_compare = compare_maps_numeric(evidence_map(base_val_1), direct_val, sorted(direct_val))
    require(direct_compare["maximum_abs_numeric_delta"] <= 3e-10,
            f"R11_R2_4_PRODUCTION_VS_DIRECT_CANDIDATE_DRIFT:{direct_compare}")

    frozen_after = frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "R11_R2_4_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R2_4_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()

    result = {
        "schema": SCHEMA,
        "status": "R11_TEACHER_DEPENDENCE_BALANCED_PRODUCTION_BINDING_PASS",
        "repo": repo,
        "runtime": runtime,
        "teacher_binding": {
            "runtime_version": R11_TEACHER_RUNTIME,
            "engine": R11_BALANCED_TEACHER_ENGINE,
            "scheduler": R11_BALANCED_SCHEDULER,
            "legacy_train_protocol_hash": TRAIN_TEACHER_CONFIG_R102.content_hash,
            "r11_train_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash,
            "legacy_validation_protocol_hash": VAL_TEACHER_CONFIG_R102.content_hash,
            "r11_validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
            "r10_2_legacy_modules_mutated": False,
        },
        "historical_scope": {
            "symbol": args.symbol,
            "train_dependence_groups": int(args.train_groups),
            "validation_dependence_groups": int(args.validation_groups),
            "base_context_identity_counts": r21.summarize_context_identity_counts(parents),
            "replica_context_identity_counts": r21.summarize_context_identity_counts(replica_parents),
            "replica_meta": replica_meta,
            "support": support,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "network_reads_by_r2_4": 0,
        },
        "sensory": sensory_receipt,
        "post_binding_checks": {
            "worker_1_vs_4_evidence_byte_identical": True,
            "original_targets_exactly_replica_invariant": True,
            "original_targets_compared": len(original_ids),
            "direct_candidate_validation_comparison": direct_compare,
            "train_evidence_count": len(base_train_1),
            "validation_evidence_count": len(base_val_1),
            "replica_world_evidence_count": len(replica_train) + len(replica_val),
        },
        "semantic_guards": {
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
        "next_legal_step": "R11_SCIENCE_G0_DISTRIBUTIONAL_STUDENT_SHADOW_R3",
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "r11_train_protocol_hash": result["teacher_binding"]["r11_train_protocol_hash"],
        "r11_validation_protocol_hash": result["teacher_binding"]["r11_validation_protocol_hash"],
        "worker_1_vs_4_evidence_byte_identical": True,
        "original_targets_exactly_replica_invariant": True,
        "direct_candidate_max_abs_numeric_delta": direct_compare["maximum_abs_numeric_delta"],
        "student_training_started": False,
        "historical_market_information_verdict_reopened": False,
        "final_holdout_payload_opened": False,
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
