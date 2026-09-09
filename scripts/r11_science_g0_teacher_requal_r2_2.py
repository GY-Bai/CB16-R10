#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 Teacher Geometry R2.2 — shadow mechanics requalification.

Status-driving question: can the replica-invariant dependence-balanced geometry
preserve the existing R6/R10.2 OOF distributional-control behavior on balanced
real history while removing exact-replica sensitivity?

This is NOT a market-information re-adjudication and cannot replace the frozen
DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED / TRUE_WORSE_THAN_SHUFFLE result.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_learning import VAL_TEACHER_CONFIG_R102
from cb16_local_opt.r102_market import load_anchor_frames
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
from cb16_local_opt.science_teacher_requalification_r11 import (
    SHADOW_GEOMETRY_VERSION,
    DependenceBalancedHistoricalControlSuiteShadowR11,
    compare_control_receipts_r11,
    compare_coverage_r11,
    coverage_diagnostics_r11,
    receipt_to_dict,
)
from cb16_local_opt.scientific_controls_r6 import (
    DependenceAwareControlSuiteConfigR6,
    DependenceAwareHistoricalControlSuiteR6,
)
from scripts import r11_science_g0_geometry_r2_1 as r21
from scripts import r11_science_g0_historical_r1 as r1

SCHEMA = "CB16_R11_SCIENCE_G0_TEACHER_REQUAL_R2_2_RESULT_V1"
R2_1_PASS_SEED = "847ef8960be2341cad1f6ad3953029a1d5490327"


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


def qmap(receipt):
    return {x.formulation: float(x.qcrps) for x in receipt.formulations}


def receipt_replica_delta(base, replica):
    b = qmap(base); r = qmap(replica)
    return {
        "replica_minus_base_qcrps": {f: float(r[f] - b[f]) for f in sorted(b)},
        "maximum_abs_qcrps_delta": float(max(abs(r[f] - b[f]) for f in b)),
        "f2_minus_f0_delta_change": float(replica.f2_minus_f0.mean_delta - base.f2_minus_f0.mean_delta),
        "f3_minus_f2_delta_change": float(replica.f3_minus_f2.mean_delta - base.f3_minus_f2.mean_delta),
        "f1_minus_f0_delta_change": float(replica.f1_minus_f0.mean_delta - base.f1_minus_f0.mean_delta),
    }


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

    work_root = args.work_root.resolve(); work_root.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve(); g0_root = args.g0_root.resolve(); package_root = args.package_root.resolve()
    require(args.symbol == "BTCUSDT", "R11_R2_2_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R2_1_PASS_SEED, "HEAD"]) == 0,
            "R11_R2_2_NOT_DESCENDED_FROM_R2_1_PASS_SEED")

    repo = r1.verify_repo_seed(); repo["r2_1_pass_seed"] = R2_1_PASS_SEED; repo["execution_head"] = git("rev-parse", "HEAD")
    runtime = r1.runtime_identity(args.device)
    lineage, g0_identity = r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_before = dict(g0_identity["authority_file_hashes"])

    market_cache = MarketRuntimeCacheR11(g0_root); market = market_cache.get(args.symbol)
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R2_2_ANCHOR_SHA_DRIFT")
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

    target_parent_ids = sorted(p.parent_id for p in parents.values() if p.split == "VALIDATION")
    train_deps = sorted({p.dependence_group_id for p in parents.values() if p.split == "TRAIN"})
    require(len(train_deps) == int(args.train_groups), "R11_R2_2_TRAIN_DEPENDENCE_GROUP_COUNT_DRIFT")
    require(len({p.dependence_group_id for p in parents.values() if p.split == "VALIDATION"}) == int(args.validation_groups),
            "R11_R2_2_VALIDATION_DEPENDENCE_GROUP_COUNT_DRIFT")

    # Current R11 sensory context is Operator48 + Medium48 + Account6 = 96 + 6.
    control_config = DependenceAwareControlSuiteConfigR6(
        teacher=VAL_TEACHER_CONFIG_R102,
        market_dim=96,
        account_dim=6,
        shuffle_seed=20260904,
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_alpha=0.05,
        minimum_scored_dependence_groups=8,
        control_version="CB16_R11_R2_2_TEACHER_GEOMETRY_REQUAL_CONTROLS_V1",
    )
    current_suite = DependenceAwareHistoricalControlSuiteR6(control_config)
    shadow_suite = DependenceBalancedHistoricalControlSuiteShadowR11(control_config)

    current_base = current_suite.evaluate(
        samples, target_parent_ids=target_parent_ids,
        eligible_train_dependence_group_ids=train_deps,
    )
    shadow_base = shadow_suite.evaluate(
        samples, target_parent_ids=target_parent_ids,
        eligible_train_dependence_group_ids=train_deps,
    )
    current_replica = current_suite.evaluate(
        replica_samples, target_parent_ids=target_parent_ids,
        eligible_train_dependence_group_ids=train_deps,
    )
    shadow_replica = shadow_suite.evaluate(
        replica_samples, target_parent_ids=target_parent_ids,
        eligible_train_dependence_group_ids=train_deps,
    )
    require(current_base.status == "PASS" and shadow_base.status == "PASS", "R11_R2_2_BASE_CONTROL_SUPPORT_NOT_PASS")
    require(current_replica.status == "PASS" and shadow_replica.status == "PASS", "R11_R2_2_REPLICA_CONTROL_SUPPORT_NOT_PASS")

    current_coverage = coverage_diagnostics_r11(
        current_suite, samples, target_parent_ids=target_parent_ids,
        eligible_train_dependence_group_ids=train_deps,
    )
    shadow_coverage = coverage_diagnostics_r11(
        shadow_suite, samples, target_parent_ids=target_parent_ids,
        eligible_train_dependence_group_ids=train_deps,
    )

    baseline_compare = compare_control_receipts_r11(current_base, shadow_base)
    coverage_compare = compare_coverage_r11(current_coverage, shadow_coverage)
    current_replica_delta = receipt_replica_delta(current_base, current_replica)
    shadow_replica_delta = receipt_replica_delta(shadow_base, shadow_replica)

    # R2.1 established mathematical equivalence on balanced six-account groups.
    # R2.2 requires that equivalence to survive formal qCRPS/control scoring.
    numerical_tol = 1e-12
    require(baseline_compare["maximum_abs_qcrps_delta"] <= numerical_tol,
            f"R11_R2_2_SHADOW_BASELINE_QCRPS_DRIFT:{baseline_compare}")
    require(coverage_compare["maximum_abs_qcrps_delta"] <= numerical_tol,
            f"R11_R2_2_SHADOW_COVERAGE_QCRPS_DRIFT:{coverage_compare}")
    require(coverage_compare["maximum_abs_coverage_delta"] <= numerical_tol,
            f"R11_R2_2_SHADOW_COVERAGE_RATE_DRIFT:{coverage_compare}")
    require(shadow_replica_delta["maximum_abs_qcrps_delta"] <= numerical_tol,
            f"R11_R2_2_SHADOW_NOT_REPLICA_INVARIANT:{shadow_replica_delta}")
    require(abs(shadow_replica_delta["f3_minus_f2_delta_change"]) <= numerical_tol,
            f"R11_R2_2_SHADOW_TRUE_VS_SHUFFLE_NOT_REPLICA_INVARIANT:{shadow_replica_delta}")
    require(current_replica_delta["maximum_abs_qcrps_delta"] > numerical_tol,
            "R11_R2_2_CURRENT_CONTROL_REPLICA_SENSITIVITY_NOT_REPRODUCED")

    frozen_after = frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "R11_R2_2_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R2_2_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()

    f2 = qmap(shadow_base)["F2_TRUE_MARKET_ACCOUNT"]
    f3 = qmap(shadow_base)["F3_SHUFFLED_MARKET"]
    bounded_relation = (
        "TRUE_MARKET_BETTER_THAN_SHUFFLE_ON_THIS_BOUNDED_R2_2_SAMPLE" if f2 < f3 else
        "TRUE_MARKET_NOT_BETTER_THAN_SHUFFLE_ON_THIS_BOUNDED_R2_2_SAMPLE"
    )
    result = {
        "schema": SCHEMA,
        "status": "TEACHER_GEOMETRY_MECHANICS_REQUALIFIED_SHADOW_ONLY",
        "repo": repo,
        "runtime": runtime,
        "historical_scope": {
            "symbol": args.symbol,
            "train_dependence_groups": int(args.train_groups),
            "validation_dependence_groups": int(args.validation_groups),
            "base_context_identity_counts": r21.summarize_context_identity_counts(parents),
            "replica_context_identity_counts": r21.summarize_context_identity_counts(replica_parents),
            "support": support,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "network_reads_by_r2_2": 0,
        },
        "sensory": sensory_receipt,
        "control_protocol": {
            "config": asdict(control_config),
            "current_geometry": "CB16_DEPENDENCE_AWARE_KNN_TEACHER_R6_PARENT_ROW_WEIGHTED_NORMALIZATION",
            "shadow_geometry": SHADOW_GEOMETRY_VERSION,
            "market_dim": 96,
            "account_dim": 6,
            "metric": "DEPENDENCE_GROUP_WEIGHTED_DISCRETE_QCRPS_LOWER_IS_BETTER",
            "paired_bootstrap_unit": "INDEPENDENT_MARKET_FUTURE_DEPENDENCE_GROUP",
        },
        "current_base": receipt_to_dict(current_base),
        "shadow_base": receipt_to_dict(shadow_base),
        "current_coverage": current_coverage,
        "shadow_coverage": shadow_coverage,
        "baseline_current_vs_shadow": baseline_compare,
        "coverage_current_vs_shadow": coverage_compare,
        "replica_canary": replica_meta,
        "current_replica_sensitivity": current_replica_delta,
        "shadow_replica_sensitivity": shadow_replica_delta,
        "bounded_true_vs_shuffle_observation": {
            "shadow_F2_qcrps": float(f2),
            "shadow_F3_qcrps": float(f3),
            "F3_minus_F2_qcrps": float(f3 - f2),
            "paired_bootstrap": asdict(shadow_base.f3_minus_f2),
            "relation": bounded_relation,
            "status_driving_for_historical_market_information_verdict": False,
        },
        "adjudication": {
            "geometry_mechanics_requalified": True,
            "shadow_only": True,
            "production_teacher_cutover_authorized": False,
            "teacher_authority_changed": False,
            "student_architecture_changed": False,
            "distributional_student_head_authorized": False,
            "qr_dqn_conversion_authorized": False,
            "historical_market_information_verdict_reopened": False,
            "next_legal_step": "NARROW_TEACHER_GEOMETRY_AUTHORITY_CUTOVER_GATE_THEN_CB16_NATIVE_DISTRIBUTIONAL_STUDENT_SHADOW",
        },
        "semantic_guards": {
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "tournament_run": False,
            "teacher_semantics_changed": False,
            "economic_objective_changed": False,
            "scientific_verdict_created": False,
            "market_information_qualified": False,
            "profitability_or_alpha_claimed": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
        },
        "authority": {
            "frozen_scientific_status_before": r1.FROZEN_SCIENTIFIC_STATUS,
            "frozen_scientific_status_after": r1.FROZEN_SCIENTIFIC_STATUS,
            "g0_authority_hashes_unchanged": True,
        },
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "current_qcrps": qmap(current_base),
        "shadow_qcrps": qmap(shadow_base),
        "baseline_max_abs_qcrps_delta": baseline_compare["maximum_abs_qcrps_delta"],
        "baseline_max_abs_coverage_delta": coverage_compare["maximum_abs_coverage_delta"],
        "current_replica_max_abs_qcrps_delta": current_replica_delta["maximum_abs_qcrps_delta"],
        "shadow_replica_max_abs_qcrps_delta": shadow_replica_delta["maximum_abs_qcrps_delta"],
        "bounded_F3_minus_F2": float(f3 - f2),
        "historical_market_information_verdict_reopened": False,
        "production_teacher_cutover_authorized": False,
        "final_holdout_payload_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
