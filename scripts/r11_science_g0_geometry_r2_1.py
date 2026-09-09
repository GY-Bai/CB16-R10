#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 Teacher Geometry R2.1 — shadow adjudication only.

R2 observed that exact AccountState replicas inside one market-future dependence
group can alter the frozen Teacher's feature normalization despite adding no
independent future support.  R2.1 reproduces that result on real historical
multi-account contexts and compares it with a dependence-balanced,
unique-context shadow metric.

No Teacher authority is changed here.  The shadow is an experimental candidate
for later requalification only.
"""

import argparse
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_common import FORBIDDEN_FINAL_START_MS, H72, HOUR_MS
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102, evidence_summary
from cb16_local_opt.r102_market import load_anchor_frames
from cb16_local_opt.r102_physics import (
    CANDIDATES_R102,
    FrozenPhysicsRuntimeR102,
    build_parent_scenarios,
    market_future_lineage_hash,
    simulate_h72_branch,
)
from cb16_local_opt.science_teacher_geometry_r11 import (
    SHADOW_ENGINE,
    compare_teacher_evidence_sets_r11,
    compile_teacher_evidence_dependence_balanced_shadow_r11,
)
from cb16_local_opt.teacher_vectorized_r11 import compile_teacher_evidence_vectorized_r11
from scripts import r11_science_g0_historical_r1 as r1


SCHEMA = "CB16_R11_SCIENCE_G0_TEACHER_GEOMETRY_R2_1_RESULT_V1"
R2_SEED = "9d04df9976321321028a4ebec548c2b5dbc01166"


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError(code)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def build_multi_account_counterfactual_support(
    *,
    symbol: str,
    selected: Sequence[tuple[str, Any]],
    encoded: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    physics: FrozenPhysicsRuntimeR102,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    train_target_groups: int,
    validation_target_groups: int,
) -> tuple[dict[str, ParentContextR102], list[CounterfactualBranchSampleR5], dict[str, Any]]:
    parents: dict[str, ParentContextR102] = {}
    samples: list[CounterfactualBranchSampleR5] = []
    target = {"TRAIN": int(train_target_groups), "VALIDATION": int(validation_target_groups)}
    accepted_groups = {"TRAIN": 0, "VALIDATION": 0}
    rejected_groups = {"TRAIN": 0, "VALIDATION": 0}
    eligible_parent_counts: list[int] = []
    scenario_counts: dict[str, int] = {}

    for split, frame in selected:
        if accepted_groups[split] >= target[split]:
            continue
        t = int(frame.decision_time_ms)
        require(t + (H72 - 1) * HOUR_MS < FORBIDDEN_FINAL_START_MS,
                f"R11_R2_1_FORBIDDEN_H72_FUTURE:{t}")
        op, med, riskctx = encoded[t]
        scenarios = build_parent_scenarios(
            physics,
            symbol=symbol,
            decision_time_ms=t,
            hourly_ts=hourly_ts,
            hourly_ohlcv=hourly_ohlcv,
            funding=funding,
            prehistory_hours=96,
        )
        group_id = f"FUT:{symbol}:{t}"
        future_hash = market_future_lineage_hash(symbol, t, hourly_ts, hourly_ohlcv, funding)
        staged_parents: list[ParentContextR102] = []
        staged_samples: list[CounterfactualBranchSampleR5] = []
        for scenario in scenarios:
            if not bool(scenario["eligible_for_economic_evidence"]):
                continue
            parent_id = f"R11R21:P:{symbol}:{t}:{scenario['scenario']}"
            parent = ParentContextR102(
                parent_id=parent_id,
                dependence_group_id=group_id,
                symbol=symbol,
                decision_time_ms=t,
                split=split,
                scenario=str(scenario["scenario"]),
                operator48=tuple(float(x) for x in op),
                medium48=tuple(float(x) for x in med),
                account6=tuple(float(x) for x in scenario["account6"]),
                ordered4h30=tuple(float(x) for x in riskctx),
                current_mark=float(scenario["current_mark"]),
                snapshot_sha256=str(scenario["snapshot_sha256"]),
                eligible_for_economic_evidence=True,
                market_lineage_hash=future_hash,
            )
            state = {
                "parent_id": parent_id,
                "account_id": scenario["account_id"],
                "snapshot": scenario["snapshot"],
                "risk_authority": scenario["risk_authority"],
                "current_mark": float(scenario["current_mark"]),
            }
            rows: list[CounterfactualBranchSampleR5] = []
            complete = True
            for direction_v55, requested_risk in CANDIDATES_R102:
                branch = simulate_h72_branch(
                    physics,
                    parent=state,
                    symbol=symbol,
                    decision_time_ms=t,
                    candidate_direction_v55=int(direction_v55),
                    candidate_risk=float(requested_risk),
                    hourly_ts=hourly_ts,
                    hourly_ohlcv=hourly_ohlcv,
                    funding=funding,
                )
                utility = branch.get("utility")
                if branch.get("status") != "MATURED" or utility is None or not math.isfinite(float(utility)):
                    complete = False
                    break
                row = CounterfactualBranchSampleR5(
                    parent_id=parent_id,
                    student_context_object_id=parent.student_context_object_id,
                    timestamp=t,
                    context_features=parent.student_features,
                    direction=int(direction_v55) - 1,
                    requested_risk=float(requested_risk),
                    realized_utility=float(utility),
                    dependence_group_id=group_id,
                    market_lineage_hash=future_hash,
                )
                row.validate(); rows.append(row)
            if complete and len(rows) == len(CANDIDATES_R102):
                staged_parents.append(parent)
                staged_samples.extend(rows)
                scenario_counts[parent.scenario] = scenario_counts.get(parent.scenario, 0) + 1

        if not staged_parents:
            rejected_groups[split] += 1
            continue
        for p in staged_parents:
            parents[p.parent_id] = p
        samples.extend(staged_samples)
        eligible_parent_counts.append(len(staged_parents))
        accepted_groups[split] += 1

    require(accepted_groups["TRAIN"] >= target["TRAIN"], f"R11_R2_1_TRAIN_GROUP_SHORTFALL:{accepted_groups}")
    require(accepted_groups["VALIDATION"] >= target["VALIDATION"], f"R11_R2_1_VALIDATION_GROUP_SHORTFALL:{accepted_groups}")
    dep_counts: dict[str, int] = {}
    for p in parents.values():
        dep_counts[p.dependence_group_id] = dep_counts.get(p.dependence_group_id, 0) + 1
    return parents, samples, {
        "accepted_dependence_groups": accepted_groups,
        "rejected_candidate_groups": rejected_groups,
        "parent_contexts": len(parents),
        "branch_samples": len(samples),
        "eligible_parents_per_group_min": min(eligible_parent_counts),
        "eligible_parents_per_group_max": max(eligible_parent_counts),
        "eligible_parents_per_group_mean": float(np.mean(np.asarray(eligible_parent_counts, dtype=np.float64))),
        "scenario_counts": scenario_counts,
        "dependence_group_parent_counts": dep_counts,
    }


def exact_replica_variant(
    parents: Mapping[str, ParentContextR102],
    samples: Sequence[CounterfactualBranchSampleR5],
    *,
    replicas: int,
) -> tuple[dict[str, ParentContextR102], list[CounterfactualBranchSampleR5], dict[str, Any]]:
    train_parents = sorted(
        (p for p in parents.values() if p.split == "TRAIN"),
        key=lambda p: (p.decision_time_ms, p.parent_id),
    )
    require(train_parents, "R11_R2_1_NO_TRAIN_PARENT_FOR_REPLICA")
    source = train_parents[0]
    source_rows = [x for x in samples if x.parent_id == source.parent_id]
    require(len(source_rows) == 9, "R11_R2_1_REPLICA_SOURCE_GRID_DRIFT")
    out_parents = dict(parents); out_samples = list(samples)
    for i in range(int(replicas)):
        pid = f"ZZ_R11_R2_1_REPLICA_{i:03d}:{source.parent_id}"
        out_parents[pid] = replace(source, parent_id=pid)
        out_samples.extend(replace(x, parent_id=pid) for x in source_rows)
    return out_parents, out_samples, {
        "source_parent_id": source.parent_id,
        "source_dependence_group_id": source.dependence_group_id,
        "source_student_context_object_id": source.student_context_object_id,
        "replicas_added": int(replicas),
    }


def summarize_context_identity_counts(parents: Mapping[str, ParentContextR102]) -> dict[str, int]:
    physical = len(parents)
    unique_pairs = {(p.dependence_group_id, p.student_context_object_id) for p in parents.values()}
    deps = {p.dependence_group_id for p in parents.values()}
    return {
        "physical_parent_rows": int(physical),
        "unique_dependence_group_context_pairs": int(len(unique_pairs)),
        "dependence_groups": int(len(deps)),
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
    ap.add_argument("--block-targets", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=32)
    args = ap.parse_args()

    work_root=args.work_root.resolve(); work_root.mkdir(parents=True, exist_ok=True)
    output=args.output.resolve(); g0_root=args.g0_root.resolve(); package_root=args.package_root.resolve()
    require(args.symbol == "BTCUSDT", "R11_R2_1_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R2_SEED, "HEAD"]) == 0,
            "R11_R2_1_NOT_DESCENDED_FROM_R2_SEED")

    repo=r1.verify_repo_seed(); repo["r2_seed"]=R2_SEED; repo["execution_head"]=git("rev-parse","HEAD")
    runtime=r1.runtime_identity(args.device)
    lineage,g0_identity=r1.verify_g0_authority(g0_root)
    frozen_before=frozen_authority_hashes(package_root)
    g0_before=dict(g0_identity["authority_file_hashes"])

    market_cache=MarketRuntimeCacheR11(g0_root); market=market_cache.get(args.symbol)
    per_asset=next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path=g0_root/"market_cache"/str(per_asset["anchors_file"])
    require(r1.sha256_file(anchor_path)==str(per_asset["anchors_sha256"]),"R11_R2_1_ANCHOR_SHA_DRIFT")
    frames=load_anchor_frames(args.symbol,anchor_path)
    selected=r1.select_candidate_frames(
        frames, train_target=int(args.train_groups), validation_target=int(args.validation_groups),
        candidate_factor=int(args.candidate_factor))
    encoded,sensory_receipt=r1.encode_selected_frames(
        package_root=package_root,device=args.device,selected=selected,batch_size=int(args.sensory_batch_size))
    physics=FrozenPhysicsRuntimeR102.load(package_root)
    parents,samples,support=build_multi_account_counterfactual_support(
        symbol=args.symbol,selected=selected,encoded=encoded,physics=physics,
        hourly_ts=market.open_time_ms,hourly_ohlcv=market.ohlcv,funding=market.funding_rate,
        train_target_groups=int(args.train_groups),validation_target_groups=int(args.validation_groups))
    del encoded; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    replica_parents,replica_samples,replica_meta=exact_replica_variant(
        parents,samples,replicas=int(args.replicas))

    current_train,current_val,current_stats=compile_teacher_evidence_vectorized_r11(
        samples=samples,parents=parents,train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,block_targets=int(args.block_targets))
    current_rep_train,current_rep_val,current_rep_stats=compile_teacher_evidence_vectorized_r11(
        samples=replica_samples,parents=replica_parents,train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,block_targets=int(args.block_targets))
    shadow_train,shadow_val,shadow_stats,shadow_counts=compile_teacher_evidence_dependence_balanced_shadow_r11(
        samples=samples,parents=parents,train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,block_targets=int(args.block_targets))
    shadow_rep_train,shadow_rep_val,shadow_rep_stats,shadow_rep_counts=compile_teacher_evidence_dependence_balanced_shadow_r11(
        samples=replica_samples,parents=replica_parents,train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,block_targets=int(args.block_targets))

    current_replica_train=compare_teacher_evidence_sets_r11(current_train,current_rep_train)
    current_replica_val=compare_teacher_evidence_sets_r11(current_val,current_rep_val)
    shadow_replica_train=compare_teacher_evidence_sets_r11(shadow_train,shadow_rep_train)
    shadow_replica_val=compare_teacher_evidence_sets_r11(shadow_val,shadow_rep_val)
    baseline_train_delta=compare_teacher_evidence_sets_r11(current_train,shadow_train)
    baseline_val_delta=compare_teacher_evidence_sets_r11(current_val,shadow_val)

    require(current_replica_train["content_hash_changed_targets"] + current_replica_val["content_hash_changed_targets"] > 0,
            "R11_R2_1_FAILED_TO_REPRODUCE_CURRENT_GEOMETRY_REPLICA_SENSITIVITY")
    require(shadow_replica_train["content_hash_changed_targets"] == 0 and shadow_replica_val["content_hash_changed_targets"] == 0,
            "R11_R2_1_SHADOW_GEOMETRY_NOT_EXACT_REPLICA_INVARIANT")
    require(shadow_replica_train["maximum_action_mean_utility_abs_delta"] == 0.0,
            "R11_R2_1_SHADOW_TRAIN_MEAN_NOT_EXACT_INVARIANT")
    require(shadow_replica_val["maximum_action_mean_utility_abs_delta"] == 0.0,
            "R11_R2_1_SHADOW_VAL_MEAN_NOT_EXACT_INVARIANT")

    frozen_after=frozen_authority_hashes(package_root)
    require(frozen_after==frozen_before,"R11_R2_1_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _,g0_after=r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"]==g0_before,"R11_R2_1_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()

    result={
        "schema":SCHEMA,
        "status":"R11_TEACHER_GEOMETRY_SHADOW_ADJUDICATION_PASS_WITH_CURRENT_AUTHORITY_FINDING",
        "repo":repo,"runtime":runtime,
        "historical_scope":{"symbol":args.symbol,"train_dependence_groups":int(args.train_groups),
            "validation_dependence_groups":int(args.validation_groups),"support":support,
            "base_context_identity_counts":summarize_context_identity_counts(parents),
            "replica_context_identity_counts":summarize_context_identity_counts(replica_parents),
            "final_holdout_payload_opened":False,"fresh_market_data_downloaded":False,"network_reads_by_r2_1":0},
        "sensory":sensory_receipt,
        "replica_canary":replica_meta,
        "current_frozen_teacher":{"engine":current_stats.engine,"train_summary":evidence_summary(current_train),
            "validation_summary":evidence_summary(current_val),"replica_train_delta":current_replica_train,
            "replica_validation_delta":current_replica_val,
            "finding":"PARENT_ROW_WEIGHTED_NORMALIZATION_ALLOWS_EXACT_SAME_FUTURE_CONTEXT_REPLICA_VOLUME_TO_CHANGE_BELIEF"},
        "dependence_balanced_shadow":{"engine":SHADOW_ENGINE,"stats":asdict(shadow_stats),"support_regime_counts":shadow_counts,
            "replica_stats":asdict(shadow_rep_stats),"replica_support_regime_counts":shadow_rep_counts,
            "replica_train_delta":shadow_replica_train,"replica_validation_delta":shadow_replica_val,
            "exact_replica_invariant":True,
            "candidate_rule":"EQUAL_TOTAL_NORMALIZATION_MASS_PER_DEPENDENCE_GROUP__DEDUP_EXACT_STUDENT_CONTEXT_ID_WITHIN_GROUP"},
        "baseline_semantic_delta_current_vs_shadow":{"train":baseline_train_delta,"validation":baseline_val_delta,
            "interpretation":"MEASURES_REQUALIFICATION_COST_ONLY__NOT_AUTHORIZATION"},
        "adjudication":{"current_teacher_authority_changed":False,"shadow_teacher_promoted":False,
            "teacher_requalification_required_before_any_cutover":True,
            "next_legal_question":"DOES_DEPENDENCE_BALANCED_GEOMETRY_PRESERVE_OR_IMPROVE_OOF_CALIBRATION_AND_TRUE_VS_SHUFFLE_WITHOUT_ERASING_ACCOUNT_STATE_STRUCTURE",
            "student_distributional_head_change_authorized":False,"qr_dqn_conversion_authorized":False},
        "semantic_guards":{"canonical_generation_advanced":False,"champion_promoted":False,"tournament_run":False,
            "student_architecture_changed":False,"teacher_semantics_changed":False,"economic_objective_changed":False,
            "scientific_verdict_created":False,"market_information_qualified":False,"profitability_or_alpha_claimed":False,
            "final_holdout_payload_opened":False,"fresh_market_data_downloaded":False},
        "authority":{"frozen_scientific_status_before":r1.FROZEN_SCIENTIFIC_STATUS,
            "frozen_scientific_status_after":r1.FROZEN_SCIENTIFIC_STATUS,"g0_authority_hashes_unchanged":True},
    }
    atomic_json(output,result)
    print(json.dumps({"status":result["status"],
        "base_physical_parent_rows":result["historical_scope"]["base_context_identity_counts"]["physical_parent_rows"],
        "base_unique_group_context_pairs":result["historical_scope"]["base_context_identity_counts"]["unique_dependence_group_context_pairs"],
        "current_replica_changed_train_targets":current_replica_train["content_hash_changed_targets"],
        "current_replica_changed_validation_targets":current_replica_val["content_hash_changed_targets"],
        "shadow_replica_changed_train_targets":shadow_replica_train["content_hash_changed_targets"],
        "shadow_replica_changed_validation_targets":shadow_replica_val["content_hash_changed_targets"],
        "baseline_shadow_train_change_rate":baseline_train_delta["content_hash_change_rate"],
        "baseline_shadow_validation_change_rate":baseline_val_delta["content_hash_change_rate"],
        "baseline_shadow_max_mean_utility_delta":max(baseline_train_delta["maximum_action_mean_utility_abs_delta"],baseline_val_delta["maximum_action_mean_utility_abs_delta"]),
        "teacher_authority_changed":False,"scientific_verdict_created":False,"final_holdout_payload_opened":False},indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
