#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 Distributional Feedback Boundary Audit R2.

R2 is diagnostic only. It does not add a quantile head, convert CB16 to
QR-DQN/IQN, change the frozen expected-log-equity objective, or advance the
canonical generation.
"""

import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import torch

from cb16_local_opt.evidence_store_r11 import EvidenceStoreR11
from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.probabilistic_teacher_r6 import DependenceAwareProbabilisticTeacherR6
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102, evidence_summary
from cb16_local_opt.r102_market import load_anchor_frames
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
from cb16_local_opt.science_distribution_boundary_r11 import (
    audit_teacher_action_laws_r11,
    classify_distributional_boundary_r11,
    compare_student_projection_paths_r11,
    projected_target_perturbation_r11,
    tail_only_law_perturbation_r11,
)
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.teacher_vectorized_r11 import compile_teacher_evidence_vectorized_r11
from cb16_local_opt.training_runtime_r11 import PreparedEvidenceR11
from scripts import r11_science_g0_historical_r1 as r1


SCHEMA = "CB16_R11_SCIENCE_G0_DISTRIBUTIONAL_BOUNDARY_R2_RESULT_V1"
R1_PASS_COMMIT = "2e213f3dffcc3c2fb13c5209b5f1e3b5ff977eb9"


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


def _by_parent(rows):
    return {e.parent_id: e for e in rows}


def crossfit_self_future_exclusion_canary(*, samples, parents, train_evidence) -> dict[str, Any]:
    admitted = [e for e in train_evidence if e.admission.admitted]
    require(admitted, "R11_R2_NO_TRAIN_EVIDENCE_FOR_CROSSFIT_CANARY")
    target = admitted[len(admitted) // 2]
    target_dep = target.target_dependence_group_id
    mutated_samples = [
        replace(s, realized_utility=float(s.realized_utility) + (10.0 if i % 2 == 0 else -10.0))
        if s.dependence_group_id == target_dep else s
        for i, s in enumerate(samples)
    ]
    base_train, _, _ = compile_teacher_evidence_vectorized_r11(
        samples=samples, parents=parents,
        train_config=TRAIN_TEACHER_CONFIG_R102, val_config=VAL_TEACHER_CONFIG_R102,
        block_targets=8,
    )
    mutant_train, _, _ = compile_teacher_evidence_vectorized_r11(
        samples=mutated_samples, parents=parents,
        train_config=TRAIN_TEACHER_CONFIG_R102, val_config=VAL_TEACHER_CONFIG_R102,
        block_targets=8,
    )
    base = _by_parent(base_train)[target.parent_id]
    mutant = _by_parent(mutant_train)[target.parent_id]
    production = _by_parent(train_evidence)[target.parent_id]
    require(production.content_hash == base.content_hash, "R11_R2_PRODUCTION_VECTORIZED_TEACHER_TARGET_DRIFT")
    require(base.content_hash == mutant.content_hash, "R11_R2_TARGET_OWN_FUTURE_LEAKS_INTO_TEACHER_BELIEF")
    return {
        "status": "PASS", "target_parent_id": target.parent_id,
        "target_dependence_group_id": target_dep,
        "production_vectorized_target_equivalent": True,
        "target_own_realized_utilities_extremely_perturbed": True,
        "target_evidence_content_hash_unchanged": True,
        "rule": "TARGET_SAME_FUTURE_GROUP_EXCLUDED_FROM_ITS_OWN_TEACHER_SUPPORT",
    }


def same_future_replica_sensitivity_canary(*, samples, parents, train_evidence, replicas: int) -> dict[str, Any]:
    require(replicas >= 1, "R11_R2_REPLICAS_MUST_BE_POSITIVE")
    admitted = [e for e in train_evidence if e.admission.admitted]
    target = admitted[len(admitted) // 2]
    oracle = DependenceAwareProbabilisticTeacherR6(TRAIN_TEACHER_CONFIG_R102)
    idx = oracle.index(samples)
    train_groups = {p.dependence_group_id for p in parents.values() if p.split == "TRAIN"}
    support_deps = oracle.train_dependence_groups_for_target(
        target_parent=target.parent_id, index=idx,
        eligible_train_dependence_groups=train_groups,
    )
    require(bool(support_deps), "R11_R2_NO_SUPPORT_DEP_FOR_REPLICA_CANARY")
    source_dep = support_deps[0]
    source_parent_id = idx.parents_by_dependence_group[source_dep][0]
    source_parent = parents[source_parent_id]
    source_rows = [s for s in samples if s.parent_id == source_parent_id]
    require(len(source_rows) == 9, "R11_R2_REPLICA_SOURCE_ACTION_GRID_DRIFT")

    replica_parents = dict(parents)
    replica_samples = list(samples)
    for n in range(int(replicas)):
        clone_id = f"ZZ_R11_R2_REPLICA_{n:03d}:{source_parent_id}"
        replica_parents[clone_id] = replace(source_parent, parent_id=clone_id)
        replica_samples.extend(replace(s, parent_id=clone_id) for s in source_rows)

    base_train, _, _ = compile_teacher_evidence_vectorized_r11(
        samples=samples, parents=parents,
        train_config=TRAIN_TEACHER_CONFIG_R102, val_config=VAL_TEACHER_CONFIG_R102,
        block_targets=8,
    )
    replica_train, _, _ = compile_teacher_evidence_vectorized_r11(
        samples=replica_samples, parents=replica_parents,
        train_config=TRAIN_TEACHER_CONFIG_R102, val_config=VAL_TEACHER_CONFIG_R102,
        block_targets=8,
    )
    base = _by_parent(base_train)[target.parent_id]
    changed = _by_parent(replica_train)[target.parent_id]
    same_hash = base.content_hash == changed.content_hash
    same_support_count = base.admission.unique_train_dependence_groups == changed.admission.unique_train_dependence_groups
    base_means = [float(x.mean_utility) for x in base.action_laws]
    changed_means = [float(x.mean_utility) for x in changed.action_laws]
    mean_max_abs_delta = max(abs(a - b) for a, b in zip(base_means, changed_means))
    return {
        "status": "INVARIANT" if same_hash else "GEOMETRY_SENSITIVE_FINDING",
        "target_parent_id": target.parent_id,
        "replicated_dependence_group_id": source_dep,
        "replica_parent_count_added": int(replicas),
        "independent_dependence_group_count_unchanged": bool(same_support_count),
        "teacher_evidence_content_hash_unchanged": bool(same_hash),
        "maximum_action_mean_utility_abs_delta": float(mean_max_abs_delta),
        "interpretation": (
            "SAME_FUTURE_REPLICA_VOLUME_DOES_NOT_CHANGE_TEACHER_BELIEF" if same_hash
            else "SAME_FUTURE_REPLICA_VOLUME_CHANGES_TEACHER_GEOMETRY_DESPITE_UNCHANGED_INDEPENDENT_SUPPORT"
        ),
        "scientific_verdict_changed": False,
    }


def replay_dedup_canary(*, work_root: Path, g0_root: Path, train_evidence, parents) -> dict[str, Any]:
    items = r1.teacher_evidence_items(train_evidence, parents)
    root = work_root / "r2_replay_dedup_store"
    store = EvidenceStoreR11(
        metadata_root=root / "metadata", payload_roots=[root / "payload0", root / "payload1"],
        segment_target_bytes=8 * 1024 * 1024, codec="zlib", sqlite_synchronous="FULL",
        read_only_source_roots=[g0_root],
    )
    try:
        _, first = store.put_evidence(items)
        _, second = store.put_evidence(items)
        audit = store.periodic_audit()
    finally:
        store.close()
    require(bool(audit.get("pass")), f"R11_R2_REPLAY_STORE_AUDIT_FAIL:{audit}")
    require(first.created_evidence_count == len(items), "R11_R2_FIRST_EVIDENCE_MATERIALIZATION_COUNT_DRIFT")
    require(second.created_evidence_count == 0, "R11_R2_REPLAY_CREATED_NEW_EVIDENCE_IDENTITY")
    require(second.created_payload_count == 0, "R11_R2_REPLAY_CREATED_NEW_PAYLOAD_IDENTITY")
    return {
        "status": "PASS", "first_materialization": asdict(first), "identical_replay": asdict(second),
        "replay_training_exposure_may_repeat": True,
        "replay_increases_independent_evidence_count": False,
        "periodic_audit": audit,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-root", type=Path, default=Path(os.environ.get("CB16_G0_ROOT", "/cb16/g0")))
    ap.add_argument("--package-root", type=Path, default=Path(os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package")))
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--train-groups", type=int, default=64)
    ap.add_argument("--validation-groups", type=int, default=16)
    ap.add_argument("--candidate-factor", type=int, default=2)
    ap.add_argument("--sensory-batch-size", type=int, default=8)
    ap.add_argument("--teacher-workers", type=int, default=4)
    ap.add_argument("--teacher-block-targets", type=int, default=8)
    ap.add_argument("--same-future-replicas", type=int, default=32)
    args = ap.parse_args()

    work_root = args.work_root.resolve(); work_root.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve(); g0_root = args.g0_root.resolve(); package_root = args.package_root.resolve()
    require(args.symbol == "BTCUSDT", "R11_R2_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R1_PASS_COMMIT, "HEAD"]) == 0,
            "R11_R2_NOT_DESCENDED_FROM_R1_PASS_COMMIT")

    repo = r1.verify_repo_seed(); repo["r1_pass_commit"] = R1_PASS_COMMIT; repo["execution_head"] = git("rev-parse", "HEAD")
    runtime = r1.runtime_identity(args.device)
    lineage, g0_identity = r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_authority_before = dict(g0_identity["authority_file_hashes"])

    market_cache = MarketRuntimeCacheR11(g0_root); market = market_cache.get(args.symbol)
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R2_ANCHOR_SHA_DRIFT")
    frames = load_anchor_frames(args.symbol, anchor_path)
    selected = r1.select_candidate_frames(frames, train_target=int(args.train_groups),
                                          validation_target=int(args.validation_groups),
                                          candidate_factor=int(args.candidate_factor))
    encoded, sensory_receipt = r1.encode_selected_frames(package_root=package_root, device=args.device,
                                                          selected=selected, batch_size=int(args.sensory_batch_size))
    physics = FrozenPhysicsRuntimeR102.load(package_root)
    parents, parent_states, samples, support = r1.build_bounded_counterfactual_support(
        symbol=args.symbol, selected=selected, encoded=encoded, physics=physics,
        hourly_ts=market.open_time_ms, hourly_ohlcv=market.ohlcv, funding=market.funding_rate,
        train_target=int(args.train_groups), validation_target=int(args.validation_groups))
    del encoded, parent_states; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    train_evidence, validation_evidence, teacher_stats = compile_teacher_evidence_r11(
        samples=samples, parents=parents, train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102, workers=int(args.teacher_workers),
        block_targets=int(args.teacher_block_targets))
    train_summary = evidence_summary(train_evidence); validation_summary = evidence_summary(validation_evidence)
    require(train_summary["admitted_dependence_groups"] >= 32, "R11_R2_TRAIN_EVIDENCE_SUPPORT_NOT_READY")
    require(validation_summary["admitted_dependence_groups"] >= 8, "R11_R2_VALIDATION_EVIDENCE_SUPPORT_NOT_READY")

    law_audit_train = audit_teacher_action_laws_r11(train_evidence)
    law_audit_validation = audit_teacher_action_laws_r11(validation_evidence)
    crossfit_canary = crossfit_self_future_exclusion_canary(samples=samples, parents=parents, train_evidence=train_evidence)
    replica_canary = same_future_replica_sensitivity_canary(samples=samples, parents=parents,
                                                             train_evidence=train_evidence,
                                                             replicas=int(args.same_future_replicas))
    replay_canary = replay_dedup_canary(work_root=work_root, g0_root=g0_root,
                                         train_evidence=train_evidence, parents=parents)

    original_prepared = PreparedEvidenceR11.from_evidence(train_evidence, parents, device=args.device)
    tail_prepared = PreparedEvidenceR11.from_evidence(tail_only_law_perturbation_r11(train_evidence),
                                                       parents, device=args.device)
    target_prepared = PreparedEvidenceR11.from_evidence(projected_target_perturbation_r11(train_evidence),
                                                         parents, device=args.device)
    model = r1.load_bootstrap_model(g0_root, args.device)
    tail_projection = compare_student_projection_paths_r11(model=model, original=original_prepared,
                                                            variant=tail_prepared, atol=1e-7)
    target_projection = compare_student_projection_paths_r11(model=model, original=original_prepared,
                                                              variant=target_prepared, atol=1e-7)
    boundary = classify_distributional_boundary_r11(teacher_law_audit=law_audit_train,
                                                      tail_projection=tail_projection,
                                                      target_projection_control=target_projection)

    frozen_after = frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "R11_R2_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_authority_before, "R11_R2_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()

    finding = replica_canary["status"] != "INVARIANT"
    result = {
        "schema": SCHEMA,
        "status": "R11_DISTRIBUTIONAL_FEEDBACK_BOUNDARY_AUDIT_COMPLETE_WITH_FINDING" if finding
                  else "R11_DISTRIBUTIONAL_FEEDBACK_BOUNDARY_AUDIT_COMPLETE",
        "repo": repo, "runtime": runtime,
        "historical_scope": {"symbol": args.symbol, "train_complete_parent_groups": int(args.train_groups),
            "validation_complete_parent_groups": int(args.validation_groups),
            "counterfactual_branch_samples": support["counterfactual_branch_samples"],
            "final_holdout_payload_opened": False, "fresh_market_data_downloaded": False, "network_reads_by_r2": 0},
        "sensory": sensory_receipt,
        "teacher": {"runtime_stats": asdict(teacher_stats), "train_summary": train_summary,
            "validation_summary": validation_summary, "train_distributional_law_audit": law_audit_train,
            "validation_distributional_law_audit": law_audit_validation},
        "crossfit_self_future_exclusion": crossfit_canary,
        "same_future_replica_sensitivity": replica_canary,
        "replay_identity_and_support": replay_canary,
        "student_distributional_projection": {"tail_only_law_perturbation": tail_projection,
            "projected_target_positive_control": target_projection, "boundary_classification": boundary},
        "qr_dqn_iqn_adjudication": {
            "convert_cb16_to_qr_dqn": "NO",
            "reason": "CB16_IS_FIXED_HORIZON_EVIDENCE_GATED_PROBABILISTIC_DECISION_LEARNING_WITH_CONTINUOUS_REQUESTED_RISK_AND_FROZEN_PERMISSION_PHYSICS__NOT_A_DISCRETE_ACTION_BELLMAN_Q_CONTROL_PROBLEM",
            "borrowable_only_after_separate_gate": ["QUANTILE_OR_CDF_BELIEF_REPRESENTATION",
                "PINBALL_OR_QUANTILE_HUBER_AUXILIARY_SCORING", "MONOTONE_QUANTILE_FAIL_CLOSED_CONSTRAINT"],
            "forbidden_without_objective_revision": ["DISTORTION_RISK_MEASURE_AS_POLICY_OBJECTIVE",
                "CVAR_POLICY_REPLACEMENT", "MEAN_VARIANCE_POLICY_REPLACEMENT", "TAIL_PENALTY_POLICY_REPLACEMENT"],
            "bellman_bootstrap_authorized": False},
        "semantic_guards": {"canonical_generation_advanced": False, "champion_promoted": False,
            "tournament_run": False, "student_architecture_changed": False, "teacher_semantics_changed": False,
            "economic_objective_changed": False, "scientific_verdict_created": False,
            "market_information_qualified": False, "profitability_or_alpha_claimed": False,
            "final_holdout_payload_opened": False, "fresh_market_data_downloaded": False},
        "authority": {"frozen_scientific_status_before": r1.FROZEN_SCIENTIFIC_STATUS,
            "frozen_scientific_status_after": r1.FROZEN_SCIENTIFIC_STATUS, "g0_authority_hashes_unchanged": True},
    }
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "distributional_gradient_diagnosis": boundary["diagnosis"],
        "tail_only_changes_student_gradient": boundary["tail_only_law_changes_student_gradient"],
        "projected_targets_change_student_gradient": boundary["projected_target_positive_control_changes_gradient"],
        "crossfit_self_future_exclusion": crossfit_canary["status"],
        "same_future_replica_sensitivity": replica_canary["status"],
        "same_future_replica_mean_max_abs_delta": replica_canary["maximum_action_mean_utility_abs_delta"],
        "replay_creates_new_evidence": replay_canary["identical_replay"]["created_evidence_count"],
        "replay_creates_new_payload": replay_canary["identical_replay"]["created_payload_count"],
        "qr_dqn_conversion": "NO", "scientific_verdict_created": False,
        "final_holdout_payload_opened": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
