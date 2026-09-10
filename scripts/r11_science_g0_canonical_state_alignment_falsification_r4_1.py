#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 R4.1 — post-R4 state-alignment falsification diagnostic."""

import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from cb16_local_opt.canonical_state_alignment_falsification_r41 import (
    R41_RUNTIME,
    R41_SCENARIOS,
    R41_SHIFTS,
    shuffle_reduced_targets_by_future_group_r41,
    summarize_state_alignment_control_r41,
)
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.training_integration_r11 import IntegratedTrainingRuntimeR11
from cb16_local_opt.training_runtime_r11 import AUTHORIZED_GRADIENT_OWNERS_R11, PreparedCampaignR11, policy_hash_r11
from scripts import r11_science_g0_canonical_historical_learning_baseline_r4 as r4

SCHEMA = "CB16_R11_SCIENCE_G0_CANONICAL_STATE_ALIGNMENT_FALSIFICATION_R4_1_RESULT_V1"
R4_PASSING_HEAD = "d6619d75787d204c72db4f85be5ccb91a695604c"
R4_ADJUDICATION_BLOB = "42efaa9723c66de1f7d0936e2e4526eb312347f9"
FROZEN_SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
R4_REFERENCE = {
    "champion": {"loss": 1.208237886428833, "direction_loss": 1.1262524127960205, "sizing_loss": 0.0819854661822319},
    "aligned": {"loss": 1.0994198322296143, "direction_loss": 1.0919666290283203, "sizing_loss": 0.007453371305018663},
}
R4_REFERENCE_ATOL = 1e-7
IMMUTABLE_BLOBS = {
    "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", r4.SEMANTIC_FREEZE_BLOB),
    "typed_central_brain": ("cb16_local_opt/typed_central_brain_r10.py", r4.TYPED_CENTRAL_BRAIN_BLOB),
    "training_runtime": ("cb16_local_opt/training_runtime_r11.py", r4.TRAINING_RUNTIME_BLOB),
    "training_integration": ("cb16_local_opt/training_integration_r11.py", r4.TRAINING_INTEGRATION_BLOB),
    "feedback_diagnostics": ("cb16_local_opt/science_feedback_diagnostics_r11.py", r4.FEEDBACK_DIAGNOSTICS_BLOB),
    "probabilistic_teacher": ("cb16_local_opt/probabilistic_teacher_r6.py", r4.PROBABILISTIC_TEACHER_BLOB),
    "teacher_runtime": ("cb16_local_opt/teacher_runtime_r11.py", r4.TEACHER_RUNTIME_BLOB),
    "teacher_authority_candidate": ("cb16_local_opt/r11_teacher_authority_candidate.py", r4.TEACHER_AUTHORITY_CANDIDATE_BLOB),
    "historical_r1_script": ("scripts/r11_science_g0_historical_r1.py", r4.R1_SCRIPT_BLOB),
    "multi_account_support_r21": ("scripts/r11_science_g0_geometry_r2_1.py", r4.R21_SCRIPT_BLOB),
    "r4_helper": ("cb16_local_opt/canonical_historical_learning_baseline_r4.py", "0180ea47b259a94b2e47b51bc186b5e43d1d96a0"),
    "r4_gate": ("authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_CANONICAL_HISTORICAL_LEARNING_BASELINE_R4_GATEWORK_V1.json", "c355dae5dc02f28844291c3d1d6439c1db14577e"),
    "r4_script": ("scripts/r11_science_g0_canonical_historical_learning_baseline_r4.py", "40454165a6d818a11242351b74af8d9cb4b26f77"),
    "r4_adjudication": ("authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_CANONICAL_HISTORICAL_LEARNING_BASELINE_R4_ADJUDICATION_V1.json", R4_ADJUDICATION_BLOB),
}


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def metric_close(a: Mapping[str, Any], b: Mapping[str, Any], *, atol: float) -> bool:
    return all(abs(float(a[k]) - float(b[k])) <= atol for k in ("loss", "direction_loss", "sizing_loss"))


def arm_snapshot_hash(base_snapshot_hash: str, *, arm: str, shift: int | None) -> str:
    core = {"schema": "CB16_R11_R4_1_ARM_SNAPSHOT_V1", "base_r4_snapshot_hash": str(base_snapshot_hash), "arm": str(arm), "shift": None if shift is None else int(shift), "canonical_generation": 0, "post_r4_diagnostic_not_status_driving": True}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_training_receipt(receipt: Mapping[str, Any], *, expected_validation_before: Mapping[str, Any]) -> None:
    require(receipt["generation_seed"] == 24680, "R11_R41_GENERATION_SEED_DRIFT")
    require(receipt["epochs"] == 12 and receipt["batch_size"] == 512, "R11_R41_TRAINING_RULE_DRIFT")
    require(receipt["lr"] == 3e-4 and receipt["weight_decay"] == 1e-4, "R11_R41_OPTIMIZER_RULE_DRIFT")
    require(receipt["optimizer"] == "AdamW_FP32" and receipt["amp"] is False and receipt["dtype"] == "torch.float32", "R11_R41_NUMERIC_RULE_DRIFT")
    require(frozenset(receipt["gradient_owner_set_last_step"]) == AUTHORIZED_GRADIENT_OWNERS_R11, "R11_R41_GRADIENT_OWNER_SET_DRIFT")
    require(all(math.isfinite(float(v)) and float(v) > 0.0 for v in receipt["gradient_group_norms_last_step"].values()), "R11_R41_NONPOSITIVE_GRADIENT_OWNER")
    require(all(math.isfinite(float(v)) and float(v) > 0.0 for v in receipt["update_group_norms"].values()), "R11_R41_NONPOSITIVE_UPDATE_OWNER")
    require(float(receipt["parameter_l2_delta"]) > 0.0, "R11_R41_ZERO_PARAMETER_DELTA")
    require(metric_close(receipt["validation_before"], expected_validation_before, atol=1e-7), "R11_R41_VALIDATION_BASELINE_DRIFT")


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
    ap.add_argument("--teacher-workers", type=int, default=4)
    ap.add_argument("--teacher-block-targets", type=int, default=32)
    args = ap.parse_args()
    work_root = args.work_root.resolve(); work_root.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve(); g0_root = args.g0_root.resolve(); package_root = args.package_root.resolve()
    require(args.symbol == "BTCUSDT", "R11_R41_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(int(args.train_groups) == 48 and int(args.validation_groups) == 12, "R11_R41_PRE_REGISTERED_SUPPORT_DRIFT")
    require(int(args.candidate_factor) == 2, "R11_R41_CANDIDATE_FACTOR_DRIFT")
    require(tuple(R41_SHIFTS) == (1, 7, 13, 23, 31), "R11_R41_SHIFT_PROTOCOL_DRIFT")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R4_PASSING_HEAD, "HEAD"]) == 0, "R11_R41_NOT_DESCENDED_FROM_R4_PASS")
    for name, (path, expected) in IMMUTABLE_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected, f"R11_R41_IMMUTABLE_BLOB_DRIFT:{name}:{observed}")
    repo = r4.r1.verify_repo_seed(); repo["r4_passing_head"] = R4_PASSING_HEAD; repo["execution_head"] = git("rev-parse", "HEAD")
    runtime = r4.r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"R11_R41_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    lineage, g0_identity = r4.r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root); g0_before = dict(g0_identity["authority_file_hashes"])
    market_cache = r4.MarketRuntimeCacheR11(g0_root); market = market_cache.get(args.symbol); market_cache.assert_read_only()
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r4.r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R41_ANCHOR_SHA_DRIFT")
    frames = r4.load_anchor_frames(args.symbol, anchor_path)
    selected = r4.r1.select_candidate_frames(frames, train_target=48, validation_target=12, candidate_factor=2)
    encoded, sensory_receipt = r4.r1.encode_selected_frames(package_root=package_root, device=args.device, selected=selected, batch_size=int(args.sensory_batch_size))
    physics = r4.FrozenPhysicsRuntimeR102.load(package_root)
    parents, samples, support = r4.r21.build_multi_account_counterfactual_support(symbol=args.symbol, selected=selected, encoded=encoded, physics=physics, hourly_ts=market.open_time_ms, hourly_ohlcv=market.ohlcv, funding=market.funding_rate, train_target_groups=48, validation_target_groups=12)
    del encoded; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    train_evidence, validation_evidence, teacher_stats = r4.compile_teacher_evidence_r11(samples=samples, parents=parents, train_config=r4.R11_TRAIN_TEACHER_CONFIG, val_config=r4.R11_VALIDATION_TEACHER_CONFIG, workers=int(args.teacher_workers), block_targets=int(args.teacher_block_targets))
    train_summary = r4.evidence_summary(train_evidence); validation_summary = r4.evidence_summary(validation_evidence)
    require(train_summary["admitted_dependence_groups"] == 48 and validation_summary["admitted_dependence_groups"] == 12, "R11_R41_TEACHER_GROUP_COUNT_DRIFT")
    campaign = r4.prepare_evidence_campaign_r11(train_evidence=train_evidence, validation_evidence=validation_evidence, parents=parents, device=args.device)
    aligned_target_audit = r4.validate_teacher_targets_r11(campaign.train); validation_target_audit = r4.validate_teacher_targets_r11(campaign.validation)
    require(aligned_target_audit["independent_dependence_groups"] == 48 and validation_target_audit["independent_dependence_groups"] == 12, "R11_R41_PREPARED_GROUP_COUNT_DRIFT")
    champion = r4.r1.load_bootstrap_model(g0_root, args.device)
    champion_state = r4.clone_state(champion); champion_hash = policy_hash_r11(champion)
    baseline_trainer = IntegratedTrainingRuntimeR11(device=args.device)
    champion_validation = baseline_trainer.evaluation_runtime.evaluate(champion, campaign.validation, use_cache=True)
    require(metric_close(champion_validation, R4_REFERENCE["champion"], atol=R4_REFERENCE_ATOL), "R11_R41_R4_CHAMPION_REFERENCE_NOT_REPRODUCED")
    behavior_before = r4.capture_behavior_r11(champion, campaign.validation)
    base_snapshot = r4.build_shadow_snapshot_r4(champion_policy_hash=champion_hash, train_evidence_hash=campaign.train.evidence_hash, validation_evidence_hash=campaign.validation.evidence_hash, train_teacher_protocol_hash=r4.R11_TRAIN_TEACHER_CONFIG.content_hash, validation_teacher_protocol_hash=r4.R11_VALIDATION_TEACHER_CONFIG.content_hash, train_parent_ids=campaign.train.parent_ids, train_dependence_group_ids=campaign.train.dependence_group_ids, validation_parent_ids=campaign.validation.parent_ids, validation_dependence_group_ids=campaign.validation.dependence_group_ids)
    aligned_model = copy.deepcopy(champion)
    aligned_trainer = IntegratedTrainingRuntimeR11(device=args.device)
    aligned_receipt = aligned_trainer.train_challenger(model=aligned_model, campaign=campaign, generation=0, snapshot_hash=base_snapshot["snapshot_hash"], receipt_dir=work_root / "aligned_training")
    validate_training_receipt(aligned_receipt, expected_validation_before=champion_validation)
    require(metric_close(aligned_receipt["validation_after"], R4_REFERENCE["aligned"], atol=R4_REFERENCE_ATOL), "R11_R41_R4_ALIGNED_REFERENCE_NOT_REPRODUCED")
    aligned_behavior = r4.behavior_delta_r11(behavior_before, r4.capture_behavior_r11(aligned_model, campaign.validation))
    shuffled_arms: dict[int, dict[str, Any]] = {}; shuffled_validations: dict[int, Mapping[str, Any]] = {}
    for shift in R41_SHIFTS:
        shuffled_train, shuffle_receipt = shuffle_reduced_targets_by_future_group_r41(campaign.train, shift=int(shift))
        shuffled_audit = r4.validate_teacher_targets_r11(shuffled_train)
        require(shuffled_audit == aligned_target_audit, f"R11_R41_TARGET_MARGINAL_AUDIT_DRIFT:{shift}")
        arm_campaign = PreparedCampaignR11(train=shuffled_train, validation=campaign.validation)
        model = copy.deepcopy(champion)
        require(policy_hash_r11(model) == champion_hash, f"R11_R41_SHUFFLE_ARM_INITIALIZATION_DRIFT:{shift}")
        trainer = IntegratedTrainingRuntimeR11(device=args.device)
        receipt = trainer.train_challenger(model=model, campaign=arm_campaign, generation=0, snapshot_hash=arm_snapshot_hash(base_snapshot["snapshot_hash"], arm="SHUFFLED", shift=int(shift)), receipt_dir=work_root / f"shuffle_{int(shift):02d}_training")
        validate_training_receipt(receipt, expected_validation_before=champion_validation)
        require(receipt["validation_evidence_hash"] == campaign.validation.evidence_hash, f"R11_R41_VALIDATION_EVIDENCE_DRIFT:{shift}")
        behavior = r4.behavior_delta_r11(behavior_before, r4.capture_behavior_r11(model, campaign.validation))
        shuffled_validations[int(shift)] = receipt["validation_after"]
        shuffled_arms[int(shift)] = {"shift": int(shift), "shuffle_receipt": shuffle_receipt, "training_receipt": receipt, "behavior_delta": behavior, "policy_hash_after": policy_hash_r11(model)}
        del model, trainer, arm_campaign, shuffled_train; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = summarize_state_alignment_control_r41(aligned_receipt["validation_after"], shuffled_validations)
    require(r4.state_equal(champion_state, champion), "R11_R41_CHAMPION_MODEL_MUTATED")
    require(policy_hash_r11(champion) == champion_hash, "R11_R41_CHAMPION_POLICY_HASH_MUTATED")
    require(frozen_authority_hashes(package_root) == frozen_before, "R11_R41_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r4.r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R41_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()
    distributional_modules_loaded = sorted(name for name in sys.modules if "distributional_student" in name or "distributional_belief_decision_bridge" in name)
    require(not distributional_modules_loaded, f"R11_R41_DISTRIBUTIONAL_R3_MODULE_IMPORTED:{distributional_modules_loaded}")
    result = {
        "schema": SCHEMA,
        "status": "R11_CANONICAL_STATE_ALIGNMENT_FALSIFICATION_R4_1_PASS",
        "repo": repo,
        "runtime": runtime,
        "authority": {"r4_passing_head": R4_PASSING_HEAD, "r4_adjudication_blob": R4_ADJUDICATION_BLOB, "immutable_blobs": {k: {"path": p, "git_blob": h} for k, (p, h) in IMMUTABLE_BLOBS.items()}, "frozen_scientific_status_before": FROZEN_SCIENTIFIC_STATUS, "frozen_scientific_status_after": FROZEN_SCIENTIFIC_STATUS},
        "protocol": {"runtime": R41_RUNTIME, "role": "POST_R4_DIAGNOSTIC_NOT_STATUS_DRIVING", "symbol": args.symbol, "train_dependence_groups": 48, "validation_dependence_groups": 12, "account_scenarios": list(R41_SCENARIOS), "pre_registered_shifts": list(R41_SHIFTS), "epochs": 12, "batch_size": 512, "lr": 3e-4, "weight_decay": 1e-4, "generation_seed": 24680, "amp": False, "dtype": "torch.float32", "validation_reused_from_r4": True, "status_driving": False, "optional_stopping": False},
        "historical_scope": {"support": support, "anchor_sha256": str(per_asset["anchors_sha256"]), "market_cache_stats": r4.market_cache_stats_receipt_r4(market_cache), "final_holdout_payload_opened": False, "fresh_market_data_downloaded": False, "network_reads_by_r41": 0},
        "sensory": sensory_receipt,
        "teacher": {"runtime_stats": asdict(teacher_stats), "train": train_summary, "validation": validation_summary, "student_gradient_consumes_reduced_direction_risk_targets_only": True, "realized_future_used_as_correct_action_label": False},
        "targets": {"aligned_train": aligned_target_audit, "original_validation": validation_target_audit},
        "champion": {"policy_hash": champion_hash, "validation": champion_validation, "state_mutated": False},
        "aligned_arm": {"starts_from_exact_champion": True, "training_receipt": aligned_receipt, "behavior_delta": aligned_behavior, "r4_reference_reproduced": True},
        "shuffled_arms": {str(k): v for k, v in shuffled_arms.items()},
        "control_summary": summary,
        "semantic_guards": {"post_r4_diagnostic_not_status_driving": True, "original_validation_used_for_all_arms": True, "target_marginals_exactly_preserved_in_all_shuffles": True, "whole_future_group_shuffle_only": True, "scenario_identity_preserved": True, "canonical_generation_advanced": False, "champion_promoted": False, "canonical_g0_state_mutated": False, "historical_market_information_verdict_reopened": False, "scientific_verdict_created": False, "profitability_or_alpha_claimed": False, "final_holdout_payload_opened": False, "fresh_market_data_downloaded": False, "production_cutover_authorized": False, "distributional_r3_modules_loaded": distributional_modules_loaded},
        "next_legal_step": "R11_SCIENCE_G0_CANONICAL_STATE_ALIGNMENT_FALSIFICATION_R4_1_ADJUDICATION__NO_PROMOTION_AUTHORIZED",
    }
    r4.r1.atomic_json(output, result)
    print(json.dumps({"status": result["status"], "aligned_validation_loss": aligned_receipt["validation_after"]["loss"], "aligned_lower_total_count": summary["aligned_lower_total_count"], "aligned_lower_direction_count": summary["aligned_lower_direction_count"], "aligned_lower_sizing_count": summary["aligned_lower_sizing_count"], "median_aligned_minus_shuffled_total": summary["median_aligned_minus_shuffled_total"], "descriptive_pattern": summary["descriptive_pattern"], "status_driving": False, "champion_promoted": False, "scientific_verdict_created": False, "next_legal_step": result["next_legal_step"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
