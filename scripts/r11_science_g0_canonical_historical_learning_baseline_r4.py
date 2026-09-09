#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 R4 — canonical six-owner historical-learning baseline.

R4 is the first post-R3 return to the canonical Student gradient path.  It uses
R2.4 production Teacher evidence on real frozen multi-account history, starts a
disposable Challenger from the immutable R11 G0 Champion, trains with the exact
R11 canonical FP32 AdamW rule, and produces a threshold-free shadow tournament
ranking.  Canonical G0/G1 state is never mutated here.
"""

import argparse
from dataclasses import asdict
import copy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from cb16_local_opt.canonical_historical_learning_baseline_r4 import (
    R4_RUNTIME,
    build_shadow_snapshot_r4,
    shadow_champion_challenger_ranking_r4,
)
from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_learning import evidence_summary
from cb16_local_opt.r102_market import load_anchor_frames
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
from cb16_local_opt.r11_teacher_authority_candidate import (
    R11_TRAIN_TEACHER_CONFIG,
    R11_VALIDATION_TEACHER_CONFIG,
)
from cb16_local_opt.science_feedback_diagnostics_r11 import (
    audit_training_feedback_r11,
    behavior_delta_r11,
    capture_behavior_r11,
    independent_loss_breakdown_r11,
    validate_teacher_targets_r11,
)
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.training_integration_r11 import IntegratedTrainingRuntimeR11
from cb16_local_opt.training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    CANONICAL_BATCH_SIZE_R11,
    CANONICAL_EPOCHS_R11,
    CANONICAL_LR_R11,
    CANONICAL_WEIGHT_DECAY_R11,
    policy_hash_r11,
    prepare_evidence_campaign_r11,
)
from scripts import r11_science_g0_geometry_r2_1 as r21
from scripts import r11_science_g0_historical_r1 as r1


SCHEMA = "CB16_R11_SCIENCE_G0_CANONICAL_HISTORICAL_LEARNING_BASELINE_R4_RESULT_V1"
R34_ADJUDICATION_SEED = "5a500e572b20f26bb597db726bb768bb46166c10"
FROZEN_SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
SEMANTIC_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
TYPED_CENTRAL_BRAIN_BLOB = "899065a9b9c015fd5a0e24fc919a79ff7d3a67ff"
TRAINING_RUNTIME_BLOB = "56c227800d7a5c82ef88c691bb69d7550aee8b39"
TRAINING_INTEGRATION_BLOB = "e604467f4288326f56d1c20dbe2634e442e37a63"
FEEDBACK_DIAGNOSTICS_BLOB = "9c9e484deb227c67211d0575eeda4756e2a6e42d"
PROBABILISTIC_TEACHER_BLOB = "3de3092d244c3fd315950c9f8b0bf4d78c50ee2b"
TEACHER_RUNTIME_BLOB = "656f1483f2e6a96daae5c98eb96cd75805d3547c"
TEACHER_AUTHORITY_CANDIDATE_BLOB = "6e77e8f989a6d4358a7feac99c269db7b7d6549e"
R1_SCRIPT_BLOB = "9fcbd58275208423ce2a6bb20ec1c54739c6e17f"
R21_SCRIPT_BLOB = "131cc17d983c8daccbc5df919ebee02b3e4f9b05"
R34_ADJUDICATION_BLOB = "fe4865c5c559f509f0dfbbcff0fc0d059f2b833c"


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_equal(before: Mapping[str, torch.Tensor], model: torch.nn.Module) -> bool:
    now = model.state_dict()
    return set(before) == set(now) and all(torch.equal(before[k], now[k].detach().cpu()) for k in before)


def metric_close(a: Mapping[str, Any], b: Mapping[str, Any], *, atol: float = 1e-6) -> bool:
    for key in ("loss", "direction_loss", "sizing_loss"):
        if abs(float(a[key]) - float(b[key])) > atol:
            return False
    return True


def market_cache_stats_receipt_r4(market_cache: MarketRuntimeCacheR11) -> dict[str, Any]:
    """Serialize cache telemetry without deep-copying its MappingProxyType field."""
    stats = market_cache.stats()
    return {
        "compressed_loads_total": int(stats.compressed_loads_total),
        "index_builds_total": int(stats.index_builds_total),
        "loaded_symbols": list(stats.loaded_symbols),
        "compressed_loads_by_symbol": {
            str(symbol): int(count)
            for symbol, count in stats.compressed_loads_by_symbol.items()
        },
    }


def save_shadow_challenger(path: Path, model: torch.nn.Module, *, champion_hash: str, snapshot_hash: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "CB16_R11_G0_R4_SHADOW_CHALLENGER_CHECKPOINT_V1",
            "role": "SHADOW_CHALLENGER__NOT_CANONICAL_CHAMPION",
            "canonical_generation_advanced": False,
            "parent_champion_policy_hash": champion_hash,
            "snapshot_hash": snapshot_hash,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        tmp,
    )
    os.replace(tmp, path)
    return {
        "path": str(path),
        "file_sha256": r1.sha256_file(path),
        "policy_semantic_sha256": policy_hash_r11(model),
        "canonical_authority": False,
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
    ap.add_argument("--teacher-workers", type=int, default=4)
    ap.add_argument("--teacher-block-targets", type=int, default=32)
    args = ap.parse_args()

    work_root = args.work_root.resolve(); work_root.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve(); g0_root = args.g0_root.resolve(); package_root = args.package_root.resolve()
    require(args.symbol == "BTCUSDT", "R11_R4_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(int(args.train_groups) == 48 and int(args.validation_groups) == 12, "R11_R4_PRE_REGISTERED_SUPPORT_DRIFT")
    require(int(args.candidate_factor) == 2, "R11_R4_CANDIDATE_FACTOR_DRIFT")
    require(work_root != g0_root and g0_root not in work_root.parents, "R11_R4_WORK_ROOT_OVERLAPS_CANONICAL_G0")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R34_ADJUDICATION_SEED, "HEAD"]) == 0, "R11_R4_NOT_DESCENDED_FROM_R34_ADJUDICATION")

    immutable_blobs = {
        "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", SEMANTIC_FREEZE_BLOB),
        "typed_central_brain": ("cb16_local_opt/typed_central_brain_r10.py", TYPED_CENTRAL_BRAIN_BLOB),
        "training_runtime": ("cb16_local_opt/training_runtime_r11.py", TRAINING_RUNTIME_BLOB),
        "training_integration": ("cb16_local_opt/training_integration_r11.py", TRAINING_INTEGRATION_BLOB),
        "feedback_diagnostics": ("cb16_local_opt/science_feedback_diagnostics_r11.py", FEEDBACK_DIAGNOSTICS_BLOB),
        "probabilistic_teacher": ("cb16_local_opt/probabilistic_teacher_r6.py", PROBABILISTIC_TEACHER_BLOB),
        "teacher_runtime": ("cb16_local_opt/teacher_runtime_r11.py", TEACHER_RUNTIME_BLOB),
        "teacher_authority_candidate": ("cb16_local_opt/r11_teacher_authority_candidate.py", TEACHER_AUTHORITY_CANDIDATE_BLOB),
        "historical_r1_script": ("scripts/r11_science_g0_historical_r1.py", R1_SCRIPT_BLOB),
        "multi_account_support_r21": ("scripts/r11_science_g0_geometry_r2_1.py", R21_SCRIPT_BLOB),
        "r34_adjudication": ("authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_DISTRIBUTIONAL_BELIEF_DECISION_BRIDGE_R3_4_ADJUDICATION_V1.json", R34_ADJUDICATION_BLOB),
    }
    for name, (path, expected) in immutable_blobs.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected, f"R11_R4_IMMUTABLE_BLOB_DRIFT:{name}:{observed}")

    repo = r1.verify_repo_seed(); repo["r34_adjudication_seed"] = R34_ADJUDICATION_SEED; repo["execution_head"] = git("rev-parse", "HEAD")
    runtime = r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"R11_R4_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    lineage, g0_identity = r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_before = dict(g0_identity["authority_file_hashes"])

    market_cache = MarketRuntimeCacheR11(g0_root)
    market = market_cache.get(args.symbol); market_cache.assert_read_only()
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R4_ANCHOR_SHA_DRIFT")
    frames = load_anchor_frames(args.symbol, anchor_path)
    selected = r1.select_candidate_frames(
        frames,
        train_target=48,
        validation_target=12,
        candidate_factor=2,
    )
    encoded, sensory_receipt = r1.encode_selected_frames(
        package_root=package_root,
        device=args.device,
        selected=selected,
        batch_size=int(args.sensory_batch_size),
    )
    physics = FrozenPhysicsRuntimeR102.load(package_root)
    parents, samples, support = r21.build_multi_account_counterfactual_support(
        symbol=args.symbol,
        selected=selected,
        encoded=encoded,
        physics=physics,
        hourly_ts=market.open_time_ms,
        hourly_ohlcv=market.ohlcv,
        funding=market.funding_rate,
        train_target_groups=48,
        validation_target_groups=12,
    )
    del encoded; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    train_evidence, validation_evidence, teacher_stats = compile_teacher_evidence_r11(
        samples=samples,
        parents=parents,
        train_config=R11_TRAIN_TEACHER_CONFIG,
        val_config=R11_VALIDATION_TEACHER_CONFIG,
        workers=int(args.teacher_workers),
        block_targets=int(args.teacher_block_targets),
    )
    train_summary = evidence_summary(train_evidence)
    validation_summary = evidence_summary(validation_evidence)
    require(train_summary["admitted_dependence_groups"] >= 32, f"R11_R4_TRAIN_TEACHER_SUPPORT_NOT_READY:{train_summary}")
    require(validation_summary["admitted_dependence_groups"] >= 8, f"R11_R4_VALIDATION_TEACHER_SUPPORT_NOT_READY:{validation_summary}")
    require({e.teacher_protocol_hash for e in train_evidence} == {R11_TRAIN_TEACHER_CONFIG.content_hash}, "R11_R4_TRAIN_TEACHER_PROTOCOL_DRIFT")
    require({e.teacher_protocol_hash for e in validation_evidence} == {R11_VALIDATION_TEACHER_CONFIG.content_hash}, "R11_R4_VALIDATION_TEACHER_PROTOCOL_DRIFT")

    campaign = prepare_evidence_campaign_r11(
        train_evidence=train_evidence,
        validation_evidence=validation_evidence,
        parents=parents,
        device=args.device,
    )
    train_target_audit = validate_teacher_targets_r11(campaign.train)
    validation_target_audit = validate_teacher_targets_r11(campaign.validation)
    require(train_target_audit["independent_dependence_groups"] == 48, f"R11_R4_TRAIN_GROUP_COUNT_DRIFT:{train_target_audit}")
    require(validation_target_audit["independent_dependence_groups"] == 12, f"R11_R4_VALIDATION_GROUP_COUNT_DRIFT:{validation_target_audit}")

    champion = r1.load_bootstrap_model(g0_root, args.device)
    champion_state_before = clone_state(champion)
    champion_policy_hash = policy_hash_r11(champion)
    challenger = copy.deepcopy(champion)
    require(policy_hash_r11(challenger) == champion_policy_hash, "R11_R4_CHALLENGER_NOT_EXACT_CHAMPION_COPY")

    trainer = IntegratedTrainingRuntimeR11(device=args.device)
    require(trainer.config.epochs == CANONICAL_EPOCHS_R11, "R11_R4_EPOCH_RULE_DRIFT")
    require(trainer.config.batch_size == CANONICAL_BATCH_SIZE_R11, "R11_R4_BATCH_RULE_DRIFT")
    require(abs(trainer.config.lr - CANONICAL_LR_R11) <= 0.0, "R11_R4_LR_RULE_DRIFT")
    require(abs(trainer.config.weight_decay - CANONICAL_WEIGHT_DECAY_R11) <= 0.0, "R11_R4_WEIGHT_DECAY_RULE_DRIFT")

    champion_validation = trainer.evaluation_runtime.evaluate(champion, campaign.validation, use_cache=True)
    independent_before = independent_loss_breakdown_r11(champion, campaign.validation)
    require(metric_close(champion_validation, independent_before), "R11_R4_CHAMPION_VALIDATION_FORMULA_MISMATCH")
    behavior_before = capture_behavior_r11(champion, campaign.validation)

    snapshot = build_shadow_snapshot_r4(
        champion_policy_hash=champion_policy_hash,
        train_evidence_hash=campaign.train.evidence_hash,
        validation_evidence_hash=campaign.validation.evidence_hash,
        train_teacher_protocol_hash=R11_TRAIN_TEACHER_CONFIG.content_hash,
        validation_teacher_protocol_hash=R11_VALIDATION_TEACHER_CONFIG.content_hash,
        train_parent_ids=campaign.train.parent_ids,
        train_dependence_group_ids=campaign.train.dependence_group_ids,
        validation_parent_ids=campaign.validation.parent_ids,
        validation_dependence_group_ids=campaign.validation.dependence_group_ids,
    )
    r1.atomic_json(work_root / "CB16_R11_G0_R4_SHADOW_FROZEN_TRAINING_SNAPSHOT.json", snapshot)

    training_receipt = trainer.train_challenger(
        model=challenger,
        campaign=campaign,
        generation=0,
        snapshot_hash=snapshot["snapshot_hash"],
        receipt_dir=work_root / "shadow_training",
    )
    require(training_receipt["generation_seed"] == 24680, "R11_R4_GENERATION_SEED_DRIFT")
    require(training_receipt["epochs"] == 12 and training_receipt["batch_size"] == 512, "R11_R4_TRAINING_RULE_RECEIPT_DRIFT")
    require(abs(float(training_receipt["lr"]) - 3e-4) <= 0.0, "R11_R4_RECEIPT_LR_DRIFT")
    require(abs(float(training_receipt["weight_decay"]) - 1e-4) <= 0.0, "R11_R4_RECEIPT_WEIGHT_DECAY_DRIFT")
    require(frozenset(training_receipt["gradient_owner_set_last_step"]) == AUTHORIZED_GRADIENT_OWNERS_R11, "R11_R4_GRADIENT_OWNER_SET_DRIFT")
    require(metric_close(training_receipt["validation_before"], champion_validation), "R11_R4_TRAINER_CHAMPION_BASELINE_DRIFT")

    independent_after = independent_loss_breakdown_r11(challenger, campaign.validation)
    behavior_after = capture_behavior_r11(challenger, campaign.validation)
    behavior_delta = behavior_delta_r11(behavior_before, behavior_after)
    feedback = audit_training_feedback_r11(
        training_receipt=training_receipt,
        independent_before=independent_before,
        independent_after=independent_after,
        behavior_delta=behavior_delta,
    )
    ranking = shadow_champion_challenger_ranking_r4(
        champion_validation,
        training_receipt["validation_after"],
    )
    require(ranking["canonical_promotion_authorized"] is False, "R11_R4_CANONICAL_PROMOTION_ACCIDENTALLY_AUTHORIZED")
    require(ranking["canonical_generation_advance_authorized"] is False, "R11_R4_CANONICAL_GENERATION_ACCIDENTALLY_AUTHORIZED")

    challenger_checkpoint = save_shadow_challenger(
        work_root / "R11_G0_R4_SHADOW_CHALLENGER.pt",
        challenger,
        champion_hash=champion_policy_hash,
        snapshot_hash=snapshot["snapshot_hash"],
    )
    require(challenger_checkpoint["policy_semantic_sha256"] == training_receipt["challenger_semantic_sha256"], "R11_R4_CHALLENGER_CHECKPOINT_HASH_DRIFT")

    require(state_equal(champion_state_before, champion), "R11_R4_CHAMPION_MODEL_MUTATED")
    require(policy_hash_r11(champion) == champion_policy_hash, "R11_R4_CHAMPION_POLICY_HASH_MUTATED")
    require(frozen_authority_hashes(package_root) == frozen_before, "R11_R4_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R4_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()
    distributional_modules_loaded = sorted(
        name for name in sys.modules
        if "distributional_student" in name or "distributional_belief_decision_bridge" in name
    )
    require(not distributional_modules_loaded, f"R11_R4_DISTRIBUTIONAL_R3_MODULE_IMPORTED:{distributional_modules_loaded}")

    result = {
        "schema": SCHEMA,
        "status": "R11_CANONICAL_HISTORICAL_LEARNING_BASELINE_R4_PASS",
        "repo": repo,
        "runtime": runtime,
        "authority": {
            "r34_adjudication_seed": R34_ADJUDICATION_SEED,
            "immutable_blobs": {k: {"path": p, "git_blob": h} for k, (p, h) in immutable_blobs.items()},
            "frozen_scientific_status_before": FROZEN_SCIENTIFIC_STATUS,
            "frozen_scientific_status_after": FROZEN_SCIENTIFIC_STATUS,
            "g0_authority_hashes_unchanged": True,
        },
        "protocol": {
            "runtime": R4_RUNTIME,
            "symbol": args.symbol,
            "train_dependence_groups": 48,
            "validation_dependence_groups": 12,
            "multi_account_contexts": True,
            "teacher": "R11_R2_4_DEPENDENCE_BALANCED_PRODUCTION_TEACHER",
            "train_teacher_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash,
            "validation_teacher_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
            "epochs": CANONICAL_EPOCHS_R11,
            "batch_size": CANONICAL_BATCH_SIZE_R11,
            "lr": CANONICAL_LR_R11,
            "weight_decay": CANONICAL_WEIGHT_DECAY_R11,
            "amp": False,
            "dtype": "torch.float32",
            "generation_seed": 24680,
            "optional_stopping": False,
            "distributional_r3_artifacts_in_gradient_graph": False,
        },
        "historical_scope": {
            "support": support,
            "anchor_file": str(anchor_path),
            "anchor_sha256": str(per_asset["anchors_sha256"]),
            "market_cache_stats": market_cache_stats_receipt_r4(market_cache),
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "network_reads_by_r4": 0,
        },
        "sensory": sensory_receipt,
        "teacher": {
            "runtime_stats": asdict(teacher_stats),
            "train": train_summary,
            "validation": validation_summary,
            "action_laws_preserved_in_evidence": True,
            "student_gradient_consumes_reduced_direction_risk_targets_only": True,
            "realized_future_used_as_correct_action_label": False,
        },
        "training_snapshot": snapshot,
        "training_targets": {
            "train": train_target_audit,
            "validation": validation_target_audit,
        },
        "champion": {
            "policy_hash": champion_policy_hash,
            "validation": champion_validation,
            "state_mutated": False,
            "canonical_role": "R11_G0_INITIAL_CHAMPION",
        },
        "challenger": {
            "starts_from_exact_champion": True,
            "training_receipt": training_receipt,
            "checkpoint": challenger_checkpoint,
            "independent_validation_after": independent_after,
        },
        "feedback_diagnostics": feedback,
        "behavior_delta": behavior_delta,
        "shadow_tournament": ranking,
        "semantic_guards": {
            "six_owner_gradient_path_exact": True,
            "distributional_r3_modules_loaded": distributional_modules_loaded,
            "distributional_r3_artifacts_in_gradient_graph": False,
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "canonical_g0_state_mutated": False,
            "numeric_promotion_threshold_selected": False,
            "legacy_r10_promotion_threshold_used": False,
            "historical_market_information_verdict_reopened": False,
            "scientific_verdict_created": False,
            "profitability_or_alpha_claimed": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "live_capital": False,
            "production_trading": False,
            "supervisor_permission_changed": False,
            "physics_changed": False,
        },
        "next_legal_step": "R11_SCIENCE_G0_CANONICAL_HISTORICAL_LEARNING_BASELINE_R4_ADJUDICATION__NO_CANONICAL_PROMOTION_AUTHORIZED",
    }
    r1.atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "champion_validation_loss": champion_validation["loss"],
        "challenger_validation_loss": training_receipt["validation_after"]["loss"],
        "relative_validation_improvement": ranking["relative_validation_improvement"],
        "direction_loss_delta": ranking["direction_loss_delta_challenger_minus_champion"],
        "sizing_loss_delta": ranking["sizing_loss_delta_challenger_minus_champion"],
        "shadow_winner": ranking["shadow_winner"],
        "optimizer_steps": training_receipt["optimizer_steps"],
        "parameter_l2_delta": training_receipt["parameter_l2_delta"],
        "learning_direction": feedback["learning_direction"],
        "canonical_generation_advanced": False,
        "champion_promoted": False,
        "scientific_verdict_created": False,
        "final_holdout_payload_opened": False,
        "fresh_market_data_downloaded": False,
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
