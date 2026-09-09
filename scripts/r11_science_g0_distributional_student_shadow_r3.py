#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 R3 — distributional Student belief shadow qualification.

R3 is a bounded mechanics/authority gate only. It consumes the already-bound
R11 Teacher ``action_laws`` quantiles and proves that a small Student-owned
belief head can learn them without changing the production Tier-1 Student,
external action contract, Supervisor/Permission, Physics, generation state, or
frozen market-information verdict. No realized future is used as a direct
Student label.
"""

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

from cb16_local_opt.distributional_student_shadow_r3 import (
    DistributionalEvidenceBatchR3,
    DistributionalStudentShadowHeadR3,
    R11_DISTRIBUTIONAL_STUDENT_SHADOW_R3,
    R3_BELIEF_SEMANTICS,
    R3_DISTANCE,
    R3_GRADIENT_OWNER,
    R3_SHARED_INPUT_GRADIENT_POLICY,
    assert_shadow_gradient_ownership_r3,
    quantile_diagnostics_r3,
    shadow_parameter_report_r3,
    shared_representation_r3,
    truncated_quantile_w1_loss_r3,
)
from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_learning import evidence_summary
from cb16_local_opt.r102_market import load_anchor_frames
from cb16_local_opt.r102_physics import CANDIDATES_R102, FrozenPhysicsRuntimeR102
from cb16_local_opt.r11_teacher_authority_candidate import (
    R11_TRAIN_TEACHER_CONFIG,
    R11_VALIDATION_TEACHER_CONFIG,
)
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.training_runtime_r11 import policy_hash_r11
from scripts import r11_science_g0_geometry_r2_1 as r21
from scripts import r11_science_g0_historical_r1 as r1


SCHEMA = "CB16_R11_SCIENCE_G0_DISTRIBUTIONAL_STUDENT_SHADOW_R3_RESULT_V1"
R2_4_PASS_SEED = "155b5ce8f444ed0e92954ac892d346812920ae90"
SEMANTIC_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
TYPED_CENTRAL_BRAIN_BLOB = "899065a9b9c015fd5a0e24fc919a79ff7d3a67ff"
TRAINING_RUNTIME_BLOB = "56c227800d7a5c82ef88c691bb69d7550aee8b39"
PROBABILISTIC_TEACHER_BLOB = "3de3092d244c3fd315950c9f8b0bf4d78c50ee2b"
TEACHER_RUNTIME_BLOB = "656f1483f2e6a96daae5c98eb96cd75805d3547c"
FROZEN_SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
SHADOW_SEED = 31_337
SHADOW_STEPS = 32
SHADOW_LR = 2e-3
SHADOW_WEIGHT_DECAY = 0.0
SHADOW_GRAD_CLIP = 10.0


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


def state_equal(before: dict[str, torch.Tensor], model: torch.nn.Module) -> bool:
    now = model.state_dict()
    return set(before) == set(now) and all(torch.equal(before[k], now[k].detach().cpu()) for k in before)


def state_l2_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in before:
        d = after[key].detach().cpu().double() - before[key].detach().cpu().double()
        total += float(torch.sum(d * d))
    return math.sqrt(total)


def production_actions(model, batch: DistributionalEvidenceBatchR3) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        out = model(batch.operator48, batch.medium48, batch.account6)
        action = model.compose_action(out)
    return {k: v.detach().cpu().clone() for k, v in action.items()}


def action_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, Any]:
    require(set(before) == set(after), "R11_R3_PRODUCTION_ACTION_FIELD_DRIFT")
    direction_changed = int(torch.sum(before["direction"] != after["direction"]).item())
    class_changed = int(torch.sum(before["direction_class"] != after["direction_class"]).item())
    risk_delta = torch.abs(before["requested_risk"] - after["requested_risk"])
    return {
        "direction_changed_rows": direction_changed,
        "direction_class_changed_rows": class_changed,
        "maximum_abs_requested_risk_delta": float(risk_delta.max().item()) if risk_delta.numel() else 0.0,
        "production_behavior_identical": direction_changed == 0 and class_changed == 0 and bool(torch.all(risk_delta == 0).item()),
    }


def batch_bytes(batch: DistributionalEvidenceBatchR3) -> int:
    tensors = (batch.operator48, batch.medium48, batch.account6, batch.teacher_quantiles, batch.group_weight)
    return sum(int(t.numel() * t.element_size()) for t in tensors)


def loss_value(head, shared, batch: DistributionalEvidenceBatchR3) -> tuple[float, dict[str, Any], torch.Tensor]:
    head.eval()
    with torch.no_grad():
        pred = head(shared)
        loss = truncated_quantile_w1_loss_r3(
            pred, batch.teacher_quantiles, batch.quantile_levels, batch.group_weight
        )
        diag = quantile_diagnostics_r3(pred, batch.teacher_quantiles)
    return float(loss.item()), diag, pred.detach().clone()


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
    ap.add_argument("--shadow-steps", type=int, default=SHADOW_STEPS)
    args = ap.parse_args()

    work_root = args.work_root.resolve(); work_root.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve(); g0_root = args.g0_root.resolve(); package_root = args.package_root.resolve()
    require(args.symbol == "BTCUSDT", "R11_R3_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(int(args.train_groups) >= 32, "R11_R3_TRAIN_GROUPS_BELOW_TEACHER_MINIMUM")
    require(int(args.validation_groups) >= 8, "R11_R3_VALIDATION_GROUPS_BELOW_REQUIRED_SUPPORT")
    require(int(args.shadow_steps) == SHADOW_STEPS, "R11_R3_SHADOW_STEP_RULE_DRIFT")
    require(work_root != g0_root and g0_root not in work_root.parents, "R11_R3_WORK_ROOT_OVERLAPS_CANONICAL_G0")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R2_4_PASS_SEED, "HEAD"]) == 0,
            "R11_R3_NOT_DESCENDED_FROM_R2_4_PASS_SEED")

    immutable_blobs = {
        "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", SEMANTIC_FREEZE_BLOB),
        "typed_central_brain": ("cb16_local_opt/typed_central_brain_r10.py", TYPED_CENTRAL_BRAIN_BLOB),
        "training_runtime": ("cb16_local_opt/training_runtime_r11.py", TRAINING_RUNTIME_BLOB),
        "probabilistic_teacher": ("cb16_local_opt/probabilistic_teacher_r6.py", PROBABILISTIC_TEACHER_BLOB),
        "teacher_runtime": ("cb16_local_opt/teacher_runtime_r11.py", TEACHER_RUNTIME_BLOB),
    }
    for name, (path, expected) in immutable_blobs.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected, f"R11_R3_IMMUTABLE_BLOB_DRIFT:{name}:{observed}")

    repo = r1.verify_repo_seed(); repo["r2_4_pass_seed"] = R2_4_PASS_SEED; repo["execution_head"] = git("rev-parse", "HEAD")
    runtime = r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"R11_R3_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    lineage, g0_identity = r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_before = dict(g0_identity["authority_file_hashes"])

    market_cache = MarketRuntimeCacheR11(g0_root)
    market = market_cache.get(args.symbol)
    market_cache.assert_read_only()
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R3_ANCHOR_SHA_DRIFT")
    frames = load_anchor_frames(args.symbol, anchor_path)
    selected = r1.select_candidate_frames(
        frames,
        train_target=int(args.train_groups),
        validation_target=int(args.validation_groups),
        candidate_factor=int(args.candidate_factor),
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
        train_target_groups=int(args.train_groups),
        validation_target_groups=int(args.validation_groups),
    )
    del encoded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    require(train_summary["admitted_dependence_groups"] >= 32, f"R11_R3_TRAIN_TEACHER_SUPPORT_NOT_READY:{train_summary}")
    require(validation_summary["admitted_dependence_groups"] >= 8, f"R11_R3_VALIDATION_TEACHER_SUPPORT_NOT_READY:{validation_summary}")
    require({e.teacher_protocol_hash for e in train_evidence} == {R11_TRAIN_TEACHER_CONFIG.content_hash}, "R11_R3_TRAIN_TEACHER_PROTOCOL_DRIFT")
    require({e.teacher_protocol_hash for e in validation_evidence} == {R11_VALIDATION_TEACHER_CONFIG.content_hash}, "R11_R3_VALIDATION_TEACHER_PROTOCOL_DRIFT")

    train_batch = DistributionalEvidenceBatchR3.from_evidence(train_evidence, parents, device=args.device)
    validation_batch = DistributionalEvidenceBatchR3.from_evidence(validation_evidence, parents, device=args.device)
    require(train_batch.action_grid == validation_batch.action_grid, "R11_R3_TRAIN_VALIDATION_ACTION_GRID_DRIFT")
    require(train_batch.quantile_levels == validation_batch.quantile_levels, "R11_R3_TRAIN_VALIDATION_QUANTILE_LEVEL_DRIFT")
    require(train_batch.action_grid == tuple((int(d), float(r)) for d, r in CANDIDATES_R102), "R11_R3_ACTION_GRID_NOT_FROZEN_PHYSICS_GRID")
    require(train_batch.quantile_levels == tuple(float(x) for x in R11_TRAIN_TEACHER_CONFIG.quantile_levels), "R11_R3_TEACHER_QUANTILE_LEVEL_NOT_BOUND")

    model = r1.load_bootstrap_model(g0_root, args.device)
    production_before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    production_policy_before = policy_hash_r11(model)
    production_actions_before = production_actions(model, validation_batch)
    train_shared = shared_representation_r3(model, train_batch)
    validation_shared = shared_representation_r3(model, validation_batch)

    torch.manual_seed(SHADOW_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SHADOW_SEED)
    head = DistributionalStudentShadowHeadR3(
        shared_dim=int(train_shared.shape[1]),
        action_count=train_batch.action_count,
        quantile_count=train_batch.quantile_count,
    ).to(args.device, dtype=torch.float32)
    parameter_report = shadow_parameter_report_r3(head)
    require(parameter_report["parameter_count"] == 20_543, f"R11_R3_SHADOW_PARAMETER_COUNT_DRIFT:{parameter_report}")
    shadow_before_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    train_loss_before, train_diag_before, train_pred_before = loss_value(head, train_shared, train_batch)
    validation_loss_before, validation_diag_before, validation_pred_before = loss_value(head, validation_shared, validation_batch)

    optimizer = torch.optim.AdamW(head.parameters(), lr=SHADOW_LR, weight_decay=SHADOW_WEIGHT_DECAY)
    gradient_audit = None
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize()
        baseline_allocated = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_allocated = 0
    t0 = time.perf_counter()
    head.train()
    for step in range(SHADOW_STEPS):
        optimizer.zero_grad(set_to_none=True)
        pred = head(train_shared)
        loss = truncated_quantile_w1_loss_r3(
            pred, train_batch.teacher_quantiles, train_batch.quantile_levels, train_batch.group_weight
        )
        loss.backward()
        if step == 0:
            gradient_audit = assert_shadow_gradient_ownership_r3(
                production_model=model,
                shadow_head=head,
                require_nonzero_shadow_gradient=True,
            )
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), SHADOW_GRAD_CLIP).item())
        require(math.isfinite(grad_norm), "R11_R3_NONFINITE_SHADOW_GRAD_NORM")
        optimizer.step()
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - t0
    require(gradient_audit is not None, "R11_R3_GRADIENT_AUDIT_MISSING")

    train_loss_after, train_diag_after, train_pred_after = loss_value(head, train_shared, train_batch)
    validation_loss_after, validation_diag_after, validation_pred_after = loss_value(head, validation_shared, validation_batch)
    shadow_after_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    shadow_delta = state_l2_delta(shadow_before_state, shadow_after_state)
    train_belief_delta = float(torch.max(torch.abs(train_pred_after - train_pred_before)).item())
    validation_belief_delta = float(torch.max(torch.abs(validation_pred_after - validation_pred_before)).item())
    require(shadow_delta > 0.0, "R11_R3_SHADOW_PARAMETER_DID_NOT_MOVE")
    require(train_belief_delta > 0.0, "R11_R3_SHADOW_BELIEF_DID_NOT_MOVE")
    require(train_loss_after < train_loss_before, f"R11_R3_TRAIN_DISTRIBUTIONAL_LOSS_DID_NOT_DECREASE:{train_loss_before}->{train_loss_after}")
    require(train_diag_after["prediction_quantiles_monotone"] is True, "R11_R3_TRAIN_PREDICTION_QUANTILE_CROSSING")
    require(validation_diag_after["prediction_quantiles_monotone"] is True, "R11_R3_VALIDATION_PREDICTION_QUANTILE_CROSSING")
    require(math.isfinite(validation_loss_after), "R11_R3_NONFINITE_VALIDATION_LOSS")

    production_policy_after = policy_hash_r11(model)
    production_actions_after = production_actions(model, validation_batch)
    behavior = action_delta(production_actions_before, production_actions_after)
    require(production_policy_after == production_policy_before, "R11_R3_PRODUCTION_POLICY_HASH_MUTATED")
    require(state_equal(production_before_state, model), "R11_R3_PRODUCTION_STUDENT_PARAMETER_MUTATED")
    require(behavior["production_behavior_identical"] is True, f"R11_R3_PRODUCTION_BEHAVIOR_CHANGED:{behavior}")

    if torch.device(args.device).type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    else:
        peak_allocated = peak_reserved = 0
    memory = {
        "train_distributional_batch_bytes": batch_bytes(train_batch),
        "validation_distributional_batch_bytes": batch_bytes(validation_batch),
        "shadow_parameter_bytes": int(parameter_report["parameter_bytes"]),
        "cuda_pretraining_baseline_allocated_bytes": baseline_allocated,
        "cuda_training_peak_allocated_bytes": peak_allocated,
        "cuda_training_peak_reserved_bytes": peak_reserved,
        "cuda_incremental_peak_over_pretraining_baseline_bytes": max(0, peak_allocated - baseline_allocated),
    }

    frozen_after = frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "R11_R3_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R3_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()

    result = {
        "schema": SCHEMA,
        "status": "R11_DISTRIBUTIONAL_STUDENT_SHADOW_R3_PASS",
        "repo": repo,
        "runtime": runtime,
        "authority": {
            "r2_4_pass_seed": R2_4_PASS_SEED,
            "immutable_blobs": {k: {"path": p, "git_blob": h} for k, (p, h) in immutable_blobs.items()},
            "frozen_scientific_status_before": FROZEN_SCIENTIFIC_STATUS,
            "frozen_scientific_status_after": FROZEN_SCIENTIFIC_STATUS,
            "g0_authority_hashes_unchanged": True,
            "frozen_package_authority_hashes_unchanged": True,
        },
        "historical_scope": {
            "symbol": args.symbol,
            "scope_role": "BOUNDED_DISTRIBUTIONAL_STUDENT_MECHANICS_SHADOW__NOT_MARKET_RULE",
            "train_dependence_groups_requested": int(args.train_groups),
            "validation_dependence_groups_requested": int(args.validation_groups),
            "support": support,
            "anchor_file": str(anchor_path),
            "anchor_sha256": str(per_asset["anchors_sha256"]),
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "network_reads_by_r3": 0,
        },
        "sensory": sensory_receipt,
        "teacher": {
            "runtime_stats": asdict(teacher_stats),
            "train": train_summary,
            "validation": validation_summary,
            "train_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash,
            "validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
            "action_laws_consumed": True,
            "realized_future_used_as_direct_student_label": False,
        },
        "distributional_target": {
            "belief_semantics": R3_BELIEF_SEMANTICS,
            "distance": R3_DISTANCE,
            "action_grid": [list(x) for x in train_batch.action_grid],
            "action_count": train_batch.action_count,
            "quantile_levels": list(train_batch.quantile_levels),
            "quantile_count": train_batch.quantile_count,
            "train_rows": train_batch.rows,
            "validation_rows": validation_batch.rows,
            "train_evidence_hash": train_batch.evidence_hash,
            "validation_evidence_hash": validation_batch.evidence_hash,
            "dependence_group_weighting": "EQUAL_TOTAL_MASS_PER_INDEPENDENT_FUTURE_GROUP",
        },
        "shadow_student": {
            "runtime": R11_DISTRIBUTIONAL_STUDENT_SHADOW_R3,
            "gradient_owner": R3_GRADIENT_OWNER,
            "shared_input_gradient_policy": R3_SHARED_INPUT_GRADIENT_POLICY,
            "parameter_report": parameter_report,
            "optimizer": "AdamW",
            "steps": SHADOW_STEPS,
            "lr": SHADOW_LR,
            "weight_decay": SHADOW_WEIGHT_DECAY,
            "grad_clip": SHADOW_GRAD_CLIP,
            "seed": SHADOW_SEED,
            "parameter_l2_delta": shadow_delta,
            "gradient_audit_first_step": gradient_audit,
            "train_belief_max_abs_delta": train_belief_delta,
            "validation_belief_max_abs_delta": validation_belief_delta,
            "train_loss_before": train_loss_before,
            "train_loss_after": train_loss_after,
            "validation_loss_before": validation_loss_before,
            "validation_loss_after": validation_loss_after,
            "train_diagnostics_before": train_diag_before,
            "train_diagnostics_after": train_diag_after,
            "validation_diagnostics_before": validation_diag_before,
            "validation_diagnostics_after": validation_diag_after,
            "training_seconds": float(training_seconds),
            "shadow_steps_per_second": float(SHADOW_STEPS / max(training_seconds, 1e-12)),
            "memory": memory,
        },
        "production_student": {
            "architecture_changed": False,
            "training_started": False,
            "parameter_state_changed": False,
            "policy_hash_before": production_policy_before,
            "policy_hash_after": production_policy_after,
            "external_behavior_delta": behavior,
            "external_action_contract_changed": False,
            "supervisor_permission_semantics_changed": False,
            "physics_changed": False,
        },
        "semantic_guards": {
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "tournament_run": False,
            "canonical_g0_state_mutated": False,
            "production_student_training_started": False,
            "shadow_student_training_started": True,
            "historical_market_information_verdict_reopened": False,
            "scientific_verdict_created": False,
            "market_information_qualified": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "teacher_future_autograd": False,
            "realized_future_used_as_correct_action_label": False,
            "production_student_received_shadow_gradient": False,
        },
        "next_legal_step": "R11_SCIENCE_G0_DISTRIBUTIONAL_STUDENT_SHADOW_R3_ADJUDICATION__NO_PRODUCTION_CUTOVER_AUTHORIZED",
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "shadow_parameter_count": parameter_report["parameter_count"],
        "action_count": train_batch.action_count,
        "quantile_count": train_batch.quantile_count,
        "train_loss_before": train_loss_before,
        "train_loss_after": train_loss_after,
        "validation_loss_before": validation_loss_before,
        "validation_loss_after": validation_loss_after,
        "shadow_parameter_l2_delta": shadow_delta,
        "production_policy_unchanged": production_policy_after == production_policy_before,
        "production_behavior_identical": behavior["production_behavior_identical"],
        "cuda_training_peak_allocated_bytes": peak_allocated,
        "training_seconds": training_seconds,
        "final_holdout_payload_opened": False,
        "fresh_market_data_downloaded": False,
        "scientific_verdict_created": False,
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
