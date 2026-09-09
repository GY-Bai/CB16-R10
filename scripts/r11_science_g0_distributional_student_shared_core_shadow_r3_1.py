#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 R3.1 — disposable Shared Decision Core shadow qualification.

R3 proved that full Teacher action-law quantiles can train a detached, head-only
Student belief representation. R3.1 keeps production untouched and compares two
pre-registered disposable arms on the same bounded frozen history:

A) distributional head only, fixed production Shared Decision Core values;
B) disposable Shared Decision Core + identical distributional head.

The existing canonical Direction/Requested-Risk loss is evaluated before/after
arm B as an architecture-compatibility diagnostic only. No market-information
verdict is created or reopened, no generation advances, and FINAL remains shut.
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
from typing import Any, Mapping

import torch

from cb16_local_opt.distributional_student_shadow_r3 import (
    DistributionalEvidenceBatchR3,
    assert_shadow_gradient_ownership_r3,
    quantile_diagnostics_r3,
    shared_representation_r3,
    truncated_quantile_w1_loss_r3,
)
from cb16_local_opt.distributional_student_shared_core_shadow_r31 import (
    DistributionalStudentSharedCoreHeadR31,
    R31_RUNTIME,
    assert_r31_gradient_ownership,
    forward_disposable_r31,
    head_parameter_report_r31,
    make_disposable_shared_core_student_r31,
    shared_core_parameter_report_r31,
    state_update_audit_r31,
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
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.training_runtime_r11 import (
    policy_hash_r11,
    prepare_evidence_campaign_r11,
    student_loss_from_outputs_r11,
)
from scripts import r11_science_g0_distributional_student_shadow_r3 as r3
from scripts import r11_science_g0_geometry_r2_1 as r21
from scripts import r11_science_g0_historical_r1 as r1


SCHEMA = "CB16_R11_SCIENCE_G0_DISTRIBUTIONAL_STUDENT_SHARED_CORE_SHADOW_R3_1_RESULT_V1"
R3_PASS_SEED = "517689b48bd01c5be4b2a2695bf38d9e8ad60ff3"
SEMANTIC_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
TYPED_CENTRAL_BRAIN_BLOB = "899065a9b9c015fd5a0e24fc919a79ff7d3a67ff"
TRAINING_RUNTIME_BLOB = "56c227800d7a5c82ef88c691bb69d7550aee8b39"
PROBABILISTIC_TEACHER_BLOB = "3de3092d244c3fd315950c9f8b0bf4d78c50ee2b"
TEACHER_RUNTIME_BLOB = "656f1483f2e6a96daae5c98eb96cd75805d3547c"
R3_SHADOW_RUNTIME_BLOB = "2e0dac6c4ddac81f3f7f20cfc8c8789ade8f1373"
R3_QUALIFICATION_SCRIPT_BLOB = "e472234a7fb96b0ecec5579662203364a01f8f31"
FROZEN_SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
SEED = 31_337
STEPS = 32
LR = 2e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 10.0


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


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_equal(before: Mapping[str, torch.Tensor], model: torch.nn.Module) -> bool:
    now = model.state_dict()
    return set(before) == set(now) and all(torch.equal(before[k], now[k].detach().cpu()) for k in before)


def state_l2_delta(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for key in before:
        d = after[key].detach().cpu().double() - before[key].detach().cpu().double()
        total += float(torch.sum(d * d))
    return math.sqrt(total)


def decision_loss_report(model: torch.nn.Module, prepared) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        outputs = model(prepared.operator48, prepared.medium48, prepared.account6)
        loss = student_loss_from_outputs_r11(outputs, prepared)
    return {
        "total": float(loss.loss.item()),
        "direction": float(loss.direction_loss.item()),
        "sizing": float(loss.sizing_loss.item()),
    }


def decision_loss_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, Any]:
    delta = {k: float(after[k] - before[k]) for k in ("total", "direction", "sizing")}
    rel = {
        k: float(delta[k] / max(abs(float(before[k])), 1e-12))
        for k in ("total", "direction", "sizing")
    }
    if delta["total"] < 0.0:
        classification = "LOWER_CANONICAL_DECISION_LOSS_ON_BOUNDED_VALIDATION"
    elif delta["total"] > 0.0:
        classification = "HIGHER_CANONICAL_DECISION_LOSS_ON_BOUNDED_VALIDATION"
    else:
        classification = "UNCHANGED_CANONICAL_DECISION_LOSS_ON_BOUNDED_VALIDATION"
    return {
        "absolute_delta": delta,
        "relative_delta": rel,
        "classification": classification,
        "scientific_market_verdict": False,
    }


def actions(model: torch.nn.Module, batch: DistributionalEvidenceBatchR3) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        out = model(batch.operator48, batch.medium48, batch.account6)
        composed = model.compose_action(out)
    return {k: v.detach().cpu().clone() for k, v in composed.items()}


def action_delta(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    require(set(before) == set(after), "R11_R31_ACTION_FIELD_DRIFT")
    direction_changed = int(torch.sum(before["direction"] != after["direction"]).item())
    class_changed = int(torch.sum(before["direction_class"] != after["direction_class"]).item())
    risk_delta = torch.abs(before["requested_risk"] - after["requested_risk"])
    return {
        "direction_changed_rows": direction_changed,
        "direction_class_changed_rows": class_changed,
        "maximum_abs_requested_risk_delta": float(risk_delta.max().item()) if risk_delta.numel() else 0.0,
        "mean_abs_requested_risk_delta": float(risk_delta.mean().item()) if risk_delta.numel() else 0.0,
        "role": "DISPOSABLE_ARCHITECTURE_COMPATIBILITY_DIAGNOSTIC_ONLY",
    }


def shared_value(model: torch.nn.Module, batch: DistributionalEvidenceBatchR3) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        shared = model(batch.operator48, batch.medium48, batch.account6)["shared"]
    return shared.detach().clone()


def representation_delta(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    require(reference.shape == candidate.shape, "R11_R31_SHARED_REPRESENTATION_SHAPE_DRIFT")
    d = torch.abs(candidate - reference)
    return {
        "mean_abs_delta": float(d.mean().item()),
        "max_abs_delta": float(d.max().item()),
        "rms_delta": float(torch.sqrt(torch.mean((candidate - reference) ** 2)).item()),
    }


def distributional_eval(head, shared: torch.Tensor, batch: DistributionalEvidenceBatchR3) -> tuple[float, dict[str, Any], torch.Tensor]:
    head.eval()
    with torch.no_grad():
        pred = head(shared)
        loss = truncated_quantile_w1_loss_r3(
            pred, batch.teacher_quantiles, batch.quantile_levels, batch.group_weight
        )
        diag = quantile_diagnostics_r3(pred, batch.teacher_quantiles)
    return float(loss.item()), diag, pred.detach().clone()


def train_head_only(production_model, train_batch, validation_batch) -> dict[str, Any]:
    train_shared = shared_representation_r3(production_model, train_batch)
    validation_shared = shared_representation_r3(production_model, validation_batch)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    head = DistributionalStudentSharedCoreHeadR31(
        shared_dim=int(train_shared.shape[1]),
        action_count=train_batch.action_count,
        quantile_count=train_batch.quantile_count,
    ).to(train_batch.device, dtype=torch.float32)
    param_report = head_parameter_report_r31(head)
    before_state = clone_state(head)
    train_before, train_diag_before, _ = distributional_eval(head, train_shared, train_batch)
    val_before, val_diag_before, _ = distributional_eval(head, validation_shared, validation_batch)

    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if train_batch.device.type == "cuda":
        torch.cuda.synchronize()
        baseline = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline = 0
    t0 = time.perf_counter()
    gradient_audit = None
    head.train()
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        pred = head(train_shared)
        loss = truncated_quantile_w1_loss_r3(
            pred, train_batch.teacher_quantiles, train_batch.quantile_levels, train_batch.group_weight
        )
        loss.backward()
        if step == 0:
            gradient_audit = assert_shadow_gradient_ownership_r3(
                production_model=production_model,
                shadow_head=head,
                require_nonzero_shadow_gradient=True,
            )
        norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP).item())
        require(math.isfinite(norm), "R11_R31_HEAD_ONLY_NONFINITE_GRAD_NORM")
        optimizer.step()
    if train_batch.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    train_after, train_diag_after, _ = distributional_eval(head, train_shared, train_batch)
    val_after, val_diag_after, _ = distributional_eval(head, validation_shared, validation_batch)
    after_state = clone_state(head)
    delta = state_l2_delta(before_state, after_state)
    require(gradient_audit is not None, "R11_R31_HEAD_ONLY_GRADIENT_AUDIT_MISSING")
    require(delta > 0.0, "R11_R31_HEAD_ONLY_PARAMETER_DID_NOT_MOVE")
    require(train_after < train_before, "R11_R31_HEAD_ONLY_TRAIN_LOSS_DID_NOT_DECREASE")
    require(train_diag_after["prediction_quantiles_monotone"] is True, "R11_R31_HEAD_ONLY_TRAIN_QUANTILE_CROSSING")
    require(val_diag_after["prediction_quantiles_monotone"] is True, "R11_R31_HEAD_ONLY_VALIDATION_QUANTILE_CROSSING")
    peak_alloc = int(torch.cuda.max_memory_allocated()) if train_batch.device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if train_batch.device.type == "cuda" else 0
    return {
        "parameter_report": param_report,
        "gradient_audit_first_step": gradient_audit,
        "parameter_l2_delta": delta,
        "train_loss_before": train_before,
        "train_loss_after": train_after,
        "validation_loss_before": val_before,
        "validation_loss_after": val_after,
        "train_diagnostics_before": train_diag_before,
        "train_diagnostics_after": train_diag_after,
        "validation_diagnostics_before": val_diag_before,
        "validation_diagnostics_after": val_diag_after,
        "training_seconds": float(seconds),
        "steps_per_second": float(STEPS / max(seconds, 1e-12)),
        "cuda_baseline_allocated_bytes": baseline,
        "cuda_peak_allocated_bytes": peak_alloc,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_incremental_peak_bytes": max(0, peak_alloc - baseline),
    }


def train_shared_core_arm(production_model, train_batch, validation_batch, decision_campaign) -> tuple[dict[str, Any], torch.nn.Module]:
    disposable = make_disposable_shared_core_student_r31(production_model).to(train_batch.device)
    shared_report = shared_core_parameter_report_r31(disposable)
    disposable_before_state = clone_state(disposable)
    production_validation_shared = shared_value(production_model, validation_batch)
    disposable_validation_shared_before = shared_value(disposable, validation_batch)
    require(torch.equal(production_validation_shared, disposable_validation_shared_before), "R11_R31_DISPOSABLE_BOOTSTRAP_SHARED_DRIFT")
    production_actions_before = actions(production_model, validation_batch)
    disposable_actions_before = actions(disposable, validation_batch)
    require(all(torch.equal(production_actions_before[k], disposable_actions_before[k]) for k in production_actions_before), "R11_R31_DISPOSABLE_BOOTSTRAP_BEHAVIOR_DRIFT")

    decision_before = decision_loss_report(disposable, decision_campaign.validation)
    production_decision = decision_loss_report(production_model, decision_campaign.validation)
    for key in decision_before:
        require(abs(decision_before[key] - production_decision[key]) <= 1e-12, f"R11_R31_DISPOSABLE_BOOTSTRAP_DECISION_LOSS_DRIFT:{key}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    head = DistributionalStudentSharedCoreHeadR31(
        shared_dim=int(disposable_validation_shared_before.shape[1]),
        action_count=train_batch.action_count,
        quantile_count=train_batch.quantile_count,
    ).to(train_batch.device, dtype=torch.float32)
    head_report = head_parameter_report_r31(head)
    head_before_state = clone_state(head)

    train_shared_before = shared_value(disposable, train_batch)
    train_loss_before, train_diag_before, _ = distributional_eval(head, train_shared_before, train_batch)
    val_loss_before, val_diag_before, _ = distributional_eval(head, disposable_validation_shared_before, validation_batch)

    params = list(disposable.shared_core.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    if train_batch.device.type == "cuda":
        torch.cuda.synchronize()
        baseline = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline = 0
    t0 = time.perf_counter()
    gradient_audit = None
    disposable.train(); head.train()
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_disposable_r31(disposable, train_batch)
        pred = head(outputs["shared"])
        loss = truncated_quantile_w1_loss_r3(
            pred, train_batch.teacher_quantiles, train_batch.quantile_levels, train_batch.group_weight
        )
        loss.backward()
        if step == 0:
            gradient_audit = assert_r31_gradient_ownership(
                production_model=production_model,
                disposable_model=disposable,
                head=head,
            )
        norm = float(torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP).item())
        require(math.isfinite(norm), "R11_R31_SHARED_ARM_NONFINITE_GRAD_NORM")
        optimizer.step()
    if train_batch.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    require(gradient_audit is not None, "R11_R31_SHARED_ARM_GRADIENT_AUDIT_MISSING")

    disposable_after_state = clone_state(disposable)
    head_after_state = clone_state(head)
    update_audit = state_update_audit_r31(disposable_before_state, disposable_after_state)
    head_delta = state_l2_delta(head_before_state, head_after_state)
    require(head_delta > 0.0, "R11_R31_SHARED_ARM_HEAD_DID_NOT_UPDATE")

    train_shared_after = shared_value(disposable, train_batch)
    validation_shared_after = shared_value(disposable, validation_batch)
    train_loss_after, train_diag_after, _ = distributional_eval(head, train_shared_after, train_batch)
    val_loss_after, val_diag_after, _ = distributional_eval(head, validation_shared_after, validation_batch)
    require(train_loss_after < train_loss_before, "R11_R31_SHARED_ARM_TRAIN_LOSS_DID_NOT_DECREASE")
    require(train_diag_after["prediction_quantiles_monotone"] is True, "R11_R31_SHARED_ARM_TRAIN_QUANTILE_CROSSING")
    require(val_diag_after["prediction_quantiles_monotone"] is True, "R11_R31_SHARED_ARM_VALIDATION_QUANTILE_CROSSING")

    decision_after = decision_loss_report(disposable, decision_campaign.validation)
    disposable_actions_after = actions(disposable, validation_batch)
    shared_delta = representation_delta(production_validation_shared, validation_shared_after)
    hypothetical_behavior = action_delta(disposable_actions_before, disposable_actions_after)

    peak_alloc = int(torch.cuda.max_memory_allocated()) if train_batch.device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if train_batch.device.type == "cuda" else 0
    return {
        "disposable_student_parameter_report": shared_report,
        "distributional_head_parameter_report": head_report,
        "authorized_trainable_parameter_count": int(shared_report["authorized_shared_core_parameter_count"] + head_report["parameter_count"]),
        "gradient_audit_first_step": gradient_audit,
        "disposable_update_audit": update_audit,
        "distributional_head_parameter_l2_delta": head_delta,
        "train_loss_before": train_loss_before,
        "train_loss_after": train_loss_after,
        "validation_loss_before": val_loss_before,
        "validation_loss_after": val_loss_after,
        "train_diagnostics_before": train_diag_before,
        "train_diagnostics_after": train_diag_after,
        "validation_diagnostics_before": val_diag_before,
        "validation_diagnostics_after": val_diag_after,
        "canonical_decision_validation_before": decision_before,
        "canonical_decision_validation_after": decision_after,
        "canonical_decision_compatibility": decision_loss_delta(decision_before, decision_after),
        "validation_shared_representation_delta": shared_delta,
        "hypothetical_disposable_behavior_delta": hypothetical_behavior,
        "training_seconds": float(seconds),
        "steps_per_second": float(STEPS / max(seconds, 1e-12)),
        "cuda_baseline_allocated_bytes": baseline,
        "cuda_peak_allocated_bytes": peak_alloc,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_incremental_peak_bytes": max(0, peak_alloc - baseline),
    }, disposable


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
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    work_root = args.work_root.resolve(); work_root.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve(); g0_root = args.g0_root.resolve(); package_root = args.package_root.resolve()
    require(args.symbol == "BTCUSDT", "R11_R31_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(int(args.train_groups) == 48 and int(args.validation_groups) == 12, "R11_R31_PRE_REGISTERED_SUPPORT_DRIFT")
    require(int(args.steps) == STEPS, "R11_R31_PRE_REGISTERED_STEP_BUDGET_DRIFT")
    require(work_root != g0_root and g0_root not in work_root.parents, "R11_R31_WORK_ROOT_OVERLAPS_CANONICAL_G0")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R3_PASS_SEED, "HEAD"]) == 0, "R11_R31_NOT_DESCENDED_FROM_R3_PASS_SEED")

    immutable_blobs = {
        "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", SEMANTIC_FREEZE_BLOB),
        "typed_central_brain": ("cb16_local_opt/typed_central_brain_r10.py", TYPED_CENTRAL_BRAIN_BLOB),
        "training_runtime": ("cb16_local_opt/training_runtime_r11.py", TRAINING_RUNTIME_BLOB),
        "probabilistic_teacher": ("cb16_local_opt/probabilistic_teacher_r6.py", PROBABILISTIC_TEACHER_BLOB),
        "teacher_runtime": ("cb16_local_opt/teacher_runtime_r11.py", TEACHER_RUNTIME_BLOB),
        "r3_shadow_runtime": ("cb16_local_opt/distributional_student_shadow_r3.py", R3_SHADOW_RUNTIME_BLOB),
        "r3_qualification_script": ("scripts/r11_science_g0_distributional_student_shadow_r3.py", R3_QUALIFICATION_SCRIPT_BLOB),
    }
    for name, (path, expected) in immutable_blobs.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected, f"R11_R31_IMMUTABLE_BLOB_DRIFT:{name}:{observed}")

    repo = r1.verify_repo_seed(); repo["r3_pass_seed"] = R3_PASS_SEED; repo["execution_head"] = git("rev-parse", "HEAD")
    runtime = r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"R11_R31_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    lineage, g0_identity = r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_before = dict(g0_identity["authority_file_hashes"])

    market_cache = MarketRuntimeCacheR11(g0_root); market = market_cache.get(args.symbol); market_cache.assert_read_only()
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R31_ANCHOR_SHA_DRIFT")
    frames = load_anchor_frames(args.symbol, anchor_path)
    selected = r1.select_candidate_frames(
        frames,
        train_target=int(args.train_groups), validation_target=int(args.validation_groups),
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
    del encoded; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    train_evidence, validation_evidence, teacher_stats = compile_teacher_evidence_r11(
        samples=samples, parents=parents,
        train_config=R11_TRAIN_TEACHER_CONFIG, val_config=R11_VALIDATION_TEACHER_CONFIG,
        workers=int(args.teacher_workers), block_targets=int(args.teacher_block_targets),
    )
    train_summary = evidence_summary(train_evidence); validation_summary = evidence_summary(validation_evidence)
    require(train_summary["admitted_dependence_groups"] >= 32, f"R11_R31_TRAIN_TEACHER_SUPPORT_NOT_READY:{train_summary}")
    require(validation_summary["admitted_dependence_groups"] >= 8, f"R11_R31_VALIDATION_TEACHER_SUPPORT_NOT_READY:{validation_summary}")
    require({e.teacher_protocol_hash for e in train_evidence} == {R11_TRAIN_TEACHER_CONFIG.content_hash}, "R11_R31_TRAIN_TEACHER_PROTOCOL_DRIFT")
    require({e.teacher_protocol_hash for e in validation_evidence} == {R11_VALIDATION_TEACHER_CONFIG.content_hash}, "R11_R31_VALIDATION_TEACHER_PROTOCOL_DRIFT")

    train_batch = DistributionalEvidenceBatchR3.from_evidence(train_evidence, parents, device=args.device)
    validation_batch = DistributionalEvidenceBatchR3.from_evidence(validation_evidence, parents, device=args.device)
    expected_grid = r3.teacher_action_grid_from_frozen_candidates()
    require(train_batch.action_grid == validation_batch.action_grid == expected_grid, "R11_R31_ACTION_GRID_AUTHORITY_DRIFT")
    require(train_batch.quantile_levels == validation_batch.quantile_levels == tuple(float(x) for x in R11_TRAIN_TEACHER_CONFIG.quantile_levels), "R11_R31_QUANTILE_LEVEL_AUTHORITY_DRIFT")

    decision_campaign = prepare_evidence_campaign_r11(
        train_evidence=train_evidence, validation_evidence=validation_evidence,
        parents=parents, device=args.device,
    )
    production = r1.load_bootstrap_model(g0_root, args.device)
    production_before_state = clone_state(production)
    production_policy_before = policy_hash_r11(production)
    production_actions_before = actions(production, validation_batch)

    arm_a = train_head_only(production, train_batch, validation_batch)
    require(state_equal(production_before_state, production), "R11_R31_PRODUCTION_MUTATED_BY_HEAD_ONLY_ARM")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    arm_b, disposable = train_shared_core_arm(
        production, train_batch, validation_batch, decision_campaign
    )
    require(state_equal(production_before_state, production), "R11_R31_PRODUCTION_MUTATED_BY_SHARED_CORE_ARM")
    production_policy_after = policy_hash_r11(production)
    production_actions_after = actions(production, validation_batch)
    require(production_policy_after == production_policy_before, "R11_R31_PRODUCTION_POLICY_HASH_MUTATED")
    require(all(torch.equal(production_actions_before[k], production_actions_after[k]) for k in production_actions_before), "R11_R31_PRODUCTION_BEHAVIOR_MUTATED")

    comparison = {
        "head_only_validation_distributional_loss_after": arm_a["validation_loss_after"],
        "shared_core_validation_distributional_loss_after": arm_b["validation_loss_after"],
        "shared_minus_head_only_validation_distributional_loss": float(arm_b["validation_loss_after"] - arm_a["validation_loss_after"]),
        "lower_distributional_validation_loss_arm": (
            "SHARED_CORE" if arm_b["validation_loss_after"] < arm_a["validation_loss_after"]
            else "HEAD_ONLY" if arm_b["validation_loss_after"] > arm_a["validation_loss_after"]
            else "TIE"
        ),
        "role": "ARCHITECTURE_MECHANICS_COMPARISON_ONLY__NOT_MARKET_INFORMATION_ADJUDICATION",
    }

    frozen_after = frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "R11_R31_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R31_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()

    result = {
        "schema": SCHEMA,
        "status": "R11_DISTRIBUTIONAL_STUDENT_SHARED_CORE_SHADOW_R3_1_PASS",
        "repo": repo,
        "runtime": runtime,
        "authority": {
            "r3_pass_seed": R3_PASS_SEED,
            "immutable_blobs": {k: {"path": p, "git_blob": h} for k, (p, h) in immutable_blobs.items()},
            "frozen_scientific_status_before": FROZEN_SCIENTIFIC_STATUS,
            "frozen_scientific_status_after": FROZEN_SCIENTIFIC_STATUS,
            "g0_authority_hashes_unchanged": True,
            "frozen_package_authority_hashes_unchanged": True,
        },
        "historical_scope": {
            "symbol": args.symbol,
            "scope_role": "BOUNDED_SHARED_CORE_ARCHITECTURE_SHADOW__NOT_MARKET_RULE",
            "train_dependence_groups": int(args.train_groups),
            "validation_dependence_groups": int(args.validation_groups),
            "support": support,
            "anchor_file": str(anchor_path),
            "anchor_sha256": str(per_asset["anchors_sha256"]),
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "network_reads_by_r3_1": 0,
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
            "action_grid": [list(x) for x in train_batch.action_grid],
            "quantile_levels": list(train_batch.quantile_levels),
            "train_rows": train_batch.rows,
            "validation_rows": validation_batch.rows,
            "loss": "TRUNCATED_QUANTILE_W1_ON_TEACHER_QUANTILE_SUPPORT",
            "dependence_group_weighting": "EQUAL_TOTAL_MASS_PER_INDEPENDENT_FUTURE_GROUP",
        },
        "pre_registered_protocol": {
            "same_seed": SEED,
            "same_steps": STEPS,
            "same_lr": LR,
            "same_weight_decay": WEIGHT_DECAY,
            "same_grad_clip": GRAD_CLIP,
            "hyperparameter_tuning": False,
        },
        "arm_a_head_only": arm_a,
        "arm_b_disposable_shared_core": arm_b,
        "arm_comparison": comparison,
        "production_student": {
            "architecture_changed": False,
            "training_started": False,
            "parameter_state_changed": False,
            "policy_hash_before": production_policy_before,
            "policy_hash_after": production_policy_after,
            "behavior_identical": True,
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
            "disposable_shadow_training_started": True,
            "historical_market_information_verdict_reopened": False,
            "scientific_verdict_created": False,
            "market_information_qualified": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "teacher_future_autograd": False,
            "realized_future_used_as_correct_action_label": False,
            "production_student_received_shadow_gradient": False,
            "production_cutover_authorized": False,
        },
        "next_legal_step": "R11_SCIENCE_G0_DISTRIBUTIONAL_STUDENT_SHARED_CORE_SHADOW_R3_1_ADJUDICATION__NO_PRODUCTION_CUTOVER_AUTHORIZED",
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "head_only_validation_loss_after": arm_a["validation_loss_after"],
        "shared_core_validation_loss_after": arm_b["validation_loss_after"],
        "shared_minus_head_only_validation_loss": comparison["shared_minus_head_only_validation_distributional_loss"],
        "decision_total_before": arm_b["canonical_decision_validation_before"]["total"],
        "decision_total_after": arm_b["canonical_decision_validation_after"]["total"],
        "decision_compatibility": arm_b["canonical_decision_compatibility"]["classification"],
        "shared_core_l2_delta": arm_b["disposable_update_audit"]["shared_core_parameter_l2_delta"],
        "hypothetical_direction_changed_rows": arm_b["hypothetical_disposable_behavior_delta"]["direction_changed_rows"],
        "production_policy_unchanged": production_policy_before == production_policy_after,
        "final_holdout_payload_opened": False,
        "fresh_market_data_downloaded": False,
        "scientific_verdict_created": False,
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    del disposable
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
