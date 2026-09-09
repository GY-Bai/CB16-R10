#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 R3.3 — dynamic compatibility projection shadow.

Three disposable arms use the same frozen historical evidence and fixed seed registry:
A) head-only distributional belief, detached Shared Core;
B) raw distributional Adam direction into a disposable Shared Core;
C) the same raw distributional Adam moments/direction, but on Shared Core only the
   actual Adam direction is projected whenever it is first-order conflicting with
   the current canonical Direction/Requested-Risk total gradient.

The distributional head always receives its full distributional gradient.  The
canonical gradient is a compatibility guard only; it is never added as a loss,
never changes Teacher truth semantics, and never touches production Student state.
"""

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Mapping

import torch

from cb16_local_opt.distributional_student_compatibility_projection_r33 import (
    ManualAdamDirectionR33,
    R33_BETA1,
    R33_BETA2,
    R33_EPS,
    R33_GRAD_CLIP,
    R33_LR,
    R33_RUNTIME,
    R33_SEEDS,
    R33_STEPS,
    R33_WEIGHT_DECAY,
    apply_adam_direction_r33,
    compatibility_project_adam_direction_r33,
    summarize_projection_steps_r33,
)
from cb16_local_opt.distributional_student_shadow_r3 import (
    DistributionalEvidenceBatchR3,
    quantile_diagnostics_r3,
    truncated_quantile_w1_loss_r3,
)
from cb16_local_opt.distributional_student_shared_core_shadow_r31 import (
    DistributionalStudentSharedCoreHeadR31,
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
from scripts import r11_science_g0_distributional_student_gradient_stability_r3_2 as r32script
from scripts import r11_science_g0_distributional_student_shadow_r3 as r3
from scripts import r11_science_g0_geometry_r2_1 as r21
from scripts import r11_science_g0_historical_r1 as r1


SCHEMA = "CB16_R11_SCIENCE_G0_DISTRIBUTIONAL_STUDENT_COMPATIBILITY_PROJECTION_R3_3_RESULT_V1"
R32_PASS_SEED = "7e4de752454bb7a57d4554592f810c33d0e97841"
FROZEN_SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
SEMANTIC_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
TYPED_CENTRAL_BRAIN_BLOB = "899065a9b9c015fd5a0e24fc919a79ff7d3a67ff"
TRAINING_RUNTIME_BLOB = "56c227800d7a5c82ef88c691bb69d7550aee8b39"
PROBABILISTIC_TEACHER_BLOB = "3de3092d244c3fd315950c9f8b0bf4d78c50ee2b"
TEACHER_RUNTIME_BLOB = "656f1483f2e6a96daae5c98eb96cd75805d3547c"
R3_SHADOW_RUNTIME_BLOB = "2e0dac6c4ddac81f3f7f20cfc8c8789ade8f1373"
R31_SHARED_RUNTIME_BLOB = "f3875d4ff873f189b06cd8b5c8784936b2828fc0"
R32_STABILITY_RUNTIME_BLOB = "34cd6c120d9dd383e1f8496dc7151c8fdd50d83c"
R32_QUALIFICATION_SCRIPT_BLOB = "5f209ad8ca1038ab0b1bf11b5979f0560ac7edcf"
R32_SEED31337_RAW_VAL_DIST = 0.011101977899670601
R32_SEED31337_RAW_CANON_TOTAL = 1.189879059791565


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


def states_equal(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def state_l2_delta(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for key in before:
        d = after[key].detach().cpu().double() - before[key].detach().cpu().double()
        total += float(torch.sum(d * d))
    return math.sqrt(total)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def decision_loss_report(model: torch.nn.Module, prepared) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        out = model(prepared.operator48, prepared.medium48, prepared.account6)
        loss = student_loss_from_outputs_r11(out, prepared)
    return {
        "total": float(loss.loss.item()),
        "direction": float(loss.direction_loss.item()),
        "sizing": float(loss.sizing_loss.item()),
    }


def actions(model: torch.nn.Module, batch: DistributionalEvidenceBatchR3) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        out = model(batch.operator48, batch.medium48, batch.account6)
        composed = model.compose_action(out)
    return {k: v.detach().cpu().clone() for k, v in composed.items()}


def action_delta(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    require(set(before) == set(after), "R11_R33_ACTION_FIELD_DRIFT")
    direction_changed = int(torch.sum(before["direction"] != after["direction"]).item())
    risk_delta = torch.abs(before["requested_risk"] - after["requested_risk"])
    return {
        "direction_changed_rows": direction_changed,
        "direction_class_changed_rows": int(torch.sum(before["direction_class"] != after["direction_class"]).item()),
        "maximum_abs_requested_risk_delta": float(risk_delta.max().item()) if risk_delta.numel() else 0.0,
        "mean_abs_requested_risk_delta": float(risk_delta.mean().item()) if risk_delta.numel() else 0.0,
    }


def distributional_eval(head, shared: torch.Tensor, batch: DistributionalEvidenceBatchR3) -> tuple[float, dict[str, Any]]:
    head.eval()
    with torch.no_grad():
        pred = head(shared)
        loss = truncated_quantile_w1_loss_r3(
            pred, batch.teacher_quantiles, batch.quantile_levels, batch.group_weight
        )
        diag = quantile_diagnostics_r3(pred, batch.teacher_quantiles)
    return float(loss.item()), diag


def shared_value(model: torch.nn.Module, batch: DistributionalEvidenceBatchR3) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(batch.operator48, batch.medium48, batch.account6)["shared"].detach().clone()


def canonical_total_guard_gradient(model: torch.nn.Module, prepared, shared_params) -> tuple[tuple[torch.Tensor, ...], dict[str, float]]:
    out = model(prepared.operator48, prepared.medium48, prepared.account6)
    losses = student_loss_from_outputs_r11(out, prepared)
    grads = torch.autograd.grad(losses.loss, shared_params, retain_graph=False, allow_unused=False)
    copied = tuple(g.detach().clone() for g in grads)
    if any(not bool(torch.isfinite(g).all().item()) for g in copied):
        raise RuntimeError("R11_R33_NONFINITE_CANONICAL_GUARD_GRADIENT")
    return copied, {
        "total": float(losses.loss.detach().item()),
        "direction": float(losses.direction_loss.detach().item()),
        "sizing": float(losses.sizing_loss.detach().item()),
    }


def train_manual_shared_arm(
    production,
    train_batch,
    validation_batch,
    decision_campaign,
    *,
    seed: int,
    expected_head_initial: Mapping[str, torch.Tensor],
    project_conflicts: bool,
) -> dict[str, Any]:
    disposable = make_disposable_shared_core_student_r31(production).to(train_batch.device)
    shared_report = shared_core_parameter_report_r31(disposable)
    disposable_before = clone_state(disposable)
    before_actions = actions(disposable, validation_batch)
    decision_before = decision_loss_report(disposable, decision_campaign.validation)

    set_seed(seed)
    initial_shared = shared_value(disposable, validation_batch)
    head = DistributionalStudentSharedCoreHeadR31(
        shared_dim=int(initial_shared.shape[1]),
        action_count=train_batch.action_count,
        quantile_count=train_batch.quantile_count,
    ).to(train_batch.device, dtype=torch.float32)
    head_initial = clone_state(head)
    require(states_equal(expected_head_initial, head_initial), f"R11_R33_A_B_C_HEAD_INITIALIZATION_DRIFT:{seed}")
    head_report = head_parameter_report_r31(head)

    shared_params = tuple(disposable.shared_core.parameters())
    head_params = tuple(head.parameters())
    manual_adam = ManualAdamDirectionR33(
        shared_params, beta1=R33_BETA1, beta2=R33_BETA2, eps=R33_EPS
    )
    head_optimizer = torch.optim.AdamW(
        head_params,
        lr=R33_LR,
        weight_decay=R33_WEIGHT_DECAY,
        betas=(R33_BETA1, R33_BETA2),
        eps=R33_EPS,
    )

    train_before, train_diag_before = distributional_eval(head, shared_value(disposable, train_batch), train_batch)
    val_before, val_diag_before = distributional_eval(head, initial_shared, validation_batch)
    projection_rows: list[dict[str, Any]] = []
    first_audit = None
    t0 = time.perf_counter()
    for step in range(R33_STEPS):
        for p in shared_params:
            p.grad = None
        head_optimizer.zero_grad(set_to_none=True)
        disposable.train(); head.train()
        out = forward_disposable_r31(disposable, train_batch)
        pred = head(out["shared"])
        dist_loss = truncated_quantile_w1_loss_r3(
            pred, train_batch.teacher_quantiles, train_batch.quantile_levels, train_batch.group_weight
        )
        dist_loss.backward()
        if step == 0:
            first_audit = assert_r31_gradient_ownership(
                production_model=production,
                disposable_model=disposable,
                head=head,
            )
        all_params = list(shared_params) + list(head_params)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(all_params, R33_GRAD_CLIP).item())
        require(math.isfinite(grad_norm), f"R11_R33_NONFINITE_DISTRIBUTIONAL_GRAD_NORM:{seed}:{step}")
        shared_dist_grads = tuple(p.grad.detach().clone() for p in shared_params)
        raw_directions = manual_adam.directions(shared_dist_grads)
        guard_grads, guard_loss = canonical_total_guard_gradient(
            disposable, decision_campaign.train, shared_params
        )
        safe_directions, projection_report = compatibility_project_adam_direction_r33(
            raw_directions, guard_grads
        )
        projection_report["step"] = int(step)
        projection_report["canonical_guard_loss"] = guard_loss
        projection_rows.append(projection_report)
        directions = safe_directions if project_conflicts else raw_directions
        apply_adam_direction_r33(shared_params, directions, lr=R33_LR)
        head_optimizer.step()

    if train_batch.device.type == "cuda": torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    require(first_audit is not None, f"R11_R33_GRADIENT_AUDIT_MISSING:{seed}")
    disposable_after = clone_state(disposable)
    update = state_update_audit_r31(disposable_before, disposable_after)
    head_delta = state_l2_delta(head_initial, clone_state(head))
    require(head_delta > 0.0, f"R11_R33_HEAD_DID_NOT_UPDATE:{seed}")

    train_after, train_diag_after = distributional_eval(head, shared_value(disposable, train_batch), train_batch)
    val_after, val_diag_after = distributional_eval(head, shared_value(disposable, validation_batch), validation_batch)
    require(train_after < train_before, f"R11_R33_TRAIN_DISTRIBUTIONAL_LOSS_DID_NOT_DECREASE:{seed}")
    require(train_diag_after["prediction_quantiles_monotone"] and val_diag_after["prediction_quantiles_monotone"], f"R11_R33_QUANTILE_CROSSING:{seed}")
    decision_after = decision_loss_report(disposable, decision_campaign.validation)
    behavior = action_delta(before_actions, actions(disposable, validation_batch))

    if project_conflicts:
        projection_summary = summarize_projection_steps_r33(projection_rows)
        require(projection_summary["all_safe_directions_nonconflicting_within_tolerance"] is True, f"R11_R33_PROJECTION_SAFETY_FAIL:{seed}")
    else:
        projection_summary = {
            "step_count": len(projection_rows),
            "raw_conflicting_step_count": int(sum(bool(x["projected"]) for x in projection_rows)),
            "raw_nonconflicting_step_count": int(sum(not bool(x["projected"]) for x in projection_rows)),
            "minimum_raw_dot": float(min(x["dot_before"] for x in projection_rows)),
            "maximum_raw_dot": float(max(x["dot_before"] for x in projection_rows)),
            "projection_applied": False,
        }

    return {
        "mode": "PROJECTED_COMPATIBILITY_GUARD" if project_conflicts else "RAW_DISTRIBUTIONAL_ADAM_DIRECTION",
        "disposable_student_parameter_report": shared_report,
        "distributional_head_parameter_report": head_report,
        "authorized_trainable_parameter_count": int(shared_report["authorized_shared_core_parameter_count"] + head_report["parameter_count"]),
        "gradient_audit_first_step": first_audit,
        "disposable_update_audit": update,
        "distributional_head_parameter_l2_delta": head_delta,
        "train_loss_before": train_before,
        "train_loss_after": train_after,
        "validation_loss_before": val_before,
        "validation_loss_after": val_after,
        "train_diagnostics_before": train_diag_before,
        "train_diagnostics_after": train_diag_after,
        "validation_diagnostics_before": val_diag_before,
        "validation_diagnostics_after": val_diag_after,
        "canonical_decision_validation_before": decision_before,
        "canonical_decision_validation_after": decision_after,
        "canonical_decision_total_delta": float(decision_after["total"] - decision_before["total"]),
        "canonical_direction_loss_delta": float(decision_after["direction"] - decision_before["direction"]),
        "canonical_sizing_loss_delta": float(decision_after["sizing"] - decision_before["sizing"]),
        "hypothetical_disposable_behavior_delta": behavior,
        "projection_summary": projection_summary,
        "training_seconds": float(seconds),
    }


def scalar_stats(values) -> dict[str, float]:
    xs = [float(x) for x in values]
    require(xs and all(math.isfinite(x) for x in xs), "R11_R33_NONFINITE_OR_EMPTY_SUMMARY")
    return {
        "min": float(min(xs)),
        "max": float(max(xs)),
        "mean": float(statistics.fmean(xs)),
        "median": float(statistics.median(xs)),
    }


def summarize(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(tuple(int(x["seed"]) for x in seed_rows) == R33_SEEDS, "R11_R33_SEED_SET_OR_ORDER_DRIFT")
    c_vs_b_dist = [r["projected_minus_raw_validation_distributional_loss"] for r in seed_rows]
    b_dec = [r["raw_shared"]["canonical_decision_total_delta"] for r in seed_rows]
    c_dec = [r["projected_shared"]["canonical_decision_total_delta"] for r in seed_rows]
    c_dir = [r["projected_shared"]["canonical_direction_loss_delta"] for r in seed_rows]
    c_size = [r["projected_shared"]["canonical_sizing_loss_delta"] for r in seed_rows]
    a_to_c = [r["projected_minus_head_validation_distributional_loss"] for r in seed_rows]
    raw_to_projected_dec = [c - b for b, c in zip(b_dec, c_dec)]
    projected_counts = [r["projected_shared"]["projection_summary"]["projected_step_count"] for r in seed_rows]
    c_direction_changes = [r["projected_shared"]["hypothetical_disposable_behavior_delta"]["direction_changed_rows"] for r in seed_rows]
    c_risk = [r["projected_shared"]["hypothetical_disposable_behavior_delta"]["mean_abs_requested_risk_delta"] for r in seed_rows]
    return {
        "seed_count": len(seed_rows),
        "seed_set": list(R33_SEEDS),
        "projected_beats_raw_distributional_seed_count": int(sum(x < 0.0 for x in c_vs_b_dist)),
        "projected_beats_head_only_distributional_seed_count": int(sum(x < 0.0 for x in a_to_c)),
        "projected_canonical_total_lower_than_bootstrap_seed_count": int(sum(x < 0.0 for x in c_dec)),
        "projected_canonical_direction_lower_seed_count": int(sum(x < 0.0 for x in c_dir)),
        "projected_canonical_sizing_lower_seed_count": int(sum(x < 0.0 for x in c_size)),
        "projected_canonical_total_better_than_raw_seed_count": int(sum(x < 0.0 for x in raw_to_projected_dec)),
        "projected_minus_raw_validation_distributional_loss": scalar_stats(c_vs_b_dist),
        "projected_minus_head_validation_distributional_loss": scalar_stats(a_to_c),
        "raw_canonical_total_delta": scalar_stats(b_dec),
        "projected_canonical_total_delta": scalar_stats(c_dec),
        "projected_canonical_direction_delta": scalar_stats(c_dir),
        "projected_canonical_sizing_delta": scalar_stats(c_size),
        "projected_minus_raw_canonical_total_delta": scalar_stats(raw_to_projected_dec),
        "projected_steps_per_seed": {
            "min": int(min(projected_counts)),
            "max": int(max(projected_counts)),
            "mean": float(statistics.fmean(projected_counts)),
            "median": float(statistics.median(projected_counts)),
        },
        "projected_hypothetical_direction_changed_rows": scalar_stats(c_direction_changes),
        "projected_mean_abs_requested_risk_delta": scalar_stats(c_risk),
        "role": "ARCHITECTURE_COMPATIBILITY_SHADOW_ONLY__NOT_MARKET_INFORMATION_ADJUDICATION",
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
    require(args.symbol == "BTCUSDT", "R11_R33_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(int(args.train_groups) == 48 and int(args.validation_groups) == 12, "R11_R33_PRE_REGISTERED_SUPPORT_DRIFT")
    require(work_root != g0_root and g0_root not in work_root.parents, "R11_R33_WORK_ROOT_OVERLAPS_CANONICAL_G0")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R32_PASS_SEED, "HEAD"]) == 0, "R11_R33_NOT_DESCENDED_FROM_R32_PASS_SEED")

    immutable_blobs = {
        "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", SEMANTIC_FREEZE_BLOB),
        "typed_central_brain": ("cb16_local_opt/typed_central_brain_r10.py", TYPED_CENTRAL_BRAIN_BLOB),
        "training_runtime": ("cb16_local_opt/training_runtime_r11.py", TRAINING_RUNTIME_BLOB),
        "probabilistic_teacher": ("cb16_local_opt/probabilistic_teacher_r6.py", PROBABILISTIC_TEACHER_BLOB),
        "teacher_runtime": ("cb16_local_opt/teacher_runtime_r11.py", TEACHER_RUNTIME_BLOB),
        "r3_shadow_runtime": ("cb16_local_opt/distributional_student_shadow_r3.py", R3_SHADOW_RUNTIME_BLOB),
        "r31_shared_runtime": ("cb16_local_opt/distributional_student_shared_core_shadow_r31.py", R31_SHARED_RUNTIME_BLOB),
        "r32_stability_runtime": ("cb16_local_opt/distributional_student_gradient_stability_r32.py", R32_STABILITY_RUNTIME_BLOB),
        "r32_qualification_script": ("scripts/r11_science_g0_distributional_student_gradient_stability_r3_2.py", R32_QUALIFICATION_SCRIPT_BLOB),
    }
    for name, (path, expected) in immutable_blobs.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected, f"R11_R33_IMMUTABLE_BLOB_DRIFT:{name}:{observed}")

    repo = r1.verify_repo_seed(); repo["r3_2_pass_seed"] = R32_PASS_SEED; repo["execution_head"] = git("rev-parse", "HEAD")
    runtime = r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3,10], f"R11_R33_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    lineage, g0_identity = r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root); g0_before = dict(g0_identity["authority_file_hashes"])

    market_cache = MarketRuntimeCacheR11(g0_root); market = market_cache.get(args.symbol); market_cache.assert_read_only()
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R33_ANCHOR_SHA_DRIFT")
    frames = load_anchor_frames(args.symbol, anchor_path)
    selected = r1.select_candidate_frames(frames, train_target=48, validation_target=12, candidate_factor=int(args.candidate_factor))
    encoded, sensory_receipt = r1.encode_selected_frames(package_root=package_root, device=args.device, selected=selected, batch_size=int(args.sensory_batch_size))
    physics = FrozenPhysicsRuntimeR102.load(package_root)
    parents, samples, support = r21.build_multi_account_counterfactual_support(
        symbol=args.symbol, selected=selected, encoded=encoded, physics=physics,
        hourly_ts=market.open_time_ms, hourly_ohlcv=market.ohlcv, funding=market.funding_rate,
        train_target_groups=48, validation_target_groups=12,
    )
    del encoded; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    train_evidence, validation_evidence, teacher_stats = compile_teacher_evidence_r11(
        samples=samples, parents=parents,
        train_config=R11_TRAIN_TEACHER_CONFIG, val_config=R11_VALIDATION_TEACHER_CONFIG,
        workers=int(args.teacher_workers), block_targets=int(args.teacher_block_targets),
    )
    train_summary = evidence_summary(train_evidence); validation_summary = evidence_summary(validation_evidence)
    require(train_summary["admitted_dependence_groups"] == 48, f"R11_R33_TRAIN_TEACHER_SUPPORT_DRIFT:{train_summary}")
    require(validation_summary["admitted_dependence_groups"] == 12, f"R11_R33_VALIDATION_TEACHER_SUPPORT_DRIFT:{validation_summary}")
    require({e.teacher_protocol_hash for e in train_evidence} == {R11_TRAIN_TEACHER_CONFIG.content_hash}, "R11_R33_TRAIN_TEACHER_PROTOCOL_DRIFT")
    require({e.teacher_protocol_hash for e in validation_evidence} == {R11_VALIDATION_TEACHER_CONFIG.content_hash}, "R11_R33_VALIDATION_TEACHER_PROTOCOL_DRIFT")

    train_batch = DistributionalEvidenceBatchR3.from_evidence(train_evidence, parents, device=args.device)
    validation_batch = DistributionalEvidenceBatchR3.from_evidence(validation_evidence, parents, device=args.device)
    require(train_batch.action_grid == validation_batch.action_grid == r3.teacher_action_grid_from_frozen_candidates(), "R11_R33_ACTION_GRID_AUTHORITY_DRIFT")
    decision_campaign = prepare_evidence_campaign_r11(
        train_evidence=train_evidence, validation_evidence=validation_evidence, parents=parents, device=args.device
    )

    production = r1.load_bootstrap_model(g0_root, args.device)
    production_before = clone_state(production)
    production_policy_before = policy_hash_r11(production)
    production_actions_before = actions(production, validation_batch)

    seed_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for seed in R33_SEEDS:
        head_only, head_initial = r32script.train_head_only_seed(
            production, train_batch, validation_batch, seed=seed
        )
        raw = train_manual_shared_arm(
            production, train_batch, validation_batch, decision_campaign,
            seed=seed, expected_head_initial=head_initial, project_conflicts=False,
        )
        projected = train_manual_shared_arm(
            production, train_batch, validation_batch, decision_campaign,
            seed=seed, expected_head_initial=head_initial, project_conflicts=True,
        )
        if int(seed) == 31_337:
            require(abs(raw["validation_loss_after"] - R32_SEED31337_RAW_VAL_DIST) <= 5e-6, f"R11_R33_MANUAL_RAW_ADAM_R32_DIST_REPRODUCTION_DRIFT:{raw['validation_loss_after']}")
            require(abs(raw["canonical_decision_validation_after"]["total"] - R32_SEED31337_RAW_CANON_TOTAL) <= 5e-5, f"R11_R33_MANUAL_RAW_ADAM_R32_CANON_REPRODUCTION_DRIFT:{raw['canonical_decision_validation_after']['total']}")
        seed_rows.append({
            "seed": int(seed),
            "head_only": head_only,
            "raw_shared": raw,
            "projected_shared": projected,
            "raw_minus_head_validation_distributional_loss": float(raw["validation_loss_after"] - head_only["validation_loss_after"]),
            "projected_minus_head_validation_distributional_loss": float(projected["validation_loss_after"] - head_only["validation_loss_after"]),
            "projected_minus_raw_validation_distributional_loss": float(projected["validation_loss_after"] - raw["validation_loss_after"]),
        })
        require(state_equal(production_before, production), f"R11_R33_PRODUCTION_MUTATED_DURING_SEED:{seed}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    shadow_seconds = time.perf_counter() - t0
    summary = summarize(seed_rows)

    production_policy_after = policy_hash_r11(production)
    production_actions_after = actions(production, validation_batch)
    require(production_policy_after == production_policy_before, "R11_R33_PRODUCTION_POLICY_HASH_MUTATED")
    require(state_equal(production_before, production), "R11_R33_PRODUCTION_PARAMETER_STATE_MUTATED")
    require(all(torch.equal(production_actions_before[k], production_actions_after[k]) for k in production_actions_before), "R11_R33_PRODUCTION_BEHAVIOR_MUTATED")
    frozen_after = frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "R11_R33_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R33_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()

    result = {
        "schema": SCHEMA,
        "status": "R11_DISTRIBUTIONAL_STUDENT_COMPATIBILITY_PROJECTION_R3_3_PASS",
        "repo": repo,
        "runtime": runtime,
        "authority": {
            "r3_2_pass_seed": R32_PASS_SEED,
            "frozen_scientific_status_before": FROZEN_SCIENTIFIC_STATUS,
            "frozen_scientific_status_after": FROZEN_SCIENTIFIC_STATUS,
            "immutable_blobs": {k: {"path": p, "git_blob": h} for k,(p,h) in immutable_blobs.items()},
            "g0_authority_hashes_unchanged": True,
            "frozen_package_authority_hashes_unchanged": True,
        },
        "historical_scope": {
            "symbol": args.symbol,
            "train_dependence_groups": 48,
            "validation_dependence_groups": 12,
            "support": support,
            "anchor_file": str(anchor_path),
            "anchor_sha256": str(per_asset["anchors_sha256"]),
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "network_reads_by_r3_3": 0,
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
        "protocol": {
            "runtime": R33_RUNTIME,
            "seed_set": list(R33_SEEDS),
            "steps_per_arm": R33_STEPS,
            "lr": R33_LR,
            "weight_decay": R33_WEIGHT_DECAY,
            "grad_clip": R33_GRAD_CLIP,
            "adam_betas": [R33_BETA1, R33_BETA2],
            "adam_eps": R33_EPS,
            "canonical_gradient_role": "DYNAMIC_COMPATIBILITY_GUARD_ONLY__NOT_ADDED_TO_OBJECTIVE",
            "distributional_head_gradient": "FULL_UNPROJECTED",
            "hyperparameter_tuning": False,
            "optional_stopping": False,
            "support_expansion": False,
        },
        "seed_results": seed_rows,
        "compatibility_summary": summary,
        "execution": {
            "seed_count_completed": len(seed_rows),
            "shadow_training_seconds": float(shadow_seconds),
        },
        "production_student": {
            "architecture_changed": False,
            "training_started": False,
            "parameter_state_changed": False,
            "policy_hash_before": production_policy_before,
            "policy_hash_after": production_policy_after,
            "behavior_identical": True,
            "production_cutover_authorized": False,
        },
        "semantic_guards": {
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "tournament_run": False,
            "production_student_training_started": False,
            "production_student_received_shadow_gradient": False,
            "realized_future_used_as_correct_action_label": False,
            "teacher_future_autograd": False,
            "historical_market_information_verdict_reopened": False,
            "scientific_verdict_created": False,
            "market_information_qualified": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "production_cutover_authorized": False,
        },
        "next_legal_step": "R11_SCIENCE_G0_DISTRIBUTIONAL_STUDENT_COMPATIBILITY_PROJECTION_R3_3_ADJUDICATION__NO_PRODUCTION_CUTOVER_AUTHORIZED",
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "projected_beats_raw_distributional_seed_count": summary["projected_beats_raw_distributional_seed_count"],
        "projected_beats_head_only_distributional_seed_count": summary["projected_beats_head_only_distributional_seed_count"],
        "projected_canonical_total_lower_than_bootstrap_seed_count": summary["projected_canonical_total_lower_than_bootstrap_seed_count"],
        "projected_canonical_sizing_lower_seed_count": summary["projected_canonical_sizing_lower_seed_count"],
        "projected_canonical_total_better_than_raw_seed_count": summary["projected_canonical_total_better_than_raw_seed_count"],
        "projected_minus_raw_distributional_median": summary["projected_minus_raw_validation_distributional_loss"]["median"],
        "projected_canonical_total_delta_median": summary["projected_canonical_total_delta"]["median"],
        "projected_canonical_sizing_delta_median": summary["projected_canonical_sizing_delta"]["median"],
        "projected_steps_median": summary["projected_steps_per_seed"]["median"],
        "production_behavior_identical": True,
        "final_holdout_payload_opened": False,
        "fresh_market_data_downloaded": False,
        "scientific_verdict_created": False,
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
