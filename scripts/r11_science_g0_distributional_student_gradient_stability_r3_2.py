#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 R3.2 — fixed-seed gradient geometry/stability shadow.

R3.1 showed, on one pre-registered seed, that allowing distributional Teacher
feedback to reshape a disposable Shared Decision Core improved both distributional
validation loss and the legacy canonical Direction/Requested-Risk validation loss.
R3.2 does not promote that architecture.  It repeats the same bounded mechanics
experiment over a fixed seed registry and measures initial Shared-Core gradient
geometry between the distributional and canonical objectives.
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

from cb16_local_opt.distributional_student_gradient_stability_r32 import (
    R32_GRAD_CLIP,
    R32_LR,
    R32_RUNTIME,
    R32_SEEDS,
    R32_STEPS,
    R32_WEIGHT_DECAY,
    shared_core_gradient_geometry_r32,
    summarize_seed_stability_r32,
)
from cb16_local_opt.distributional_student_shadow_r3 import (
    DistributionalEvidenceBatchR3,
    assert_shadow_gradient_ownership_r3,
    quantile_diagnostics_r3,
    shared_representation_r3,
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
from scripts import r11_science_g0_distributional_student_shadow_r3 as r3
from scripts import r11_science_g0_geometry_r2_1 as r21
from scripts import r11_science_g0_historical_r1 as r1


SCHEMA = "CB16_R11_SCIENCE_G0_DISTRIBUTIONAL_STUDENT_GRADIENT_STABILITY_R3_2_RESULT_V1"
R31_PASS_SEED = "2647abd5bb3d9079739631c3f691e9409cf26b1a"
FROZEN_SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
SEMANTIC_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
TYPED_CENTRAL_BRAIN_BLOB = "899065a9b9c015fd5a0e24fc919a79ff7d3a67ff"
TRAINING_RUNTIME_BLOB = "56c227800d7a5c82ef88c691bb69d7550aee8b39"
PROBABILISTIC_TEACHER_BLOB = "3de3092d244c3fd315950c9f8b0bf4d78c50ee2b"
TEACHER_RUNTIME_BLOB = "656f1483f2e6a96daae5c98eb96cd75805d3547c"
R3_SHADOW_RUNTIME_BLOB = "2e0dac6c4ddac81f3f7f20cfc8c8789ade8f1373"
R31_SHARED_RUNTIME_BLOB = "f3875d4ff873f189b06cd8b5c8784936b2828fc0"
R31_QUALIFICATION_SCRIPT_BLOB = "449278dc7d97628f05f8180c5f948ec5d55cf187"


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
        outputs = model(prepared.operator48, prepared.medium48, prepared.account6)
        loss = student_loss_from_outputs_r11(outputs, prepared)
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
    require(set(before) == set(after), "R11_R32_ACTION_FIELD_DRIFT")
    direction_changed = int(torch.sum(before["direction"] != after["direction"]).item())
    class_changed = int(torch.sum(before["direction_class"] != after["direction_class"]).item())
    risk_delta = torch.abs(before["requested_risk"] - after["requested_risk"])
    return {
        "direction_changed_rows": direction_changed,
        "direction_class_changed_rows": class_changed,
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


def train_head_only_seed(production, train_batch, validation_batch, *, seed: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    train_shared = shared_representation_r3(production, train_batch)
    validation_shared = shared_representation_r3(production, validation_batch)
    set_seed(seed)
    head = DistributionalStudentSharedCoreHeadR31(
        shared_dim=int(train_shared.shape[1]),
        action_count=train_batch.action_count,
        quantile_count=train_batch.quantile_count,
    ).to(train_batch.device, dtype=torch.float32)
    initial_state = clone_state(head)
    report = head_parameter_report_r31(head)
    train_before, train_diag_before = distributional_eval(head, train_shared, train_batch)
    val_before, val_diag_before = distributional_eval(head, validation_shared, validation_batch)
    opt = torch.optim.AdamW(head.parameters(), lr=R32_LR, weight_decay=R32_WEIGHT_DECAY)
    audit = None
    head.train(); t0 = time.perf_counter()
    for step in range(R32_STEPS):
        opt.zero_grad(set_to_none=True)
        pred = head(train_shared)
        loss = truncated_quantile_w1_loss_r3(
            pred, train_batch.teacher_quantiles, train_batch.quantile_levels, train_batch.group_weight
        )
        loss.backward()
        if step == 0:
            audit = assert_shadow_gradient_ownership_r3(
                production_model=production,
                shadow_head=head,
                require_nonzero_shadow_gradient=True,
            )
        norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), R32_GRAD_CLIP).item())
        require(math.isfinite(norm), f"R11_R32_HEAD_ONLY_NONFINITE_GRAD_NORM:{seed}")
        opt.step()
    if train_batch.device.type == "cuda": torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    train_after, train_diag_after = distributional_eval(head, train_shared, train_batch)
    val_after, val_diag_after = distributional_eval(head, validation_shared, validation_batch)
    delta = state_l2_delta(initial_state, clone_state(head))
    require(audit is not None, f"R11_R32_HEAD_ONLY_AUDIT_MISSING:{seed}")
    require(delta > 0.0 and train_after < train_before, f"R11_R32_HEAD_ONLY_LEARNING_PATH_FAIL:{seed}")
    require(train_diag_after["prediction_quantiles_monotone"] and val_diag_after["prediction_quantiles_monotone"], f"R11_R32_HEAD_ONLY_QUANTILE_CROSSING:{seed}")
    return {
        "parameter_report": report,
        "gradient_audit_first_step": audit,
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
    }, initial_state


def train_shared_seed(production, train_batch, validation_batch, decision_campaign, *, seed: int, expected_head_initial: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    disposable = make_disposable_shared_core_student_r31(production).to(train_batch.device)
    shared_report = shared_core_parameter_report_r31(disposable)
    disposable_before = clone_state(disposable)
    before_actions = actions(disposable, validation_batch)
    decision_before = decision_loss_report(disposable, decision_campaign.validation)

    set_seed(seed)
    head = DistributionalStudentSharedCoreHeadR31(
        shared_dim=256,
        action_count=train_batch.action_count,
        quantile_count=train_batch.quantile_count,
    ).to(train_batch.device, dtype=torch.float32)
    head_initial = clone_state(head)
    require(states_equal(expected_head_initial, head_initial), f"R11_R32_A_B_HEAD_INITIALIZATION_DRIFT:{seed}")
    head_report = head_parameter_report_r31(head)

    geometry = shared_core_gradient_geometry_r32(
        disposable_model=disposable,
        distributional_head=head,
        distributional_batch=train_batch,
        canonical_prepared_batch=decision_campaign.train,
    )
    require(geometry["shared_core_parameter_count"] == int(shared_report["authorized_shared_core_parameter_count"]), f"R11_R32_GEOMETRY_SHARED_COUNT_DRIFT:{seed}")

    with torch.no_grad():
        train_shared_before = disposable(train_batch.operator48, train_batch.medium48, train_batch.account6)["shared"].detach().clone()
        val_shared_before = disposable(validation_batch.operator48, validation_batch.medium48, validation_batch.account6)["shared"].detach().clone()
    train_before, train_diag_before = distributional_eval(head, train_shared_before, train_batch)
    val_before, val_diag_before = distributional_eval(head, val_shared_before, validation_batch)

    params = list(disposable.shared_core.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=R32_LR, weight_decay=R32_WEIGHT_DECAY)
    audit = None
    disposable.train(); head.train(); t0 = time.perf_counter()
    for step in range(R32_STEPS):
        opt.zero_grad(set_to_none=True)
        outputs = forward_disposable_r31(disposable, train_batch)
        pred = head(outputs["shared"])
        loss = truncated_quantile_w1_loss_r3(
            pred, train_batch.teacher_quantiles, train_batch.quantile_levels, train_batch.group_weight
        )
        loss.backward()
        if step == 0:
            audit = assert_r31_gradient_ownership(
                production_model=production,
                disposable_model=disposable,
                head=head,
            )
        norm = float(torch.nn.utils.clip_grad_norm_(params, R32_GRAD_CLIP).item())
        require(math.isfinite(norm), f"R11_R32_SHARED_ARM_NONFINITE_GRAD_NORM:{seed}")
        opt.step()
    if train_batch.device.type == "cuda": torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    require(audit is not None, f"R11_R32_SHARED_ARM_AUDIT_MISSING:{seed}")

    update = state_update_audit_r31(disposable_before, clone_state(disposable))
    head_delta = state_l2_delta(head_initial, clone_state(head))
    require(head_delta > 0.0, f"R11_R32_SHARED_HEAD_DID_NOT_UPDATE:{seed}")
    with torch.no_grad():
        train_shared_after = disposable(train_batch.operator48, train_batch.medium48, train_batch.account6)["shared"].detach().clone()
        val_shared_after = disposable(validation_batch.operator48, validation_batch.medium48, validation_batch.account6)["shared"].detach().clone()
    train_after, train_diag_after = distributional_eval(head, train_shared_after, train_batch)
    val_after, val_diag_after = distributional_eval(head, val_shared_after, validation_batch)
    require(train_after < train_before, f"R11_R32_SHARED_ARM_TRAIN_LOSS_DID_NOT_DECREASE:{seed}")
    require(train_diag_after["prediction_quantiles_monotone"] and val_diag_after["prediction_quantiles_monotone"], f"R11_R32_SHARED_ARM_QUANTILE_CROSSING:{seed}")

    decision_after = decision_loss_report(disposable, decision_campaign.validation)
    after_actions = actions(disposable, validation_batch)
    behavior = action_delta(before_actions, after_actions)
    return {
        "disposable_student_parameter_report": shared_report,
        "distributional_head_parameter_report": head_report,
        "authorized_trainable_parameter_count": int(shared_report["authorized_shared_core_parameter_count"] + head_report["parameter_count"]),
        "gradient_geometry": geometry,
        "gradient_audit_first_step": audit,
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
        "hypothetical_disposable_behavior_delta": behavior,
        "training_seconds": float(seconds),
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
    require(args.symbol == "BTCUSDT", "R11_R32_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(int(args.train_groups) == 48 and int(args.validation_groups) == 12, "R11_R32_PRE_REGISTERED_SUPPORT_DRIFT")
    require(work_root != g0_root and g0_root not in work_root.parents, "R11_R32_WORK_ROOT_OVERLAPS_CANONICAL_G0")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R31_PASS_SEED, "HEAD"]) == 0, "R11_R32_NOT_DESCENDED_FROM_R31_PASS_SEED")

    immutable_blobs = {
        "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", SEMANTIC_FREEZE_BLOB),
        "typed_central_brain": ("cb16_local_opt/typed_central_brain_r10.py", TYPED_CENTRAL_BRAIN_BLOB),
        "training_runtime": ("cb16_local_opt/training_runtime_r11.py", TRAINING_RUNTIME_BLOB),
        "probabilistic_teacher": ("cb16_local_opt/probabilistic_teacher_r6.py", PROBABILISTIC_TEACHER_BLOB),
        "teacher_runtime": ("cb16_local_opt/teacher_runtime_r11.py", TEACHER_RUNTIME_BLOB),
        "r3_shadow_runtime": ("cb16_local_opt/distributional_student_shadow_r3.py", R3_SHADOW_RUNTIME_BLOB),
        "r31_shared_runtime": ("cb16_local_opt/distributional_student_shared_core_shadow_r31.py", R31_SHARED_RUNTIME_BLOB),
        "r31_qualification_script": ("scripts/r11_science_g0_distributional_student_shared_core_shadow_r3_1.py", R31_QUALIFICATION_SCRIPT_BLOB),
    }
    for name, (path, expected) in immutable_blobs.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected, f"R11_R32_IMMUTABLE_BLOB_DRIFT:{name}:{observed}")

    repo = r1.verify_repo_seed(); repo["r3_1_pass_seed"] = R31_PASS_SEED; repo["execution_head"] = git("rev-parse", "HEAD")
    runtime = r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"R11_R32_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    lineage, g0_identity = r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root); g0_before = dict(g0_identity["authority_file_hashes"])

    market_cache = MarketRuntimeCacheR11(g0_root); market = market_cache.get(args.symbol); market_cache.assert_read_only()
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R32_ANCHOR_SHA_DRIFT")
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
    require(train_summary["admitted_dependence_groups"] == 48, f"R11_R32_TRAIN_TEACHER_SUPPORT_DRIFT:{train_summary}")
    require(validation_summary["admitted_dependence_groups"] == 12, f"R11_R32_VALIDATION_TEACHER_SUPPORT_DRIFT:{validation_summary}")
    require({e.teacher_protocol_hash for e in train_evidence} == {R11_TRAIN_TEACHER_CONFIG.content_hash}, "R11_R32_TRAIN_TEACHER_PROTOCOL_DRIFT")
    require({e.teacher_protocol_hash for e in validation_evidence} == {R11_VALIDATION_TEACHER_CONFIG.content_hash}, "R11_R32_VALIDATION_TEACHER_PROTOCOL_DRIFT")

    train_batch = DistributionalEvidenceBatchR3.from_evidence(train_evidence, parents, device=args.device)
    validation_batch = DistributionalEvidenceBatchR3.from_evidence(validation_evidence, parents, device=args.device)
    expected_grid = r3.teacher_action_grid_from_frozen_candidates()
    require(train_batch.action_grid == validation_batch.action_grid == expected_grid, "R11_R32_ACTION_GRID_AUTHORITY_DRIFT")
    decision_campaign = prepare_evidence_campaign_r11(
        train_evidence=train_evidence, validation_evidence=validation_evidence, parents=parents, device=args.device
    )

    production = r1.load_bootstrap_model(g0_root, args.device)
    production_before_state = clone_state(production)
    production_policy_before = policy_hash_r11(production)
    production_actions_before = actions(production, validation_batch)

    seed_results: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()
    for seed in R32_SEEDS:
        head_only, head_initial = train_head_only_seed(
            production, train_batch, validation_batch, seed=seed
        )
        shared = train_shared_seed(
            production, train_batch, validation_batch, decision_campaign,
            seed=seed, expected_head_initial=head_initial,
        )
        before_decision = shared["canonical_decision_validation_before"]
        after_decision = shared["canonical_decision_validation_after"]
        behavior = shared["hypothetical_disposable_behavior_delta"]
        seed_results.append({
            "seed": int(seed),
            "arm_a_head_only": head_only,
            "arm_b_shared_core": shared,
            "shared_minus_head_validation_distributional_loss": float(shared["validation_loss_after"] - head_only["validation_loss_after"]),
            "canonical_decision_total_delta": float(after_decision["total"] - before_decision["total"]),
            "canonical_direction_loss_delta": float(after_decision["direction"] - before_decision["direction"]),
            "canonical_sizing_loss_delta": float(after_decision["sizing"] - before_decision["sizing"]),
            "gradient_geometry": shared["gradient_geometry"],
            "hypothetical_direction_changed_rows": int(behavior["direction_changed_rows"]),
            "mean_abs_requested_risk_delta": float(behavior["mean_abs_requested_risk_delta"]),
            "maximum_abs_requested_risk_delta": float(behavior["maximum_abs_requested_risk_delta"]),
        })
        require(state_equal(production_before_state, production), f"R11_R32_PRODUCTION_MUTATED_DURING_SEED:{seed}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    total_seconds = time.perf_counter() - total_t0
    summary = summarize_seed_stability_r32(seed_results)

    production_policy_after = policy_hash_r11(production)
    production_actions_after = actions(production, validation_batch)
    require(production_policy_after == production_policy_before, "R11_R32_PRODUCTION_POLICY_HASH_MUTATED")
    require(state_equal(production_before_state, production), "R11_R32_PRODUCTION_PARAMETER_STATE_MUTATED")
    require(all(torch.equal(production_actions_before[k], production_actions_after[k]) for k in production_actions_before), "R11_R32_PRODUCTION_BEHAVIOR_MUTATED")

    frozen_after = frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "R11_R32_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R32_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()

    result = {
        "schema": SCHEMA,
        "status": "R11_DISTRIBUTIONAL_STUDENT_GRADIENT_STABILITY_R3_2_PASS",
        "repo": repo,
        "runtime": runtime,
        "authority": {
            "r3_1_pass_seed": R31_PASS_SEED,
            "frozen_scientific_status_before": FROZEN_SCIENTIFIC_STATUS,
            "frozen_scientific_status_after": FROZEN_SCIENTIFIC_STATUS,
            "immutable_blobs": {k: {"path": p, "git_blob": h} for k, (p, h) in immutable_blobs.items()},
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
            "network_reads_by_r3_2": 0,
            "role": "FIXED_SUPPORT_ARCHITECTURE_STABILITY_SHADOW__NOT_MARKET_INFORMATION_RULE",
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
        "pre_registered_protocol": {
            "seed_set": list(R32_SEEDS),
            "seed_order_is_authoritative": True,
            "steps_per_arm": R32_STEPS,
            "lr": R32_LR,
            "weight_decay": R32_WEIGHT_DECAY,
            "grad_clip": R32_GRAD_CLIP,
            "hyperparameter_tuning": False,
            "optional_stopping": False,
            "support_expansion": False,
        },
        "gradient_geometry_contract": {
            "runtime": R32_RUNTIME,
            "parameter_surface": "DISPOSABLE_SHARED_DECISION_CORE_ONLY",
            "measurement_point": "INITIAL_BOOTSTRAP_BEFORE_ANY_R3_2_UPDATE",
            "distributional_objective": "TRUNCATED_QUANTILE_W1_ACTION_LAW_DISTILLATION",
            "canonical_objectives": ["TOTAL", "DIRECTION_SOFT_TARGET_CE", "REQUESTED_RISK_SMOOTHL1"],
            "gradient_geometry_is_diagnostic_only": True,
        },
        "seed_results": seed_results,
        "stability_summary": summary,
        "execution": {
            "seed_count_completed": len(seed_results),
            "total_shadow_training_and_geometry_seconds": float(total_seconds),
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
        "next_legal_step": "R11_SCIENCE_G0_DISTRIBUTIONAL_STUDENT_GRADIENT_STABILITY_R3_2_ADJUDICATION__NO_PRODUCTION_CUTOVER_AUTHORIZED",
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "architecture_pattern": summary["architecture_pattern"],
        "shared_core_better_distributional_seed_count": summary["shared_core_better_distributional_seed_count"],
        "canonical_total_loss_lower_seed_count": summary["canonical_total_loss_lower_seed_count"],
        "gradient_total_cosine_median": summary["gradient_cosine_distributional_vs_canonical_total"]["median"],
        "gradient_direction_cosine_median": summary["gradient_cosine_distributional_vs_direction"]["median"],
        "gradient_sizing_cosine_median": summary["gradient_cosine_distributional_vs_sizing"]["median"],
        "shared_minus_head_validation_loss_median": summary["shared_minus_head_validation_distributional_loss"]["median"],
        "canonical_decision_total_delta_median": summary["canonical_decision_total_delta"]["median"],
        "hypothetical_direction_changed_rows_median": summary["hypothetical_direction_changed_rows"]["median"],
        "production_behavior_identical": True,
        "final_holdout_payload_opened": False,
        "fresh_market_data_downloaded": False,
        "scientific_verdict_created": False,
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
