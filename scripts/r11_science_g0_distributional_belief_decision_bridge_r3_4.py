#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 R3.4 — distributional belief to decision bridge shadow.

For each pre-registered seed:
1) train the already-qualified detached distributional belief head on Teacher action laws;
2) freeze that belief head and the entire production Student;
3) compare two byte-identically initialized disposable residual decision bridges:
   A) state-aligned distributional belief;
   B) whole-dependence-group shuffled belief negative control.

This is an incremental-information architecture gate, not a market-information
verdict and not a production cutover.
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

from cb16_local_opt.distributional_belief_decision_bridge_r34 import (
    DistributionalBeliefDecisionBridgeR34,
    R34_GRAD_CLIP,
    R34_LR,
    R34_RUNTIME,
    R34_SEEDS,
    R34_STEPS,
    R34_WEIGHT_DECAY,
    apply_bridge_r34,
    assert_bridge_gradient_ownership_r34,
    assert_zero_residual_equivalence_r34,
    bridge_parameter_report_r34,
    clone_bridge_r34,
    group_block_shuffle_r34,
    state_l2_delta_r34,
)
from cb16_local_opt.distributional_student_shadow_r3 import (
    DistributionalEvidenceBatchR3,
    DistributionalStudentShadowHeadR3,
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


SCHEMA = "CB16_R11_SCIENCE_G0_DISTRIBUTIONAL_BELIEF_DECISION_BRIDGE_R3_4_RESULT_V1"
R33_ADJUDICATION_SEED = "a4de92de34e8ae27c20c607cf5574473892214ed"
FROZEN_SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
SEMANTIC_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
TYPED_CENTRAL_BRAIN_BLOB = "899065a9b9c015fd5a0e24fc919a79ff7d3a67ff"
TRAINING_RUNTIME_BLOB = "56c227800d7a5c82ef88c691bb69d7550aee8b39"
PROBABILISTIC_TEACHER_BLOB = "3de3092d244c3fd315950c9f8b0bf4d78c50ee2b"
TEACHER_RUNTIME_BLOB = "656f1483f2e6a96daae5c98eb96cd75805d3547c"
R3_SHADOW_RUNTIME_BLOB = "2e0dac6c4ddac81f3f7f20cfc8c8789ade8f1373"
R31_SHARED_RUNTIME_BLOB = "f3875d4ff873f189b06cd8b5c8784936b2828fc0"
R32_STABILITY_RUNTIME_BLOB = "34cd6c120d9dd383e1f8496dc7151c8fdd50d83c"
R33_PROJECTION_RUNTIME_BLOB = "2e1422e7959d8d64b7594d46329f7a19e2d760e8"
R33_QUALIFICATION_SCRIPT_BLOB = "1ab1e63d3fa69f7206f58b61b79607c3a2ab6c8e"
HEAD_STEPS = 32
HEAD_LR = 2e-3
HEAD_WEIGHT_DECAY = 0.0
HEAD_GRAD_CLIP = 10.0


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


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_equal(before: Mapping[str, torch.Tensor], model: torch.nn.Module) -> bool:
    now = model.state_dict()
    return set(before) == set(now) and all(torch.equal(before[k], now[k].detach().cpu()) for k in before)


def states_equal(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def detached_production_outputs(model: torch.nn.Module, prepared) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        out = model(prepared.operator48, prepared.medium48, prepared.account6)
    return {
        "direction_logits": out["direction_logits"].detach().clone(),
        "direction_probs": out["direction_probs"].detach().clone(),
        "requested_risk_raw": out["requested_risk_raw"].detach().clone(),
    }


def decision_loss(outputs: Mapping[str, torch.Tensor], prepared) -> dict[str, float]:
    with torch.no_grad():
        loss = student_loss_from_outputs_r11(outputs, prepared)
    return {
        "total": float(loss.loss.item()),
        "direction": float(loss.direction_loss.item()),
        "sizing": float(loss.sizing_loss.item()),
    }


def actions(model: torch.nn.Module, outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        act = model.compose_action(outputs)
    return {k: v.detach().cpu().clone() for k, v in act.items()}


def action_delta(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    require(set(before) == set(after), "R11_R34_ACTION_FIELD_DRIFT")
    risk_delta = torch.abs(before["requested_risk"] - after["requested_risk"])
    return {
        "direction_changed_rows": int(torch.sum(before["direction"] != after["direction"]).item()),
        "direction_class_changed_rows": int(torch.sum(before["direction_class"] != after["direction_class"]).item()),
        "maximum_abs_requested_risk_delta": float(risk_delta.max().item()) if risk_delta.numel() else 0.0,
        "mean_abs_requested_risk_delta": float(risk_delta.mean().item()) if risk_delta.numel() else 0.0,
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


def train_distributional_head(
    production: torch.nn.Module,
    train_batch: DistributionalEvidenceBatchR3,
    validation_batch: DistributionalEvidenceBatchR3,
    *,
    seed: int,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, dict[str, Any]]:
    train_shared = shared_representation_r3(production, train_batch)
    val_shared = shared_representation_r3(production, validation_batch)
    set_seed(seed)
    head = DistributionalStudentShadowHeadR3(
        shared_dim=int(train_shared.shape[1]),
        action_count=train_batch.action_count,
        quantile_count=train_batch.quantile_count,
    ).to(train_batch.device, dtype=torch.float32)
    report = shadow_parameter_report_r3(head)
    require(report["parameter_count"] == 20_543, f"R11_R34_DISTRIBUTIONAL_HEAD_PARAMETER_DRIFT:{report}")
    before = clone_state(head)
    train_before, _, _ = distributional_eval(head, train_shared, train_batch)
    val_before, _, _ = distributional_eval(head, val_shared, validation_batch)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    audit = None
    t0 = time.perf_counter()
    for step in range(HEAD_STEPS):
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
        norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), HEAD_GRAD_CLIP).item())
        require(math.isfinite(norm), f"R11_R34_NONFINITE_HEAD_GRAD_NORM:{seed}")
        opt.step()
    if train_batch.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    require(audit is not None, f"R11_R34_DISTRIBUTIONAL_HEAD_GRAD_AUDIT_MISSING:{seed}")
    train_after, train_diag, train_pred = distributional_eval(head, train_shared, train_batch)
    val_after, val_diag, val_pred = distributional_eval(head, val_shared, validation_batch)
    delta = state_l2_delta_r34(before, clone_state(head))
    require(delta > 0.0, f"R11_R34_DISTRIBUTIONAL_HEAD_DID_NOT_UPDATE:{seed}")
    require(train_after < train_before, f"R11_R34_DISTRIBUTIONAL_HEAD_TRAIN_LOSS_NOT_LOWER:{seed}")
    require(train_diag["prediction_quantiles_monotone"] is True, f"R11_R34_TRAIN_QUANTILE_CROSSING:{seed}")
    require(val_diag["prediction_quantiles_monotone"] is True, f"R11_R34_VALIDATION_QUANTILE_CROSSING:{seed}")
    for p in head.parameters():
        p.grad = None
        p.requires_grad_(False)
    return head, train_pred.reshape(train_pred.shape[0], -1), val_pred.reshape(val_pred.shape[0], -1), {
        "parameter_report": report,
        "gradient_audit_first_step": audit,
        "parameter_l2_delta": delta,
        "train_loss_before": train_before,
        "train_loss_after": train_after,
        "validation_loss_before": val_before,
        "validation_loss_after": val_after,
        "training_seconds": float(seconds),
    }


def bridge_eval(
    bridge: DistributionalBeliefDecisionBridgeR34,
    *,
    shared: torch.Tensor,
    belief: torch.Tensor,
    base_outputs: Mapping[str, torch.Tensor],
    prepared,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    bridge.eval()
    with torch.no_grad():
        out = apply_bridge_r34(base_outputs, bridge(shared, belief))
        loss = student_loss_from_outputs_r11(out, prepared)
    return {
        "total": float(loss.loss.item()),
        "direction": float(loss.direction_loss.item()),
        "sizing": float(loss.sizing_loss.item()),
    }, {k: v.detach().clone() for k, v in out.items()}


def train_bridge(
    *,
    production: torch.nn.Module,
    head: torch.nn.Module,
    bridge: DistributionalBeliefDecisionBridgeR34,
    train_shared: torch.Tensor,
    val_shared: torch.Tensor,
    train_belief: torch.Tensor,
    val_belief: torch.Tensor,
    train_base: Mapping[str, torch.Tensor],
    val_base: Mapping[str, torch.Tensor],
    decision_campaign,
    seed: int,
    arm: str,
) -> dict[str, Any]:
    report = bridge_parameter_report_r34(bridge)
    before_state = clone_state(bridge)
    assert_zero_residual_equivalence_r34(bridge, train_shared, train_belief, train_base)
    assert_zero_residual_equivalence_r34(bridge, val_shared, val_belief, val_base)
    train_before, _ = bridge_eval(
        bridge, shared=train_shared, belief=train_belief, base_outputs=train_base, prepared=decision_campaign.train
    )
    val_before, _ = bridge_eval(
        bridge, shared=val_shared, belief=val_belief, base_outputs=val_base, prepared=decision_campaign.validation
    )
    production_val = decision_loss(val_base, decision_campaign.validation)
    require(abs(val_before["total"] - production_val["total"]) <= 2e-6, f"R11_R34_ZERO_RESIDUAL_TOTAL_DRIFT:{seed}:{arm}")

    opt = torch.optim.AdamW(bridge.parameters(), lr=R34_LR, weight_decay=R34_WEIGHT_DECAY)
    audit = None
    t0 = time.perf_counter()
    for step in range(R34_STEPS):
        opt.zero_grad(set_to_none=True)
        out = apply_bridge_r34(train_base, bridge(train_shared, train_belief))
        loss = student_loss_from_outputs_r11(out, decision_campaign.train)
        loss.loss.backward()
        if step == 0:
            audit = assert_bridge_gradient_ownership_r34(
                production_model=production,
                distributional_head=head,
                bridge=bridge,
            )
        norm = float(torch.nn.utils.clip_grad_norm_(bridge.parameters(), R34_GRAD_CLIP).item())
        require(math.isfinite(norm), f"R11_R34_NONFINITE_BRIDGE_GRAD_NORM:{seed}:{arm}:{step}")
        opt.step()
    if train_shared.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    require(audit is not None, f"R11_R34_BRIDGE_GRAD_AUDIT_MISSING:{seed}:{arm}")
    train_after, _ = bridge_eval(
        bridge, shared=train_shared, belief=train_belief, base_outputs=train_base, prepared=decision_campaign.train
    )
    val_after, val_outputs = bridge_eval(
        bridge, shared=val_shared, belief=val_belief, base_outputs=val_base, prepared=decision_campaign.validation
    )
    delta = state_l2_delta_r34(before_state, clone_state(bridge))
    require(delta > 0.0, f"R11_R34_BRIDGE_DID_NOT_UPDATE:{seed}:{arm}")
    require(train_after["total"] < train_before["total"], f"R11_R34_BRIDGE_TRAIN_LOSS_NOT_LOWER:{seed}:{arm}")
    require(all(math.isfinite(v) for v in val_after.values()), f"R11_R34_NONFINITE_BRIDGE_VALIDATION:{seed}:{arm}")
    baseline_actions = actions(production, val_base)
    bridged_actions = actions(production, val_outputs)
    return {
        "arm": arm,
        "parameter_report": report,
        "gradient_audit_first_step": audit,
        "parameter_l2_delta": delta,
        "train_loss_before": train_before,
        "train_loss_after": train_after,
        "validation_loss_before": val_before,
        "validation_loss_after": val_after,
        "validation_delta_vs_production": {
            k: float(val_after[k] - production_val[k]) for k in ("total", "direction", "sizing")
        },
        "hypothetical_behavior_delta": action_delta(baseline_actions, bridged_actions),
        "training_seconds": float(seconds),
    }


def median(values) -> float:
    return float(statistics.median(float(x) for x in values))


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
    require(args.symbol == "BTCUSDT", "R11_R34_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")
    require(int(args.train_groups) == 48 and int(args.validation_groups) == 12, "R11_R34_PRE_REGISTERED_SUPPORT_DRIFT")
    require(int(args.candidate_factor) == 2, "R11_R34_CANDIDATE_FACTOR_DRIFT")
    require(work_root != g0_root and g0_root not in work_root.parents, "R11_R34_WORK_ROOT_OVERLAPS_CANONICAL_G0")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", R33_ADJUDICATION_SEED, "HEAD"]) == 0, "R11_R34_NOT_DESCENDED_FROM_R33_ADJUDICATION")

    immutable_blobs = {
        "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", SEMANTIC_FREEZE_BLOB),
        "typed_central_brain": ("cb16_local_opt/typed_central_brain_r10.py", TYPED_CENTRAL_BRAIN_BLOB),
        "training_runtime": ("cb16_local_opt/training_runtime_r11.py", TRAINING_RUNTIME_BLOB),
        "probabilistic_teacher": ("cb16_local_opt/probabilistic_teacher_r6.py", PROBABILISTIC_TEACHER_BLOB),
        "teacher_runtime": ("cb16_local_opt/teacher_runtime_r11.py", TEACHER_RUNTIME_BLOB),
        "r3_shadow_runtime": ("cb16_local_opt/distributional_student_shadow_r3.py", R3_SHADOW_RUNTIME_BLOB),
        "r31_shared_runtime": ("cb16_local_opt/distributional_student_shared_core_shadow_r31.py", R31_SHARED_RUNTIME_BLOB),
        "r32_stability_runtime": ("cb16_local_opt/distributional_student_gradient_stability_r32.py", R32_STABILITY_RUNTIME_BLOB),
        "r33_projection_runtime": ("cb16_local_opt/distributional_student_compatibility_projection_r33.py", R33_PROJECTION_RUNTIME_BLOB),
        "r33_qualification_script": ("scripts/r11_science_g0_distributional_student_compatibility_projection_r3_3.py", R33_QUALIFICATION_SCRIPT_BLOB),
    }
    for name, (path, expected) in immutable_blobs.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected, f"R11_R34_IMMUTABLE_BLOB_DRIFT:{name}:{observed}")

    repo = r1.verify_repo_seed(); repo["r33_adjudication_seed"] = R33_ADJUDICATION_SEED; repo["execution_head"] = git("rev-parse", "HEAD")
    runtime = r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"R11_R34_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    lineage, g0_identity = r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_before = dict(g0_identity["authority_file_hashes"])

    market_cache = MarketRuntimeCacheR11(g0_root)
    market = market_cache.get(args.symbol); market_cache.assert_read_only()
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R34_ANCHOR_SHA_DRIFT")
    frames = load_anchor_frames(args.symbol, anchor_path)
    selected = r1.select_candidate_frames(
        frames, train_target=48, validation_target=12, candidate_factor=2
    )
    encoded, sensory_receipt = r1.encode_selected_frames(
        package_root=package_root, device=args.device, selected=selected, batch_size=int(args.sensory_batch_size)
    )
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
    train_summary = evidence_summary(train_evidence); val_summary = evidence_summary(validation_evidence)
    require(train_summary["admitted_dependence_groups"] >= 32, f"R11_R34_TRAIN_TEACHER_SUPPORT_NOT_READY:{train_summary}")
    require(val_summary["admitted_dependence_groups"] >= 8, f"R11_R34_VALIDATION_TEACHER_SUPPORT_NOT_READY:{val_summary}")

    train_dist = DistributionalEvidenceBatchR3.from_evidence(train_evidence, parents, device=args.device)
    val_dist = DistributionalEvidenceBatchR3.from_evidence(validation_evidence, parents, device=args.device)
    decision_campaign = prepare_evidence_campaign_r11(
        train_evidence=train_evidence, validation_evidence=validation_evidence, parents=parents, device=args.device
    )
    require(train_dist.parent_ids == decision_campaign.train.parent_ids, "R11_R34_TRAIN_ROW_ALIGNMENT_DRIFT")
    require(val_dist.parent_ids == decision_campaign.validation.parent_ids, "R11_R34_VALIDATION_ROW_ALIGNMENT_DRIFT")
    require(train_dist.dependence_group_ids == decision_campaign.train.dependence_group_ids, "R11_R34_TRAIN_GROUP_ALIGNMENT_DRIFT")
    require(val_dist.dependence_group_ids == decision_campaign.validation.dependence_group_ids, "R11_R34_VALIDATION_GROUP_ALIGNMENT_DRIFT")
    require(train_dist.action_grid == val_dist.action_grid == r3.teacher_action_grid_from_frozen_candidates(), "R11_R34_ACTION_GRID_AUTHORITY_DRIFT")

    production = r1.load_bootstrap_model(g0_root, args.device)
    production_before = clone_state(production)
    policy_before = policy_hash_r11(production)
    train_shared = shared_representation_r3(production, train_dist)
    val_shared = shared_representation_r3(production, val_dist)
    train_base = detached_production_outputs(production, decision_campaign.train)
    val_base = detached_production_outputs(production, decision_campaign.validation)
    production_val = decision_loss(val_base, decision_campaign.validation)
    production_actions_before = actions(production, val_base)

    seed_results: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(R34_SEEDS):
        head, train_belief, val_belief, head_report = train_distributional_head(
            production, train_dist, val_dist, seed=int(seed)
        )
        require(train_belief.requires_grad is False and val_belief.requires_grad is False, f"R11_R34_BELIEF_AUTOGRAD_TAINT:{seed}")
        shift = seed_index + 1
        train_shuffled, train_shuffle = group_block_shuffle_r34(
            train_belief, decision_campaign.train.dependence_group_ids, shift=shift
        )
        val_shuffled, val_shuffle = group_block_shuffle_r34(
            val_belief, decision_campaign.validation.dependence_group_ids, shift=shift
        )

        set_seed(int(seed) + 34)
        aligned = DistributionalBeliefDecisionBridgeR34(
            shared_dim=int(train_shared.shape[1]), belief_dim=int(train_belief.shape[1])
        ).to(train_shared.device, dtype=torch.float32)
        shuffled = clone_bridge_r34(aligned)
        require(states_equal(clone_state(aligned), clone_state(shuffled)), f"R11_R34_ARM_INITIALIZATION_DRIFT:{seed}")
        aligned_report = train_bridge(
            production=production, head=head, bridge=aligned,
            train_shared=train_shared, val_shared=val_shared,
            train_belief=train_belief, val_belief=val_belief,
            train_base=train_base, val_base=val_base,
            decision_campaign=decision_campaign, seed=int(seed), arm="ALIGNED_BELIEF",
        )
        shuffled_report = train_bridge(
            production=production, head=head, bridge=shuffled,
            train_shared=train_shared, val_shared=val_shared,
            train_belief=train_shuffled, val_belief=val_shuffled,
            train_base=train_base, val_base=val_base,
            decision_campaign=decision_campaign, seed=int(seed), arm="GROUP_SHUFFLED_BELIEF_CONTROL",
        )
        require(aligned_report["parameter_report"] == shuffled_report["parameter_report"], f"R11_R34_ARM_PARAMETER_REPORT_DRIFT:{seed}")
        a = aligned_report["validation_loss_after"]; s = shuffled_report["validation_loss_after"]
        seed_results.append({
            "seed": int(seed),
            "distributional_head": head_report,
            "shuffle_control": {
                "train": {k: v for k, v in train_shuffle.items() if k != "mapping"},
                "validation": {k: v for k, v in val_shuffle.items() if k != "mapping"},
                "shift": shift,
            },
            "aligned_belief_bridge": aligned_report,
            "shuffled_belief_bridge": shuffled_report,
            "aligned_minus_shuffled_validation": {
                k: float(a[k] - s[k]) for k in ("total", "direction", "sizing")
            },
            "aligned_minus_production_validation": {
                k: float(a[k] - production_val[k]) for k in ("total", "direction", "sizing")
            },
            "shuffled_minus_production_validation": {
                k: float(s[k] - production_val[k]) for k in ("total", "direction", "sizing")
            },
        })
        del head, aligned, shuffled, train_belief, val_belief, train_shuffled, val_shuffled
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    aligned_minus_shuffled_total = [x["aligned_minus_shuffled_validation"]["total"] for x in seed_results]
    aligned_minus_shuffled_direction = [x["aligned_minus_shuffled_validation"]["direction"] for x in seed_results]
    aligned_minus_shuffled_sizing = [x["aligned_minus_shuffled_validation"]["sizing"] for x in seed_results]
    aligned_minus_prod_total = [x["aligned_minus_production_validation"]["total"] for x in seed_results]
    shuffled_minus_prod_total = [x["shuffled_minus_production_validation"]["total"] for x in seed_results]
    summary = {
        "seed_count": len(seed_results),
        "aligned_better_than_shuffled_total_seed_count": int(sum(v < 0.0 for v in aligned_minus_shuffled_total)),
        "aligned_better_than_shuffled_direction_seed_count": int(sum(v < 0.0 for v in aligned_minus_shuffled_direction)),
        "aligned_better_than_shuffled_sizing_seed_count": int(sum(v < 0.0 for v in aligned_minus_shuffled_sizing)),
        "aligned_better_than_production_total_seed_count": int(sum(v < 0.0 for v in aligned_minus_prod_total)),
        "shuffled_better_than_production_total_seed_count": int(sum(v < 0.0 for v in shuffled_minus_prod_total)),
        "aligned_minus_shuffled_total_median": median(aligned_minus_shuffled_total),
        "aligned_minus_shuffled_direction_median": median(aligned_minus_shuffled_direction),
        "aligned_minus_shuffled_sizing_median": median(aligned_minus_shuffled_sizing),
        "aligned_minus_production_total_median": median(aligned_minus_prod_total),
        "shuffled_minus_production_total_median": median(shuffled_minus_prod_total),
        "distributional_head_validation_loss_median": median(x["distributional_head"]["validation_loss_after"] for x in seed_results),
    }

    policy_after = policy_hash_r11(production)
    production_actions_after = actions(production, detached_production_outputs(production, decision_campaign.validation))
    require(policy_after == policy_before, "R11_R34_PRODUCTION_POLICY_HASH_MUTATED")
    require(state_equal(production_before, production), "R11_R34_PRODUCTION_PARAMETER_MUTATED")
    require(all(torch.equal(production_actions_before[k], production_actions_after[k]) for k in production_actions_before), "R11_R34_PRODUCTION_BEHAVIOR_CHANGED")
    require(frozen_authority_hashes(package_root) == frozen_before, "R11_R34_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R34_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()

    result = {
        "schema": SCHEMA,
        "status": "R11_DISTRIBUTIONAL_BELIEF_DECISION_BRIDGE_R3_4_PASS",
        "repo": repo,
        "runtime": runtime,
        "authority": {
            "r33_adjudication_seed": R33_ADJUDICATION_SEED,
            "immutable_blobs": {k: {"path": p, "git_blob": h} for k, (p, h) in immutable_blobs.items()},
            "frozen_scientific_status_before": FROZEN_SCIENTIFIC_STATUS,
            "frozen_scientific_status_after": FROZEN_SCIENTIFIC_STATUS,
        },
        "protocol": {
            "runtime": R34_RUNTIME,
            "seed_set": list(R34_SEEDS),
            "distributional_head_steps": HEAD_STEPS,
            "decision_bridge_steps": R34_STEPS,
            "lr": R34_LR,
            "weight_decay": R34_WEIGHT_DECAY,
            "optional_stopping": False,
            "negative_control": "WHOLE_DEPENDENCE_GROUP_SHUFFLED_BELIEF_WITH_IDENTICAL_BRIDGE_INIT",
        },
        "historical_scope": {
            "symbol": args.symbol,
            "train_dependence_groups_requested": 48,
            "validation_dependence_groups_requested": 12,
            "support": support,
            "anchor_file": str(anchor_path),
            "anchor_sha256": str(per_asset["anchors_sha256"]),
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
        },
        "sensory": sensory_receipt,
        "teacher": {
            "runtime_stats": asdict(teacher_stats),
            "train": train_summary,
            "validation": val_summary,
            "train_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash,
            "validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
            "action_laws_consumed": True,
            "realized_future_used_as_correct_action_label": False,
        },
        "production_validation_baseline": production_val,
        "seed_results": seed_results,
        "bridge_information_summary": summary,
        "production_student": {
            "architecture_changed": False,
            "training_started": False,
            "parameter_state_changed": False,
            "policy_hash_before": policy_before,
            "policy_hash_after": policy_after,
            "behavior_identical": True,
        },
        "semantic_guards": {
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "tournament_run": False,
            "production_student_received_shadow_gradient": False,
            "shared_core_received_distributional_gradient": False,
            "distributional_head_received_bridge_gradient": False,
            "historical_market_information_verdict_reopened": False,
            "scientific_verdict_created": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "production_cutover_authorized": False,
            "supervisor_permission_changed": False,
            "physics_changed": False,
        },
        "next_legal_step": "R11_SCIENCE_G0_DISTRIBUTIONAL_BELIEF_DECISION_BRIDGE_R3_4_ADJUDICATION__NO_PRODUCTION_CUTOVER_AUTHORIZED",
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "aligned_better_than_shuffled_total_seed_count": summary["aligned_better_than_shuffled_total_seed_count"],
        "aligned_better_than_production_total_seed_count": summary["aligned_better_than_production_total_seed_count"],
        "shuffled_better_than_production_total_seed_count": summary["shuffled_better_than_production_total_seed_count"],
        "aligned_minus_shuffled_total_median": summary["aligned_minus_shuffled_total_median"],
        "aligned_minus_shuffled_direction_median": summary["aligned_minus_shuffled_direction_median"],
        "aligned_minus_shuffled_sizing_median": summary["aligned_minus_shuffled_sizing_median"],
        "aligned_minus_production_total_median": summary["aligned_minus_production_total_median"],
        "production_behavior_identical": True,
        "final_holdout_payload_opened": False,
        "fresh_market_data_downloaded": False,
        "scientific_verdict_created": False,
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
