#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cb16_local_opt.frozen_sensory_stack_r10 import FrozenSensoryStackR10
from cb16_local_opt.full_minute_long_trajectory_r1 import FINAL_HOLDOUT_START_MS
from cb16_local_opt.longtraj_infra_closure_r0 import find_contiguous_prefinal_run_r0, funding_events_by_minute_r0
from cb16_local_opt.minute_physics_binding_r2 import canonical_hash
from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.r102_learning import evidence_summary
from cb16_local_opt.r102_physics import CANDIDATES_R102, FrozenPhysicsRuntimeR102
from cb16_local_opt.r11_teacher_authority_candidate import R11_TRAIN_TEACHER_CONFIG, R11_VALIDATION_TEACHER_CONFIG
from cb16_local_opt.science_feedback_diagnostics_r11 import validate_teacher_targets_r11
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.training_runtime_r11 import (
    EvaluationRuntimeR11,
    SMOOTH_L1_BETA_R11,
    policy_hash_r11,
    prepare_evidence_campaign_r11,
)
from scripts import r11_longtraj_training_admission_r0 as adm
from scripts import r11_longtraj_training_campaign_r0 as camp
from scripts.r11_longtraj_training_admission_r0_entry import build_sensory_frame_exact_r0
from scripts import r11_science_g0_historical_r1 as r1
from scripts.r11_science_g0_canonical_historical_learning_baseline_r4 import clone_state, state_equal

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_GENERALIZATION_R0_EVALUATION_SPEC_V1.json"
COHORT = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_GENERALIZATION_R0_COHORT_V1.json"
SELECTION_RECEIPT = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_GENERALIZATION_R0_SELECTION_FREEZE_RECEIPT_V1.json"
SCHEMA = "CB16_R11_LONGTRAJ_GENERALIZATION_R0_EVALUATION_RESULT_V1"
EXPECTED_CHAMPION = "d0e8c01cc58a1e70936f94f58eeca794eae2e4f89036ec26ccb49ae5d5219886"
EXPECTED_CHALLENGER = "4c4bdc1b61da306777e02b9841b5aa4b9b04f4fb9d896ba089df10148f682297"
EXPECTED_CHALLENGER_FILE = "81ea58ce1cbc90fdf912d106ec6e99186be2c3b7a28ee3497cca12613a97f246"
EXPECTED_TRAIN_EVIDENCE = "a8f293ac0c01edb5a9e39b34c8d0d1b1f62f8b0041d9606eefe735e77992c492"
EXPECTED_CAMPAIGN_DECISION_TIMES = "dba104c55dd6c9987f9ff0231a075363c69ea18338acd24467af3726e117217c"
DIRECTION_TO_TEACHER = camp.DIRECTION_TO_TEACHER


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def smooth_l1_rows(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, reduction="none", beta=SMOOTH_L1_BETA_R11)


def row_losses(outputs: Mapping[str, torch.Tensor], target_p: torch.Tensor, risk_target: torch.Tensor) -> dict[str, np.ndarray]:
    logp = F.log_softmax(outputs["direction_logits"], dim=-1)
    d = -(target_p * logp).sum(dim=-1)
    s = smooth_l1_rows(outputs["requested_risk_raw"], risk_target)
    t = d + s
    return {"total": t.detach().cpu().numpy().astype(np.float64), "direction": d.detach().cpu().numpy().astype(np.float64), "sizing": s.detach().cpu().numpy().astype(np.float64)}


def smooth_l1_constant_minimizer(values: np.ndarray, beta: float = SMOOTH_L1_BETA_R11) -> float:
    y = np.asarray(values, dtype=np.float64)
    require(y.ndim == 1 and len(y) > 0 and np.isfinite(y).all(), "GENERALIZATION_CLIMATOLOGY_RISK_VALUES_INVALID")
    def grad(x: float) -> float:
        d = x - y
        g = np.where(d < -beta, -1.0, np.where(d > beta, 1.0, d / beta))
        return float(g.sum())
    if grad(0.0) >= 0.0:
        return 0.0
    if grad(1.0) <= 0.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) * 0.5
        if grad(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) * 0.5


def paired_bootstrap(diff: np.ndarray, *, seed: int, reps: int) -> dict[str, float]:
    x = np.asarray(diff, dtype=np.float64)
    require(x.ndim == 1 and len(x) == 48 and np.isfinite(x).all(), "GENERALIZATION_BOOTSTRAP_INPUT_INVALID")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(x), size=(int(reps), len(x)), endpoint=False)
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"point": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi), "replicates": int(reps), "seed": int(seed)}


def outputs_exact(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    keys = ("direction_logits", "direction_probs", "requested_risk_raw")
    return all(k in a and k in b and torch.equal(a[k].detach().cpu(), b[k].detach().cpu()) for k in keys)


def load_challenger(path: Path, champion: torch.nn.Module, device: str) -> torch.nn.Module:
    require(sha256_file(path) == EXPECTED_CHALLENGER_FILE, "GENERALIZATION_CHALLENGER_CHECKPOINT_FILE_DRIFT")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("schema") == "CB16_R11_G0_R4_SHADOW_CHALLENGER_CHECKPOINT_V1", "GENERALIZATION_CHALLENGER_SCHEMA_DRIFT")
    require(payload.get("role") == "SHADOW_CHALLENGER__NOT_CANONICAL_CHAMPION", "GENERALIZATION_CHALLENGER_ROLE_DRIFT")
    require(payload.get("canonical_generation_advanced") is False, "GENERALIZATION_CHALLENGER_GENERATION_DRIFT")
    require(payload.get("parent_champion_policy_hash") == EXPECTED_CHAMPION, "GENERALIZATION_CHALLENGER_PARENT_DRIFT")
    out = copy.deepcopy(champion)
    out.load_state_dict(payload["state_dict"], strict=True)
    out.to(device)
    require(policy_hash_r11(out) == EXPECTED_CHALLENGER, "GENERALIZATION_CHALLENGER_POLICY_DRIFT")
    return out


def _scientific_support_fail(output: Path, *, spec: Mapping[str, Any], cohort: Mapping[str, Any], train_summary: Mapping[str, Any], validation_summary: Mapping[str, Any], teacher_stats: Any) -> int:
    result = {
        "schema": SCHEMA,
        "status": "SCIENTIFIC_FAIL",
        "classification": "SCIENTIFIC_FAIL__FROZEN_GENERALIZATION_COHORT_TEACHER_SUPPORT_NOT_QUALIFIED",
        "spec_sha256": sha256_file(SPEC),
        "cohort_file_sha256": sha256_file(COHORT),
        "cohort_decision_times_sha256": cohort["cohort"]["decision_times_canonical_sha256"],
        "teacher": {"train_summary": dict(train_summary), "validation_summary": dict(validation_summary), "runtime_stats": str(teacher_stats)},
        "models_loaded": False,
        "models_scored": False,
        "cohort_resampled": False,
        "support_thresholds_relaxed": False,
        "final_holdout_touched": False,
        "fresh_market_data_downloaded": False,
        "canonical_promotion_authorized": False,
        "canonical_generation_advance_authorized": False,
        "scientific_market_verdict": None,
        "next_gate": "STOP__NO_RESCUE_OR_COHORT_REPLACEMENT_UNDER_R0"
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--g0-root", default=os.environ.get("CB16_G0_ROOT", "/cb16/g0"))
    ap.add_argument("--challenger-checkpoint", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    output = Path(args.output).resolve()
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.challenger_checkpoint).resolve()
    package_root = Path(args.package_root).resolve()
    g0_root = Path(args.g0_root).resolve()

    spec = json.loads(SPEC.read_text())
    cohort = json.loads(COHORT.read_text())
    sel = json.loads(SELECTION_RECEIPT.read_text())
    require(spec["status"] == "FROZEN_BEFORE_FIRST_EVALUATION_EXECUTION", "GENERALIZATION_SPEC_NOT_FROZEN")
    require(cohort["status"] == "FROZEN_EXACT_COHORT", "GENERALIZATION_COHORT_NOT_FROZEN")
    require(sel["status"] == "FROZEN_AND_QUALIFIED", "GENERALIZATION_SELECTION_NOT_QUALIFIED")
    require(args.symbol == "BTCUSDT", "GENERALIZATION_SCOPE_BTC_ONLY")
    require(args.workers == 8, "GENERALIZATION_REQUIRES_8_WORKERS")
    require(args.device == "cuda", "GENERALIZATION_CANONICAL_DEVICE_CUDA")
    require(sha256_file(COHORT) == spec["selection_authority"]["cohort_file_sha256"], "GENERALIZATION_COHORT_FILE_HASH_DRIFT")
    require(cohort["cohort"]["decision_times_canonical_sha256"] == spec["selection_authority"]["cohort_decision_times_sha256"], "GENERALIZATION_COHORT_DECISION_HASH_DRIFT")
    require(cohort["cohort"]["zero_overlap_with_prior_consumed_tail"] is True, "GENERALIZATION_PRIOR_OVERLAP")
    require(int(cohort["cohort"]["dependence_groups"]) == 48, "GENERALIZATION_COHORT_COUNT_DRIFT")
    require(int(cohort["cohort"]["future_end_times_ms"][-1]) < FINAL_HOLDOUT_START_MS, "GENERALIZATION_FINAL_TOUCHED_BY_COHORT")

    source = adm.BinanceUSDMArchiveSourceR10(args.raw_root)
    layout = source.validate_layout()
    require(args.symbol in layout["symbols"], "GENERALIZATION_SYMBOL_MISSING")
    segment_start = int(cohort["cohort"]["used_segments"][0]["start_ms"])
    last_future = int(cohort["cohort"]["future_end_times_ms"][-1])
    required_rows = (last_future - segment_start) // adm.MINUTE_MS + 1
    records = find_contiguous_prefinal_run_r0(source, args.symbol, required_rows=int(required_rows))
    require(int(records[0].open_time) == segment_start, "GENERALIZATION_CONTIGUOUS_SEGMENT_START_DRIFT")
    require(int(records[-1].open_time) >= last_future, "GENERALIZATION_CONTIGUOUS_SEGMENT_TOO_SHORT")
    require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in records), "GENERALIZATION_FINAL_PAYLOAD_TOUCHED")

    def idx_for(t: int) -> int:
        d = int(t) - int(records[0].open_time)
        require(d >= 0 and d % adm.MINUTE_MS == 0, f"GENERALIZATION_DECISION_NOT_ON_SEGMENT:{t}")
        i = d // adm.MINUTE_MS
        require(0 <= i < len(records) and int(records[i].open_time) == int(t), f"GENERALIZATION_DECISION_MISSING:{t}")
        return int(i)

    campaign_indices = adm.select_parent_indices_r0(records, adm.TOTAL_GROUPS)
    campaign_decisions = [int(records[i].open_time) for i in campaign_indices]
    require(camp.canonical_json_sha256(campaign_decisions) == EXPECTED_CAMPAIGN_DECISION_TIMES, "GENERALIZATION_PRIOR_CAMPAIGN_COHORT_DRIFT")
    train_times = campaign_decisions[:adm.TRAIN_GROUPS]
    eval_times = [int(x) for x in cohort["cohort"]["decision_times_ms"]]
    require(len(train_times) == 48 and len(eval_times) == 48, "GENERALIZATION_SPLIT_COUNT_DRIFT")
    require(max(train_times) + adm.H72_MINUTES * adm.MINUTE_MS < int(cohort["cohort"]["first_new_sensory_start_ms"]), "GENERALIZATION_SUPPORT_OVERLAPS_NEW_SENSORY")

    train_indices = [idx_for(t) for t in train_times]
    eval_indices = [idx_for(t) for t in eval_times]
    all_entries = [("TRAIN", i, j) for j, i in enumerate(train_indices)] + [("VALIDATION", i, 48 + j) for j, i in enumerate(eval_indices)]
    first_data_idx = min(i for _, i, _ in all_entries) - (adm.SENSORY_PREFIX_MINUTES - 1)
    start_ms = int(records[first_data_idx].open_time)
    end_ms = max(int(records[i + adm.H72_MINUTES].open_time) for _, i, _ in all_entries)
    funding = funding_events_by_minute_r0(source, args.symbol, start_ms=start_ms, end_ms=end_ms)
    physics = FrozenPhysicsRuntimeR102.load(package_root)

    frames = [build_sensory_frame_exact_r0(records, i, args.symbol) for _, i, _ in all_entries]
    sensory = FrozenSensoryStackR10(package_root, device=args.device, verify_hashes=True)
    encoded: dict[int, tuple[Any, Any, Any]] = {}
    for start in range(0, len(frames), 8):
        chunk = frames[start:start + 8]
        enc = sensory.encode_frames(chunk)
        for j in range(len(chunk)):
            encoded[start + j] = (enc.operator48[j].copy(), enc.medium48[j].copy(), enc.ordered4h30[j].copy())
    frozen_trainable = sum(int(p.requires_grad) for module in (sensory.operator.tok, sensory.operator.model, sensory.medium.model) for p in module.parameters())
    require(frozen_trainable == 0, "GENERALIZATION_FROZEN_SENSORY_TRAINABLE")
    del sensory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    parents: dict[str, ParentContextR102] = {}
    payloads: list[dict[str, Any]] = []
    for row_ord, (split, idx, _) in enumerate(all_entries):
        parent_row = records[idx]
        future_rows = tuple(records[idx + 1: idx + 1 + adm.H72_MINUTES])
        require(len(future_rows) == adm.H72_MINUTES, "GENERALIZATION_H72_SLICE_LENGTH")
        require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in future_rows), "GENERALIZATION_FUTURE_FINAL_TOUCHED")
        prefix_start = idx - (adm.e6.PARENT_PREFIX_MINUTES - 1)
        prefix = tuple(records[prefix_start:idx + 1])
        parent_id = (f"ADMR0:{args.symbol}:{int(parent_row.open_time)}" if split == "TRAIN" else f"GENR0:{args.symbol}:{int(parent_row.open_time)}")
        parent_state, risk_authority, account6 = adm.e6._build_parent_state(physics, symbol=args.symbol, account_id=parent_id, prefix_rows=prefix, funding=funding)
        lineage_hash = adm.e6._future_hash(args.symbol, int(parent_row.open_time), future_rows, funding)
        op, med, ordered = encoded[row_ord]
        dep_id = f"FUT:{args.symbol}:{int(parent_row.open_time)}:{lineage_hash[:16]}"
        pc = ParentContextR102(
            parent_id=parent_id,
            dependence_group_id=dep_id,
            symbol=args.symbol,
            decision_time_ms=int(parent_row.open_time),
            split=split,
            scenario=("FLAT_MINUTE_R2_ADMISSION" if split == "TRAIN" else "GENERALIZATION_R0_UNCONSUMED"),
            operator48=tuple(float(x) for x in op),
            medium48=tuple(float(x) for x in med),
            account6=tuple(float(x) for x in account6),
            ordered4h30=tuple(float(x) for x in ordered),
            current_mark=float(parent_row.close),
            snapshot_sha256=canonical_hash(parent_state),
            eligible_for_economic_evidence=True,
            market_lineage_hash=lineage_hash,
        )
        parents[parent_id] = pc
        payloads.append({"ordinal": row_ord, "package_root": str(package_root), "symbol": args.symbol, "parent_id": parent_id, "decision_time_ms": int(parent_row.open_time), "parent_state": parent_state, "risk_authority": risk_authority, "future_rows": future_rows, "funding": dict(funding)})

    eval_first = 48
    causal = adm.e6._prefix_causality_canaries(str(package_root), args.symbol, payloads[eval_first]["parent_state"], payloads[eval_first]["risk_authority"], payloads[eval_first]["future_rows"], funding)
    require(all(causal.values()), f"GENERALIZATION_CAUSALITY_CANARY_FAIL:{causal}")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        executed = list(pool.map(adm._simulate_parent, payloads))
    executed.sort(key=lambda x: int(x["ordinal"]))
    require(len(executed) == 96 and all(len(x["branches"]) == len(CANDIDATES_R102) for x in executed), "GENERALIZATION_EXECUTION_GRID_DRIFT")

    samples: list[CounterfactualBranchSampleR5] = []
    for x in executed:
        pc = parents[str(x["parent_id"])]
        for b in x["branches"]:
            sample = CounterfactualBranchSampleR5(parent_id=pc.parent_id, student_context_object_id=pc.student_context_object_id, timestamp=int(pc.decision_time_ms), context_features=tuple(float(v) for v in pc.student_features), direction=DIRECTION_TO_TEACHER[int(b["direction"])], requested_risk=float(b["requested_risk"]), realized_utility=float(b["utility"]), dependence_group_id=pc.dependence_group_id, market_lineage_hash=pc.market_lineage_hash)
            sample.validate()
            samples.append(sample)
    require(len(samples) == 864, "GENERALIZATION_SAMPLE_COUNT_DRIFT")

    train_evidence, val_evidence, teacher_stats = compile_teacher_evidence_r11(samples=samples, parents=parents, train_config=R11_TRAIN_TEACHER_CONFIG, val_config=R11_VALIDATION_TEACHER_CONFIG, workers=8, block_targets=32)
    train_summary = evidence_summary(train_evidence)
    val_summary = evidence_summary(val_evidence)
    require(int(train_summary["admitted_dependence_groups"]) == 48 and int(train_summary["rejected"]) == 0, f"GENERALIZATION_PRIOR_TRAIN_SUPPORT_DRIFT:{train_summary}")
    if not (int(val_summary["admitted_dependence_groups"]) == 48 and int(val_summary["rejected"]) == 0):
        return _scientific_support_fail(output, spec=spec, cohort=cohort, train_summary=train_summary, validation_summary=val_summary, teacher_stats=teacher_stats)

    campaign = prepare_evidence_campaign_r11(train_evidence=train_evidence, validation_evidence=val_evidence, parents=parents, device=args.device)
    require(campaign.train.evidence_hash == EXPECTED_TRAIN_EVIDENCE, "GENERALIZATION_PRIOR_TRAIN_EVIDENCE_HASH_DRIFT")
    require(campaign.validation.rows == 48, "GENERALIZATION_VALIDATION_ROWS_DRIFT")
    train_audit = validate_teacher_targets_r11(campaign.train)
    val_audit = validate_teacher_targets_r11(campaign.validation)
    require(train_audit["independent_dependence_groups"] == 48 and val_audit["independent_dependence_groups"] == 48, "GENERALIZATION_DEPENDENCE_GROUP_AUDIT_DRIFT")
    require(bool(torch.all(campaign.validation.group_weight == 1.0).item()), "GENERALIZATION_EVAL_GROUP_WEIGHTS_NOT_ONE")

    train_p = campaign.train.direction_target_probs.detach().cpu().numpy().astype(np.float64)
    train_r = campaign.train.requested_risk_target.detach().cpu().numpy().astype(np.float64)
    climatology_p = train_p.mean(axis=0)
    climatology_p = climatology_p / climatology_p.sum()
    climatology_r = smooth_l1_constant_minimizer(train_r)

    champion_a = r1.load_bootstrap_model(g0_root, args.device)
    champion_b = r1.load_bootstrap_model(g0_root, args.device)
    require(policy_hash_r11(champion_a) == EXPECTED_CHAMPION and policy_hash_r11(champion_b) == EXPECTED_CHAMPION, "GENERALIZATION_CHAMPION_HASH_DRIFT")
    challenger_a = load_challenger(checkpoint, champion_a, args.device)
    challenger_b = load_challenger(checkpoint, champion_b, args.device)
    champ_state = clone_state(champion_a)
    chall_state = clone_state(challenger_a)

    for m in (champion_a, champion_b, challenger_a, challenger_b):
        m.eval()
    with torch.inference_mode():
        ca = champion_a(campaign.validation.operator48, campaign.validation.medium48, campaign.validation.account6)
        cb = champion_b(campaign.validation.operator48, campaign.validation.medium48, campaign.validation.account6)
        xa = challenger_a(campaign.validation.operator48, campaign.validation.medium48, campaign.validation.account6)
        xb = challenger_b(campaign.validation.operator48, campaign.validation.medium48, campaign.validation.account6)
    require(outputs_exact(ca, cb), "GENERALIZATION_CHAMPION_RELOAD_OUTPUT_DRIFT")
    require(outputs_exact(xa, xb), "GENERALIZATION_CHALLENGER_RELOAD_OUTPUT_DRIFT")

    evaluator = EvaluationRuntimeR11(enable_cuda_graph=False)
    champion_eval = evaluator.evaluate(champion_a, campaign.validation, use_cache=False)
    challenger_eval = evaluator.evaluate(challenger_a, campaign.validation, use_cache=False)

    target_p = campaign.validation.direction_target_probs.detach()
    risk_target = campaign.validation.requested_risk_target.detach()
    champ_rows = row_losses(ca, target_p, risk_target)
    chall_rows = row_losses(xa, target_p, risk_target)
    require(abs(float(champ_rows["total"].mean()) - float(champion_eval["loss"])) < 2e-6, "GENERALIZATION_CHAMPION_CANONICAL_LOSS_MISMATCH")
    require(abs(float(chall_rows["total"].mean()) - float(challenger_eval["loss"])) < 2e-6, "GENERALIZATION_CHALLENGER_CANONICAL_LOSS_MISMATCH")

    cp = torch.tensor(climatology_p, dtype=torch.float32, device=target_p.device).clamp_min(1e-12)
    cr = torch.tensor(float(climatology_r), dtype=torch.float32, device=risk_target.device)
    clim_direction = -(target_p * torch.log(cp)[None, :]).sum(dim=-1)
    clim_sizing = smooth_l1_rows(torch.full_like(risk_target, cr), risk_target)
    clim_total = (clim_direction + clim_sizing).detach().cpu().numpy().astype(np.float64)

    g1_diff = chall_rows["total"] - champ_rows["total"]
    g2_diff = chall_rows["total"] - clim_total
    g1 = paired_bootstrap(g1_diff, seed=20260911, reps=10000)
    g2 = paired_bootstrap(g2_diff, seed=20260912, reps=10000)

    shift_deltas: list[float] = []
    n = 48
    base_idx = torch.arange(n, device=target_p.device)
    for k in range(1, n):
        j = (base_idx + k) % n
        c = row_losses(ca, target_p[j], risk_target[j])["total"]
        x = row_losses(xa, target_p[j], risk_target[j])["total"]
        shift_deltas.append(float((x - c).mean()))
    identity_delta = float(g1["point"])
    count_le = sum(float(v) <= identity_delta for v in shift_deltas)
    exact_rank_p = float((1 + count_le) / 48.0)

    g1_pass = bool(g1["point"] < 0.0 and g1["ci_high"] < 0.0)
    g2_pass = bool(g2["point"] < 0.0 and g2["ci_high"] < 0.0)
    g3_pass = bool(count_le == 0 and identity_delta < min(shift_deltas) and abs(exact_rank_p - (1.0 / 48.0)) < 1e-15)

    if not g1_pass:
        classification = "SCIENTIFIC_FAIL__FROZEN_SHADOW_CHALLENGER_DID_NOT_GENERALIZE_VS_G0_CHAMPION"
        status = "SCIENTIFIC_FAIL"
    elif not g2_pass:
        classification = "SCIENTIFIC_FAIL__CHALLENGER_IMPROVEMENT_DID_NOT_BEAT_PRIOR_NO_STATE_CLIMATOLOGY"
        status = "SCIENTIFIC_FAIL"
    elif not g3_pass:
        classification = "SCIENTIFIC_FAIL__CHALLENGER_ADVANTAGE_NOT_SPECIFIC_TO_TRUE_STATE_TARGET_CORRESPONDENCE"
        status = "SCIENTIFIC_FAIL"
    else:
        classification = "GENERALIZATION_TARGET_CORRESPONDENCE_QUALIFIED_FOR_SEPARATE_NEXT_GATE__NO_PROMOTION__NO_MARKET_VERDICT"
        status = "PASS"

    require(state_equal(champ_state, champion_a), "GENERALIZATION_CHAMPION_MUTATED")
    require(state_equal(chall_state, challenger_a), "GENERALIZATION_CHALLENGER_MUTATED")
    require(policy_hash_r11(champion_a) == EXPECTED_CHAMPION and policy_hash_r11(challenger_a) == EXPECTED_CHALLENGER, "GENERALIZATION_POLICY_HASH_POST_SCORE_DRIFT")

    result = {
        "schema": SCHEMA,
        "status": status,
        "classification": classification,
        "spec_sha256": sha256_file(SPEC),
        "cohort": {
            "path": str(COHORT.relative_to(ROOT)),
            "file_sha256": sha256_file(COHORT),
            "decision_times_sha256": cohort["cohort"]["decision_times_canonical_sha256"],
            "dependence_groups": 48,
            "first_decision_time_ms": eval_times[0],
            "last_decision_time_ms": eval_times[-1],
            "cohort_resampled": False,
        },
        "teacher": {
            "train_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash,
            "validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
            "prior_train_evidence_hash": campaign.train.evidence_hash,
            "new_validation_evidence_hash": campaign.validation.evidence_hash,
            "train_summary": train_summary,
            "validation_summary": val_summary,
            "train_target_audit": train_audit,
            "validation_target_audit": val_audit,
            "runtime_stats": str(teacher_stats),
            "support_thresholds_relaxed": False,
            "outcome_as_label_used": False,
        },
        "models": {
            "champion_policy_hash": policy_hash_r11(champion_a),
            "challenger_policy_hash": policy_hash_r11(challenger_a),
            "challenger_checkpoint_sha256": sha256_file(checkpoint),
            "independent_champion_reload_outputs_exact": True,
            "independent_challenger_reload_outputs_exact": True,
            "training_or_tuning_on_generalization_cohort": False,
        },
        "canonical_true_alignment": {
            "champion": {"loss": float(champion_eval["loss"]), "direction_loss": float(champion_eval["direction_loss"]), "sizing_loss": float(champion_eval["sizing_loss"]), "behavior_fingerprint": champion_eval["behavior_fingerprint"]},
            "challenger": {"loss": float(challenger_eval["loss"]), "direction_loss": float(challenger_eval["direction_loss"]), "sizing_loss": float(challenger_eval["sizing_loss"]), "behavior_fingerprint": challenger_eval["behavior_fingerprint"]},
        },
        "prior_train_climatology": {"direction_probs": [float(x) for x in climatology_p], "requested_risk": float(climatology_r), "new_cohort_total_loss": float(clim_total.mean())},
        "G1": {**g1, "pass": g1_pass, "metric": "challenger_minus_champion_mean_total_loss"},
        "G2": {**g2, "pass": g2_pass, "metric": "challenger_minus_prior_train_climatology_mean_total_loss"},
        "G3": {
            "pass": g3_pass,
            "identity_delta_challenger_minus_champion": identity_delta,
            "nonidentity_cyclic_shift_deltas": shift_deltas,
            "best_nonidentity_shift_delta": float(min(shift_deltas)),
            "worst_nonidentity_shift_delta": float(max(shift_deltas)),
            "nonidentity_shifts_with_delta_le_identity": int(count_le),
            "exact_one_sided_rank_p": exact_rank_p,
            "permutation_family": "ALL_47_NONIDENTITY_CYCLIC_TARGET_ROW_SHIFTS",
            "models_refit": False,
        },
        "causality_canaries": causal,
        "firewalls": {
            "final_holdout_touched": False,
            "fresh_market_data_downloaded": False,
            "canonical_promotion_authorized": False,
            "canonical_generation_advance_authorized": False,
            "new_market_information_verdict": False,
            "scientific_market_verdict": None,
        },
        "next_gate": ("SEPARATE_PROMOTION_OR_REPLICATION_DESIGN_PREREGISTRATION_REQUIRED" if status == "PASS" else "STOP__NO_POST_RESULT_RESCUE_UNDER_R0"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
