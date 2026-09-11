#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import torch

from cb16_local_opt.canonical_historical_learning_baseline_r4 import (
    build_shadow_snapshot_r4,
    shadow_champion_challenger_ranking_r4,
)
from cb16_local_opt.frozen_sensory_stack_r10 import FrozenSensoryStackR10
from cb16_local_opt.full_minute_long_trajectory_r1 import FINAL_HOLDOUT_START_MS
from cb16_local_opt.longtraj_infra_closure_r0 import (
    find_contiguous_prefinal_run_r0,
    funding_events_by_minute_r0,
)
from cb16_local_opt.minute_physics_binding_r2 import canonical_hash
from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.r102_learning import evidence_summary
from cb16_local_opt.r102_physics import CANDIDATES_R102, FLAT, LONG, SHORT, FrozenPhysicsRuntimeR102
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
from scripts import r11_longtraj_training_admission_r0 as adm
from scripts.r11_longtraj_training_admission_r0_entry import build_sensory_frame_exact_r0
from scripts import r11_science_g0_historical_r1 as r1
from scripts.r11_science_g0_canonical_historical_learning_baseline_r4 import (
    clone_state,
    metric_close,
    save_shadow_challenger,
    state_equal,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_TRAINING_CAMPAIGN_R0_SPEC_V1.json"
ADMISSION_RECEIPT = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_TRAINING_ADMISSION_R0_FREEZE_RECEIPT_V1.json"
E6_RECEIPT = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_E6_FEEDBACK_BINDING_R2_FREEZE_RECEIPT_V1.json"
SCHEMA = "CB16_R11_LONGTRAJ_TRAINING_CAMPAIGN_R0_RESULT_V1"
DIRECTION_TO_TEACHER = {SHORT: -1, FLAT: 0, LONG: 1}
EXPECTED_G0_CHAMPION = "d0e8c01cc58a1e70936f94f58eeca794eae2e4f89036ec26ccb49ae5d5219886"
EXPECTED_PARENT_RECEIPTS_SHA = "b8a5878a07689677967dcf92b42c44903d987fd2f7c5fd19517d93c1fd70daca"
EXPECTED_PARENT_IDS_SHA = "8a15f139047e0cfbaf1c47a3d2794e07df6cd1bc0f88870414c1efd36408061b"
EXPECTED_DECISION_TIMES_SHA = "dba104c55dd6c9987f9ff0231a075363c69ea18338acd24467af3726e117217c"
EXPECTED_TRAIN_EVIDENCE = "a8f293ac0c01edb5a9e39b34c8d0d1b1f62f8b0041d9606eefe735e77992c492"
EXPECTED_VALIDATION_EVIDENCE = "0ac7a0490e0df3c97150c14e84ca378fe3cc33d767ec26a907aad3a781297131"


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    raw = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _reload_shadow_exact(
    path: Path,
    champion: torch.nn.Module,
    challenger: torch.nn.Module,
    *,
    device: str,
    expected_snapshot_hash: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("snapshot_hash") == expected_snapshot_hash, "CAMPAIGN_SHADOW_RELOAD_SNAPSHOT_DRIFT")
    require(payload.get("parent_champion_policy_hash") == policy_hash_r11(champion), "CAMPAIGN_SHADOW_RELOAD_PARENT_DRIFT")
    restored = copy.deepcopy(champion)
    restored.load_state_dict(payload["state_dict"], strict=True)
    restored.to(device)
    expected_hash = policy_hash_r11(challenger)
    observed_hash = policy_hash_r11(restored)
    exact = expected_hash == observed_hash and state_equal(clone_state(challenger), restored)
    return {
        "exact": bool(exact),
        "checkpoint_sha256": sha256_file(path),
        "expected_policy_hash": expected_hash,
        "restored_policy_hash": observed_hash,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--g0-root", default=os.environ.get("CB16_G0_ROOT", "/cb16/g0"))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    output = Path(args.output).resolve()
    work_root = Path(args.work_root).resolve()
    package_root = Path(args.package_root).resolve()
    g0_root = Path(args.g0_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    require(work_root != g0_root and g0_root not in work_root.parents, "CAMPAIGN_WORK_ROOT_OVERLAPS_G0")

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    require(spec["schema"] == "CB16_R11_LONGTRAJ_TRAINING_CAMPAIGN_R0_SPEC_V1", "CAMPAIGN_SPEC_SCHEMA")
    require(spec["status"] == "FROZEN_BEFORE_FIRST_CAMPAIGN_EXECUTION", "CAMPAIGN_SPEC_NOT_PREREGISTERED")
    require(spec["classification"] == "ONE_BOUNDED_GENERATION0_SHADOW_CAMPAIGN__NO_CANONICAL_PROMOTION__NO_SCIENTIFIC_MARKET_VERDICT", "CAMPAIGN_SPEC_CLASSIFICATION_DRIFT")
    require(args.symbol == "BTCUSDT", "CAMPAIGN_SCOPE_BTC_ONLY")
    require(int(args.workers) == 8, "CAMPAIGN_REQUIRES_8X1_SCALAR_PHYSICS_FARM")
    require(args.device == "cuda", "CAMPAIGN_CANONICAL_DEVICE_CUDA")
    require(tuple(CANDIDATES_R102) == ((FLAT, 0.0), (SHORT, 0.25), (SHORT, 0.5), (SHORT, 0.75), (SHORT, 1.0), (LONG, 0.25), (LONG, 0.5), (LONG, 0.75), (LONG, 1.0)), "CAMPAIGN_CANDIDATE_GRID_DRIFT")

    admission_receipt = json.loads(ADMISSION_RECEIPT.read_text(encoding="utf-8"))
    e6_receipt = json.loads(E6_RECEIPT.read_text(encoding="utf-8"))
    require(admission_receipt.get("status") == "FROZEN_AND_QUALIFIED", "CAMPAIGN_ADMISSION_NOT_FROZEN")
    require(admission_receipt.get("qualified_code_sha") == "d9b0f3213c51ff78748619eb78434f63d3b58db9", "CAMPAIGN_ADMISSION_SHA_DRIFT")
    require(e6_receipt.get("status") == "FROZEN_AND_QUALIFIED", "CAMPAIGN_E6_NOT_FROZEN")
    require(e6_receipt.get("qualified_code_sha") == "72005b9d7eca06be99a4cebdbdaad48126b9d061", "CAMPAIGN_E6_SHA_DRIFT")

    lineage, g0_identity = r1.verify_g0_authority(g0_root)
    g0_before = dict(g0_identity["authority_file_hashes"])
    frozen_before = frozen_authority_hashes(package_root)

    source = adm.BinanceUSDMArchiveSourceR10(args.raw_root)
    layout = source.validate_layout()
    require(args.symbol in layout["symbols"], f"CAMPAIGN_SYMBOL_MISSING:{args.symbol}")
    required_rows = adm.SENSORY_PREFIX_MINUTES + (adm.TOTAL_GROUPS - 1) * 73 * 60 + adm.H72_MINUTES + 180
    records = find_contiguous_prefinal_run_r0(source, args.symbol, required_rows=required_rows)
    require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in records), "CAMPAIGN_FINAL_TOUCHED")
    require(all(int(b.open_time) - int(a.open_time) == adm.MINUTE_MS for a, b in zip(records, records[1:])), "CAMPAIGN_ARCHIVE_NOT_CONTIGUOUS")
    indices = adm.select_parent_indices_r0(records, adm.TOTAL_GROUPS)
    require(len(indices) == adm.TOTAL_GROUPS, "CAMPAIGN_PARENT_COUNT")
    require(int(records[indices[0]].open_time) == int(spec["cohort_identity"]["first_decision_time_ms"]), "CAMPAIGN_FIRST_PARENT_DRIFT")
    require(int(records[indices[-1]].open_time) == int(spec["cohort_identity"]["last_decision_time_ms"]), "CAMPAIGN_LAST_PARENT_DRIFT")

    first_data_idx = indices[0] - (adm.SENSORY_PREFIX_MINUTES - 1)
    start_ms = int(records[first_data_idx].open_time)
    end_ms = int(records[indices[-1] + adm.H72_MINUTES].open_time)
    funding = funding_events_by_minute_r0(source, args.symbol, start_ms=start_ms, end_ms=end_ms)
    physics = FrozenPhysicsRuntimeR102.load(package_root)

    frames = [build_sensory_frame_exact_r0(records, idx, args.symbol) for idx in indices]
    sensory = FrozenSensoryStackR10(package_root, device=args.device, verify_hashes=True)
    encoded_by_ordinal: dict[int, tuple[Any, Any, Any]] = {}
    for start in range(0, len(frames), 8):
        chunk = frames[start:start + 8]
        enc = sensory.encode_frames(chunk)
        for j in range(len(chunk)):
            encoded_by_ordinal[start + j] = (
                enc.operator48[j].copy(), enc.medium48[j].copy(), enc.ordered4h30[j].copy()
            )
    frozen_trainable = sum(
        int(p.requires_grad)
        for module in (sensory.operator.tok, sensory.operator.model, sensory.medium.model)
        for p in module.parameters()
    )
    require(frozen_trainable == 0, "CAMPAIGN_FROZEN_SENSORY_TRAINABLE_PARAMETER")
    del sensory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payloads: list[dict[str, Any]] = []
    parents: dict[str, ParentContextR102] = {}
    for ordinal, idx in enumerate(indices):
        parent_row = records[idx]
        future_rows = tuple(records[idx + 1:idx + 1 + adm.H72_MINUTES])
        require(len(future_rows) == adm.H72_MINUTES, "CAMPAIGN_H72_SLICE_LENGTH")
        require(int(future_rows[0].open_time) - int(parent_row.open_time) == adm.MINUTE_MS, "CAMPAIGN_NEXT_MINUTE_DRIFT")
        require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in future_rows), "CAMPAIGN_FUTURE_FINAL_TOUCHED")

        physics_prefix_start = idx - (adm.e6.PARENT_PREFIX_MINUTES - 1)
        physics_prefix = tuple(records[physics_prefix_start:idx + 1])
        parent_id = f"ADMR0:{args.symbol}:{int(parent_row.open_time)}"
        parent_state, risk_authority, account6 = adm.e6._build_parent_state(
            physics,
            symbol=args.symbol,
            account_id=parent_id,
            prefix_rows=physics_prefix,
            funding=funding,
        )
        lineage_hash = adm.e6._future_hash(args.symbol, int(parent_row.open_time), future_rows, funding)
        op, med, ordered = encoded_by_ordinal[ordinal]
        split = "TRAIN" if ordinal < adm.TRAIN_GROUPS else "VALIDATION"
        dep_id = f"FUT:{args.symbol}:{int(parent_row.open_time)}:{lineage_hash[:16]}"
        pc = ParentContextR102(
            parent_id=parent_id,
            dependence_group_id=dep_id,
            symbol=args.symbol,
            decision_time_ms=int(parent_row.open_time),
            split=split,
            scenario="FLAT_MINUTE_R2_ADMISSION",
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
        payloads.append({
            "ordinal": ordinal,
            "package_root": str(package_root),
            "symbol": args.symbol,
            "parent_id": parent_id,
            "decision_time_ms": int(parent_row.open_time),
            "parent_state": parent_state,
            "risk_authority": risk_authority,
            "future_rows": future_rows,
            "funding": dict(funding),
        })

    pairwise_nonoverlap = all(
        int(records[b].open_time) - int(records[a].open_time) >= adm.PARENT_SPACING_MS
        for a, b in zip(indices, indices[1:])
    )
    require(pairwise_nonoverlap, "CAMPAIGN_PARENT_FUTURES_OVERLAP")
    require(all(frames[i].decision_time_ms == int(records[indices[i]].open_time) + adm.MINUTE_MS for i in range(adm.TOTAL_GROUPS)), "CAMPAIGN_SENSORY_NOMINAL_TIME_DRIFT")

    causal = adm.e6._prefix_causality_canaries(
        str(package_root),
        args.symbol,
        payloads[0]["parent_state"],
        payloads[0]["risk_authority"],
        payloads[0]["future_rows"],
        funding,
    )
    require(all(causal.values()), f"CAMPAIGN_CAUSALITY_CANARY_FAIL:{causal}")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        executed = list(pool.map(adm._simulate_parent, payloads))
    executed.sort(key=lambda x: int(x["ordinal"]))
    require(len(executed) == adm.TOTAL_GROUPS, "CAMPAIGN_EXECUTED_PARENT_COUNT")
    require(all(len(x["branches"]) == len(CANDIDATES_R102) for x in executed), "CAMPAIGN_BRANCH_GRID_COUNT")

    parent_receipts = [
        {
            "parent_id": x["parent_id"],
            "decision_time_ms": x["decision_time_ms"],
            "first_execution_time_ms": x["first_execution_time_ms"],
            "split": parents[x["parent_id"]].split,
            "dependence_group_id": parents[x["parent_id"]].dependence_group_id,
            "student_context_object_id": parents[x["parent_id"]].student_context_object_id,
            "parent_snapshot_sha256": parents[x["parent_id"]].snapshot_sha256,
            "branch_utility_sha256": canonical_hash([(b["direction"], b["requested_risk"], b["utility"]) for b in x["branches"]]),
        }
        for x in executed
    ]
    cohort_hashes = {
        "parent_receipts_sha256": canonical_json_sha256(parent_receipts),
        "parent_ids_sha256": canonical_json_sha256([p["parent_id"] for p in parent_receipts]),
        "decision_times_sha256": canonical_json_sha256([p["decision_time_ms"] for p in parent_receipts]),
    }
    require(cohort_hashes["parent_receipts_sha256"] == EXPECTED_PARENT_RECEIPTS_SHA, "CAMPAIGN_PARENT_RECEIPTS_IDENTITY_DRIFT")
    require(cohort_hashes["parent_ids_sha256"] == EXPECTED_PARENT_IDS_SHA, "CAMPAIGN_PARENT_IDS_IDENTITY_DRIFT")
    require(cohort_hashes["decision_times_sha256"] == EXPECTED_DECISION_TIMES_SHA, "CAMPAIGN_DECISION_TIMES_IDENTITY_DRIFT")

    samples: list[CounterfactualBranchSampleR5] = []
    for x in executed:
        pc = parents[str(x["parent_id"])]
        for b in x["branches"]:
            sample = CounterfactualBranchSampleR5(
                parent_id=pc.parent_id,
                student_context_object_id=pc.student_context_object_id,
                timestamp=int(pc.decision_time_ms),
                context_features=tuple(float(v) for v in pc.student_features),
                direction=DIRECTION_TO_TEACHER[int(b["direction"])],
                requested_risk=float(b["requested_risk"]),
                realized_utility=float(b["utility"]),
                dependence_group_id=pc.dependence_group_id,
                market_lineage_hash=pc.market_lineage_hash,
            )
            sample.validate()
            samples.append(sample)
    require(len(samples) == 540, "CAMPAIGN_SAMPLE_COUNT")

    train_evidence, validation_evidence, teacher_stats = compile_teacher_evidence_r11(
        samples=samples,
        parents=parents,
        train_config=R11_TRAIN_TEACHER_CONFIG,
        val_config=R11_VALIDATION_TEACHER_CONFIG,
        workers=8,
        block_targets=32,
    )
    train_summary = evidence_summary(train_evidence)
    validation_summary = evidence_summary(validation_evidence)
    require(int(train_summary["admitted_dependence_groups"]) == 48 and int(train_summary["rejected"]) == 0, f"CAMPAIGN_TRAIN_SUPPORT_DRIFT:{train_summary}")
    require(int(validation_summary["admitted_dependence_groups"]) == 12 and int(validation_summary["rejected"]) == 0, f"CAMPAIGN_VALIDATION_SUPPORT_DRIFT:{validation_summary}")
    require({e.teacher_protocol_hash for e in train_evidence} == {R11_TRAIN_TEACHER_CONFIG.content_hash}, "CAMPAIGN_TRAIN_TEACHER_PROTOCOL_DRIFT")
    require({e.teacher_protocol_hash for e in validation_evidence} == {R11_VALIDATION_TEACHER_CONFIG.content_hash}, "CAMPAIGN_VALIDATION_TEACHER_PROTOCOL_DRIFT")

    campaign = prepare_evidence_campaign_r11(
        train_evidence=train_evidence,
        validation_evidence=validation_evidence,
        parents=parents,
        device=args.device,
    )
    require(campaign.train.evidence_hash == EXPECTED_TRAIN_EVIDENCE, "CAMPAIGN_TRAIN_EVIDENCE_IDENTITY_DRIFT")
    require(campaign.validation.evidence_hash == EXPECTED_VALIDATION_EVIDENCE, "CAMPAIGN_VALIDATION_EVIDENCE_IDENTITY_DRIFT")
    train_target_audit = validate_teacher_targets_r11(campaign.train)
    validation_target_audit = validate_teacher_targets_r11(campaign.validation)
    require(train_target_audit["independent_dependence_groups"] == 48, "CAMPAIGN_TRAIN_GROUP_AUDIT_DRIFT")
    require(validation_target_audit["independent_dependence_groups"] == 12, "CAMPAIGN_VALIDATION_GROUP_AUDIT_DRIFT")

    champion = r1.load_bootstrap_model(g0_root, args.device)
    champion_state_before = clone_state(champion)
    champion_hash = policy_hash_r11(champion)
    require(champion_hash == EXPECTED_G0_CHAMPION, f"CAMPAIGN_G0_CHAMPION_IDENTITY_DRIFT:{champion_hash}")
    challenger = copy.deepcopy(champion)
    require(policy_hash_r11(challenger) == champion_hash, "CAMPAIGN_CHALLENGER_NOT_EXACT_CHAMPION_COPY")

    trainer = IntegratedTrainingRuntimeR11(device=args.device)
    require(trainer.config.epochs == CANONICAL_EPOCHS_R11 == 12, "CAMPAIGN_EPOCH_RULE_DRIFT")
    require(trainer.config.batch_size == CANONICAL_BATCH_SIZE_R11 == 512, "CAMPAIGN_BATCH_RULE_DRIFT")
    require(abs(float(trainer.config.lr) - CANONICAL_LR_R11) <= 0.0 and abs(float(trainer.config.lr) - 3e-4) <= 0.0, "CAMPAIGN_LR_RULE_DRIFT")
    require(abs(float(trainer.config.weight_decay) - CANONICAL_WEIGHT_DECAY_R11) <= 0.0 and abs(float(trainer.config.weight_decay) - 1e-4) <= 0.0, "CAMPAIGN_WEIGHT_DECAY_RULE_DRIFT")
    require(int(trainer.config.generation_base_seed) == 24680, "CAMPAIGN_SEED_RULE_DRIFT")

    champion_validation = trainer.evaluation_runtime.evaluate(champion, campaign.validation, use_cache=True)
    independent_before = independent_loss_breakdown_r11(champion, campaign.validation)
    require(metric_close(champion_validation, independent_before), "CAMPAIGN_CHAMPION_VALIDATION_FORMULA_MISMATCH")
    behavior_before = capture_behavior_r11(champion, campaign.validation)

    snapshot = build_shadow_snapshot_r4(
        champion_policy_hash=champion_hash,
        train_evidence_hash=campaign.train.evidence_hash,
        validation_evidence_hash=campaign.validation.evidence_hash,
        train_teacher_protocol_hash=R11_TRAIN_TEACHER_CONFIG.content_hash,
        validation_teacher_protocol_hash=R11_VALIDATION_TEACHER_CONFIG.content_hash,
        train_parent_ids=campaign.train.parent_ids,
        train_dependence_group_ids=campaign.train.dependence_group_ids,
        validation_parent_ids=campaign.validation.parent_ids,
        validation_dependence_group_ids=campaign.validation.dependence_group_ids,
    )
    r1.atomic_json(work_root / "CAMPAIGN_R0_FROZEN_TRAINING_SNAPSHOT.json", snapshot)

    training_receipt = trainer.train_challenger(
        model=challenger,
        campaign=campaign,
        generation=0,
        snapshot_hash=snapshot["snapshot_hash"],
        receipt_dir=work_root / "shadow_training",
    )
    expected_steps = 12
    require(training_receipt["optimizer_steps"] == expected_steps, f"CAMPAIGN_OPTIMIZER_STEP_DRIFT:{training_receipt['optimizer_steps']}")
    require(training_receipt["generation_seed"] == 24680, "CAMPAIGN_GENERATION_SEED_DRIFT")
    require(training_receipt["epochs"] == 12 and training_receipt["batch_size"] == 512, "CAMPAIGN_TRAINING_RULE_RECEIPT_DRIFT")
    require(frozenset(training_receipt["gradient_owner_set_last_step"]) == AUTHORIZED_GRADIENT_OWNERS_R11, "CAMPAIGN_GRADIENT_OWNER_SET_DRIFT")
    require(all(math.isfinite(float(v)) and float(v) > 0.0 for v in training_receipt["update_group_norms"].values()), "CAMPAIGN_SIX_OWNER_UPDATE_DISCONNECT")
    require(metric_close(training_receipt["validation_before"], champion_validation), "CAMPAIGN_TRAINER_CHAMPION_BASELINE_DRIFT")

    independent_after = independent_loss_breakdown_r11(challenger, campaign.validation)
    require(metric_close(training_receipt["validation_after"], independent_after), "CAMPAIGN_CHALLENGER_VALIDATION_FORMULA_MISMATCH")
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
    require(ranking["canonical_promotion_authorized"] is False, "CAMPAIGN_PROMOTION_ACCIDENTALLY_AUTHORIZED")
    require(ranking["canonical_generation_advance_authorized"] is False, "CAMPAIGN_GENERATION_ACCIDENTALLY_AUTHORIZED")

    checkpoint = save_shadow_challenger(
        work_root / "CAMPAIGN_R0_SHADOW_CHALLENGER.pt",
        challenger,
        champion_hash=champion_hash,
        snapshot_hash=snapshot["snapshot_hash"],
    )
    restart = _reload_shadow_exact(
        Path(checkpoint["path"]),
        champion,
        challenger,
        device=args.device,
        expected_snapshot_hash=snapshot["snapshot_hash"],
    )
    require(restart["exact"], "CAMPAIGN_SHADOW_CHECKPOINT_RESTART_NOT_EXACT")

    require(state_equal(champion_state_before, champion), "CAMPAIGN_CHAMPION_MODEL_MUTATED")
    require(policy_hash_r11(champion) == champion_hash, "CAMPAIGN_CHAMPION_POLICY_HASH_MUTATED")
    require(frozen_authority_hashes(package_root) == frozen_before, "CAMPAIGN_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "CAMPAIGN_G0_AUTHORITY_MUTATED")
    require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in records), "CAMPAIGN_FINAL_FIREWALL_DRIFT")

    gates = {
        "ADMISSION_FROZEN_LINEAGE_BOUND": True,
        "E6_FROZEN_LINEAGE_BOUND": True,
        "ADMISSION_COHORT_PARENT_RECEIPTS_EXACT": cohort_hashes["parent_receipts_sha256"] == EXPECTED_PARENT_RECEIPTS_SHA,
        "ADMISSION_COHORT_PARENT_IDS_EXACT": cohort_hashes["parent_ids_sha256"] == EXPECTED_PARENT_IDS_SHA,
        "ADMISSION_COHORT_DECISION_TIMES_EXACT": cohort_hashes["decision_times_sha256"] == EXPECTED_DECISION_TIMES_SHA,
        "ADMISSION_TRAIN_EVIDENCE_EXACT": campaign.train.evidence_hash == EXPECTED_TRAIN_EVIDENCE,
        "ADMISSION_VALIDATION_EVIDENCE_EXACT": campaign.validation.evidence_hash == EXPECTED_VALIDATION_EVIDENCE,
        "PRODUCTION_TEACHER_PROTOCOL_HASHES_EXACT": True,
        "TRAIN_SUPPORT_48_OF_48": int(train_summary["admitted_dependence_groups"]) == 48 and int(train_summary["rejected"]) == 0,
        "VALIDATION_SUPPORT_12_OF_12": int(validation_summary["admitted_dependence_groups"]) == 12 and int(validation_summary["rejected"]) == 0,
        "IMMUTABLE_G0_CHAMPION_EXACT": champion_hash == EXPECTED_G0_CHAMPION,
        "CHALLENGER_STARTS_FROM_EXACT_CHAMPION": True,
        "FULL_CANONICAL_12_EPOCH_RULE": training_receipt["epochs"] == 12 and training_receipt["optimizer_steps"] == 12,
        "EXACT_SIX_GRADIENT_OWNERS": frozenset(training_receipt["gradient_owner_set_last_step"]) == AUTHORIZED_GRADIENT_OWNERS_R11,
        "SIX_OWNER_PARAMETER_UPDATES_NONZERO": all(float(v) > 0.0 for v in training_receipt["update_group_norms"].values()),
        "CHAMPION_VALIDATION_FORMULA_EXACT": metric_close(champion_validation, independent_before),
        "CHALLENGER_VALIDATION_FORMULA_EXACT": metric_close(training_receipt["validation_after"], independent_after),
        "SHADOW_CHECKPOINT_RESTART_EXACT": bool(restart["exact"]),
        "CHAMPION_IMMUTABLE": state_equal(champion_state_before, champion),
        "NO_CANONICAL_PROMOTION": ranking["canonical_promotion_authorized"] is False,
        "NO_CANONICAL_GENERATION_ADVANCE": ranking["canonical_generation_advance_authorized"] is False,
        "FUTURE_SUFFIX_MUTATION_INVARIANCE": bool(causal["future_suffix_mutation_invariance"]),
        "FUTURE_ACCOUNT_POISON_INVARIANCE": bool(causal["future_account_poison_invariance"]),
        "NEXT_BAR_ISOLATION": bool(causal["next_bar_isolation"]),
        "FINAL_FIREWALL_CLOSED": True,
        "FRESH_DATA_FIREWALL_CLOSED": True,
    }
    require(all(gates.values()), f"CAMPAIGN_GATE_FAIL:{gates}")

    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": "ONE_BOUNDED_GENERATION0_MINUTE_CAMPAIGN_COMPLETED__SHADOW_ONLY__NO_MARKET_VERDICT",
        "spec_sha256": sha256_file(SPEC),
        "lineage": {
            "admission_freeze_receipt_sha256": sha256_file(ADMISSION_RECEIPT),
            "e6_freeze_receipt_sha256": sha256_file(E6_RECEIPT),
            "admission_qualified_code_sha": admission_receipt["qualified_code_sha"],
            "e6_qualified_code_sha": e6_receipt["qualified_code_sha"],
        },
        "archive": {
            "symbol": args.symbol,
            "first_ms": start_ms,
            "last_ms": end_ms,
            "contiguous_observed_1m_only": True,
            "final_holdout_touched": False,
            "fresh_market_data_downloaded": False,
        },
        "cohort": {
            "train_groups": 48,
            "validation_groups": 12,
            "total_groups": 60,
            "branches_per_parent": 9,
            "total_branches": 540,
            "horizon_minutes": 4320,
            "parent_spacing_hours": 73,
            "decision_to_execution_minutes": 1,
            "hashes": cohort_hashes,
            "parent_receipts": parent_receipts,
        },
        "sensory": {
            "nominal_time": "H",
            "max_visible_market_time": "H-1m",
            "frozen_trainable_parameters": frozen_trainable,
        },
        "teacher": {
            "runtime_stats": str(teacher_stats),
            "train_summary": train_summary,
            "validation_summary": validation_summary,
            "train_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash,
            "validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
            "support_thresholds_relaxed": False,
            "realized_outcome_used_as_correct_action_label": False,
        },
        "training_snapshot": snapshot,
        "training_targets": {
            "train": train_target_audit,
            "validation": validation_target_audit,
        },
        "champion": {
            "canonical_role": "R11_G0_INITIAL_CHAMPION",
            "policy_hash": champion_hash,
            "validation": champion_validation,
            "state_mutated": False,
        },
        "challenger": {
            "starts_from_exact_champion": True,
            "training_receipt": training_receipt,
            "checkpoint": checkpoint,
            "checkpoint_restart": restart,
            "independent_validation_after": independent_after,
        },
        "feedback_diagnostics": feedback,
        "behavior_delta": behavior_delta,
        "shadow_tournament": ranking,
        "causality_canaries": causal,
        "gates": gates,
        "canonical_promotion_authorized": False,
        "canonical_generation_advance_authorized": False,
        "production_cutover_authorized": False,
        "new_scientific_market_verdict": False,
        "scientific_verdict": None,
        "final_holdout_touched": False,
        "fresh_market_data_downloaded": False,
        "next_gate": "LONGTRAJ_TRAINING_CAMPAIGN_R0_ADJUDICATION__NO_AUTOMATIC_PROMOTION",
    }
    r1.atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "classification": result["classification"],
        "champion_validation_loss": champion_validation["loss"],
        "challenger_validation_loss": training_receipt["validation_after"]["loss"],
        "relative_validation_improvement": ranking["relative_validation_improvement"],
        "shadow_winner": ranking["shadow_winner"],
        "optimizer_steps": training_receipt["optimizer_steps"],
        "parameter_l2_delta": training_receipt["parameter_l2_delta"],
        "checkpoint_restart_exact": restart["exact"],
        "gates": gates,
        "canonical_promotion_authorized": False,
        "canonical_generation_advance_authorized": False,
        "scientific_verdict": None,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
