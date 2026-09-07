#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_local_opt.checkpoint_store_r11 import CheckpointStoreR11
from cb16_local_opt.evidence_store_r11 import EvidenceItemR11, EvidenceStoreR11
from cb16_local_opt.event_journal_r11 import EventJournalR11
from cb16_local_opt.integration_adapters_r11 import (
    CheckpointStoreProtocolAdapterR11,
    EvidenceStoreProtocolAdapterR11,
    EventJournalProtocolAdapterR11,
)
from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.orchestrator_r11 import R11Orchestrator
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_common import G0_FILE_SHA256, model_state_semantic_sha256, sha256_file, sha256_obj
from cb16_local_opt.r102_evidence_cache import load_parent_physics_states, load_teacher_samples
from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
from cb16_local_opt.r102_policy_trace import run_real_on_policy_trace
from cb16_local_opt.r102_teacher_incremental import compile_teacher_evidence_incremental
from cb16_local_opt.r2_training_incremental import (
    evaluate_policy_r2,
    prepare_evidence_batch_r2,
    train_challenger_r2,
)
from cb16_local_opt.rearchitecture_authority_r11 import verify_static_semantic_contracts
from cb16_local_opt.runtime_events_r11 import (
    AuthorityStamp,
    EvidenceRef,
    FrozenInputReceipt,
    PolicyResult,
    SnapshotSeal,
    TournamentDecision,
    TournamentResult,
    TrainingResult,
    ValidationResult,
)
from cb16_local_opt.sharded_experience_lake import ShardedExperienceLake
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.trace_runtime_r11 import TraceRuntimeR11, run_real_on_policy_trace_r11
from cb16_local_opt.training_runtime_r11 import (
    EvaluationRuntimeR11,
    TrainingRuntimeR11,
    policy_hash_r11,
    prepare_evidence_campaign_r11,
)
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10
from scripts.run_r11_teacher_vectorized_qualification import _compare_evidence


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, Mapping) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, Mapping):
        raise RuntimeError("R11_TASK_F_G0_STATE_NOT_FOUND")
    return obj


def state_compare(a: torch.nn.Module, b: torch.nn.Module) -> dict[str, Any]:
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys():
        return {"pass": False, "reason": "STATE_KEY_MISMATCH"}
    exact = True
    max_abs = 0.0
    values = 0
    for key in sa:
        x = sa[key].detach().cpu()
        y = sb[key].detach().cpu()
        if x.shape != y.shape or x.dtype != y.dtype:
            return {"pass": False, "reason": f"STATE_LAYOUT_MISMATCH:{key}"}
        exact = exact and torch.equal(x, y)
        if x.numel():
            d = float((x.double() - y.double()).abs().max())
            max_abs = max(max_abs, d)
            values += x.numel()
    return {
        "pass": bool(max_abs <= 1e-7),
        "bitwise_equal": bool(exact),
        "max_abs_diff": max_abs,
        "values_checked": int(values),
        "atol": 1e-7,
    }


def validation_compare(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("loss", "direction_loss", "sizing_loss")
    diffs = {k: abs(float(old[k]) - float(new[k])) for k in keys}
    return {"pass": max(diffs.values(), default=0.0) <= 1e-7, "diffs": diffs, "atol": 1e-7}


def promotion(before: Mapping[str, Any], after: Mapping[str, Any]) -> TournamentDecision:
    b, a = float(before["loss"]), float(after["loss"])
    rel = (b - a) / max(abs(b), 1e-12)
    return TournamentDecision.PROMOTE if math.isfinite(a) and a < b and rel >= 0.001 else TournamentDecision.REJECT


def main() -> int:
    ap = argparse.ArgumentParser(description="CB16 R11 Task F real-stack acceptance")
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--legacy-r104-root", required=True)
    ap.add_argument("--teacher-cache-root", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher-workers", type=int, default=12)
    ap.add_argument("--trace-workers", type=int, default=8)
    args = ap.parse_args()

    package_root = Path(args.package_root).resolve()
    legacy_root = Path(args.legacy_r104_root).resolve()
    teacher_cache_root = Path(args.teacher_cache_root).resolve()
    work = Path(args.work_root).resolve()
    out = Path(args.out).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    static_guard = verify_static_semantic_contracts(ROOT)
    frozen_before = frozen_authority_hashes(package_root)
    manifest_path = legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("stride_hours", -1)) != 256:
        raise RuntimeError("R11_TASK_F_EXPECTED_R104_STRIDE_256")
    if bool(manifest.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_TASK_F_REFUSES_FINAL_HOLDOUT_CACHE")
    parents, samples = load_teacher_samples(manifest["parents_file"], manifest["branches_file"])
    parent_states = load_parent_physics_states(manifest["parent_states_file"])

    # A: full real Teacher evidence against the already-published legacy oracle.
    old_train, old_val, oracle_receipt = compile_teacher_evidence_incremental(
        samples=samples,
        parents=parents,
        source_identity=manifest,
        cache_root=teacher_cache_root,
        train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,
        workers=1,
        threads_per_worker=1,
        max_in_flight=1,
    )
    if oracle_receipt.get("mode") != "REUSED_VERIFIED_AUTHORITY":
        raise RuntimeError("R11_TASK_F_TEACHER_ORACLE_NOT_WARM_VERIFIED")
    t0 = time.perf_counter()
    new_train, new_val, teacher_stats = compile_teacher_evidence_r11(
        samples=samples,
        parents=parents,
        train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,
        workers=int(args.teacher_workers),
        block_targets=64,
    )
    teacher_seconds = time.perf_counter() - t0
    teacher_train_cmp = _compare_evidence(old_train, new_train)
    teacher_val_cmp = _compare_evidence(old_val, new_val)
    if not teacher_train_cmp["pass"] or not teacher_val_cmp["pass"]:
        raise RuntimeError("R11_TASK_F_TEACHER_EQUIVALENCE_FAIL")

    # B: full canonical 12-epoch AdamW training, same evidence and same G0 state.
    device = "cuda"
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (6, 1):
        raise RuntimeError("R11_TASK_F_REQUIRES_GTX1060_SM61")
    g0_path = package_root / "authority/g0_parent/central_brain_g0_r10_2_parent.pt"
    if sha256_file(g0_path) != G0_FILE_SHA256:
        raise RuntimeError("R11_TASK_F_G0_FILE_SHA_DRIFT")
    g0_state = load_checkpoint_state(g0_path)
    legacy_model = build_g0_brain_r10("TIER_1", seed=24680, device=device)
    r11_model = build_g0_brain_r10("TIER_1", seed=24680, device=device)
    trace_model = build_g0_brain_r10("TIER_1", seed=24680, device=device)
    for model in (legacy_model, r11_model, trace_model):
        model.load_state_dict(g0_state, strict=True)

    old_train_batch = prepare_evidence_batch_r2(old_train, parents, device=device)
    old_val_batch = prepare_evidence_batch_r2(old_val, parents, device=device)
    legacy_val_before, legacy_behavior_before = evaluate_policy_r2(legacy_model, old_val_batch)
    legacy_receipt, legacy_behavior_after = train_challenger_r2(
        model=legacy_model,
        train_batch=old_train_batch,
        val_batch=old_val_batch,
        validation_before=legacy_val_before,
        device=device,
        generation=1,
        snapshot_hash="TASK_F_REAL_SNAPSHOT",
        receipt_dir=work / "legacy_training",
    )

    prepared = prepare_evidence_campaign_r11(
        train_evidence=old_train,
        validation_evidence=old_val,
        parents=parents,
        device=device,
        pin_memory=True,
    )
    evaluation_runtime = EvaluationRuntimeR11(enable_cuda_graph=True)
    r11_training = TrainingRuntimeR11(device=device, evaluation_runtime=evaluation_runtime)
    r11_receipt = r11_training.train_challenger(
        model=r11_model,
        campaign=prepared,
        generation=1,
        snapshot_hash="TASK_F_REAL_SNAPSHOT",
        receipt_dir=work / "r11_training",
    )
    training_state_cmp = state_compare(legacy_model, r11_model)
    training_val_cmp = validation_compare(legacy_receipt["validation_after"], r11_receipt["validation_after"])
    behavior_after_old = legacy_behavior_after or evaluate_policy_r2(legacy_model, old_val_batch)[1]
    behavior_after_new = r11_receipt["validation_after"]["behavior_fingerprint"]
    behavior_equal = behavior_after_old["sha256"] == behavior_after_new["sha256"]
    if not training_state_cmp["pass"] or not training_val_cmp["pass"] or not behavior_equal:
        raise RuntimeError("R11_TASK_F_TRAINING_EQUIVALENCE_FAIL")

    # C: exact frozen Supervisor/Physics trace, legacy serial vs R11 shared-cache parallel runtime.
    physics = FrozenPhysicsRuntimeR102.load(package_root)
    policy_hash = model_state_semantic_sha256(trace_model)
    legacy_lake = ShardedExperienceLake(work / "legacy_trace_lake", shards=4, synchronous="FULL")
    r11_lake = ShardedExperienceLake(work / "r11_trace_lake", shards=4, synchronous="FULL")
    market_cache = MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    trace_runtime = TraceRuntimeR11(
        physics=physics, market_cache=market_cache, max_workers=int(args.trace_workers)
    )
    try:
        legacy_trace = run_real_on_policy_trace(
            model=trace_model,
            policy_hash=policy_hash,
            generation=1,
            parents=parents,
            parent_states=parent_states,
            cache_dir=legacy_root / "evidence_cache",
            physics=physics,
            lake=legacy_lake,
            device=device,
            max_groups=24,
        )
        r11_trace = run_real_on_policy_trace_r11(
            model=trace_model,
            policy_hash=policy_hash,
            generation=1,
            parents=parents,
            parent_states=parent_states,
            trace_runtime=trace_runtime,
            lake=r11_lake,
            device=device,
            max_groups=24,
        )
        market_cache.assert_read_only()
        trace_cache_stats = asdict(market_cache.stats())
    finally:
        trace_runtime.close()
        market_cache.close()
    legacy_lake_audit = legacy_lake.audit(verify_payloads=True)
    r11_lake_audit = r11_lake.audit(verify_payloads=True)
    legacy_lake.close(); r11_lake.close()
    trace_equal = legacy_trace == r11_trace
    if not trace_equal or not legacy_lake_audit["pass"] or not r11_lake_audit["pass"]:
        raise RuntimeError("R11_TASK_F_TRACE_EQUIVALENCE_FAIL")

    # D: materialize the real admitted Teacher evidence, seal, replay, fast-open, forensic audit.
    admitted = [e for e in new_train if e.admission.admitted]
    evidence_items = [
        EvidenceItemR11(
            evidence_id=e.evidence_id,
            parent_snapshot_hash=parents[e.parent_id].snapshot_sha256,
            lineage_hash=e.content_hash,
            teacher_protocol_hash=e.teacher_protocol_hash,
            payload=asdict(e),
        )
        for e in admitted
    ]
    evidence_store = EvidenceStoreR11(
        metadata_root=work / "storage_meta",
        payload_roots=[work / "storage_payload"],
        segment_target_bytes=1 << 20,
        codec="zlib",
        read_only_source_roots=[legacy_root],
    )
    _, first_storage = evidence_store.put_evidence(evidence_items)
    evidence_set = evidence_store.seal_evidence_set("TASK_F_REAL_TRAIN_SET", [x.evidence_id for x in evidence_items])
    evidence_store.seal_all_active_segments()
    periodic = evidence_store.periodic_audit()
    forensic_first = evidence_store.full_forensic_audit()
    evidence_store.close()

    evidence_store = EvidenceStoreR11(
        metadata_root=work / "storage_meta",
        payload_roots=[work / "storage_payload"],
        segment_target_bytes=1 << 20,
        codec="zlib",
        read_only_source_roots=[legacy_root],
    )
    startup = dict(evidence_store.startup_stats)
    _, replay_storage = evidence_store.put_evidence(evidence_items)
    forensic_replay = evidence_store.full_forensic_audit()
    storage_pass = bool(
        first_storage.created_payload_count > 0
        and replay_storage.created_payload_count == 0
        and replay_storage.created_evidence_count == 0
        and startup["sealed_payload_bytes_read"] == 0
        and periodic["pass"] and forensic_first["pass"] and forensic_replay["pass"]
    )
    if not storage_pass:
        raise RuntimeError("R11_TASK_F_STORAGE_REPLAY_FAIL")

    # D+E: real checkpoint objects plus durable generation barriers through concrete adapters.
    checkpoint_store = CheckpointStoreR11(work / "checkpoints")
    parent_obj = checkpoint_store.put_state_dict(trace_model.state_dict())
    challenger_obj = checkpoint_store.put_state_dict(r11_model.state_dict())
    parent_dup = checkpoint_store.put_state_dict(trace_model.state_dict())
    if parent_dup.created or parent_obj.semantic_sha256 != parent_dup.semantic_sha256:
        raise RuntimeError("R11_TASK_F_CHECKPOINT_DEDUP_FAIL")
    event_journal = EventJournalR11(work / "orchestrator_journal")
    evidence_adapter = EvidenceStoreProtocolAdapterR11(evidence_store)
    journal_adapter = EventJournalProtocolAdapterR11(event_journal)
    checkpoint_adapter = CheckpointStoreProtocolAdapterR11(checkpoint_store)
    authority = AuthorityStamp(
        generation=1,
        champion_id=parent_obj.semantic_sha256,
        champion_hash=parent_obj.semantic_sha256,
        teacher_authority_id=str(oracle_receipt["authority_hash"]),
        physics_authority_id=sha256_obj(frozen_before),
    )
    orchestrator = R11Orchestrator(
        authority=authority,
        evidence_store=evidence_adapter,
        journal=journal_adapter,
        checkpoint_store=checkpoint_adapter,
    )
    orchestrator.verify_frozen_inputs(FrozenInputReceipt("R104_FROZEN_CACHE", sha256_obj(manifest), True))
    set_ref = EvidenceRef(
        evidence_id="TRAIN_SET:" + evidence_set["evidence_set_hash"],
        payload_hash=evidence_set["evidence_set_hash"],
        source_generation=0,
        producer_champion_id=parent_obj.semantic_sha256,
        teacher_authority_id=authority.teacher_authority_id,
        physics_authority_id=authority.physics_authority_id,
        teacher_generation=0,
    )
    orchestrator.accept_teacher_evidence(set_ref)
    snapshot_hash = sha256_obj({"parent": authority.champion_hash, "evidence_set": evidence_set["evidence_set_hash"]})
    snapshot = SnapshotSeal(
        "TASK_F_SNAPSHOT_G1", snapshot_hash, 1, authority.champion_id, (set_ref.evidence_id,)
    )
    orchestrator.seal_training_snapshot(snapshot)
    orchestrator.accept_policy_result(PolicyResult("POLICY_G1", 1, authority.champion_id, sha256_obj(legacy_behavior_before)))
    orchestrator.start_trace("TRACE_G1")
    trace_ref = EvidenceRef(
        evidence_id="TRACE_SET_G1",
        payload_hash=sha256_obj(r11_trace),
        source_generation=1,
        producer_champion_id=authority.champion_id,
        teacher_authority_id=authority.teacher_authority_id,
        physics_authority_id=authority.physics_authority_id,
        teacher_generation=1,
    )
    orchestrator.materialize_trace_evidence(trace_ref, work_id="TRACE_COMPLETE_G1")
    orchestrator.start_challenger_training(work_id="TRAIN_G1")
    orchestrator.complete_training(TrainingResult(
        "TRAIN_G1", 1, challenger_obj.semantic_sha256, challenger_obj.semantic_sha256,
        authority.champion_id, snapshot.snapshot_id, sha256_obj(r11_receipt)
    ))
    orchestrator.complete_validation(ValidationResult(
        "VALIDATE_G1", 1, challenger_obj.semantic_sha256, authority.champion_id,
        snapshot.snapshot_id, "VALIDATION_G1", sha256_obj(r11_receipt["validation_after"])
    ))
    decision = promotion(r11_receipt["validation_before"], r11_receipt["validation_after"])
    orchestrator.decide_tournament(TournamentResult(
        "TOURNAMENT_G1", 1, challenger_obj.semantic_sha256, authority.champion_id,
        "VALIDATION_G1", decision, sha256_obj({"decision": decision.value, "validation": r11_receipt["validation_after"]})
    ))
    commit = orchestrator.commit_tournament()
    orchestrator.seal_checkpoint(commit.next_champion_hash)
    next_authority = orchestrator.release_next_generation()
    recovered = R11Orchestrator(
        authority=authority,
        evidence_store=EvidenceStoreProtocolAdapterR11(evidence_store),
        journal=EventJournalProtocolAdapterR11(event_journal),
        checkpoint_store=CheckpointStoreProtocolAdapterR11(checkpoint_store),
        recover=True,
    )
    orchestrator_pass = bool(
        recovered.record.commit == orchestrator.record.commit
        and recovered.record.checkpoint == orchestrator.record.checkpoint
        and next_authority.champion_hash == commit.next_champion_hash
    )
    journal_audit = event_journal.audit()
    if not orchestrator_pass or not journal_audit["pass"]:
        raise RuntimeError("R11_TASK_F_ORCHESTRATOR_DURABILITY_FAIL")

    checkpoint_store.close(); event_journal.close(); evidence_store.close()
    frozen_after = frozen_authority_hashes(package_root)
    if frozen_before != frozen_after:
        raise RuntimeError("R11_TASK_F_FROZEN_AUTHORITY_DRIFT")

    result = {
        "schema": "CB16_R11_TASK_F_INTEGRATION_ACCEPTANCE_V1",
        "status": "PASS",
        "scientific_semantics_changed": False,
        "static_semantic_guard": static_guard,
        "hardware": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "training_dtype": "FP32",
            "amp": False,
        },
        "teacher": {
            "workers": int(args.teacher_workers),
            "seconds": teacher_seconds,
            "train": teacher_train_cmp,
            "validation": teacher_val_cmp,
            "stats": asdict(teacher_stats),
        },
        "training": {
            "state_equivalence": training_state_cmp,
            "validation_equivalence": training_val_cmp,
            "behavior_sha_equal": behavior_equal,
            "legacy_optimizer_steps": legacy_receipt["optimizer_steps"],
            "r11_optimizer_steps": r11_receipt["optimizer_steps"],
            "r11_execution_mode_before": r11_receipt["validation_before"]["execution_mode"],
            "r11_execution_mode_after": r11_receipt["validation_after"]["execution_mode"],
        },
        "trace": {
            "receipt_exact_equal": trace_equal,
            "trace_count": r11_trace["trace_count"],
            "matured": r11_trace["matured"],
            "market_cache": trace_cache_stats,
        },
        "storage": {
            "first": asdict(first_storage),
            "replay": asdict(replay_storage),
            "startup": startup,
            "periodic_pass": periodic["pass"],
            "forensic_pass": forensic_replay["pass"],
            "evidence_set_hash": evidence_set["evidence_set_hash"],
        },
        "checkpoint_orchestrator": {
            "parent_checkpoint": parent_obj.semantic_sha256,
            "challenger_checkpoint": challenger_obj.semantic_sha256,
            "decision": decision.value,
            "next_champion": next_authority.champion_hash,
            "durable_recovery_equal": orchestrator_pass,
            "journal_audit_pass": journal_audit["pass"],
        },
        "frozen_authority_unchanged": True,
        "forbidden_work": {
            "raw_1m_archive_read": False,
            "raw_market_data_modified": False,
            "final_holdout_2025_09_read": False,
            "frozen_sensory_modified": False,
            "frozen_physics_or_supervisor_modified": False,
            "legacy_python_runtime_made_authority": False,
        },
    }
    atomic_json(out, result)
    print(json.dumps({
        "status": "PASS",
        "teacher_seconds": teacher_seconds,
        "training_bitwise_equal": training_state_cmp["bitwise_equal"],
        "training_max_abs_diff": training_state_cmp["max_abs_diff"],
        "trace_exact_equal": trace_equal,
        "storage_replay_new_payloads": replay_storage.created_payload_count,
        "decision": decision.value,
        "next_champion": next_authority.champion_hash,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
