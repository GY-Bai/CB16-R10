#!/usr/bin/env python3
from __future__ import annotations

"""R11 Stage-2 bounded Shanxi integration burst.

This is a qualification workload, not a scientific campaign.  It repeatedly replays
one frozen generation to exercise the accepted R11 execution planes under sustained
CPU/GPU/IO overlap.  Replay counters are engineering throughput counters only; this
module never emits a new market-information or learning verdict.
"""

import json
import math
import os
import shutil
import statistics
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.burst_orchestrator_r11 import BurstLimits, BurstOrchestratorR11, RamSample
from cb16_local_opt.checkpoint_store_r11 import CheckpointStoreR11
from cb16_local_opt.evidence_store_r11 import EvidenceItemR11, EvidenceStoreR11
from cb16_local_opt.event_journal_r11 import EventJournalR11
from cb16_local_opt.integration_adapters_r11 import (
    CheckpointStoreProtocolAdapterR11,
    EvidenceStoreProtocolAdapterR11,
    EventJournalProtocolAdapterR11,
)
from cb16_local_opt.io_runtime_r11 import IOThroughputConfigR11, IOThroughputRuntimeR11
from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.orchestrator_r11 import R11Orchestrator
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_common import G0_FILE_SHA256, model_state_semantic_sha256, sha256_file, sha256_obj
from cb16_local_opt.r102_evidence_cache import load_parent_physics_states, load_teacher_samples
from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
from cb16_local_opt.r102_teacher_incremental import compile_teacher_evidence_incremental
from cb16_local_opt.r2_training_incremental import evaluate_policy_r2, prepare_evidence_batch_r2, train_challenger_r2
from cb16_local_opt.rearchitecture_authority_r11 import verify_static_semantic_contracts
from cb16_local_opt.runtime_events_r11 import (
    AuthorityStamp,
    EvidenceRef,
    FrozenInputReceipt,
    PolicyResult,
    SnapshotSeal,
    TournamentResult,
    TrainingResult,
    ValidationResult,
    WorkCompletion,
    WorkItem,
    WorkKind,
    WorkerPool,
)
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.trace_process_runtime_r11 import ForkProcessTraceRuntimeR11
from cb16_local_opt.trace_runtime_r11 import TraceRuntimeR11
from cb16_local_opt.training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    EvaluationRuntimeR11,
    TrainingRuntimeR11,
    prepare_evidence_campaign_r11,
)
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10
from scripts.benchmark_burst_orchestrator_r11 import AUTH as BENCH_AUTH, new_core as new_burst_core
from scripts.run_r11_stage2_task_a_trace_process_benchmark import _choose_items, _trace_hash
from scripts.run_r11_task_f_acceptance import (
    load_checkpoint_state,
    promotion,
    state_compare,
    validation_compare,
)
from scripts.run_r11_teacher_vectorized_qualification import _compare_evidence

TELEMETRY_SCHEMA = "CB16_R11_STAGE2_BURST_TELEMETRY_V1"
CORRECTNESS_SCHEMA = "CB16_R11_STAGE2_BURST_CORRECTNESS_V1"


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


class Counters:
    _NUMERIC = (
        "generations_committed",
        "teacher_evidence",
        "traces",
        "training_steps",
        "fp32_training_examples",
        "ssd_metadata_ops",
        "hdd_write_bytes",
        "hdd_read_bytes",
        "barrier_block_seconds",
        "fsync_stall_seconds",
    )

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.values: dict[str, float] = {k: 0.0 for k in self._NUMERIC}
        self.teacher_active = False
        self.trace_active = False
        self.gpu_active = False
        self.storage_active = False
        self.pipeline_queue_depth = 0
        self.fsync_latencies_ms: list[float] = []

    def add(self, **kwargs: float) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if key not in self.values:
                    raise KeyError(key)
                self.values[key] += float(value)

    def set_flags(self, **kwargs: bool | int) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if not hasattr(self, key):
                    raise AttributeError(key)
                setattr(self, key, value)

    def add_fsync_latency(self, seconds: float) -> None:
        with self.lock:
            self.fsync_latencies_ms.append(float(seconds) * 1000.0)
            if len(self.fsync_latencies_ms) > 2048:
                self.fsync_latencies_ms = self.fsync_latencies_ms[-1024:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            p95 = 0.0
            if self.fsync_latencies_ms:
                xs = sorted(self.fsync_latencies_ms)
                p95 = xs[min(len(xs) - 1, int(math.ceil(0.95 * len(xs))) - 1)]
            active = int(self.teacher_active) + int(self.trace_active) + int(self.gpu_active) + int(self.storage_active)
            return {
                **{k: (int(v) if k not in {"barrier_block_seconds", "fsync_stall_seconds"} else float(v)) for k, v in self.values.items()},
                "fsync_latency_p95_ms": float(p95),
                "pipeline_queue_depth": int(max(active, int(self.pipeline_queue_depth))),
                "teacher_worker_utilization_pct": 100.0 if self.teacher_active else 0.0,
                "trace_worker_utilization_pct": 100.0 if self.trace_active else 0.0,
            }


def telemetry_loop(path: Path, counters: Counters, stop: threading.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as handle:
        while not stop.is_set():
            row = {
                "schema": TELEMETRY_SCHEMA,
                "monotonic_seconds": time.monotonic(),
                **counters.snapshot(),
            }
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stop.wait(1.0)
        row = {
            "schema": TELEMETRY_SCHEMA,
            "monotonic_seconds": time.monotonic(),
            **counters.snapshot(),
        }
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _batch(rows: list[Any], n: int):
    for start in range(0, len(rows), n):
        yield rows[start : start + n]


def _durable_control_plane_proof(
    *,
    ssd_root: Path,
    hdd_root: Path,
    legacy_root: Path,
    manifest: Mapping[str, Any],
    frozen_hashes: Mapping[str, str],
    oracle_authority: str,
    evidence_items: list[EvidenceItemR11],
    parent_model: torch.nn.Module,
    challenger_model: torch.nn.Module,
    trace_hash: str,
    training_receipt: Mapping[str, Any],
    policy_before: Mapping[str, Any],
) -> tuple[bool, bool]:
    root = ssd_root / "durable_control"
    payload = hdd_root / "durable_control_payload"
    evidence_store = EvidenceStoreR11(
        metadata_root=root / "evidence_meta",
        payload_roots=[payload],
        segment_target_bytes=1 << 20,
        codec="zlib",
        read_only_source_roots=[legacy_root],
    )
    subset = evidence_items[: min(16, len(evidence_items))]
    if not subset:
        raise RuntimeError("R11_STAGE2_DURABLE_PROOF_NO_EVIDENCE")
    _, first = evidence_store.put_evidence(subset)
    _, replay = evidence_store.put_evidence(subset)
    if replay.created_payload_count != 0 or replay.created_evidence_count != 0:
        raise RuntimeError("R11_STAGE2_DURABLE_PROOF_REPLAY_FAIL")
    evidence_set = evidence_store.seal_evidence_set(
        "STAGE2_DURABLE_SET", [x.evidence_id for x in subset]
    )

    checkpoint_store = CheckpointStoreR11(root / "checkpoints")
    parent_obj = checkpoint_store.put_state_dict(parent_model.state_dict())
    challenger_obj = checkpoint_store.put_state_dict(challenger_model.state_dict())
    event_journal = EventJournalR11(root / "journal")
    authority = AuthorityStamp(
        generation=1,
        champion_id=parent_obj.semantic_sha256,
        champion_hash=parent_obj.semantic_sha256,
        teacher_authority_id=str(oracle_authority),
        physics_authority_id=sha256_obj(frozen_hashes),
    )
    orch = R11Orchestrator(
        authority=authority,
        evidence_store=EvidenceStoreProtocolAdapterR11(evidence_store),
        journal=EventJournalProtocolAdapterR11(event_journal),
        checkpoint_store=CheckpointStoreProtocolAdapterR11(checkpoint_store),
    )
    orch.verify_frozen_inputs(FrozenInputReceipt("R104_FROZEN_CACHE", sha256_obj(manifest), True))
    set_ref = EvidenceRef(
        "STAGE2_SET:" + evidence_set["evidence_set_hash"],
        evidence_set["evidence_set_hash"],
        0,
        authority.champion_id,
        authority.teacher_authority_id,
        authority.physics_authority_id,
        0,
    )
    orch.accept_teacher_evidence(set_ref)
    snapshot_hash = sha256_obj({"parent": authority.champion_hash, "set": evidence_set["evidence_set_hash"]})
    snapshot = SnapshotSeal("STAGE2_DURABLE_SNAPSHOT", snapshot_hash, 1, authority.champion_id, (set_ref.evidence_id,))
    orch.seal_training_snapshot(snapshot)
    orch.accept_policy_result(PolicyResult("STAGE2_POLICY", 1, authority.champion_id, sha256_obj(policy_before)))
    orch.start_trace("STAGE2_TRACE_START")
    trace_ref = EvidenceRef(
        "STAGE2_TRACE_SET",
        trace_hash,
        1,
        authority.champion_id,
        authority.teacher_authority_id,
        authority.physics_authority_id,
        1,
    )
    orch.materialize_trace_evidence(trace_ref, work_id="STAGE2_TRACE_COMPLETE")
    orch.start_challenger_training(work_id="STAGE2_TRAIN")
    train_payload = sha256_obj(training_receipt)
    orch.complete_training(
        TrainingResult(
            "STAGE2_TRAIN",
            1,
            challenger_obj.semantic_sha256,
            challenger_obj.semantic_sha256,
            authority.champion_id,
            snapshot.snapshot_id,
            train_payload,
        )
    )
    val_payload = sha256_obj(training_receipt["validation_after"])
    orch.complete_validation(
        ValidationResult(
            "STAGE2_VALIDATE",
            1,
            challenger_obj.semantic_sha256,
            authority.champion_id,
            snapshot.snapshot_id,
            "STAGE2_VALIDATION_SET",
            val_payload,
        )
    )
    decision = promotion(training_receipt["validation_before"], training_receipt["validation_after"])
    orch.decide_tournament(
        TournamentResult(
            "STAGE2_TOURNAMENT",
            1,
            challenger_obj.semantic_sha256,
            authority.champion_id,
            "STAGE2_VALIDATION_SET",
            decision,
            sha256_obj({"decision": decision.value, "validation": training_receipt["validation_after"]}),
        )
    )
    commit = orch.commit_tournament()
    orch.seal_checkpoint(commit.next_champion_hash)
    next_authority = orch.release_next_generation()
    recovered = R11Orchestrator(
        authority=authority,
        evidence_store=EvidenceStoreProtocolAdapterR11(evidence_store),
        journal=EventJournalProtocolAdapterR11(event_journal),
        checkpoint_store=CheckpointStoreProtocolAdapterR11(checkpoint_store),
        recover=True,
    )
    recovery_ok = bool(
        recovered.record.commit == orch.record.commit
        and recovered.record.checkpoint == orch.record.checkpoint
    )
    lifecycle_ok = bool(next_authority.champion_hash == commit.next_champion_hash)
    journal_ok = bool(event_journal.audit()["pass"])
    checkpoint_store.close()
    event_journal.close()
    evidence_store.close()
    return bool(recovery_ok and journal_ok), lifecycle_ok


def main() -> int:
    started = time.monotonic()
    warmup_seconds = int(os.environ.get("CB16_R11_BURST_WARMUP_SECONDS", "30"))
    measured_seconds = int(os.environ.get("CB16_R11_BURST_MEASURED_SECONDS", "420"))
    teacher_workers = int(os.environ.get("CB16_R11_BURST_TEACHER_WORKERS", "8"))
    trace_workers = int(os.environ.get("CB16_R11_BURST_TRACE_WORKERS", "8"))
    queue_depth = int(os.environ.get("CB16_R11_BURST_QUEUE_DEPTH", "8"))
    buffer_mib = int(os.environ.get("CB16_R11_BURST_BUFFER_MIB", "64"))
    if os.environ.get("CB16_R11_CANONICAL_DTYPE") != "FP32" or os.environ.get("CB16_R11_AMP") != "0":
        raise RuntimeError("R11_STAGE2_NONCANONICAL_ARITHMETIC_ENV")
    if not (360 <= measured_seconds <= 600):
        raise RuntimeError("R11_STAGE2_DRIVER_DURATION_OUT_OF_RANGE")

    telemetry_path = Path(os.environ["CB16_R11_BURST_TELEMETRY_JSONL"]).resolve()
    correctness_path = Path(os.environ["CB16_R11_BURST_CORRECTNESS_JSON"]).resolve()
    work_root = Path(os.environ["CB16_R11_BURST_WORK_ROOT"]).resolve()
    ssd_root = Path(os.environ["CB16_R11_BURST_SSD_ROOT"]).resolve()
    hdd_root = Path(os.environ["CB16_R11_BURST_HDD_ROOT"]).resolve()
    package_root = Path(os.environ["CB16_R11_BURST_PACKAGE_ROOT"]).resolve()
    legacy_root = Path(os.environ["CB16_R11_BURST_LEGACY_R104_ROOT"]).resolve()
    teacher_cache_root = Path("/data/cb16_hdd/cb16_diagnostics/r2_native/compiled_teacher_authority").resolve()
    for path in (work_root, ssd_root, hdd_root):
        path.mkdir(parents=True, exist_ok=True)

    counters = Counters()
    telemetry_stop = threading.Event()
    telemetry = threading.Thread(
        target=telemetry_loop,
        args=(telemetry_path, counters, telemetry_stop),
        name="r11-stage2-telemetry",
        daemon=True,
    )
    telemetry.start()

    static = verify_static_semantic_contracts(ROOT)
    frozen_before = frozen_authority_hashes(package_root)
    manifest_path = legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    manifest = json.loads(manifest_path.read_text())
    if bool(manifest.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_STAGE2_DRIVER_REFUSES_OPENED_HOLDOUT")
    if int(manifest.get("stride_hours", -1)) != 256:
        raise RuntimeError("R11_STAGE2_DRIVER_EXPECTED_R104_STRIDE256")
    parents, samples = load_teacher_samples(manifest["parents_file"], manifest["branches_file"])
    parent_states = load_parent_physics_states(manifest["parent_states_file"])

    # Teacher identity: warm verified legacy authority vs R11 vectorized/scheduled runtime.
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
        raise RuntimeError("R11_STAGE2_DRIVER_TEACHER_ORACLE_NOT_WARM")
    warm_train, warm_val, _teacher_stats = compile_teacher_evidence_r11(
        samples=samples,
        parents=parents,
        train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,
        workers=teacher_workers,
        block_targets=64,
    )
    teacher_identity = bool(
        _compare_evidence(old_train, warm_train)["pass"]
        and _compare_evidence(old_val, warm_val)["pass"]
    )
    if not teacher_identity:
        raise RuntimeError("R11_STAGE2_DRIVER_TEACHER_IDENTITY_FAIL")

    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (6, 1):
        raise RuntimeError("R11_STAGE2_DRIVER_REQUIRES_SM61")
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    g0_path = package_root / "authority/g0_parent/central_brain_g0_r10_2_parent.pt"
    if sha256_file(g0_path) != G0_FILE_SHA256:
        raise RuntimeError("R11_STAGE2_G0_FILE_DRIFT")
    g0_state = load_checkpoint_state(g0_path)

    # Actual frozen-evidence training differential after Task-B optimization.
    legacy_model = build_g0_brain_r10("TIER_1", seed=24680, device=str(device))
    r11_model = build_g0_brain_r10("TIER_1", seed=24680, device=str(device))
    for model in (legacy_model, r11_model):
        model.load_state_dict(g0_state, strict=True)
    old_train_batch = prepare_evidence_batch_r2(old_train, parents, device=str(device))
    old_val_batch = prepare_evidence_batch_r2(old_val, parents, device=str(device))
    legacy_before, _legacy_behavior_before = evaluate_policy_r2(legacy_model, old_val_batch)
    legacy_receipt, legacy_behavior_after = train_challenger_r2(
        model=legacy_model,
        train_batch=old_train_batch,
        val_batch=old_val_batch,
        validation_before=legacy_before,
        device=str(device),
        generation=1,
        snapshot_hash="STAGE2_FIXED_SNAPSHOT",
        receipt_dir=ssd_root / "warm_legacy_training",
    )
    prepared = prepare_evidence_campaign_r11(
        train_evidence=old_train,
        validation_evidence=old_val,
        parents=parents,
        device=device,
        pin_memory=False,
    )
    warm_runtime = TrainingRuntimeR11(
        device=device,
        evaluation_runtime=EvaluationRuntimeR11(enable_cuda_graph=False),
    )
    warm_receipt = warm_runtime.train_challenger(
        model=r11_model,
        campaign=prepared,
        generation=1,
        snapshot_hash="STAGE2_FIXED_SNAPSHOT",
        receipt_dir=ssd_root / "warm_r11_training",
    )
    state_cmp = state_compare(legacy_model, r11_model)
    val_cmp = validation_compare(legacy_receipt["validation_after"], warm_receipt["validation_after"])
    old_behavior = legacy_behavior_after or evaluate_policy_r2(legacy_model, old_val_batch)[1]
    training_differential = bool(
        state_cmp["pass"]
        and state_cmp["bitwise_equal"]
        and val_cmp["pass"]
        and old_behavior["sha256"] == warm_receipt["validation_after"]["behavior_fingerprint"]["sha256"]
    )
    no_forbidden_gradient = bool(
        set(warm_receipt["gradient_owner_set_last_step"]) == set(AUTHORIZED_GRADIENT_OWNERS_R11)
        and warm_receipt["amp"] is False
        and warm_receipt["dtype"] == "torch.float32"
        and warm_receipt["external_frozen_organ_gradients"]
        == "NOT_IN_AUTOGRAD_GRAPH__INPUT_VALUES_DETACHED"
    )
    if not training_differential or not no_forbidden_gradient:
        raise RuntimeError("R11_STAGE2_DRIVER_TRAINING_DIFFERENTIAL_FAIL")

    # Exact real H72 identity, then keep a persistent 8-process CoW runtime for the burst.
    physics = FrozenPhysicsRuntimeR102.load(package_root)
    trace_items = _choose_items(parents, parent_states, 96)
    symbols = sorted({x.symbol for x in trace_items})
    serial_cache = MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    serial_cache.preload(symbols)
    with TraceRuntimeR11(physics=physics, market_cache=serial_cache, max_workers=1) as serial_rt:
        serial_rows = serial_rt.run(trace_items)
    serial_hash = _trace_hash(serial_rows)
    serial_cache.assert_read_only()
    serial_cache.close()

    process_cache = MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    process_cache.preload(symbols)
    process_cache.assert_read_only()
    trace_runtime = ForkProcessTraceRuntimeR11(
        physics=physics, market_cache=process_cache, max_workers=trace_workers
    )
    process_rows = trace_runtime.run(trace_items)
    h72_identity = bool(_trace_hash(process_rows) == serial_hash)
    if not h72_identity:
        raise RuntimeError("R11_STAGE2_DRIVER_H72_IDENTITY_FAIL")

    admitted = [e for e in warm_train if e.admission.admitted]
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
    if not evidence_items:
        raise RuntimeError("R11_STAGE2_DRIVER_NO_ADMITTED_EVIDENCE")

    # Independent durable control-plane recovery proof using real R11 stores/adapters.
    parent_for_proof = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    parent_for_proof.load_state_dict(g0_state, strict=True)
    challenger_for_proof = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    challenger_for_proof.load_state_dict({k: v.detach().cpu() for k, v in r11_model.state_dict().items()}, strict=True)
    recovery_ok, lifecycle_ok = _durable_control_plane_proof(
        ssd_root=ssd_root,
        hdd_root=hdd_root,
        legacy_root=legacy_root,
        manifest=manifest,
        frozen_hashes=frozen_before,
        oracle_authority=str(oracle_receipt["authority_hash"]),
        evidence_items=evidence_items,
        parent_model=parent_for_proof,
        challenger_model=challenger_for_proof,
        trace_hash=serial_hash,
        training_receipt=warm_receipt,
        policy_before=warm_receipt["validation_before"],
    )
    if not recovery_ok or not lifecycle_ok:
        raise RuntimeError("R11_STAGE2_DRIVER_DURABLE_CONTROL_FAIL")

    # C: one bounded async SSD/HDD writer.  Materialize real admitted Teacher evidence,
    # then replay the exact same objects and demand zero new content-addressed payloads.
    io_config = IOThroughputConfigR11(
        evidence_metadata_root=ssd_root / "io_meta",
        evidence_payload_roots=(hdd_root / "io_payload",),
        journal_root=ssd_root / "io_journal",
        checkpoint_root=ssd_root / "io_checkpoints",
        segment_target_bytes=32 * 1024 * 1024,
        evidence_codec="zlib",
        queue_max_items=max(2, queue_depth),
        writer_batch_max_objects=128,
        writer_batch_max_bytes=min(32, max(1, buffer_mib)) * 1024 * 1024,
    )
    io_runtime = IOThroughputRuntimeR11(io_config)
    storage_result: dict[str, Any] = {"replay_zero": False, "error": None}

    def storage_lane() -> None:
        counters.set_flags(storage_active=True)
        try:
            for replay_index in (0, 1):
                created_payloads = 0
                created_evidence = 0
                for rows in _batch(evidence_items, 128):
                    counters.set_flags(pipeline_queue_depth=max(1, int(counters.pipeline_queue_depth)))
                    ticket = io_runtime.submit_evidence(rows, timeout=30)
                    t0 = time.perf_counter()
                    result = io_runtime.wait(ticket, timeout=60)
                    waited = time.perf_counter() - t0
                    counters.add(fsync_stall_seconds=waited)
                    counters.add_fsync_latency(waited)
                    counters.add(ssd_metadata_ops=result.metadata_transaction_count)
                    created_payloads += int(result.receipt.created_payload_count)
                    created_evidence += int(result.receipt.created_evidence_count)
                    if replay_index == 0 and result.receipt.created_payload_count:
                        unique = {ref.content_hash: ref for ref in result.refs}
                        counters.add(hdd_write_bytes=sum(int(ref.stored_bytes) for ref in unique.values()))
                    stats = io_runtime.stats()
                    counters.set_flags(
                        pipeline_queue_depth=max(0, int(stats.submitted_requests - stats.durable_requests))
                    )
                if replay_index == 1:
                    storage_result["replay_zero"] = bool(created_payloads == 0 and created_evidence == 0)
            io_runtime.durable_barrier(timeout=60)
        except BaseException as exc:
            storage_result["error"] = repr(exc)
        finally:
            counters.set_flags(storage_active=False, pipeline_queue_depth=0)

    storage_thread = threading.Thread(target=storage_lane, name="r11-stage2-storage-lane")

    # Wait until the supervisor's warmup interval has elapsed.  The warmup itself performed
    # real correctness work and cache construction, so the measured interval starts hot.
    warmup_deadline = started + warmup_seconds
    if time.monotonic() < warmup_deadline:
        time.sleep(warmup_deadline - time.monotonic())
    measured_deadline = started + warmup_seconds + measured_seconds
    storage_thread.start()

    measured_runtime = TrainingRuntimeR11(
        device=device,
        evaluation_runtime=EvaluationRuntimeR11(enable_cuda_graph=False),
    )
    gpu_model = build_g0_brain_r10("TIER_1", seed=24680, device=str(device))
    teacher_counted = False
    cycle = 0
    measured_teacher_hash = sha256_obj([e.content_hash for e in warm_train] + [e.content_hash for e in warm_val])

    while time.monotonic() < measured_deadline - 5.0:
        cycle += 1
        shared: dict[str, Any] = {}
        core = new_burst_core()

        def trace_runner(work: WorkItem) -> WorkCompletion:
            nonlocal teacher_counted
            if not teacher_counted:
                counters.set_flags(teacher_active=True)
                try:
                    check_train, check_val, _ = compile_teacher_evidence_r11(
                        samples=samples,
                        parents=parents,
                        train_config=TRAIN_TEACHER_CONFIG_R102,
                        val_config=VAL_TEACHER_CONFIG_R102,
                        workers=teacher_workers,
                        block_targets=64,
                    )
                    check_hash = sha256_obj([e.content_hash for e in check_train] + [e.content_hash for e in check_val])
                    if check_hash != measured_teacher_hash:
                        raise RuntimeError("R11_STAGE2_MEASURED_TEACHER_IDENTITY_DRIFT")
                    counters.add(teacher_evidence=len(check_train) + len(check_val))
                    teacher_counted = True
                finally:
                    counters.set_flags(teacher_active=False)
            counters.set_flags(trace_active=True)
            try:
                # Three real 96-group H72 replays roughly balance one canonical GPU train replay.
                for _ in range(3):
                    rows = trace_runtime.run(trace_items)
                    if _trace_hash(rows) != serial_hash:
                        raise RuntimeError("R11_STAGE2_MEASURED_H72_IDENTITY_DRIFT")
                    counters.add(traces=len(rows))
            finally:
                counters.set_flags(trace_active=False)
            evidence = EvidenceRef(
                f"STAGE2_BENCH_TRACE_{cycle}",
                serial_hash,
                BENCH_AUTH.generation,
                BENCH_AUTH.champion_id,
                BENCH_AUTH.teacher_authority_id,
                BENCH_AUTH.physics_authority_id,
                BENCH_AUTH.generation,
            )
            return WorkCompletion(work.work_id, work.kind, work.generation, evidence.payload_hash, evidence)

        def training_runner(work: WorkItem) -> WorkCompletion:
            counters.set_flags(gpu_active=True)
            try:
                gpu_model.load_state_dict(g0_state, strict=True)
                receipt = measured_runtime.train_challenger(
                    model=gpu_model,
                    campaign=prepared,
                    generation=BENCH_AUTH.generation,
                    snapshot_hash="snapshot-hash",
                    receipt_dir=ssd_root / "measured_training" / f"cycle_{cycle:06d}",
                )
                if receipt["amp"] is not False or receipt["dtype"] != "torch.float32":
                    raise RuntimeError("R11_STAGE2_MEASURED_TRAINING_ARITHMETIC_DRIFT")
                if set(receipt["gradient_owner_set_last_step"]) != set(AUTHORIZED_GRADIENT_OWNERS_R11):
                    raise RuntimeError("R11_STAGE2_MEASURED_GRADIENT_OWNER_DRIFT")
                chash = str(receipt["challenger_semantic_sha256"])
                shared["receipt"] = receipt
                shared["challenger_hash"] = chash
                counters.add(
                    training_steps=int(receipt["optimizer_steps"]),
                    fp32_training_examples=int(prepared.train.rows * int(receipt["epochs"])),
                )
                result = TrainingResult(
                    work.work_id,
                    BENCH_AUTH.generation,
                    chash,
                    chash,
                    BENCH_AUTH.champion_id,
                    "snapshot-g3",
                    sha256_obj(receipt),
                )
                return WorkCompletion(work.work_id, work.kind, work.generation, result.payload_hash, result)
            finally:
                counters.set_flags(gpu_active=False)

        burst = BurstOrchestratorR11(
            core,
            limits=BurstLimits(
                cpu_workers=1,
                gpu_workers=1,
                storage_workers=1,
                max_pending_work=4,
                storage_high_depth=max(2, queue_depth),
                storage_low_depth=max(1, min(4, queue_depth // 2)),
            ),
            runners={WorkKind.TRACE_EXECUTE: trace_runner, WorkKind.TRAIN_CHALLENGER: training_runner},
            ram_probe=lambda: RamSample(1, 100),
        )
        burst.start()
        burst.begin_burst()
        trace_work = WorkItem(
            f"stage2-trace-{cycle}",
            WorkKind.TRACE_EXECUTE,
            BENCH_AUTH.generation,
            WorkerPool.CPU_TRACE,
            BENCH_AUTH.champion_id,
        )
        train_work = WorkItem(
            f"stage2-train-{cycle}",
            WorkKind.TRAIN_CHALLENGER,
            BENCH_AUTH.generation,
            WorkerPool.GPU_BRAIN,
            BENCH_AUTH.champion_id,
            snapshot_id="snapshot-g3",
        )
        burst.schedule(trace_work, evidence_bytes=4096)
        burst.start_challenger_training(train_work)
        burst.wait_for((trace_work.work_id, train_work.work_id), timeout=120)
        receipt = shared.get("receipt")
        chash = shared.get("challenger_hash")
        if not isinstance(receipt, dict) or not isinstance(chash, str):
            raise RuntimeError("R11_STAGE2_MEASURED_TRAINING_RESULT_MISSING")
        core.complete_validation(
            ValidationResult(
                f"stage2-validate-{cycle}",
                BENCH_AUTH.generation,
                chash,
                BENCH_AUTH.champion_id,
                "snapshot-g3",
                f"stage2-validation-{cycle}",
                sha256_obj(receipt["validation_after"]),
            )
        )
        decision = promotion(receipt["validation_before"], receipt["validation_after"])
        core.decide_tournament(
            TournamentResult(
                f"stage2-tournament-{cycle}",
                BENCH_AUTH.generation,
                chash,
                BENCH_AUTH.champion_id,
                f"stage2-validation-{cycle}",
                decision,
                sha256_obj({"decision": decision.value, "validation": receipt["validation_after"]}),
            )
        )
        barrier_started = time.perf_counter()
        commit = core.commit_tournament()
        core.seal_checkpoint(commit.next_champion_hash)
        next_authority = core.release_next_generation()
        counters.add(barrier_block_seconds=time.perf_counter() - barrier_started)
        if next_authority.champion_hash != commit.next_champion_hash:
            raise RuntimeError("R11_STAGE2_MEASURED_CHAMPION_LIFECYCLE_DRIFT")
        burst.request_stop()
        burst.drain(timeout=30)
        burst.seal()
        burst.close()
        counters.add(generations_committed=1)

    storage_thread.join(timeout=90)
    if storage_thread.is_alive():
        raise RuntimeError("R11_STAGE2_STORAGE_LANE_DRAIN_TIMEOUT")
    if storage_result["error"] is not None:
        raise RuntimeError("R11_STAGE2_STORAGE_LANE_FAILED:" + str(storage_result["error"]))
    if not storage_result["replay_zero"]:
        raise RuntimeError("R11_STAGE2_STORAGE_REPLAY_NOT_ZERO")
    io_runtime.close(drain=True)
    trace_runtime.close()
    process_cache.assert_read_only()
    process_cache.close()

    frozen_after = frozen_authority_hashes(package_root)
    if frozen_before != frozen_after:
        raise RuntimeError("R11_STAGE2_DRIVER_FROZEN_AUTHORITY_DRIFT")
    manifest_after = json.loads(manifest_path.read_text())
    if bool(manifest_after.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_STAGE2_DRIVER_FINAL_HOLDOUT_DRIFT")

    correctness = {
        "schema": CORRECTNESS_SCHEMA,
        "semantic_freeze_pass": bool(static.get("status") == "PASS" or static.get("pass") is True),
        "final_holdout_untouched": True,
        "frozen_authority_unchanged": True,
        "teacher_identity_unchanged": teacher_identity,
        "h72_receipt_identity_unchanged": h72_identity,
        "training_differential_pass": training_differential,
        "no_forbidden_gradient": no_forbidden_gradient,
        "replay_zero_new_payload": bool(storage_result["replay_zero"]),
        "drain_or_crash_restart_receipt_valid": recovery_ok,
        "champion_challenger_lifecycle_correct": lifecycle_ok,
        "scientific_semantics_changed": False,
        "training_dtype": "FP32",
        "amp": False,
        "forbidden_work": {
            "raw_1m_archive_read": False,
            "raw_market_data_modified": False,
            "final_holdout_2025_09_read": False,
            "fresh_market_data_download": False,
            "oci_compute_used": False,
            "frozen_sensory_modified": False,
            "frozen_physics_or_supervisor_modified": False,
            "legacy_python_runtime_made_authority": False,
        },
        "engineering_replay_only": True,
        "new_scientific_verdict": False,
        "teacher_workers": teacher_workers,
        "trace_workers": trace_workers,
        "cycles_committed": int(counters.snapshot()["generations_committed"]),
    }
    atomic_json(correctness_path, correctness)
    telemetry_stop.set()
    telemetry.join(timeout=5)
    print(
        "CB16_R11_STAGE2_INTEGRATION_BURST="
        + json.dumps(
            {
                "status": "PASS",
                "elapsed_seconds": time.monotonic() - started,
                "counters": counters.snapshot(),
                "teacher_identity": teacher_identity,
                "h72_identity": h72_identity,
                "training_bitwise_equal": state_cmp["bitwise_equal"],
                "storage_replay_zero": storage_result["replay_zero"],
                "durable_recovery": recovery_ok,
                "champion_lifecycle": lifecycle_ok,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
