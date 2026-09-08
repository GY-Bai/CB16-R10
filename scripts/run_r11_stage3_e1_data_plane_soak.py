#!/usr/bin/env python3
from __future__ import annotations

"""CB16 R11 Stage-3 E1 persistent data-plane frozen-replay soak."""

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.stage3_e1_endurance_r11 import (
    CounterSnapshot, E1Config, E1Failure, LinuxResourceSampler,
    PersistentTeacherReplayR11, SCHEMA, atomic_json, make_report,
)


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"R11_E1_ENV_MISSING:{name}")
    return value


def _trace_processes_alive(rt: Any) -> int | None:
    executor = getattr(rt, "_executor", None)
    if executor is None:
        return None
    processes = getattr(executor, "_processes", None)
    if not isinstance(processes, dict):
        return None
    return sum(1 for p in processes.values() if p.is_alive())


def _queue(rt: Any) -> tuple[int, int]:
    q = getattr(rt, "_queue", None)
    return int(getattr(q, "items", 0)), int(getattr(q, "max_items", 1))


def _atomic_failure(out: Path, exc: BaseException) -> None:
    atomic_json(out, {
        "schema": SCHEMA,
        "status": "FAIL",
        "correctness_identity": {"verdict": "FAIL", "failures": [], "checks": {}},
        "runtime_endurance": {"verdict": "FAIL", "failures": [], "sample_count": 0},
        "fatal_error": {"type": type(exc).__name__, "message": str(exc)},
        "failure_taxonomy": [x.value for x in E1Failure],
        "scientific_semantics_changed": False,
        "new_scientific_verdict": False,
    })


def run() -> int:
    package_root = Path(_required("CB16_R11_E1_PACKAGE_ROOT")).resolve()
    legacy_root = Path(_required("CB16_R11_E1_LEGACY_R104_ROOT")).resolve()
    ssd_root = Path(_required("CB16_R11_E1_SSD_ROOT")).resolve()
    hdd_root = Path(_required("CB16_R11_E1_HDD_ROOT")).resolve()
    out_path = Path(_required("CB16_R11_E1_OUT")).resolve()
    config = E1Config(
        duration_seconds=int(_required("CB16_R11_E1_DURATION_SECONDS")),
        sample_seconds=float(os.environ.get("CB16_R11_E1_SAMPLE_SECONDS", "5")),
        teacher_workers=int(os.environ.get("CB16_R11_E1_TEACHER_WORKERS", "8")),
        trace_workers=int(os.environ.get("CB16_R11_E1_TRACE_WORKERS", "8")),
        queue_depth=int(os.environ.get("CB16_R11_E1_QUEUE_DEPTH", "8")),
        buffer_mib=int(os.environ.get("CB16_R11_E1_BUFFER_MIB", "64")),
    )
    config.validate()
    if os.environ.get("CB16_R11_CANONICAL_DTYPE") != "FP32" or os.environ.get("CB16_R11_AMP") != "0":
        raise RuntimeError("R11_E1_NONCANONICAL_ARITHMETIC")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(k) != "1":
            raise RuntimeError(f"R11_E1_NESTED_THREAD_POLICY_DRIFT:{k}")
    for p in (ssd_root, hdd_root, out_path.parent):
        p.mkdir(parents=True, exist_ok=True)

    # Force the persistent H72 fork pool live before CUDA-bearing imports.
    from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
    from cb16_local_opt.r102_evidence_cache import load_parent_physics_states, load_teacher_samples
    from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
    from cb16_local_opt.trace_process_runtime_r11 import ForkProcessTraceRuntimeR11
    from scripts.run_r11_stage2_task_a_trace_process_benchmark import _choose_items, _trace_hash

    manifest_path = legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    manifest_before = json.loads(manifest_path.read_text())
    if bool(manifest_before.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_E1_REFUSES_OPENED_FINAL_HOLDOUT")
    if int(manifest_before.get("stride_hours", -1)) != 256:
        raise RuntimeError("R11_E1_EXPECTED_R104_STRIDE256")
    parents, samples = load_teacher_samples(manifest_before["parents_file"], manifest_before["branches_file"])
    parent_states = load_parent_physics_states(manifest_before["parent_states_file"])
    physics = FrozenPhysicsRuntimeR102.load(package_root)
    trace_items = _choose_items(parents, parent_states, 96)
    symbols = sorted({x.symbol for x in trace_items})
    process_cache = MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    process_cache.preload(symbols)
    process_cache.assert_read_only()
    trace_runtime = ForkProcessTraceRuntimeR11(
        physics=physics, market_cache=process_cache, max_workers=config.trace_workers
    )
    first_trace = trace_runtime.run(trace_items)
    trace_identity = _trace_hash(first_trace)
    trace_executor_identity = trace_runtime.executor_identity
    if trace_executor_identity is None:
        raise RuntimeError("R11_E1_TRACE_POOL_NOT_LIVE")
    if _trace_processes_alive(trace_runtime) != config.trace_workers:
        raise RuntimeError("R11_E1_TRACE_PROCESS_COUNT_DRIFT_AT_WARMUP")

    import scripts.run_r11_stage2_integration_burst as b

    static = b.verify_static_semantic_contracts(ROOT)
    frozen_before = b.frozen_authority_hashes(package_root)
    teacher_cache_root = Path(
        "/data/cb16_hdd/cb16_diagnostics/r2_native/compiled_teacher_authority"
    ).resolve()
    oracle_train, oracle_val, oracle_receipt = b.compile_teacher_evidence_incremental(
        samples=samples,
        parents=parents,
        source_identity=manifest_before,
        cache_root=teacher_cache_root,
        train_config=b.TRAIN_TEACHER_CONFIG_R102,
        val_config=b.VAL_TEACHER_CONFIG_R102,
        workers=1,
        threads_per_worker=1,
        max_in_flight=1,
    )
    if oracle_receipt.get("mode") != "REUSED_VERIFIED_AUTHORITY":
        raise RuntimeError("R11_E1_TEACHER_ORACLE_NOT_WARM")
    teacher_runtime = PersistentTeacherReplayR11(
        samples=samples,
        parents=parents,
        train_config=b.TRAIN_TEACHER_CONFIG_R102,
        val_config=b.VAL_TEACHER_CONFIG_R102,
        workers=config.teacher_workers,
        block_targets=64,
    )
    teacher_pool_identity = teacher_runtime.pool_identity
    warm_train, warm_val = teacher_runtime.compile()
    teacher_ok = bool(
        b._compare_evidence(oracle_train, warm_train)["pass"]
        and b._compare_evidence(oracle_val, warm_val)["pass"]
    )
    if not teacher_ok:
        raise RuntimeError("R11_E1_TEACHER_WARMUP_IDENTITY_FAIL")

    serial_cache = MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    serial_cache.preload(symbols)
    with b.TraceRuntimeR11(physics=physics, market_cache=serial_cache, max_workers=1) as serial_rt:
        serial_rows = serial_rt.run(trace_items)
    serial_cache.assert_read_only()
    serial_cache.close()
    if _trace_hash(serial_rows) != trace_identity:
        raise RuntimeError("R11_E1_H72_SERIAL_PROCESS_IDENTITY_FAIL")

    if not b.torch.cuda.is_available() or tuple(b.torch.cuda.get_device_capability(0)) != (6, 1):
        raise RuntimeError("R11_E1_REQUIRES_SM61")
    device = b.torch.device("cuda:0")
    b.torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(b.torch.backends, "cudnn"):
        b.torch.backends.cudnn.allow_tf32 = False
    if b.torch.get_default_dtype() != b.torch.float32:
        raise RuntimeError("R11_E1_DEFAULT_DTYPE_DRIFT")
    g0_path = package_root / "authority/g0_parent/central_brain_g0_r10_2_parent.pt"
    if b.sha256_file(g0_path) != b.G0_FILE_SHA256:
        raise RuntimeError("R11_E1_G0_FILE_DRIFT")
    g0_state = b.load_checkpoint_state(g0_path)
    campaign = b.prepare_evidence_campaign_r11(
        train_evidence=oracle_train,
        validation_evidence=oracle_val,
        parents=parents,
        device=device,
        pin_memory=False,
    )
    training_runtime = b.TrainingRuntimeR11(
        device=device,
        evaluation_runtime=b.EvaluationRuntimeR11(enable_cuda_graph=False),
    )
    training_runtime_identity = id(training_runtime)

    admitted = [e for e in oracle_train if e.admission.admitted]
    evidence_items = [
        b.EvidenceItemR11(
            evidence_id=e.evidence_id,
            parent_snapshot_hash=parents[e.parent_id].snapshot_sha256,
            lineage_hash=e.content_hash,
            teacher_protocol_hash=e.teacher_protocol_hash,
            payload=asdict(e),
        )
        for e in admitted[: min(256, len(admitted))]
    ]
    if not evidence_items:
        raise RuntimeError("R11_E1_NO_ADMITTED_EVIDENCE")
    io_runtime = b.IOThroughputRuntimeR11(
        b.IOThroughputConfigR11(
            evidence_metadata_root=ssd_root / "e1_io_meta",
            evidence_payload_roots=(hdd_root / "e1_io_payload",),
            journal_root=ssd_root / "e1_io_journal",
            checkpoint_root=ssd_root / "e1_io_checkpoints",
            segment_target_bytes=8 * 1024 * 1024,
            evidence_codec="zlib",
            queue_max_items=config.queue_depth,
            writer_batch_max_objects=128,
            writer_batch_max_bytes=config.buffer_mib * 1024 * 1024,
        )
    )
    io_runtime_identity = id(io_runtime)
    io_logical_bytes_per_replay = sum(
        len(b.ImmutableEvidenceObjectR11.from_item(x).payload_bytes) for x in evidence_items
    )
    for rows in b._batch(evidence_items, 128):
        ticket = io_runtime.submit_evidence(rows, timeout=30)
        io_runtime.wait(ticket, timeout=60)
    io_runtime.durable_barrier(timeout=60)

    # Warm the single persistent TrainingRuntime using a disposable non-authority model.
    warm_model = b.build_g0_brain_r10("TIER_1", seed=24680, device=str(device))
    warm_model.load_state_dict(g0_state, strict=True)
    warm_opt = training_runtime.build_optimizer(warm_model)
    n = min(training_runtime.config.batch_size, campaign.train.rows)
    ids = b.torch.arange(n, dtype=b.torch.long, device=device)
    warm_step = training_runtime.train_one_step(
        model=warm_model, optimizer=warm_opt, prepared=campaign.train, ids=ids
    )
    if set(warm_step.gradient_owner_set) != set(b.AUTHORIZED_GRADIENT_OWNERS_R11):
        raise RuntimeError("R11_E1_WARMUP_GRADIENT_OWNER_DRIFT")
    del warm_model, warm_opt
    b.torch.cuda.synchronize(device)

    started = time.monotonic()
    sampler = LinuxResourceSampler(ssd_path=ssd_root, hdd_path=hdd_root)
    samples_out = []
    counters = CounterSnapshot()
    next_sample = started
    last_fsync_ms = None
    last_barrier_ms = None
    replay_created_payloads = 0
    replay_created_evidence = 0
    gradient_ok = True
    teacher_all_ok = True
    trace_all_ok = True
    worker_deaths = 0
    restart_count = 0
    cycles = 0

    def snap(force: bool = False) -> None:
        nonlocal next_sample, worker_deaths, counters
        now = time.monotonic()
        if not force and now < next_sample:
            return
        alive = _trace_processes_alive(trace_runtime)
        storage_alive = bool(getattr(io_runtime, "_thread").is_alive())
        if alive is not None and alive < config.trace_workers:
            worker_deaths += config.trace_workers - alive
        if not storage_alive:
            worker_deaths += 1
        counters = CounterSnapshot(
            cycles=counters.cycles,
            traces=counters.traces,
            optimizer_steps=counters.optimizer_steps,
            examples=counters.examples,
            io_requests=counters.io_requests,
            io_bytes_logical=counters.io_bytes_logical,
            worker_deaths=worker_deaths,
            restart_count=restart_count,
        )
        qd, qc = _queue(io_runtime)
        samples_out.append(sampler.sample(
            started=started,
            counters=counters,
            queue_depth=qd,
            queue_capacity=qc,
            fsync_latency_ms=last_fsync_ms,
            barrier_latency_ms=last_barrier_ms,
            trace_processes_alive=alive,
            storage_writer_alive=storage_alive,
        ))
        next_sample = now + config.sample_seconds

    snap(force=True)
    deadline = started + config.duration_seconds
    while time.monotonic() < deadline:
        train_rows, val_rows = teacher_runtime.compile()
        ok = bool(
            b._compare_evidence(oracle_train, train_rows)["pass"]
            and b._compare_evidence(oracle_val, val_rows)["pass"]
        )
        teacher_all_ok = teacher_all_ok and ok
        if not ok:
            raise RuntimeError("R11_E1_TEACHER_IDENTITY_DRIFT")
        snap()

        trace_rows = trace_runtime.run(trace_items)
        trace_ok = _trace_hash(trace_rows) == trace_identity
        trace_all_ok = trace_all_ok and trace_ok
        if not trace_ok:
            raise RuntimeError("R11_E1_H72_IDENTITY_DRIFT")
        counters = CounterSnapshot(**{**asdict(counters), "traces": counters.traces + len(trace_rows)})
        snap()

        model = b.build_g0_brain_r10("TIER_1", seed=24680, device=str(device))
        model.load_state_dict(g0_state, strict=True)
        optimizer = training_runtime.build_optimizer(model)
        for _ in range(4):
            step = training_runtime.train_one_step(
                model=model, optimizer=optimizer, prepared=campaign.train, ids=ids
            )
            if set(step.gradient_owner_set) != set(b.AUTHORIZED_GRADIENT_OWNERS_R11):
                gradient_ok = False
                raise RuntimeError("R11_E1_GRADIENT_OWNER_DRIFT")
            counters = CounterSnapshot(**{
                **asdict(counters),
                "optimizer_steps": counters.optimizer_steps + 1,
                "examples": counters.examples + int(n),
            })
            snap()
        del optimizer, model

        for rows in b._batch(evidence_items, 128):
            t0 = time.monotonic()
            ticket = io_runtime.submit_evidence(rows, timeout=30)
            result = io_runtime.wait(ticket, timeout=60)
            last_fsync_ms = (time.monotonic() - t0) * 1000.0
            replay_created_payloads += int(result.receipt.created_payload_count)
            replay_created_evidence += int(result.receipt.created_evidence_count)
            logical = sum(len(b.ImmutableEvidenceObjectR11.from_item(x).payload_bytes) for x in rows)
            counters = CounterSnapshot(**{
                **asdict(counters),
                "io_requests": counters.io_requests + 1,
                "io_bytes_logical": counters.io_bytes_logical + logical,
            })
            if result.receipt.created_payload_count or result.receipt.created_evidence_count:
                raise RuntimeError("R11_E1_REPLAY_CREATED_NEW_EVIDENCE")
            snap()
        t0 = time.monotonic()
        io_runtime.durable_barrier(timeout=60)
        last_barrier_ms = (time.monotonic() - t0) * 1000.0
        cycles += 1
        counters = CounterSnapshot(**{**asdict(counters), "cycles": cycles})
        snap(force=True)

    snap(force=True)
    teacher_runtime.close()
    io_runtime.close(drain=True)
    trace_runtime.close()
    process_cache.assert_read_only()
    process_cache.close()

    frozen_after = b.frozen_authority_hashes(package_root)
    manifest_after = json.loads(manifest_path.read_text())
    final_holdout_untouched = bool(
        not manifest_before.get("final_holdout_2025_09_accessed", False)
        and not manifest_after.get("final_holdout_2025_09_accessed", False)
    )
    frozen_unchanged = frozen_before == frozen_after
    static_pass = bool(static.get("status") == "PASS" or static.get("pass") is True)
    precision_ok = bool(
        os.environ["CB16_R11_CANONICAL_DTYPE"] == "FP32"
        and os.environ["CB16_R11_AMP"] == "0"
        and b.torch.get_default_dtype() == b.torch.float32
        and b.torch.backends.cuda.matmul.allow_tf32 is False
    )
    replay_zero = replay_created_payloads == 0 and replay_created_evidence == 0
    correctness = {
        "semantic_freeze_pass": static_pass,
        "teacher_identity_unchanged": teacher_all_ok,
        "h72_identity_unchanged": trace_all_ok,
        "fp32_amp_false": precision_ok,
        "gradient_authority_unchanged": gradient_ok,
        "frozen_authority_unchanged": frozen_unchanged,
        "final_holdout_untouched": final_holdout_untouched,
        "replay_created_zero": replay_zero,
    }
    identities = {
        "stage2_qualified_base": "92f1ca011aec27bbe9e89ac5ba9afe74817ee2de",
        "teacher_authority_hash": str(oracle_receipt["authority_hash"]),
        "teacher_pool_identity": teacher_pool_identity,
        "trace_hash": trace_identity,
        "trace_executor_identity": trace_executor_identity,
        "training_runtime_identity": training_runtime_identity,
        "io_runtime_identity": io_runtime_identity,
        "g0_file_sha256": b.G0_FILE_SHA256,
        "frozen_authority_hashes": frozen_before,
    }
    report = make_report(
        config=config,
        samples=samples_out,
        correctness_checks=correctness,
        identities=identities,
        extra={
            "measured_seconds": time.monotonic() - started,
            "engineering_cycles": cycles,
            "teacher_pool_executions": teacher_runtime.pool.executions,
            "trace_pool_executions": trace_runtime.executions,
            "replay_created_payloads": replay_created_payloads,
            "replay_created_evidence": replay_created_evidence,
            "training_role": "DISPOSABLE_ENGINEERING_MODEL_ONLY__NO_CHECKPOINT__NO_TOURNAMENT",
            "io_logical_bytes_per_replay": io_logical_bytes_per_replay,
        },
    )
    atomic_json(out_path, report)
    print("CB16_R11_STAGE3_E1=" + report["status"] + " " + json.dumps({
        "correctness": report["correctness_identity"]["verdict"],
        "endurance": report["runtime_endurance"]["verdict"],
        "cycles": cycles,
        "ready_for_30m": report["readiness"]["ready_for_30m"],
    }, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


def main() -> int:
    out = Path(os.environ.get(
        "CB16_R11_E1_OUT", "ci_evidence/CB16_R11_STAGE3_E1_ENDURANCE_REPORT_V1.json"
    ))
    try:
        return run()
    except BaseException as exc:
        _atomic_failure(out.resolve(), exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
