#!/usr/bin/env python3
from __future__ import annotations

"""Short real-machine R11 Stage-2 pipeline qualification.

This is deliberately NOT a performance or endurance test.  It executes one complete
real integration path across Teacher -> prepared snapshot -> FP32 training -> H72 trace
-> durable evidence/storage -> validation/tournament/checkpoint/recovery and records
only correctness/identity handoffs.  Resource-utilization thresholds are forbidden here.

The H72 process pool is created before this module imports the CUDA-bearing integration
module, preserving the CPU-only pre-CUDA fork island proved by Stage-2 Task A/Task F.
"""

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

SCHEMA = "CB16_R11_STAGE2_PIPELINE_QUALIFICATION_V1"


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> int:
    started = time.monotonic()
    required = (
        "CB16_R11_BURST_PACKAGE_ROOT",
        "CB16_R11_BURST_LEGACY_R104_ROOT",
        "CB16_R11_BURST_SSD_ROOT",
        "CB16_R11_BURST_HDD_ROOT",
        "CB16_R11_BURST_TRACE_WORKERS",
        "CB16_R11_BURST_TEACHER_WORKERS",
        "CB16_R11_PIPELINE_OUT",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("R11_STAGE2_PIPELINE_ENV_INCOMPLETE:" + ",".join(missing))
    if os.environ.get("CB16_R11_CANONICAL_DTYPE") != "FP32" or os.environ.get("CB16_R11_AMP") != "0":
        raise RuntimeError("R11_STAGE2_PIPELINE_NONCANONICAL_ARITHMETIC")

    # IMPORTANT: run before importing the integration driver (which imports torch/CUDA).
    import sitecustomize

    sitecustomize._install_stage2_prefork()

    import scripts.run_r11_stage2_integration_burst as b

    package_root = Path(os.environ["CB16_R11_BURST_PACKAGE_ROOT"]).resolve()
    legacy_root = Path(os.environ["CB16_R11_BURST_LEGACY_R104_ROOT"]).resolve()
    ssd_root = Path(os.environ["CB16_R11_BURST_SSD_ROOT"]).resolve()
    hdd_root = Path(os.environ["CB16_R11_BURST_HDD_ROOT"]).resolve()
    out_path = Path(os.environ["CB16_R11_PIPELINE_OUT"]).resolve()
    teacher_workers = int(os.environ["CB16_R11_BURST_TEACHER_WORKERS"])
    trace_workers = int(os.environ["CB16_R11_BURST_TRACE_WORKERS"])
    for p in (ssd_root, hdd_root, out_path.parent):
        p.mkdir(parents=True, exist_ok=True)

    handoffs: dict[str, bool] = {}
    identities: dict[str, Any] = {}

    static = b.verify_static_semantic_contracts(ROOT)
    frozen_before = b.frozen_authority_hashes(package_root)
    manifest_path = legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    manifest = json.loads(manifest_path.read_text())
    if bool(manifest.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_STAGE2_PIPELINE_REFUSES_OPENED_HOLDOUT")
    if int(manifest.get("stride_hours", -1)) != 256:
        raise RuntimeError("R11_STAGE2_PIPELINE_EXPECTED_R104_STRIDE256")
    parents, samples = b.load_teacher_samples(manifest["parents_file"], manifest["branches_file"])
    parent_states = b.load_parent_physics_states(manifest["parent_states_file"])

    # A -> B handoff: exact Teacher identity, then prepare the exact admitted evidence.
    teacher_cache_root = Path(
        "/data/cb16_hdd/cb16_diagnostics/r2_native/compiled_teacher_authority"
    ).resolve()
    old_train, old_val, oracle_receipt = b.compile_teacher_evidence_incremental(
        samples=samples,
        parents=parents,
        source_identity=manifest,
        cache_root=teacher_cache_root,
        train_config=b.TRAIN_TEACHER_CONFIG_R102,
        val_config=b.VAL_TEACHER_CONFIG_R102,
        workers=1,
        threads_per_worker=1,
        max_in_flight=1,
    )
    if oracle_receipt.get("mode") != "REUSED_VERIFIED_AUTHORITY":
        raise RuntimeError("R11_STAGE2_PIPELINE_TEACHER_ORACLE_NOT_WARM")
    warm_train, warm_val, _ = b.compile_teacher_evidence_r11(
        samples=samples,
        parents=parents,
        train_config=b.TRAIN_TEACHER_CONFIG_R102,
        val_config=b.VAL_TEACHER_CONFIG_R102,
        workers=teacher_workers,
        block_targets=64,
    )
    teacher_identity = bool(
        b._compare_evidence(old_train, warm_train)["pass"]
        and b._compare_evidence(old_val, warm_val)["pass"]
    )
    if not teacher_identity:
        raise RuntimeError("R11_STAGE2_PIPELINE_TEACHER_IDENTITY_FAIL")
    handoffs["teacher_to_training_input"] = True
    identities["teacher_authority_hash"] = str(oracle_receipt["authority_hash"])
    identities["teacher_evidence_count"] = len(warm_train) + len(warm_val)

    # B: exact real GTX1060 FP32 training differential and gradient ownership.
    if not b.torch.cuda.is_available() or tuple(b.torch.cuda.get_device_capability(0)) != (6, 1):
        raise RuntimeError("R11_STAGE2_PIPELINE_REQUIRES_SM61")
    device = b.torch.device("cuda:0")
    b.torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(b.torch.backends, "cudnn"):
        b.torch.backends.cudnn.allow_tf32 = False

    g0_path = package_root / "authority/g0_parent/central_brain_g0_r10_2_parent.pt"
    if b.sha256_file(g0_path) != b.G0_FILE_SHA256:
        raise RuntimeError("R11_STAGE2_PIPELINE_G0_FILE_DRIFT")
    g0_state = b.load_checkpoint_state(g0_path)

    legacy_model = b.build_g0_brain_r10("TIER_1", seed=24680, device=str(device))
    r11_model = b.build_g0_brain_r10("TIER_1", seed=24680, device=str(device))
    for model in (legacy_model, r11_model):
        model.load_state_dict(g0_state, strict=True)
    old_train_batch = b.prepare_evidence_batch_r2(old_train, parents, device=str(device))
    old_val_batch = b.prepare_evidence_batch_r2(old_val, parents, device=str(device))
    legacy_before, _ = b.evaluate_policy_r2(legacy_model, old_val_batch)
    legacy_receipt, legacy_behavior_after = b.train_challenger_r2(
        model=legacy_model,
        train_batch=old_train_batch,
        val_batch=old_val_batch,
        validation_before=legacy_before,
        device=str(device),
        generation=1,
        snapshot_hash="STAGE2_PIPELINE_SNAPSHOT",
        receipt_dir=ssd_root / "legacy_training",
    )
    prepared = b.prepare_evidence_campaign_r11(
        train_evidence=old_train,
        validation_evidence=old_val,
        parents=parents,
        device=device,
        pin_memory=False,
    )
    runtime = b.TrainingRuntimeR11(
        device=device,
        evaluation_runtime=b.EvaluationRuntimeR11(enable_cuda_graph=False),
    )
    r11_receipt = runtime.train_challenger(
        model=r11_model,
        campaign=prepared,
        generation=1,
        snapshot_hash="STAGE2_PIPELINE_SNAPSHOT",
        receipt_dir=ssd_root / "r11_training",
    )
    state_cmp = b.state_compare(legacy_model, r11_model)
    val_cmp = b.validation_compare(
        legacy_receipt["validation_after"], r11_receipt["validation_after"]
    )
    old_behavior = legacy_behavior_after or b.evaluate_policy_r2(legacy_model, old_val_batch)[1]
    training_identity = bool(
        state_cmp["pass"]
        and state_cmp["bitwise_equal"]
        and val_cmp["pass"]
        and old_behavior["sha256"]
        == r11_receipt["validation_after"]["behavior_fingerprint"]["sha256"]
    )
    gradient_identity = bool(
        set(r11_receipt["gradient_owner_set_last_step"])
        == set(b.AUTHORIZED_GRADIENT_OWNERS_R11)
        and r11_receipt["amp"] is False
        and r11_receipt["dtype"] == "torch.float32"
        and r11_receipt["external_frozen_organ_gradients"]
        == "NOT_IN_AUTOGRAD_GRAPH__INPUT_VALUES_DETACHED"
    )
    if not training_identity or not gradient_identity:
        raise RuntimeError("R11_STAGE2_PIPELINE_TRAINING_HANDOFF_FAIL")
    handoffs["training_input_to_fp32_challenger"] = True
    handoffs["training_to_validation"] = True
    identities["challenger_hash"] = str(r11_receipt["challenger_semantic_sha256"])
    identities["training_bitwise_equal"] = bool(state_cmp["bitwise_equal"])

    # A/C handoff: exact serial-vs-prefork H72 identity after CUDA is now active.
    # No new fork occurs here; sitecustomize bound the already-live CPU-only pool.
    physics = b.FrozenPhysicsRuntimeR102.load(package_root)
    trace_items = b._choose_items(parents, parent_states, 96)
    symbols = sorted({x.symbol for x in trace_items})
    serial_cache = b.MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    serial_cache.preload(symbols)
    with b.TraceRuntimeR11(physics=physics, market_cache=serial_cache, max_workers=1) as serial_rt:
        serial_rows = serial_rt.run(trace_items)
    serial_hash = b._trace_hash(serial_rows)
    serial_cache.assert_read_only()
    serial_cache.close()

    process_cache = b.MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    process_cache.preload(symbols)
    process_cache.assert_read_only()
    trace_runtime = b.ForkProcessTraceRuntimeR11(
        physics=physics, market_cache=process_cache, max_workers=trace_workers
    )
    process_rows = trace_runtime.run(trace_items)
    h72_identity = bool(b._trace_hash(process_rows) == serial_hash)
    if not h72_identity:
        raise RuntimeError("R11_STAGE2_PIPELINE_H72_HANDOFF_FAIL")
    handoffs["champion_policy_to_h72_trace"] = True
    handoffs["h72_trace_to_evidence_ref"] = True
    identities["trace_hash"] = serial_hash

    admitted = [e for e in warm_train if e.admission.admitted]
    evidence_items = [
        b.EvidenceItemR11(
            evidence_id=e.evidence_id,
            parent_snapshot_hash=parents[e.parent_id].snapshot_sha256,
            lineage_hash=e.content_hash,
            teacher_protocol_hash=e.teacher_protocol_hash,
            payload=asdict(e),
        )
        for e in admitted
    ]
    if not evidence_items:
        raise RuntimeError("R11_STAGE2_PIPELINE_NO_ADMITTED_EVIDENCE")

    # C -> E durable handoff: real stores, journal, checkpoints, restart/recovery.
    parent_for_proof = b.build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    parent_for_proof.load_state_dict(g0_state, strict=True)
    challenger_for_proof = b.build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    challenger_for_proof.load_state_dict(
        {k: v.detach().cpu() for k, v in r11_model.state_dict().items()}, strict=True
    )
    recovery_ok, lifecycle_ok = b._durable_control_plane_proof(
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
        training_receipt=r11_receipt,
        policy_before=r11_receipt["validation_before"],
    )
    if not recovery_ok or not lifecycle_ok:
        raise RuntimeError("R11_STAGE2_PIPELINE_DURABLE_CONTROL_FAIL")
    handoffs["evidence_to_durable_store"] = True
    handoffs["tournament_to_checkpoint"] = True
    handoffs["checkpoint_to_restart_recovery"] = True

    # C writer contract: bounded batched write + exact replay-zero on a real subset.
    io_config = b.IOThroughputConfigR11(
        evidence_metadata_root=ssd_root / "pipeline_io_meta",
        evidence_payload_roots=(hdd_root / "pipeline_io_payload",),
        journal_root=ssd_root / "pipeline_io_journal",
        checkpoint_root=ssd_root / "pipeline_io_checkpoints",
        segment_target_bytes=8 * 1024 * 1024,
        evidence_codec="zlib",
        queue_max_items=4,
        writer_batch_max_objects=128,
        writer_batch_max_bytes=8 * 1024 * 1024,
    )
    io_runtime = b.IOThroughputRuntimeR11(io_config)
    io_subset = evidence_items[: min(256, len(evidence_items))]
    for rows in b._batch(io_subset, 128):
        ticket = io_runtime.submit_evidence(rows, timeout=30)
        io_runtime.wait(ticket, timeout=60)
    io_runtime.durable_barrier(timeout=60)
    replay_created_payloads = 0
    replay_created_evidence = 0
    for rows in b._batch(io_subset, 128):
        ticket = io_runtime.submit_evidence(rows, timeout=30)
        result = io_runtime.wait(ticket, timeout=60)
        replay_created_payloads += int(result.receipt.created_payload_count)
        replay_created_evidence += int(result.receipt.created_evidence_count)
    io_runtime.durable_barrier(timeout=60)
    replay_zero = bool(replay_created_payloads == 0 and replay_created_evidence == 0)
    if not replay_zero:
        raise RuntimeError("R11_STAGE2_PIPELINE_STORAGE_REPLAY_NOT_ZERO")
    handoffs["durable_store_to_replay"] = True

    # D/E: one complete concurrent TRACE + TRAIN orchestration cycle and tournament barrier.
    core = b.new_burst_core()
    shared: dict[str, Any] = {}
    gpu_model = b.build_g0_brain_r10("TIER_1", seed=24680, device=str(device))

    def trace_runner(work: b.WorkItem) -> b.WorkCompletion:
        rows = trace_runtime.run(trace_items)
        if b._trace_hash(rows) != serial_hash:
            raise RuntimeError("R11_STAGE2_PIPELINE_CONCURRENT_TRACE_DRIFT")
        evidence = b.EvidenceRef(
            "STAGE2_PIPELINE_TRACE",
            serial_hash,
            b.BENCH_AUTH.generation,
            b.BENCH_AUTH.champion_id,
            b.BENCH_AUTH.teacher_authority_id,
            b.BENCH_AUTH.physics_authority_id,
            b.BENCH_AUTH.generation,
        )
        return b.WorkCompletion(
            work.work_id, work.kind, work.generation, evidence.payload_hash, evidence
        )

    def training_runner(work: b.WorkItem) -> b.WorkCompletion:
        gpu_model.load_state_dict(g0_state, strict=True)
        receipt = runtime.train_challenger(
            model=gpu_model,
            campaign=prepared,
            generation=b.BENCH_AUTH.generation,
            snapshot_hash="snapshot-hash",
            receipt_dir=ssd_root / "pipeline_concurrent_training",
        )
        if receipt["amp"] is not False or receipt["dtype"] != "torch.float32":
            raise RuntimeError("R11_STAGE2_PIPELINE_CONCURRENT_TRAINING_ARITHMETIC_DRIFT")
        if set(receipt["gradient_owner_set_last_step"]) != set(b.AUTHORIZED_GRADIENT_OWNERS_R11):
            raise RuntimeError("R11_STAGE2_PIPELINE_CONCURRENT_GRADIENT_OWNER_DRIFT")
        chash = str(receipt["challenger_semantic_sha256"])
        shared["receipt"] = receipt
        shared["challenger_hash"] = chash
        result = b.TrainingResult(
            work.work_id,
            b.BENCH_AUTH.generation,
            chash,
            chash,
            b.BENCH_AUTH.champion_id,
            "snapshot-g3",
            b.sha256_obj(receipt),
        )
        return b.WorkCompletion(work.work_id, work.kind, work.generation, result.payload_hash, result)

    burst = b.BurstOrchestratorR11(
        core,
        limits=b.BurstLimits(
            cpu_workers=1,
            gpu_workers=1,
            storage_workers=1,
            max_pending_work=4,
            storage_high_depth=4,
            storage_low_depth=2,
        ),
        runners={b.WorkKind.TRACE_EXECUTE: trace_runner, b.WorkKind.TRAIN_CHALLENGER: training_runner},
        ram_probe=lambda: b.RamSample(1, 100),
    )
    burst.start()
    burst.begin_burst()
    trace_work = b.WorkItem(
        "stage2-pipeline-trace",
        b.WorkKind.TRACE_EXECUTE,
        b.BENCH_AUTH.generation,
        b.WorkerPool.CPU_TRACE,
        b.BENCH_AUTH.champion_id,
    )
    train_work = b.WorkItem(
        "stage2-pipeline-train",
        b.WorkKind.TRAIN_CHALLENGER,
        b.BENCH_AUTH.generation,
        b.WorkerPool.GPU_BRAIN,
        b.BENCH_AUTH.champion_id,
        snapshot_id="snapshot-g3",
    )
    burst.schedule(trace_work, evidence_bytes=4096)
    burst.start_challenger_training(train_work)
    burst.wait_for((trace_work.work_id, train_work.work_id), timeout=120)
    receipt = shared.get("receipt")
    chash = shared.get("challenger_hash")
    if not isinstance(receipt, dict) or not isinstance(chash, str):
        raise RuntimeError("R11_STAGE2_PIPELINE_CONCURRENT_RESULT_MISSING")
    core.complete_validation(
        b.ValidationResult(
            "stage2-pipeline-validate",
            b.BENCH_AUTH.generation,
            chash,
            b.BENCH_AUTH.champion_id,
            "snapshot-g3",
            "stage2-pipeline-validation",
            b.sha256_obj(receipt["validation_after"]),
        )
    )
    decision = b.promotion(receipt["validation_before"], receipt["validation_after"])
    core.decide_tournament(
        b.TournamentResult(
            "stage2-pipeline-tournament",
            b.BENCH_AUTH.generation,
            chash,
            b.BENCH_AUTH.champion_id,
            "stage2-pipeline-validation",
            decision,
            b.sha256_obj({"decision": decision.value, "validation": receipt["validation_after"]}),
        )
    )
    commit = core.commit_tournament()
    core.seal_checkpoint(commit.next_champion_hash)
    next_authority = core.release_next_generation()
    if next_authority.champion_hash != commit.next_champion_hash:
        raise RuntimeError("R11_STAGE2_PIPELINE_CHAMPION_LIFECYCLE_DRIFT")
    burst.request_stop()
    burst.drain(timeout=30)
    burst.seal()
    burst.close()
    handoffs["trace_and_training_to_tournament_barrier"] = True
    handoffs["tournament_to_next_generation"] = True

    io_runtime.close(drain=True)
    trace_runtime.close()
    process_cache.assert_read_only()
    process_cache.close()

    frozen_after = b.frozen_authority_hashes(package_root)
    manifest_after = json.loads(manifest_path.read_text())
    final_holdout_untouched = bool(
        not manifest.get("final_holdout_2025_09_accessed", False)
        and not manifest_after.get("final_holdout_2025_09_accessed", False)
    )
    frozen_unchanged = frozen_before == frozen_after
    if not final_holdout_untouched or not frozen_unchanged:
        raise RuntimeError("R11_STAGE2_PIPELINE_FROZEN_BOUNDARY_DRIFT")

    required_handoffs = (
        "teacher_to_training_input",
        "training_input_to_fp32_challenger",
        "training_to_validation",
        "champion_policy_to_h72_trace",
        "h72_trace_to_evidence_ref",
        "evidence_to_durable_store",
        "durable_store_to_replay",
        "trace_and_training_to_tournament_barrier",
        "tournament_to_checkpoint",
        "checkpoint_to_restart_recovery",
        "tournament_to_next_generation",
    )
    missing_handoffs = [name for name in required_handoffs if handoffs.get(name) is not True]
    if missing_handoffs:
        raise RuntimeError("R11_STAGE2_PIPELINE_HANDOFF_INCOMPLETE:" + ",".join(missing_handoffs))

    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "purpose": "PIPELINE_CORRECTNESS_ONLY__NOT_PERFORMANCE_QUALIFICATION",
        "elapsed_seconds": time.monotonic() - started,
        "hardware": {
            "gpu": b.torch.cuda.get_device_name(0),
            "capability": list(b.torch.cuda.get_device_capability(0)),
            "training_dtype": "FP32",
            "amp": False,
        },
        "semantic": {
            "semantic_freeze_pass": bool(static.get("status") == "PASS" or static.get("pass") is True),
            "scientific_semantics_changed": False,
            "final_holdout_untouched": final_holdout_untouched,
            "frozen_authority_unchanged": frozen_unchanged,
            "teacher_identity_unchanged": teacher_identity,
            "training_differential_pass": training_identity,
            "gradient_ownership_pass": gradient_identity,
            "h72_receipt_identity_unchanged": h72_identity,
            "storage_replay_zero": replay_zero,
            "durable_recovery_pass": recovery_ok,
            "champion_challenger_lifecycle_correct": lifecycle_ok,
            "new_scientific_verdict": False,
        },
        "handoffs": handoffs,
        "identities": identities,
        "pipeline_cycles": 1,
        "performance_thresholds_evaluated": False,
        "burst_or_endurance_test": False,
    }
    atomic_json(out_path, report)
    print("CB16_R11_STAGE2_PIPELINE_QUALIFICATION=PASS " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
