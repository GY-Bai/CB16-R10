#!/usr/bin/env python3
from __future__ import annotations

"""Short real-machine calibration for Stage-2 CPU/GPU overlap.

This is engineering-only and runs after the independent pipeline qualification has
passed. It measures the real frozen H72 lane and canonical GTX1060 FP32 training lane
under the same process, first in isolation and then concurrently. It does not create a
scientific generation, promotion verdict, or new evidence authority.
"""

import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "CB16_R11_STAGE2_OVERLAP_CALIBRATION_V1"


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def main() -> int:
    required = (
        "CB16_R11_BURST_PACKAGE_ROOT",
        "CB16_R11_BURST_LEGACY_R104_ROOT",
        "CB16_R11_BURST_SSD_ROOT",
        "CB16_R11_BURST_HDD_ROOT",
        "CB16_R11_BURST_TRACE_WORKERS",
        "CB16_R11_CALIBRATION_OUT",
    )
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError("R11_STAGE2_CALIBRATION_ENV_INCOMPLETE:" + ",".join(missing))
    if os.environ.get("CB16_R11_CANONICAL_DTYPE") != "FP32" or os.environ.get("CB16_R11_AMP") != "0":
        raise RuntimeError("R11_STAGE2_CALIBRATION_NONCANONICAL_ARITHMETIC")

    workers = int(os.environ["CB16_R11_BURST_TRACE_WORKERS"])
    if workers not in {4, 6, 8}:
        raise RuntimeError(f"R11_STAGE2_CALIBRATION_WORKERS_NOT_ALLOWED:{workers}")

    # Fork the CPU-only H72 island before importing the CUDA-bearing integration module.
    import sitecustomize
    sitecustomize._install_stage2_prefork()

    import torch
    import scripts.run_r11_stage2_integration_burst as b
    from cb16_local_opt.gpu_runtime_r11 import sample_nvidia_smi_r11

    package_root = Path(os.environ["CB16_R11_BURST_PACKAGE_ROOT"]).resolve()
    legacy_root = Path(os.environ["CB16_R11_BURST_LEGACY_R104_ROOT"]).resolve()
    ssd_root = Path(os.environ["CB16_R11_BURST_SSD_ROOT"]).resolve()
    out_path = Path(os.environ["CB16_R11_CALIBRATION_OUT"]).resolve()
    ssd_root.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    static = b.verify_static_semantic_contracts(ROOT)
    if static.get("status") != "PASS":
        raise RuntimeError("R11_STAGE2_CALIBRATION_SEMANTIC_GUARD_FAIL")
    manifest_path = legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    manifest = json.loads(manifest_path.read_text())
    if bool(manifest.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_STAGE2_CALIBRATION_REFUSES_OPENED_HOLDOUT")
    parents, samples = b.load_teacher_samples(manifest["parents_file"], manifest["branches_file"])
    parent_states = b.load_parent_physics_states(manifest["parent_states_file"])

    # Reuse the already sealed Teacher authority; calibration is not a Teacher benchmark.
    teacher_cache_root = Path(
        "/data/cb16_hdd/cb16_diagnostics/r2_native/compiled_teacher_authority"
    ).resolve()
    train_evidence, val_evidence, oracle = b.compile_teacher_evidence_incremental(
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
    if oracle.get("mode") != "REUSED_VERIFIED_AUTHORITY":
        raise RuntimeError("R11_STAGE2_CALIBRATION_TEACHER_ORACLE_NOT_WARM")

    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (6, 1):
        raise RuntimeError("R11_STAGE2_CALIBRATION_REQUIRES_SM61")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    prepared = b.prepare_evidence_campaign_r11(
        train_evidence=train_evidence,
        validation_evidence=val_evidence,
        parents=parents,
        device=device,
        pin_memory=False,
    )
    g0_path = package_root / "authority/g0_parent/central_brain_g0_r10_2_parent.pt"
    if b.sha256_file(g0_path) != b.G0_FILE_SHA256:
        raise RuntimeError("R11_STAGE2_CALIBRATION_G0_DRIFT")
    g0_state = b.load_checkpoint_state(g0_path)
    training_runtime = b.TrainingRuntimeR11(
        device=device,
        evaluation_runtime=b.EvaluationRuntimeR11(enable_cuda_graph=False),
    )

    # Bind the already-preforked H72 runtime and establish exact serial identity.
    physics = b.FrozenPhysicsRuntimeR102.load(package_root)
    trace_items = b._choose_items(parents, parent_states, 96)
    symbols = sorted({x.symbol for x in trace_items})
    serial_cache = b.MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    serial_cache.preload(symbols)
    with b.TraceRuntimeR11(physics=physics, market_cache=serial_cache, max_workers=1) as serial_rt:
        serial_rows = serial_rt.run(trace_items)
    serial_hash = b._trace_hash(serial_rows)
    serial_cache.close()
    process_cache = b.MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    process_cache.preload(symbols)
    process_cache.assert_read_only()
    trace_runtime = b.ForkProcessTraceRuntimeR11(
        physics=physics, market_cache=process_cache, max_workers=workers
    )
    if b._trace_hash(trace_runtime.run(trace_items)) != serial_hash:
        raise RuntimeError("R11_STAGE2_CALIBRATION_H72_IDENTITY_FAIL")

    def run_trace(repeats: int) -> tuple[float, str]:
        t0 = time.perf_counter()
        last_hash = ""
        for _ in range(repeats):
            rows = trace_runtime.run(trace_items)
            last_hash = b._trace_hash(rows)
            if last_hash != serial_hash:
                raise RuntimeError("R11_STAGE2_CALIBRATION_H72_IDENTITY_DRIFT")
        return time.perf_counter() - t0, last_hash

    train_seq = 0
    expected_challenger: str | None = None
    train_lock = threading.Lock()

    def run_train(repeats: int, label: str) -> tuple[float, str, int]:
        nonlocal train_seq, expected_challenger
        t0 = time.perf_counter()
        last_hash = ""
        total_steps = 0
        for _ in range(repeats):
            with train_lock:
                train_seq += 1
                seq = train_seq
            model = b.build_g0_brain_r10("TIER_1", seed=24680, device=str(device))
            model.load_state_dict(g0_state, strict=True)
            receipt = training_runtime.train_challenger(
                model=model,
                campaign=prepared,
                generation=b.BENCH_AUTH.generation,
                snapshot_hash="STAGE2_CALIBRATION_SNAPSHOT",
                receipt_dir=ssd_root / f"{label}_{seq:04d}",
            )
            if receipt["amp"] is not False or receipt["dtype"] != "torch.float32":
                raise RuntimeError("R11_STAGE2_CALIBRATION_ARITHMETIC_DRIFT")
            if set(receipt["gradient_owner_set_last_step"]) != set(b.AUTHORIZED_GRADIENT_OWNERS_R11):
                raise RuntimeError("R11_STAGE2_CALIBRATION_GRADIENT_OWNER_DRIFT")
            last_hash = str(receipt["challenger_semantic_sha256"])
            total_steps += int(receipt["optimizer_steps"])
            if expected_challenger is None:
                expected_challenger = last_hash
            elif last_hash != expected_challenger:
                raise RuntimeError("R11_STAGE2_CALIBRATION_TRAINING_IDENTITY_DRIFT")
        torch.cuda.synchronize(device)
        return time.perf_counter() - t0, last_hash, total_steps

    # Isolated unit latency: enough repeats to suppress one-off launch/cache noise.
    trace_samples = [run_trace(1)[0] for _ in range(3)]
    train_samples = [run_train(1, "isolated_train")[0] for _ in range(2)]
    isolated_trace = _median(trace_samples)
    isolated_train = _median(train_samples)

    def monitor_gpu(stop: threading.Event, samples_out: list[float]) -> None:
        while not stop.is_set():
            s = sample_nvidia_smi_r11(0)
            busy = s.get("gpu_busy_percent")
            if busy is not None:
                samples_out.append(float(busy))
            stop.wait(0.25)

    cases: list[dict[str, Any]] = []
    # The old long burst hard-coded 3:1.  Calibrate 1:1, 2:1 and 3:1 under the
    # selected worker count so the final integration ratio comes from this machine.
    for trace_repeats in (1, 2, 3):
        gpu_samples: list[float] = []
        stop = threading.Event()
        monitor = threading.Thread(
            target=monitor_gpu,
            args=(stop, gpu_samples),
            name=f"stage2-calibration-gpu-{workers}-{trace_repeats}",
            daemon=True,
        )
        monitor.start()
        case_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as ex:
            tf = ex.submit(run_trace, trace_repeats)
            gf = ex.submit(run_train, 1, f"concurrent_w{workers}_t{trace_repeats}")
            trace_s, trace_hash = tf.result(timeout=120)
            train_s, challenger_hash, optimizer_steps = gf.result(timeout=120)
        wall_s = time.perf_counter() - case_started
        stop.set()
        monitor.join(timeout=2)
        avg_gpu = statistics.fmean(gpu_samples) if gpu_samples else None
        idle_ratio = (
            statistics.fmean(1.0 if x <= 5.0 else 0.0 for x in gpu_samples)
            if gpu_samples else None
        )
        cases.append(
            {
                "trace_repeats": trace_repeats,
                "train_repeats": 1,
                "wall_seconds": wall_s,
                "trace_lane_seconds": trace_s,
                "train_lane_seconds": train_s,
                "lane_balance_ratio_min_over_max": min(trace_s, train_s) / max(trace_s, train_s),
                "gpu_avg_pct": avg_gpu,
                "gpu_idle_ratio": idle_ratio,
                "gpu_samples": len(gpu_samples),
                "trace_identity_equal": trace_hash == serial_hash,
                "challenger_hash": challenger_hash,
                "optimizer_steps": optimizer_steps,
            }
        )

    # Select engineering candidate by lane balance first, then wall time. This does not
    # become scientific identity; Task F may still reject it during the long burst.
    candidate = max(
        cases,
        key=lambda x: (
            float(x["lane_balance_ratio_min_over_max"]),
            -float(x["wall_seconds"]),
        ),
    )
    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "scientific_semantics_changed": False,
        "new_scientific_verdict": False,
        "worker_count": workers,
        "trace_items": len(trace_items),
        "serial_trace_hash": serial_hash,
        "challenger_hash": expected_challenger,
        "isolated": {
            "trace_seconds_samples": trace_samples,
            "trace_seconds_median": isolated_trace,
            "train_seconds_samples": train_samples,
            "train_seconds_median": isolated_train,
            "train_to_trace_wall_ratio": isolated_train / isolated_trace,
        },
        "concurrent_cases": cases,
        "recommended_trace_repeats_per_train": int(candidate["trace_repeats"]),
        "selection_rule": "MAX_LANE_BALANCE_THEN_MIN_WALL__ENGINEERING_ONLY",
        "performance_only": True,
        "final_holdout_read": False,
        "oci_compute_used": False,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, out_path)
    trace_runtime.close()
    process_cache.assert_read_only()
    process_cache.close()
    print("CB16_R11_STAGE2_OVERLAP_CALIBRATION=PASS " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
