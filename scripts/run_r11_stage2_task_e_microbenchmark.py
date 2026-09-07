#!/usr/bin/env python3
from __future__ import annotations

"""Task E-only Shanxi microbenchmark for Stage-2 qualification-harness overhead.

This is not a scientific workload and creates no scientific verdict. It verifies the
exact Shanxi hardware/runtime identity, freezes all scientific authorities read-only,
and compares a canonical FP32 CUDA probe with and without 1 Hz Task E host sampling.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.rearchitecture_authority_r11 import verify_static_semantic_contracts
from scripts.run_r11_stage2_burst_qualification import (
    _device_for_path,
    sample_host,
    summarize_host,
    verify_hardware,
)

SCHEMA = "CB16_R11_STAGE2_TASK_E_MICROBENCH_V1"
EXACT_TASK_BRANCH = "ai/r11-stage2-task-e-burst-harness-r1"
EXACT_RUNNER_LABELS = {"self-hosted", "shanxi", "cb16-wss-qualification"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _guard_execution_context() -> dict[str, Any]:
    branch = os.environ.get("GITHUB_REF_NAME", "")
    if branch != EXACT_TASK_BRANCH:
        raise RuntimeError(f"R11_TASK_E_MICROBENCH_BRANCH_REFUSED:{branch}")
    raw = os.environ.get("CB16_RUNNER_LABELS", "")
    labels = {x.strip() for x in raw.replace(";", ",").split(",") if x.strip()}
    if labels != EXACT_RUNNER_LABELS:
        raise RuntimeError(f"R11_TASK_E_MICROBENCH_RUNNER_SELECTOR_REFUSED:{sorted(labels)}")
    if "cb16-r10-canonical" in labels:
        raise RuntimeError("R11_TASK_E_MICROBENCH_CANONICAL_RUNNER_FORBIDDEN")
    return {"branch": branch, "runner_labels": sorted(labels)}


def _run_fp32_probe(torch: Any, seconds: float, *, sampler: bool, ssd_path: Path, hdd_path: Path) -> tuple[dict[str, Any], list[Any]]:
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if torch.is_autocast_enabled():
        raise RuntimeError("R11_TASK_E_MICROBENCH_AUTOCAST_ENABLED")
    x = torch.randn((1024, 1024), device=device, dtype=torch.float32)
    y = torch.randn((1024, 1024), device=device, dtype=torch.float32)
    z = torch.empty((1024, 1024), device=device, dtype=torch.float32)
    for _ in range(8):
        torch.mm(x, y, out=z)
    torch.cuda.synchronize()

    ssd_dev = _device_for_path(ssd_path)
    hdd_dev = _device_for_path(hdd_path)
    samples: list[Any] = []
    iterations = 0
    batch = 32
    t0 = time.monotonic()
    next_sample = t0
    while True:
        for _ in range(batch):
            torch.mm(x, y, out=z)
        iterations += batch
        torch.cuda.synchronize()
        now = time.monotonic()
        if sampler and now >= next_sample:
            samples.append(sample_host(os.getpid(), ssd_dev, hdd_dev))
            next_sample += 1.0
        if now - t0 >= seconds:
            break
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    return {
        "seconds": elapsed,
        "iterations": iterations,
        "iterations_per_sec": iterations / max(elapsed, 1e-9),
        "matrix_shape": [1024, 1024],
        "dtype": "FP32",
        "amp": False,
        "tf32": False,
        "sampler_enabled": sampler,
    }, samples


def main() -> int:
    ap = argparse.ArgumentParser(description="CB16 R11 Task E Shanxi microbenchmark")
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--legacy-r104-root", required=True)
    ap.add_argument("--ssd-path", default="/data/cb16_ci")
    ap.add_argument("--hdd-path", default="/data/cb16_hdd")
    ap.add_argument("--baseline-seconds", type=float, default=20.0)
    ap.add_argument("--sampled-seconds", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not (10.0 <= args.baseline_seconds <= 60.0):
        raise RuntimeError("R11_TASK_E_BASELINE_DURATION_OUT_OF_POLICY")
    if not (30.0 <= args.sampled_seconds <= 180.0):
        raise RuntimeError("R11_TASK_E_MEASURED_DURATION_OUT_OF_POLICY")

    context = _guard_execution_context()
    static_guard = verify_static_semantic_contracts(ROOT)
    hardware = verify_hardware()
    import torch
    if torch.__version__ != "2.8.0+cu126" or torch.version.cuda != "12.6":
        raise RuntimeError("R11_TASK_E_CANONICAL_TORCH_ENV_DRIFT")

    package_root = Path(args.package_root).resolve()
    legacy_root = Path(args.legacy_r104_root).resolve()
    manifest_path = legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    manifest_before = json.loads(manifest_path.read_text())
    if manifest_before.get("final_holdout_2025_09_accessed", False):
        raise RuntimeError("R11_TASK_E_REFUSES_OPENED_FINAL_HOLDOUT")
    frozen_before = frozen_authority_hashes(package_root)
    freeze_path = ROOT / "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
    freeze_sha_before = _sha256(freeze_path)

    baseline, _ = _run_fp32_probe(
        torch, args.baseline_seconds, sampler=False,
        ssd_path=Path(args.ssd_path), hdd_path=Path(args.hdd_path),
    )
    sampled, samples = _run_fp32_probe(
        torch, args.sampled_seconds, sampler=True,
        ssd_path=Path(args.ssd_path), hdd_path=Path(args.hdd_path),
    )
    if len(samples) < max(20, int(args.sampled_seconds * 0.75)):
        raise RuntimeError(f"R11_TASK_E_HOST_SAMPLE_COVERAGE_LOW:{len(samples)}")
    host = summarize_host(samples, sampled["seconds"])

    frozen_after = frozen_authority_hashes(package_root)
    freeze_sha_after = _sha256(freeze_path)
    manifest_after = json.loads(manifest_path.read_text())
    overhead_ratio = sampled["iterations_per_sec"] / max(baseline["iterations_per_sec"], 1e-9)
    correctness = {
        "semantic_freeze_pass": static_guard.get("status") == "PASS",
        "semantic_freeze_sha_unchanged": freeze_sha_before == freeze_sha_after,
        "frozen_authority_unchanged": frozen_before == frozen_after,
        "final_holdout_untouched": not manifest_before.get("final_holdout_2025_09_accessed", False) and not manifest_after.get("final_holdout_2025_09_accessed", False),
        "training_dtype": "FP32",
        "amp": False,
        "tf32": False,
        "fresh_market_data_download": False,
        "scientific_verdict_created": False,
        "scientific_semantics_changed": False,
    }
    acceptance_failures = [k for k, v in correctness.items() if isinstance(v, bool) and k not in {"amp", "tf32", "fresh_market_data_download", "scientific_verdict_created", "scientific_semantics_changed"} and not v]
    if any(correctness[k] is not False for k in ("amp", "tf32", "fresh_market_data_download", "scientific_verdict_created", "scientific_semantics_changed")):
        acceptance_failures.append("forbidden_semantic_or_runtime_flag")
    if overhead_ratio < 0.90:
        acceptance_failures.append(f"sampler_throughput_retention:{overhead_ratio:.4f}<0.90")

    result = {
        "schema": SCHEMA,
        "status": "PASS" if not acceptance_failures else "FAIL",
        "context": context,
        "hardware": hardware,
        "correctness_identity": correctness,
        "baseline_fp32_probe": baseline,
        "sampled_fp32_probe": sampled,
        "sampler_throughput_retention_ratio": overhead_ratio,
        "sampler_overhead_fraction": max(0.0, 1.0 - overhead_ratio),
        "host_telemetry": host,
        "storage_devices": {
            "ssd_path": str(Path(args.ssd_path)),
            "ssd_device": _device_for_path(Path(args.ssd_path)),
            "hdd_path": str(Path(args.hdd_path)),
            "hdd_device": _device_for_path(Path(args.hdd_path)),
        },
        "acceptance": {
            "pass": not acceptance_failures,
            "failures": acceptance_failures,
            "sampler_throughput_retention_min": 0.90,
        },
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"schema": SCHEMA, "status": result["status"], "retention": overhead_ratio, "samples": len(samples)}, sort_keys=True))
    return 0 if not acceptance_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
