from __future__ import annotations

"""
GTX 1060 FP32 autotuner.

The tuner does not assume AMP is beneficial on Pascal.  It benchmarks FP32 first,
records real throughput/VRAM, and emits an immutable tuning receipt that later runs
can bind to.

It benchmarks:
- inference/rollout batch candidates;
- training batch candidates;
- optional CPU thread counts;
- model tiers.

A non-GTX1060 or CPU-only machine can run in diagnostic mode, but only a real GTX1060
CC 6.1 CUDA run with a binary-compatible Pascal cubin and CB16 workload canary produces
`GTX1060_AUTHORITATIVE_TUNING`.
"""

import dataclasses
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .hardware_profile import detect_hardware, validate_pascal_torch
from .pascal_cuda_compat_r81 import assess_cuda_binary_compatibility_r81, cb16_cuda_workload_canary_r81
from .trader_capacity_ladder import build_trader, parameter_report
from .gpu_training_policy import (
    ProbabilisticTrainingBatch,
    TrainingPolicy,
    build_optimizer,
    train_step,
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class BatchBenchmark:
    tier: str
    mode: str
    batch_size: int
    status: str
    samples_per_second: float | None
    ms_per_batch: float | None
    peak_vram_bytes: int
    error: str | None = None


@dataclass(frozen=True)
class TuningChoice:
    tier: str
    rollout_batch: int
    train_batch: int
    cpu_torch_threads: int
    dtype: str
    amp_enabled: bool


@dataclass(frozen=True)
class AutotuneReceipt:
    schema: str
    authority: str
    hardware_name: str
    compute_capability: tuple[int, int] | None
    torch_version: str
    torch_cuda_version: str | None
    compiled_arches: tuple[str, ...]
    model_reports: dict[str, dict[str, int]]
    benchmarks: tuple[BatchBenchmark, ...]
    choice: TuningChoice
    tuning_policy_hash: str
    created_at_unix: float

    @property
    def content_hash(self) -> str:
        # Exclude wall-clock from status-driving content hash.
        d = asdict(self)
        d.pop("created_at_unix", None)
        return canonical_hash(d)


def estimate_fp32_training_static_bytes(params: int, optimizer: str = "adamw") -> int:
    # Param + grad + Adam first moment + Adam second moment.
    if optimizer.lower() == "adamw":
        return int(params * 4 * 4)
    if optimizer.lower() == "sgd":
        return int(params * 4 * 2)
    return int(params * 4 * 4)


def _cuda_timed(fn, *, warmup: int, repeats: int) -> tuple[float, int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms_total = float(start.elapsed_time(end))
    peak = int(torch.cuda.max_memory_allocated())
    return ms_total / repeats, peak


@torch.no_grad()
def _benchmark_inference(
    model: torch.nn.Module,
    *,
    tier: str,
    batch_size: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> BatchBenchmark:
    try:
        market = torch.randn(batch_size, model.spec.market_dim, device=device, dtype=torch.float32)
        account = torch.randn(batch_size, model.spec.account_dim, device=device, dtype=torch.float32)
        model.eval()
        if device.type == "cuda":
            ms, peak = _cuda_timed(
                lambda: model(market, account),
                warmup=warmup,
                repeats=repeats,
            )
        else:
            for _ in range(warmup):
                model(market, account)
            t0 = time.perf_counter()
            for _ in range(repeats):
                model(market, account)
            ms = (time.perf_counter() - t0) * 1000 / repeats
            peak = 0
        return BatchBenchmark(
            tier=tier,
            mode="ROLLOUT_FP32",
            batch_size=batch_size,
            status="PASS",
            samples_per_second=batch_size / (ms / 1000),
            ms_per_batch=ms,
            peak_vram_bytes=peak,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return BatchBenchmark(
                tier=tier,
                mode="ROLLOUT_FP32",
                batch_size=batch_size,
                status="OOM",
                samples_per_second=None,
                ms_per_batch=None,
                peak_vram_bytes=0,
                error="CUDA_OOM",
            )
        raise


def _make_training_batch(model, batch_size: int, device: str = "cpu") -> ProbabilisticTrainingBatch:
    # Keep source batch on CPU. train_step moves market/account tensors as required.
    p = torch.rand(batch_size, 3)
    p /= p.sum(dim=-1, keepdim=True)
    return ProbabilisticTrainingBatch(
        market_latent=torch.randn(batch_size, model.spec.market_dim),
        account_state=torch.randn(batch_size, model.spec.account_dim),
        direction_target_probs=p,
        requested_risk_target=torch.rand(batch_size),
        direction_weight=torch.ones(batch_size),
        sizing_weight=torch.ones(batch_size),
        admitted=torch.ones(batch_size, dtype=torch.bool),
    )


def _benchmark_training(
    *,
    tier: str,
    batch_size: int,
    device: torch.device,
    cpu_threads: int,
    warmup: int,
    repeats: int,
) -> BatchBenchmark:
    try:
        model = build_trader(tier).to(device)
        policy = TrainingPolicy(
            device=str(device),
            dtype="fp32",
            amp_enabled=False,
            cpu_torch_threads=cpu_threads,
        )
        opt = build_optimizer(model, policy)

        b = _make_training_batch(model, batch_size)
        if device.type == "cuda":
            b = b.pin_memory()

        # Reuse one legal benchmark batch; this measures optimizer/model/device
        # throughput instead of CPU random-number generation.
        for _ in range(warmup):
            train_step(model=model, optimizer=opt, batch=b, policy=policy)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(repeats):
            train_step(model=model, optimizer=opt, batch=b, policy=policy)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
        else:
            peak = 0
        seconds = time.perf_counter() - t0
        ms = seconds * 1000 / repeats
        return BatchBenchmark(
            tier=tier,
            mode="TRAIN_FP32",
            batch_size=batch_size,
            status="PASS",
            samples_per_second=batch_size * repeats / max(seconds, 1e-12),
            ms_per_batch=ms,
            peak_vram_bytes=peak,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return BatchBenchmark(
                tier=tier,
                mode="TRAIN_FP32",
                batch_size=batch_size,
                status="OOM",
                samples_per_second=None,
                ms_per_batch=None,
                peak_vram_bytes=0,
                error="CUDA_OOM",
            )
        raise


@dataclass(frozen=True)
class AutotunePolicy:
    tiers: tuple[str, ...] = ("TIER_1", "TIER_2")
    rollout_batches: tuple[int, ...] = (4096, 8192, 16384, 32768)
    train_batches: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
    cpu_thread_candidates: tuple[int, ...] = (1, 2)
    warmup: int = 2
    repeats: int = 5
    min_vram_headroom_fraction: float = 0.15
    prefer_smallest_tier_within_throughput_ratio: float = 0.85

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


class GTX1060Autotuner:
    def __init__(self, policy: AutotunePolicy | None = None):
        self.policy = policy or AutotunePolicy()

    def run(self, *, allow_cpu_diagnostic: bool = False) -> AutotuneReceipt:
        hw = detect_hardware()
        issues = validate_pascal_torch(hw, strict=False)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif allow_cpu_diagnostic:
            device = torch.device("cpu")
        else:
            raise RuntimeError("CUDA_REQUIRED_FOR_GPU_AUTOTUNE")

        compatibility = assess_cuda_binary_compatibility_r81(
            device_cc=hw.gpu.compute_capability,
            compiled_arches=hw.gpu.compiled_arches,
            expected_cc=(6, 1),
        )
        cb16_canary = cb16_cuda_workload_canary_r81(
            rollout_rows=2048,
            training_rows=512,
        ) if device.type == "cuda" else None
        authoritative = (
            device.type == "cuda"
            and hw.gpu.compute_capability == (6, 1)
            and compatibility.status == "PASS"
            and cb16_canary is not None
            and cb16_canary.status == "PASS"
        )
        authority = (
            "GTX1060_AUTHORITATIVE_TUNING"
            if authoritative
            else "NONAUTHORITATIVE_DIAGNOSTIC"
        )

        benchmarks: list[BatchBenchmark] = []
        model_reports: dict[str, dict[str, int]] = {}

        for tier in self.policy.tiers:
            m = build_trader(tier)
            model_reports[tier] = parameter_report(m)
            m = m.to(device)
            for b in self.policy.rollout_batches:
                benchmarks.append(
                    _benchmark_inference(
                        m,
                        tier=tier,
                        batch_size=int(b),
                        device=device,
                        warmup=self.policy.warmup,
                        repeats=self.policy.repeats,
                    )
                )
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Training benchmarks are more expensive; test CPU-thread candidates.
        train_records: list[tuple[int, BatchBenchmark]] = []
        for threads in self.policy.cpu_thread_candidates:
            for tier in self.policy.tiers:
                for b in self.policy.train_batches:
                    rec = _benchmark_training(
                        tier=tier,
                        batch_size=int(b),
                        device=device,
                        cpu_threads=int(threads),
                        warmup=self.policy.warmup,
                        repeats=self.policy.repeats,
                    )
                    benchmarks.append(rec)
                    train_records.append((int(threads), rec))

        choice = self._choose(benchmarks, train_records, hw)
        return AutotuneReceipt(
            schema="CB16_GTX1060_AUTOTUNE_RECEIPT_R3",
            authority=authority,
            hardware_name=hw.gpu.name or "NO_CUDA_GPU",
            compute_capability=hw.gpu.compute_capability,
            torch_version=str(torch.__version__),
            torch_cuda_version=torch.version.cuda,
            compiled_arches=hw.gpu.compiled_arches,
            model_reports=model_reports,
            benchmarks=tuple(benchmarks),
            choice=choice,
            tuning_policy_hash=self.policy.content_hash,
            created_at_unix=time.time(),
        )

    def _choose(
        self,
        benchmarks: Sequence[BatchBenchmark],
        train_records: Sequence[tuple[int, BatchBenchmark]],
        hw,
    ) -> TuningChoice:
        # GTX1060 R3 default starts at TIER_1. Tuning does not authorize a scientific
        # capacity escalation merely because a larger tier benchmarks well.
        tier = self.policy.tiers[0]

        rollout = [
            r for r in benchmarks
            if r.tier == tier and r.mode == "ROLLOUT_FP32" and r.status == "PASS"
        ]
        if not rollout:
            raise RuntimeError("NO_STABLE_ROLLOUT_BATCH")
        # Avoid selecting a batch that consumes nearly all VRAM even if throughput wins.
        vram = hw.gpu.vram_bytes or (1 << 60)
        safe_rollout = [
            r for r in rollout
            if r.peak_vram_bytes <= vram * (1 - self.policy.min_vram_headroom_fraction)
        ] or rollout
        best_rollout = max(
            safe_rollout,
            key=lambda r: (r.samples_per_second or 0.0, r.batch_size),
        )

        train = [
            (threads, r)
            for threads, r in train_records
            if r.tier == tier and r.status == "PASS"
        ]
        if not train:
            raise RuntimeError("NO_STABLE_TRAIN_BATCH")
        safe_train = [
            (t, r) for t, r in train
            if r.peak_vram_bytes <= vram * (1 - self.policy.min_vram_headroom_fraction)
        ] or train
        best_threads, best_train = max(
            safe_train,
            key=lambda tr: (tr[1].samples_per_second or 0.0, tr[1].batch_size),
        )

        return TuningChoice(
            tier=tier,
            rollout_batch=int(best_rollout.batch_size),
            train_batch=int(best_train.batch_size),
            cpu_torch_threads=int(best_threads),
            dtype="fp32",
            amp_enabled=False,
        )

    @staticmethod
    def write_receipt(receipt: AutotuneReceipt, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(receipt)
        payload["content_hash"] = receipt.content_hash
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return path
