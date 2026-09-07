#!/usr/bin/env python3
from __future__ import annotations

"""Short real-machine micro-benchmark for R11 Stage-2 Task B.

Synthetic fixed-seed tensors only. This script never reads market data, frozen
assets, or holdout payloads. It benchmarks runtime-only choices while requiring
FP32 scientific identity between the compared execution paths.
"""

import json
import math
import statistics
import time
from typing import Any, Callable

import torch

from cb16_local_opt.gpu_runtime_r11 import (
    GTX1060_SM61_FP32_PATH,
    benchmark_h2d_policy_r11,
    inspect_gpu_execution_profile_r11,
    prepare_epoch_permutations_r11,
    sample_nvidia_smi_r11,
    transfer_host_tensor_r11,
)
from cb16_local_opt.training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    EvaluationRuntimeR11,
    PreparedEvidenceR11,
    TrainingRuntimeR11,
)
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10


DEVICE = torch.device("cuda:0")
BATCH = 512
EVAL_ROWS = 4096
TRAIN_ROWS = 4096
SEED = 24680


def _sync() -> None:
    torch.cuda.synchronize(DEVICE)


def _make_prepared(rows: int, *, seed: int, evidence_hash: str) -> PreparedEvidenceR11:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    features = torch.randn((rows, 102), generator=g, dtype=torch.float32)
    target_logits = torch.randn((rows, 3), generator=g, dtype=torch.float32)
    target_probs = torch.softmax(target_logits, dim=-1)
    risk = torch.rand((rows, 1), generator=g, dtype=torch.float32)
    weight = torch.ones((rows, 1), dtype=torch.float32)
    host = torch.cat((features, target_probs, risk, weight), dim=1).contiguous()
    packed, transfer = transfer_host_tensor_r11(host, DEVICE, pin_memory=False)
    prepared = PreparedEvidenceR11(
        parent_ids=tuple(f"SYNTH-P-{i:06d}" for i in range(rows)),
        dependence_group_ids=tuple(f"SYNTH-DG-{i % 64:03d}" for i in range(rows)),
        packed=packed,
        evidence_hash=evidence_hash,
        host_to_device_transfers=1,
        h2d_strategy=transfer.strategy,
        h2d_non_blocking=transfer.non_blocking,
        h2d_latency_ms=transfer.transfer_ms,
        h2d_benchmark=None,
    )
    prepared.validate()
    return prepared


def _elapsed(fn: Callable[[], Any], iterations: int) -> float:
    _sync()
    t0 = time.perf_counter()
    for _ in range(int(iterations)):
        fn()
    _sync()
    return time.perf_counter() - t0


def _median_ms_per_iter(fn: Callable[[], Any], iterations: int, repeats: int = 2) -> float:
    values = [1000.0 * _elapsed(fn, iterations) / iterations for _ in range(repeats)]
    return float(statistics.median(values))


def _calibrated_iterations(fn: Callable[[], Any], *, target_seconds: float, cap: int) -> int:
    probe = 40
    elapsed = _elapsed(fn, probe)
    per = max(elapsed / probe, 1e-5)
    return max(100, min(int(cap), int(target_seconds / per)))


def _state_max_abs_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    worst = 0.0
    for key, av in a.state_dict().items():
        bv = b.state_dict()[key]
        diff = float((av.detach() - bv.detach()).abs().max().cpu())
        worst = max(worst, diff)
    return worst


def _benchmark_permutations(rows: int = 65536, repeats: int = 3) -> dict[str, Any]:
    seed = SEED + 7

    expected_g = torch.Generator(device="cpu")
    expected_g.manual_seed(seed)
    expected_cpu = torch.stack(
        [torch.randperm(rows, generator=expected_g) for _ in range(12)], dim=0
    )
    optimized, transfers = prepare_epoch_permutations_r11(
        rows=rows, epochs=12, seed=seed, device=DEVICE
    )
    identity = bool(torch.equal(optimized.cpu(), expected_cpu))

    baseline_times = []
    optimized_times = []
    for _ in range(repeats):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        _sync()
        t0 = time.perf_counter()
        resident = []
        for _epoch in range(12):
            resident.append(torch.randperm(rows, generator=g).to(DEVICE))
        _sync()
        baseline_times.append(time.perf_counter() - t0)
        del resident

        _sync()
        t0 = time.perf_counter()
        stacked, _ = prepare_epoch_permutations_r11(
            rows=rows, epochs=12, seed=seed, device=DEVICE
        )
        _sync()
        optimized_times.append(time.perf_counter() - t0)
        del stacked

    base_ms = statistics.median(baseline_times) * 1000.0
    opt_ms = statistics.median(optimized_times) * 1000.0
    return {
        "rows": rows,
        "epochs": 12,
        "identity_equal": identity,
        "baseline_epoch_h2d_transfers": 12,
        "optimized_generation_h2d_transfers": int(transfers),
        "baseline_total_ms": base_ms,
        "optimized_total_ms": opt_ms,
        "speedup_x": base_ms / opt_ms if opt_ms > 0 else None,
    }


def _benchmark_eval(prepared: PreparedEvidenceR11) -> dict[str, Any]:
    model = build_g0_brain_r10("TIER_1", seed=13579, device=str(DEVICE))
    model.eval()
    eager_runtime = EvaluationRuntimeR11(enable_cuda_graph=False)
    graph_runtime = EvaluationRuntimeR11(enable_cuda_graph=True)

    eager_result = eager_runtime.evaluate(model, prepared, use_cache=False)
    graph_result = graph_runtime.evaluate(model, prepared, use_cache=False)
    loss_abs = abs(float(eager_result["loss"]) - float(graph_result["loss"]))
    behavior_equal = eager_result["behavior_fingerprint"] == graph_result["behavior_fingerprint"]
    if loss_abs > 1e-7 or not behavior_equal:
        raise RuntimeError(
            f"TASK_B_EVAL_IDENTITY_FAIL:loss_abs={loss_abs}:behavior_equal={behavior_equal}"
        )

    with torch.inference_mode():
        eager_fn = lambda: eager_runtime._eager_forward(model, prepared)
        graph_fn = lambda: graph_runtime._graph_forward(model, prepared)
        for _ in range(10):
            eager_fn()
            graph_fn()
        _sync()
        iterations = _calibrated_iterations(eager_fn, target_seconds=5.0, cap=20000)
        eager_ms = _median_ms_per_iter(eager_fn, iterations, repeats=2)
        graph_ms = _median_ms_per_iter(graph_fn, iterations, repeats=2)

    graph_active = graph_result["execution_mode"] == "CUDA_GRAPH_FP32"
    return {
        "rows": prepared.rows,
        "iterations_per_pass": iterations,
        "graph_active": graph_active,
        "graph_status": graph_result.get("cuda_graph_status"),
        "identity_loss_abs": loss_abs,
        "identity_behavior_equal": behavior_equal,
        "eager_forward_ms": eager_ms,
        "graph_forward_ms": graph_ms,
        "speedup_x": eager_ms / graph_ms if graph_active and graph_ms > 0 else None,
    }


def _run_baseline_steps(
    runtime: TrainingRuntimeR11,
    model: torch.nn.Module,
    prepared: PreparedEvidenceR11,
    ids: list[torch.Tensor],
    steps: int,
) -> tuple[float, Any]:
    optimizer = runtime.build_optimizer(model)
    result = None
    _sync()
    t0 = time.perf_counter()
    for i in range(steps):
        result = runtime.train_one_step(
            model=model,
            optimizer=optimizer,
            prepared=prepared,
            ids=ids[i % len(ids)],
        )
    _sync()
    return time.perf_counter() - t0, result


def _run_optimized_steps(
    runtime: TrainingRuntimeR11,
    model: torch.nn.Module,
    prepared: PreparedEvidenceR11,
    ids: list[torch.Tensor],
    steps: int,
) -> tuple[float, Any]:
    optimizer = runtime.build_optimizer(model)
    runtime._validate_step_boundary(model, prepared)
    result = None
    _sync()
    t0 = time.perf_counter()
    for i in range(steps):
        result = runtime._train_step(
            model=model,
            optimizer=optimizer,
            prepared=prepared,
            ids=ids[i % len(ids)],
            runtime_prevalidated=True,
            collect_diagnostics=bool(i == 0 or i == steps - 1),
            materialize_result=bool(i == steps - 1),
        )
    _sync()
    return time.perf_counter() - t0, result


def _benchmark_training(prepared: PreparedEvidenceR11) -> dict[str, Any]:
    ids = [
        torch.arange(start, start + BATCH, device=DEVICE, dtype=torch.long)
        for start in range(0, TRAIN_ROWS, BATCH)
    ]

    # Calibrate with a throwaway model so both measured trajectories start from
    # exactly the same model and optimizer state.
    probe_model = build_g0_brain_r10("TIER_1", seed=97531, device=str(DEVICE))
    probe_runtime = TrainingRuntimeR11(device=DEVICE)
    probe_optimizer = probe_runtime.build_optimizer(probe_model)
    probe_runtime._validate_step_boundary(probe_model, prepared)

    def probe_step() -> None:
        probe_runtime._train_step(
            model=probe_model,
            optimizer=probe_optimizer,
            prepared=prepared,
            ids=ids[probe_runtime._step_index % len(ids)],
            runtime_prevalidated=True,
            collect_diagnostics=False,
            materialize_result=False,
        )

    steps = _calibrated_iterations(probe_step, target_seconds=5.0, cap=2500)
    del probe_model, probe_runtime, probe_optimizer

    baseline_model = build_g0_brain_r10("TIER_1", seed=86420, device=str(DEVICE))
    optimized_model = build_g0_brain_r10("TIER_1", seed=86420, device=str(DEVICE))
    baseline_runtime = TrainingRuntimeR11(device=DEVICE)
    optimized_runtime = TrainingRuntimeR11(device=DEVICE)

    baseline_s, baseline_last = _run_baseline_steps(
        baseline_runtime, baseline_model, prepared, ids, steps
    )
    optimized_s, optimized_last = _run_optimized_steps(
        optimized_runtime, optimized_model, prepared, ids, steps
    )

    if baseline_last is None or optimized_last is None:
        raise RuntimeError("TASK_B_TRAINING_RESULT_MISSING")
    state_diff = _state_max_abs_diff(baseline_model, optimized_model)
    owners_equal = (
        baseline_last.gradient_owner_set
        == optimized_last.gradient_owner_set
        == AUTHORIZED_GRADIENT_OWNERS_R11
    )

    evaluator = EvaluationRuntimeR11(enable_cuda_graph=False)
    baseline_eval = evaluator.evaluate(baseline_model, prepared, use_cache=False)
    optimized_eval = evaluator.evaluate(optimized_model, prepared, use_cache=False)
    behavior_equal = (
        baseline_eval["behavior_fingerprint"] == optimized_eval["behavior_fingerprint"]
    )
    eval_loss_abs = abs(float(baseline_eval["loss"]) - float(optimized_eval["loss"]))
    if state_diff > 1e-7 or not owners_equal or not behavior_equal or eval_loss_abs > 1e-7:
        raise RuntimeError(
            "TASK_B_TRAINING_IDENTITY_FAIL:"
            f"state_diff={state_diff}:owners_equal={owners_equal}:"
            f"behavior_equal={behavior_equal}:eval_loss_abs={eval_loss_abs}"
        )

    baseline_ms = baseline_s * 1000.0 / steps
    optimized_ms = optimized_s * 1000.0 / steps
    return {
        "steps_each": steps,
        "optimizer_steps_equal": True,
        "gradient_owner_set_equal": owners_equal,
        "state_max_abs_diff": state_diff,
        "behavior_fingerprint_equal": behavior_equal,
        "eval_loss_abs": eval_loss_abs,
        "baseline_full_sync_diagnostics_ms_per_step": baseline_ms,
        "optimized_production_cadence_ms_per_step": optimized_ms,
        "speedup_x": baseline_ms / optimized_ms if optimized_ms > 0 else None,
        "baseline_total_s": baseline_s,
        "optimized_total_s": optimized_s,
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("TASK_B_REAL_MACHINE_REQUIRES_CUDA")
    profile = inspect_gpu_execution_profile_r11(DEVICE)
    if profile.execution_path != GTX1060_SM61_FP32_PATH or profile.compute_capability != (6, 1):
        raise RuntimeError(f"TASK_B_WRONG_GPU_PROFILE:{profile.as_dict()}")
    if profile.amp_enabled or profile.canonical_dtype != "torch.float32":
        raise RuntimeError("TASK_B_NONCANONICAL_ARITHMETIC_PROFILE")

    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    torch.cuda.reset_peak_memory_stats(DEVICE)
    started = time.perf_counter()
    before_smi = sample_nvidia_smi_r11(0)

    h2d_host = torch.randn((32768, 107), dtype=torch.float32)
    h2d = benchmark_h2d_policy_r11(
        h2d_host,
        DEVICE,
        repeats=9,
        warmups=3,
        min_total_improvement_fraction=0.03,
    )

    eval_prepared = _make_prepared(EVAL_ROWS, seed=SEED + 1, evidence_hash="TASK_B_SYNTH_EVAL")
    train_prepared = _make_prepared(TRAIN_ROWS, seed=SEED + 2, evidence_hash="TASK_B_SYNTH_TRAIN")

    permutations = _benchmark_permutations()
    if not permutations["identity_equal"]:
        raise RuntimeError("TASK_B_PERMUTATION_IDENTITY_FAIL")

    evaluation = _benchmark_eval(eval_prepared)
    training = _benchmark_training(train_prepared)

    after_smi = sample_nvidia_smi_r11(0)
    elapsed = time.perf_counter() - started
    result = {
        "schema": "CB16_R11_STAGE2_TASK_B_REAL_MACHINE_MICROBENCH_V1",
        "status": "PASS_CORRECTNESS_IDENTITY",
        "scientific_verdict_changed": False,
        "data_source": "SYNTHETIC_FIXED_SEED_ONLY",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "execution_profile": profile.as_dict(),
        "fp32": True,
        "amp": False,
        "tf32": False,
        "elapsed_s": elapsed,
        "h2d": h2d.as_dict(),
        "permutation": permutations,
        "evaluation": evaluation,
        "training": training,
        "gpu_sample_before": before_smi,
        "gpu_sample_after": after_smi,
        "torch_peak_allocated_mb": torch.cuda.max_memory_allocated(DEVICE) / (1024.0 * 1024.0),
        "torch_peak_reserved_mb": torch.cuda.max_memory_reserved(DEVICE) / (1024.0 * 1024.0),
    }
    print("TASK_B_REAL_MACHINE_RESULT=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
