from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cb16_local_opt.gpu_runtime_r11 import (
    CPU_FP32_FALLBACK_PATH,
    CUDA_GRAPH_DISABLED_FALLBACK,
    GTX1060_SM61_FP32_PATH,
    TelemetryConfigR11,
    benchmark_h2d_policy_r11,
    classify_cuda_capability_r11,
    prepare_epoch_permutations_r11,
)
from cb16_local_opt.training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    EvaluationRuntimeR11,
    PreparedCampaignR11,
    PreparedEvidenceR11,
    R11TrainingConfig,
    TrainingRuntimeR11,
)
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10


@dataclass(frozen=True)
class _Admission:
    admitted: bool = True


@dataclass(frozen=True)
class _Evidence:
    parent_id: str
    target_dependence_group_id: str
    direction_target_probs: tuple[float, float, float]
    requested_risk_target: float
    admission: _Admission = _Admission()
    evidence_id: str = ""
    teacher_protocol_hash: str = "TASK_B_SYNTHETIC_PROTOCOL"


def _fixture(rows: int = 64, groups: int = 32):
    rng = np.random.default_rng(20260907)
    parents = {}
    evidence = []
    for i in range(rows):
        parent_id = f"P{i:04d}"
        raw = rng.uniform(0.1, 1.0, size=3)
        probs = raw / raw.sum()
        parents[parent_id] = SimpleNamespace(
            operator48=rng.normal(size=48).astype(np.float32).tolist(),
            medium48=rng.normal(size=48).astype(np.float32).tolist(),
            account6=rng.normal(size=6).astype(np.float32).tolist(),
        )
        evidence.append(
            _Evidence(
                parent_id=parent_id,
                target_dependence_group_id=f"DG{i % groups:03d}",
                direction_target_probs=tuple(float(x) for x in probs),
                requested_risk_target=float(rng.uniform(0.05, 0.95)),
                evidence_id=f"E{i:04d}",
            )
        )
    return evidence, parents


def _state_clone(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _baseline_train(model, prepared: PreparedEvidenceR11, *, generation: int) -> int:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(24680 + int(generation))
    steps = 0
    for _ in range(12):
        permutation = torch.randperm(prepared.rows, generator=generator)
        for start in range(0, prepared.rows, 512):
            ids = permutation[start : start + 512]
            batch = prepared.packed.index_select(0, ids)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch[:, :48], batch[:, 48:96], batch[:, 96:102])
            target_p = batch[:, 102:105]
            risk_target = batch[:, 105]
            weight = batch[:, 106]
            row_direction = -(target_p * F.log_softmax(outputs["direction_logits"], -1)).sum(-1)
            row_sizing = F.smooth_l1_loss(
                outputs["requested_risk_raw"],
                risk_target,
                reduction="none",
                beta=0.05,
            )
            loss = ((row_direction + row_sizing) * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            steps += 1
    return steps


def test_sm61_explicit_fp32_path_and_forbidden_acceleration_assumptions():
    profile = classify_cuda_capability_r11(
        (6, 1), device_name="NVIDIA GeForce GTX 1060 6GB", device_index=0
    )
    assert profile.execution_path == GTX1060_SM61_FP32_PATH
    assert profile.is_sm61 is True
    assert profile.canonical_dtype == "torch.float32"
    assert profile.amp_enabled is False
    assert profile.fp16_fast_path_assumed is False
    assert profile.bf16_enabled is False
    assert profile.tf32_enabled is False
    assert profile.tensor_core_assumed is False
    assert profile.triton_required is False
    assert profile.inductor_cuda_required is False

    cpu = classify_cuda_capability_r11(None, device_name="cpu", device_index=None)
    assert cpu.execution_path == CPU_FP32_FALLBACK_PATH


def test_cpu_h2d_policy_is_conservative_and_amp_remains_false():
    host = torch.zeros((32, 107), dtype=torch.float32)
    bench = benchmark_h2d_policy_r11(host, "cpu")
    assert bench.selected_pinned is False
    assert bench.selected_non_blocking is False
    assert bench.reason == "CUDA_UNAVAILABLE_OR_NOT_REQUESTED"

    config = R11TrainingConfig()
    assert config.amp_enabled is False
    assert config.dtype == torch.float32
    with pytest.raises(RuntimeError, match="AMP_FORBIDDEN"):
        R11TrainingConfig(amp_enabled=True).validate()


def test_generation_permutation_stack_is_exact_sequence_and_single_cpu_resident_copy():
    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(24684)
    expected = torch.stack(
        [torch.randperm(73, generator=expected_generator) for _ in range(12)], dim=0
    )
    actual, transfers = prepare_epoch_permutations_r11(
        rows=73, epochs=12, seed=24684, device="cpu"
    )
    assert torch.equal(actual, expected)
    assert transfers == 0


def test_baseline_vs_optimized_full_trajectory_state_behavior_and_step_count(tmp_path):
    evidence, parents = _fixture(rows=64, groups=32)
    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    campaign = PreparedCampaignR11(train=prepared, validation=prepared)

    baseline = build_g0_brain_r10("TIER_1", seed=123, device="cpu")
    optimized = build_g0_brain_r10("TIER_1", seed=123, device="cpu")
    baseline_steps = _baseline_train(baseline, prepared, generation=4)

    runtime = TrainingRuntimeR11(device="cpu")
    receipt = runtime.train_challenger(
        model=optimized,
        campaign=campaign,
        generation=4,
        snapshot_hash="TASK-B-SNAPSHOT",
        receipt_dir=tmp_path,
    )

    assert baseline_steps == receipt["optimizer_steps"] == 12
    assert receipt["optimizer"] == "AdamW_FP32"
    assert receipt["amp"] is False
    assert receipt["dtype"] == "torch.float32"
    assert receipt["gradient_clip_max_norm"] == 10.0
    assert set(receipt["gradient_owner_set_last_step"]) == AUTHORIZED_GRADIENT_OWNERS_R11
    assert receipt["permutation_host_to_device_transfers"] == 0
    assert prepared.requires_grad is False if hasattr(prepared, "requires_grad") else True
    assert prepared.packed.requires_grad is False

    baseline_state = _state_clone(baseline)
    optimized_state = _state_clone(optimized)
    assert baseline_state.keys() == optimized_state.keys()
    for key in baseline_state:
        assert torch.equal(baseline_state[key], optimized_state[key]), key

    evaluator = EvaluationRuntimeR11(enable_cuda_graph=False)
    baseline_eval = evaluator.evaluate(baseline, prepared, use_cache=False)
    optimized_eval = evaluator.evaluate(optimized, prepared, use_cache=False)
    assert baseline_eval["behavior_fingerprint"] == optimized_eval["behavior_fingerprint"]
    assert baseline_eval["loss"] == optimized_eval["loss"]
    assert baseline_eval["direction_loss"] == optimized_eval["direction_loss"]
    assert baseline_eval["sizing_loss"] == optimized_eval["sizing_loss"]


def test_cuda_graph_disabled_fallback_is_explicit_and_equal():
    evidence, parents = _fixture()
    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    model = build_g0_brain_r10("TIER_1", seed=777, device="cpu")

    eager = EvaluationRuntimeR11(enable_cuda_graph=False).evaluate(
        model, prepared, use_cache=False
    )
    fallback = EvaluationRuntimeR11(
        enable_cuda_graph=True, cuda_graph_capability=lambda _device: False
    ).evaluate(model, prepared, use_cache=False)

    assert fallback["execution_mode"] == "EAGER_FP32_CUDA_GRAPH_FALLBACK"
    assert fallback["cuda_graph_status"] == CUDA_GRAPH_DISABLED_FALLBACK
    for key in ("loss", "direction_loss", "sizing_loss", "behavior_fingerprint"):
        assert fallback[key] == eager[key]


def test_cuda_unavailable_is_fail_closed_when_cuda_requested():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this runner")
    with pytest.raises(RuntimeError, match="CUDA_REQUESTED_BUT_UNAVAILABLE"):
        TrainingRuntimeR11(device="cuda")


def test_static_buffers_telemetry_and_bounded_caches_do_not_grow_unbounded():
    evidence, parents = _fixture()
    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    evaluator = EvaluationRuntimeR11(
        enable_cuda_graph=False,
        max_cached_results=2,
        max_cuda_graphs=1,
        max_behavior_buffers=1,
    )
    runtime = TrainingRuntimeR11(
        device="cpu",
        evaluation_runtime=evaluator,
        telemetry_config=TelemetryConfigR11(enabled=True, sample_every_steps=1),
    )
    ptr_before = runtime.resource_stats()["static_batch_buffer_data_ptr"]

    model = build_g0_brain_r10("TIER_1", seed=9, device="cpu")
    optimizer = runtime.build_optimizer(model)
    for _ in range(3):
        runtime.train_one_step(
            model=model,
            optimizer=optimizer,
            prepared=prepared,
            ids=torch.arange(prepared.rows, dtype=torch.long),
        )
    ptr_after = runtime.resource_stats()["static_batch_buffer_data_ptr"]
    assert ptr_after == ptr_before

    telemetry = runtime.telemetry.snapshot()
    assert telemetry["sampled_steps"] == 3
    assert telemetry["batch_latency_ms_median"] is not None
    assert telemetry["forward_ms_median"] is not None
    assert telemetry["backward_ms_median"] is not None
    assert telemetry["optimizer_ms_median"] is not None

    for seed in range(5):
        probe = build_g0_brain_r10("TIER_1", seed=seed, device="cpu")
        evaluator.evaluate(probe, prepared, use_cache=True)
    stats = evaluator.resource_stats()
    assert stats["evaluation_cache_entries"] == 2
    assert stats["cuda_graph_entries"] == 0
    assert stats["behavior_buffer_entries"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_vs_eager_output_equivalence_on_available_gpu():
    evidence, parents = _fixture()
    prepared = PreparedEvidenceR11.from_evidence(
        evidence, parents, device="cuda:0", pin_memory=False
    )
    model = build_g0_brain_r10("TIER_1", seed=2026, device="cuda:0")

    eager = EvaluationRuntimeR11(enable_cuda_graph=False).evaluate(
        model, prepared, use_cache=False
    )
    graphed = EvaluationRuntimeR11(enable_cuda_graph=True).evaluate(
        model, prepared, use_cache=False
    )
    assert graphed["execution_mode"] == "CUDA_GRAPH_FP32"
    assert graphed["cuda_graph_status"] == "CUDA_GRAPH_ACTIVE"
    for key in ("loss", "direction_loss", "sizing_loss", "behavior_fingerprint"):
        assert graphed[key] == eager[key]
