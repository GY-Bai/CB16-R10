from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cb16_local_opt.r102_learning import (
    _evidence_tensors as legacy_evidence_tensors,
    _group_weights as legacy_group_weights,
    policy_behavior_fingerprint as legacy_behavior_fingerprint,
    soft_teacher_loss as legacy_soft_teacher_loss,
)
from cb16_local_opt.training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    EvaluationRuntimeR11,
    PreparedEvidenceR11,
    R11TrainingConfig,
    SnapshotConsumptionGuardR11,
    TrainingRuntimeR11,
    assert_tier1_fp32_runtime_r11,
    forward_prepared_r11,
    group_weights_r11,
    policy_hash_r11,
    prepare_evidence_campaign_r11,
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
    teacher_protocol_hash: str = "SYNTHETIC_TEST_PROTOCOL"


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


def _assert_state_tight(a, b, *, atol=1e-7, rtol=1e-6):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.allclose(a[key], b[key], atol=atol, rtol=rtol), key


def test_old_new_forward_output_tight_equivalence():
    evidence, parents = _fixture()
    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    model = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")

    _ev, op, med, acc, *_ = legacy_evidence_tensors(evidence, parents, "cpu")
    legacy = model(op, med, acc)
    new = forward_prepared_r11(model, prepared)

    assert legacy.keys() == new.keys()
    for key in legacy:
        assert torch.equal(legacy[key], new[key]), key


def test_old_new_soft_teacher_loss_tight_equivalence_and_shared_eval_forward():
    evidence, parents = _fixture()
    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    model = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")

    legacy_loss, legacy_metrics = legacy_soft_teacher_loss(
        model, evidence, parents, device="cpu"
    )
    legacy_fp = legacy_behavior_fingerprint(model, evidence, parents, device="cpu")
    runtime = EvaluationRuntimeR11(enable_cuda_graph=False)
    result = runtime.evaluate(model, prepared, use_cache=False)

    assert result["loss"] == pytest.approx(float(legacy_loss.detach()), abs=1e-7, rel=0)
    assert result["direction_loss"] == pytest.approx(
        legacy_metrics["direction_loss"], abs=1e-7, rel=0
    )
    assert result["sizing_loss"] == pytest.approx(
        legacy_metrics["sizing_loss"], abs=1e-7, rel=0
    )
    assert result["behavior_fingerprint"] == legacy_fp


def test_dependence_group_weighting_exact_equivalence():
    evidence, parents = _fixture(rows=73, groups=17)
    admitted = [e for e in evidence if e.admission.admitted]
    expected = legacy_group_weights(admitted)
    actual = group_weights_r11([e.target_dependence_group_id for e in admitted])
    assert np.array_equal(actual, expected)

    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    assert np.array_equal(prepared.group_weight.numpy(), expected)


def test_gradient_owner_set_exact_and_frozen_inputs_forbidden_from_autograd():
    evidence, parents = _fixture()
    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    model = build_g0_brain_r10("TIER_1", seed=17, device="cpu")
    runtime = TrainingRuntimeR11(device="cpu", config=R11TrainingConfig())
    optimizer = runtime.build_optimizer(model)
    step = runtime.train_one_step(
        model=model,
        optimizer=optimizer,
        prepared=prepared,
        ids=torch.arange(len(evidence), dtype=torch.long),
    )
    assert step.gradient_owner_set == AUTHORIZED_GRADIENT_OWNERS_R11
    assert all(v > 0.0 for v in step.gradient_group_norms.values())

    _ev, op, med, acc, *_ = legacy_evidence_tensors(evidence, parents, "cpu")
    op.requires_grad_(True)
    med.requires_grad_(True)
    acc.requires_grad_(True)
    probe = build_g0_brain_r10("TIER_1", seed=19, device="cpu")
    out = probe(op, med, acc)
    out["direction_logits"].sum().backward()
    assert op.grad is None
    assert med.grad is None
    assert acc.grad is None
    assert not prepared.packed.requires_grad
    assert not prepared.direction_target_probs.requires_grad
    assert not prepared.requested_risk_target.requires_grad


def test_fixed_minibatch_one_step_adamw_update_tight_equivalence():
    evidence, parents = _fixture()
    old_model = build_g0_brain_r10("TIER_1", seed=12345, device="cpu")
    new_model = build_g0_brain_r10("TIER_1", seed=12345, device="cpu")
    old_opt = torch.optim.AdamW(old_model.parameters(), lr=3e-4, weight_decay=1e-4)
    new_runtime = TrainingRuntimeR11(device="cpu", config=R11TrainingConfig())
    new_opt = new_runtime.build_optimizer(new_model)

    ev, op, med, acc, dp, rt, w = legacy_evidence_tensors(evidence, parents, "cpu")
    assert len(ev) == len(evidence)
    old_opt.zero_grad(set_to_none=True)
    out = old_model(op, med, acc)
    direction = -(dp * F.log_softmax(out["direction_logits"], -1)).sum(-1)
    sizing = F.smooth_l1_loss(
        out["requested_risk_raw"], rt, reduction="none", beta=0.05
    )
    old_loss = ((direction + sizing) * w).sum() / w.sum()
    old_loss.backward()
    torch.nn.utils.clip_grad_norm_(old_model.parameters(), max_norm=10.0)
    old_opt.step()

    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    step = new_runtime.train_one_step(
        model=new_model,
        optimizer=new_opt,
        prepared=prepared,
        ids=torch.arange(len(evidence), dtype=torch.long),
    )
    assert step.loss == pytest.approx(float(old_loss.detach()), abs=1e-7, rel=0)
    _assert_state_tight(_state_clone(old_model), _state_clone(new_model))


def test_repeated_validation_is_state_safe_and_cached_equals_uncached():
    evidence, parents = _fixture()
    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    model = build_g0_brain_r10("TIER_1", seed=888, device="cpu")
    model.train()
    before = _state_clone(model)
    runtime = EvaluationRuntimeR11(enable_cuda_graph=False)

    uncached = runtime.evaluate(model, prepared, use_cache=False)
    first_cached = runtime.evaluate(model, prepared, use_cache=True)
    second_cached = runtime.evaluate(model, prepared, use_cache=True)

    assert uncached == first_cached == second_cached
    assert runtime.cache_hits == 1
    assert model.training is True
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value.detach().cpu()), key


def test_evaluation_cache_key_changes_with_policy_hash():
    evidence, parents = _fixture()
    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    model = build_g0_brain_r10("TIER_1", seed=999, device="cpu")
    runtime = EvaluationRuntimeR11(enable_cuda_graph=False)

    first = runtime.evaluate(model, prepared, use_cache=True)
    assert runtime.cache_misses == 1
    with torch.no_grad():
        next(model.parameters()).add_(1e-4)
    second = runtime.evaluate(model, prepared, use_cache=True)
    assert runtime.cache_misses == 2
    assert first["policy_hash"] != second["policy_hash"]


def test_cuda_graph_unavailable_falls_back_without_semantic_change():
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
    for key in ("loss", "direction_loss", "sizing_loss", "behavior_fingerprint"):
        assert fallback[key] == eager[key]


def test_no_amp_no_fp16_and_canonical_training_rule_fail_closed():
    with pytest.raises(RuntimeError, match="AMP_FORBIDDEN"):
        R11TrainingConfig(amp_enabled=True).validate()
    with pytest.raises(RuntimeError, match="NON_FP32_RUNTIME_DTYPE"):
        R11TrainingConfig(dtype=torch.float16).validate()
    with pytest.raises(RuntimeError, match="GENERATION_SEED_RULE_DRIFT"):
        R11TrainingConfig(generation_base_seed=1).validate()
    with pytest.raises(RuntimeError, match="ADAMW_TRAINING_RULE_DRIFT"):
        R11TrainingConfig(batch_size=256).validate()

    model = build_g0_brain_r10("TIER_1", seed=1, device="cpu").half()
    with pytest.raises(RuntimeError, match="NON_FP32_PARAMETER"):
        assert_tier1_fp32_runtime_r11(model)


def test_campaign_prepare_once_and_snapshot_recovery_no_double_consume(tmp_path):
    evidence, parents = _fixture(rows=64, groups=32)
    campaign = prepare_evidence_campaign_r11(
        train_evidence=evidence,
        validation_evidence=evidence,
        parents=parents,
        device="cpu",
    )
    assert campaign.train.host_to_device_transfers == 0
    assert campaign.validation.host_to_device_transfers == 0
    assert campaign.train.packed.data_ptr() == campaign.train.operator48.data_ptr()

    model = build_g0_brain_r10("TIER_1", seed=2026, device="cpu")
    original_hash = policy_hash_r11(model)
    guard = SnapshotConsumptionGuardR11(tmp_path, generation=4)
    receipt = {"schema": "TEST_RECEIPT", "snapshot_hash": "SNAP-A"}
    guard.commit(
        model=model,
        snapshot_hash="SNAP-A",
        generation=4,
        receipt=receipt,
    )
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    recovered = guard.recover_if_consumed(
        model=model, snapshot_hash="SNAP-A", device="cpu"
    )
    assert recovered == receipt
    assert policy_hash_r11(model) == original_hash
    with pytest.raises(RuntimeError, match="CONFLICT"):
        guard.recover_if_consumed(model=model, snapshot_hash="SNAP-B", device="cpu")
    with pytest.raises(RuntimeError, match="ALREADY_CONSUMED"):
        guard.commit(
            model=model,
            snapshot_hash="SNAP-A",
            generation=4,
            receipt=receipt,
        )
