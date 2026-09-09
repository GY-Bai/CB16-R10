from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cb16_local_opt.science_feedback_diagnostics_r11 import (
    audit_training_feedback_r11,
    behavior_delta_r11,
    capture_behavior_r11,
    independent_loss_breakdown_r11,
    validate_teacher_targets_r11,
)
from cb16_local_opt.training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    EvaluationRuntimeR11,
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
    evidence_id: str
    teacher_protocol_hash: str = "R11_DIAGNOSTIC_TEST_TEACHER"
    admission: _Admission = _Admission()


def _fixture(rows: int = 64):
    rng = np.random.default_rng(20260909)
    parents = {}
    evidence = []
    for i in range(rows):
        pid = f"P{i:04d}"
        raw = rng.uniform(0.05, 1.0, size=3)
        probs = raw / raw.sum()
        parents[pid] = SimpleNamespace(
            operator48=rng.normal(size=48).astype(np.float32).tolist(),
            medium48=rng.normal(size=48).astype(np.float32).tolist(),
            account6=rng.normal(size=6).astype(np.float32).tolist(),
        )
        evidence.append(
            _Evidence(
                parent_id=pid,
                target_dependence_group_id=f"DG{i:04d}",
                direction_target_probs=tuple(float(x) for x in probs),
                requested_risk_target=float(rng.uniform(0.05, 0.95)),
                evidence_id=f"E{i:04d}",
            )
        )
    return evidence, parents


def _state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _state_delta(a, b):
    total = 0.0
    for key in a:
        d = b[key].double() - a[key].double()
        total += float(torch.sum(d * d))
    return total ** 0.5


def test_independent_formula_matches_qualified_evaluation_runtime():
    evidence, parents = _fixture()
    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    model = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")

    independent = independent_loss_breakdown_r11(model, prepared)
    runtime = EvaluationRuntimeR11(enable_cuda_graph=False)
    observed = runtime.evaluate(model, prepared, use_cache=False)

    assert independent["loss"] == pytest.approx(observed["loss"], abs=1e-7, rel=0)
    assert independent["direction_loss"] == pytest.approx(observed["direction_loss"], abs=1e-7, rel=0)
    assert independent["sizing_loss"] == pytest.approx(observed["sizing_loss"], abs=1e-7, rel=0)
    target_audit = validate_teacher_targets_r11(prepared)
    assert target_audit["targets_detached_from_autograd"] is True
    assert target_audit["independent_dependence_groups"] == 64


def test_detector_proves_parameter_to_behavior_connectivity_after_real_gradient_step():
    evidence, parents = _fixture()
    prepared = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    model = build_g0_brain_r10("TIER_1", seed=12345, device="cpu")
    runtime = TrainingRuntimeR11(device="cpu", config=R11TrainingConfig())
    optimizer = runtime.build_optimizer(model)

    before_state = _state(model)
    independent_before = independent_loss_breakdown_r11(model, prepared)
    behavior_before = capture_behavior_r11(model, prepared)
    step = runtime.train_one_step(
        model=model,
        optimizer=optimizer,
        prepared=prepared,
        ids=torch.arange(prepared.rows, dtype=torch.long),
    )
    after_state = _state(model)
    independent_after = independent_loss_breakdown_r11(model, prepared)
    behavior_after = capture_behavior_r11(model, prepared)
    delta = behavior_delta_r11(behavior_before, behavior_after)

    assert delta["behavior_fingerprint_changed"] is True
    assert delta["any_numeric_behavior_change"] is True
    assert _state_delta(before_state, after_state) > 0.0

    positive_updates = {owner: 1e-12 for owner in AUTHORIZED_GRADIENT_OWNERS_R11}
    receipt = {
        "schema": "CB16_R11_CHALLENGER_TRAINING_RECEIPT_V1",
        "amp": False,
        "dtype": "torch.float32",
        "optimizer_steps": 1,
        "parameter_l2_delta": _state_delta(before_state, after_state),
        "gradient_owner_set_last_step": sorted(step.gradient_owner_set),
        "gradient_group_norms_last_step": step.gradient_group_norms,
        "update_group_norms": positive_updates,
        "validation_before": independent_before,
        "validation_after": independent_after,
    }
    audit = audit_training_feedback_r11(
        training_receipt=receipt,
        independent_before=independent_before,
        independent_after=independent_after,
        behavior_delta=delta,
    )
    assert audit["connectivity_pass"] is True
    assert audit["scientific_market_information_qualification_claimed"] is False
    assert audit["qcrps_claimed"] is False


def test_negative_probability_target_fails_closed_even_when_sum_is_one():
    evidence, parents = _fixture(rows=4)
    good = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    packed = good.packed.clone()
    # DP columns follow 48 Operator + 48 Medium + 6 Account = offset 102.
    packed[0, 102:105] = torch.tensor([-0.1, 0.5, 0.6], dtype=torch.float32)
    bad = replace(good, packed=packed)
    with pytest.raises(RuntimeError, match="DIRECTION_TARGET_NEGATIVE"):
        validate_teacher_targets_r11(bad)


def test_missing_gradient_owner_fails_closed():
    before = {"loss": 1.0, "direction_loss": 0.8, "sizing_loss": 0.2}
    after = {"loss": 0.9, "direction_loss": 0.7, "sizing_loss": 0.2}
    owners = sorted(AUTHORIZED_GRADIENT_OWNERS_R11)
    receipt = {
        "schema": "CB16_R11_CHALLENGER_TRAINING_RECEIPT_V1",
        "amp": False,
        "dtype": "torch.float32",
        "optimizer_steps": 1,
        "parameter_l2_delta": 0.1,
        "gradient_owner_set_last_step": owners[:-1],
        "gradient_group_norms_last_step": {x: 0.1 for x in owners[:-1]},
        "update_group_norms": {x: 0.1 for x in owners},
        "validation_before": before,
        "validation_after": after,
    }
    with pytest.raises(RuntimeError, match="GRADIENT_OWNER_SET_DRIFT"):
        audit_training_feedback_r11(
            training_receipt=receipt,
            independent_before=before,
            independent_after=after,
            behavior_delta={
                "behavior_fingerprint_changed": True,
                "any_numeric_behavior_change": True,
            },
        )
