from __future__ import annotations

import copy

import pytest
import torch

from cb16_local_opt.distributional_student_shadow_r3 import truncated_quantile_w1_loss_r3
from cb16_local_opt.distributional_student_shared_core_shadow_r31 import (
    DistributionalStudentSharedCoreHeadR31,
    assert_r31_gradient_ownership,
    head_parameter_report_r31,
    make_disposable_shared_core_student_r31,
    shared_core_parameter_report_r31,
    state_update_audit_r31,
)
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10


LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def monotone_teacher(rows: int, actions: int = 9, quantiles: int = 7) -> torch.Tensor:
    base = torch.linspace(-0.01, 0.01, steps=rows * actions).view(rows, actions, 1)
    gaps = torch.linspace(0.0, 0.012, steps=quantiles).view(1, 1, quantiles)
    return (base + gaps).float()


def test_r31_head_is_monotone_and_allows_shared_input_gradient():
    torch.manual_seed(13)
    head = DistributionalStudentSharedCoreHeadR31(shared_dim=256, action_count=9, quantile_count=7)
    shared = torch.randn(5, 256, requires_grad=True)
    q = head(shared)
    assert q.shape == (5, 9, 7)
    assert torch.all(q[..., 1:] >= q[..., :-1])
    q.sum().backward()
    assert shared.grad is not None
    assert torch.any(shared.grad != 0)


def test_r31_disposable_model_authorizes_only_shared_core():
    production = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    disposable = make_disposable_shared_core_student_r31(production)
    report = shared_core_parameter_report_r31(disposable)
    assert report["model_parameter_count"] == 189_052
    assert report["authorized_shared_core_parameter_count"] == 107_008
    trainable = [name for name, p in disposable.named_parameters() if p.requires_grad]
    assert trainable
    assert all(name.startswith("shared_core") for name in trainable)
    assert all(p.requires_grad for p in disposable.shared_core.parameters())
    assert all(
        not p.requires_grad
        for name, p in disposable.named_parameters()
        if not name.startswith("shared_core")
    )


def test_r31_head_parameter_count_matches_r3_head_shape():
    head = DistributionalStudentSharedCoreHeadR31(shared_dim=256, action_count=9, quantile_count=7)
    report = head_parameter_report_r31(head)
    assert report["parameter_count"] == 20_543
    assert report["trainable_parameter_count"] == 20_543
    assert report["parameter_bytes"] == 82_172


def test_r31_gradient_ownership_and_update_are_isolated():
    torch.manual_seed(17)
    production = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    production_before = copy.deepcopy(production.state_dict())
    disposable = make_disposable_shared_core_student_r31(production)
    disposable_before = copy.deepcopy(disposable.state_dict())
    head = DistributionalStudentSharedCoreHeadR31(shared_dim=256, action_count=9, quantile_count=7)

    op = torch.randn(8, 48)
    med = torch.randn(8, 48)
    acc = torch.randn(8, 6)
    teacher = monotone_teacher(8)
    weight = torch.ones(8)

    optimizer = torch.optim.AdamW(
        list(disposable.shared_core.parameters()) + list(head.parameters()),
        lr=1e-3,
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    shared = disposable(op, med, acc)["shared"]
    pred = head(shared)
    loss = truncated_quantile_w1_loss_r3(pred, teacher, LEVELS, weight)
    loss.backward()
    audit = assert_r31_gradient_ownership(
        production_model=production,
        disposable_model=disposable,
        head=head,
    )
    optimizer.step()

    assert audit["production_student_gradient_parameter_count"] == 0
    assert audit["shared_core_gradient_parameter_tensors"] == 4
    assert audit["distributional_head_gradient_parameter_tensors"] == 4
    assert audit["forbidden_disposable_gradient_parameter_count"] == 0

    update = state_update_audit_r31(disposable_before, disposable.state_dict())
    assert update["shared_core_parameter_l2_delta"] > 0.0
    assert update["forbidden_parameter_update_count"] == 0
    for name, tensor in production.state_dict().items():
        assert torch.equal(tensor, production_before[name])


def test_r31_gradient_audit_fails_closed_on_forbidden_disposable_gradient():
    torch.manual_seed(19)
    production = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    disposable = make_disposable_shared_core_student_r31(production)
    disposable.direction_out.weight.requires_grad_(True)
    head = DistributionalStudentSharedCoreHeadR31(shared_dim=256, action_count=9, quantile_count=7)
    op = torch.randn(4, 48)
    med = torch.randn(4, 48)
    acc = torch.randn(4, 6)
    out = disposable(op, med, acc)
    # Force the forbidden Direction Head into the graph in addition to the R3.1 objective.
    loss = head(out["shared"]).sum() + out["direction_logits"].sum()
    loss.backward()
    with pytest.raises(RuntimeError, match="FORBIDDEN_DISPOSABLE_GRADIENT"):
        assert_r31_gradient_ownership(
            production_model=production,
            disposable_model=disposable,
            head=head,
        )
