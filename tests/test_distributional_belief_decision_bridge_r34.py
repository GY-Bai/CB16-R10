from __future__ import annotations

import copy

import torch

from cb16_local_opt.distributional_belief_decision_bridge_r34 import (
    DistributionalBeliefDecisionBridgeR34,
    apply_bridge_r34,
    assert_bridge_gradient_ownership_r34,
    assert_zero_residual_equivalence_r34,
    bridge_parameter_report_r34,
    clone_bridge_r34,
    group_block_shuffle_r34,
)


class _TinyProduction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(2, 2))


class _TinyHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))


def _base(rows: int = 4):
    return {
        "direction_logits": torch.tensor([[0.2, -0.1, 0.3]]).repeat(rows, 1),
        "requested_risk_raw": torch.tensor([[0.35]]).repeat(rows, 1),
    }


def test_zero_residual_is_production_equivalent_and_report_stable():
    torch.manual_seed(7)
    bridge = DistributionalBeliefDecisionBridgeR34(shared_dim=5, belief_dim=6)
    shared = torch.randn(4, 5)
    belief = torch.randn(4, 6)
    base = _base()
    assert_zero_residual_equivalence_r34(bridge, shared, belief, base)
    report = bridge_parameter_report_r34(bridge)
    assert report["parameter_count"] == report["trainable_parameter_count"]
    assert report["authorized_owner"] == "Disposable Belief-to-Decision Residual Bridge"


def test_clone_is_byte_identical_then_independent():
    torch.manual_seed(11)
    a = DistributionalBeliefDecisionBridgeR34(shared_dim=5, belief_dim=6)
    b = clone_bridge_r34(a)
    for x, y in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(x, y)
    with torch.no_grad():
        next(b.parameters()).add_(1.0)
    assert any(not torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_group_block_shuffle_rotates_whole_groups_without_fixed_points():
    belief = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    gids = ("g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3")
    out, receipt = group_block_shuffle_r34(belief, gids, shift=1)
    assert receipt["group_count"] == 4
    assert receipt["rows_per_group"] == 2
    assert receipt["fixed_point_groups"] == 0
    assert torch.equal(out[0:2], belief[2:4])
    assert torch.equal(out[2:4], belief[4:6])
    assert torch.equal(out[4:6], belief[6:8])
    assert torch.equal(out[6:8], belief[0:2])


def test_group_shuffle_rejects_nonuniform_groups():
    belief = torch.randn(5, 3)
    gids = ("g0", "g0", "g1", "g1", "g1")
    try:
        group_block_shuffle_r34(belief, gids, shift=1)
    except RuntimeError as exc:
        assert "NONUNIFORM_GROUP_SIZE" in str(exc)
    else:
        raise AssertionError("expected fail-closed nonuniform group rejection")


def test_bridge_gradient_ownership_keeps_production_and_belief_head_frozen():
    torch.manual_seed(13)
    production = _TinyProduction()
    head = _TinyHead()
    for p in production.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(False)
    bridge = DistributionalBeliefDecisionBridgeR34(shared_dim=5, belief_dim=6)
    shared = torch.randn(4, 5)
    belief = torch.randn(4, 6)
    base = _base()
    out = apply_bridge_r34(base, bridge(shared, belief))
    loss = out["direction_logits"].pow(2).mean() + out["requested_risk_raw"].pow(2).mean()
    loss.backward()
    audit = assert_bridge_gradient_ownership_r34(
        production_model=production,
        distributional_head=head,
        bridge=bridge,
    )
    assert audit["production_student_gradient_parameter_count"] == 0
    assert audit["distributional_head_gradient_parameter_count"] == 0
    assert audit["bridge_gradient_parameter_tensors"] > 0


def test_aligned_and_control_can_start_from_identical_parameters_and_diverge_only_by_input():
    torch.manual_seed(17)
    aligned = DistributionalBeliefDecisionBridgeR34(shared_dim=5, belief_dim=6)
    shuffled = clone_bridge_r34(aligned)
    shared = torch.randn(4, 5)
    belief = torch.randn(4, 6)
    control = torch.roll(belief, 1, 0)
    base = _base()
    opt_a = torch.optim.AdamW(aligned.parameters(), lr=2e-3, weight_decay=0.0)
    opt_b = torch.optim.AdamW(shuffled.parameters(), lr=2e-3, weight_decay=0.0)
    target = torch.tensor([0, 1, 2, 0])
    for model, inp, opt in ((aligned, belief, opt_a), (shuffled, control, opt_b)):
        opt.zero_grad(set_to_none=True)
        out = apply_bridge_r34(base, model(shared, inp))
        loss = torch.nn.functional.cross_entropy(out["direction_logits"], target)
        loss.backward(); opt.step()
    assert any(not torch.equal(x, y) for x, y in zip(aligned.state_dict().values(), shuffled.state_dict().values()))
