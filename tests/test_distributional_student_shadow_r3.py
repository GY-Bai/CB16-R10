from __future__ import annotations

from dataclasses import dataclass
import copy

import pytest
import torch
from torch import nn

from cb16_local_opt import distributional_student_shadow_r3 as r3


@dataclass(frozen=True)
class Admission:
    admitted: bool = True


@dataclass(frozen=True)
class Law:
    direction: int
    requested_risk: float
    quantile_levels: tuple[float, ...]
    quantiles: tuple[float, ...]


@dataclass(frozen=True)
class Evidence:
    parent_id: str
    target_dependence_group_id: str
    action_laws: tuple[Law, ...]
    admission: Admission = Admission()
    evidence_id: str = "E"
    teacher_protocol_hash: str = "T"

    @property
    def content_hash(self) -> str:
        return self.parent_id + ":content"


@dataclass(frozen=True)
class Parent:
    operator48: tuple[float, ...]
    medium48: tuple[float, ...]
    account6: tuple[float, ...]


LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
GRID = ((1, 0.0), (0, 0.25), (2, 0.25))


def make_laws(offset: float = 0.0):
    out = []
    for j, (direction, risk) in enumerate(GRID):
        base = offset + j * 0.01
        qs = tuple(base + k * 0.001 for k in range(len(LEVELS)))
        out.append(Law(direction, risk, LEVELS, qs))
    return tuple(out)


def make_world():
    evidence = [
        Evidence("P0", "G0", make_laws(0.0), evidence_id="E0"),
        Evidence("P1", "G0", make_laws(0.002), evidence_id="E1"),
        Evidence("P2", "G1", make_laws(-0.001), evidence_id="E2"),
    ]
    parents = {}
    for i, e in enumerate(evidence):
        parents[e.parent_id] = Parent(
            tuple(float(i) / 10 + j / 1000 for j in range(48)),
            tuple(float(i) / 20 + j / 1200 for j in range(48)),
            tuple(float(i) / 30 + j / 100 for j in range(6)),
        )
    return evidence, parents


class FakeProductionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(48 + 48 + 6, 256)
        self.direction = nn.Linear(256, 3)
        self.risk = nn.Linear(256, 1)

    def forward(self, operator48, medium48, account6):
        h = torch.tanh(self.proj(torch.cat([operator48, medium48, account6], dim=-1)))
        logits = self.direction(h)
        return {
            "shared": h,
            "direction_logits": logits,
            "direction_probs": torch.softmax(logits, dim=-1),
            "requested_risk_raw": torch.sigmoid(self.risk(h)).squeeze(-1),
        }


def test_batch_consumes_action_law_quantiles_and_equal_group_mass():
    evidence, parents = make_world()
    batch = r3.DistributionalEvidenceBatchR3.from_evidence(evidence, parents, device="cpu")
    batch.validate()
    assert batch.rows == 3
    assert batch.action_grid == GRID
    assert batch.quantile_levels == LEVELS
    assert batch.teacher_quantiles.shape == (3, 3, 7)
    assert batch.group_weight[0].item() == pytest.approx(batch.group_weight[1].item())
    assert batch.group_weight[2].item() == pytest.approx(2.0 * batch.group_weight[0].item())
    assert batch.group_weight[:2].sum().item() == pytest.approx(batch.group_weight[2].item())
    assert not batch.teacher_quantiles.requires_grad


def test_batch_fails_closed_on_teacher_quantile_crossing():
    evidence, parents = make_world()
    bad = list(evidence)
    laws = list(bad[0].action_laws)
    qs = list(laws[0].quantiles)
    qs[3] = qs[2] - 1.0
    laws[0] = Law(laws[0].direction, laws[0].requested_risk, LEVELS, tuple(qs))
    bad[0] = Evidence("P0", "G0", tuple(laws), evidence_id="E0")
    with pytest.raises(RuntimeError, match="TEACHER_QUANTILE_CROSSING"):
        r3.DistributionalEvidenceBatchR3.from_evidence(bad, parents, device="cpu")


def test_batch_fails_closed_on_action_grid_drift():
    evidence, parents = make_world()
    bad = list(evidence)
    laws = list(bad[1].action_laws)
    laws[-1] = Law(2, 0.50, LEVELS, laws[-1].quantiles)
    bad[1] = Evidence("P1", "G0", tuple(laws), evidence_id="E1")
    with pytest.raises(RuntimeError, match="ACTION_GRID_DRIFT"):
        r3.DistributionalEvidenceBatchR3.from_evidence(bad, parents, device="cpu")


def test_shadow_head_quantiles_monotone_by_construction():
    torch.manual_seed(7)
    head = r3.DistributionalStudentShadowHeadR3(shared_dim=256, action_count=9, quantile_count=7)
    q = head(torch.randn(11, 256, requires_grad=True))
    assert q.shape == (11, 9, 7)
    assert torch.all(q[..., 1:] >= q[..., :-1])


def test_truncated_w1_is_zero_for_exact_teacher_quantiles():
    q = torch.tensor([[[0.0, 0.1, 0.2], [0.0, 0.2, 0.4]]], dtype=torch.float32)
    w = torch.ones(1, dtype=torch.float32)
    loss = r3.truncated_quantile_w1_loss_r3(q, q.detach(), (0.1, 0.5, 0.9), w)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_shadow_training_has_head_only_gradient_and_leaves_production_state_unchanged():
    torch.manual_seed(11)
    evidence, parents = make_world()
    batch = r3.DistributionalEvidenceBatchR3.from_evidence(evidence, parents, device="cpu")
    production = FakeProductionModel()
    production_before = copy.deepcopy(production.state_dict())
    shared = r3.shared_representation_r3(production, batch)
    assert not shared.requires_grad

    head = r3.DistributionalStudentShadowHeadR3(
        shared_dim=shared.shape[1], action_count=batch.action_count, quantile_count=batch.quantile_count
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    pred = head(shared)
    loss = r3.truncated_quantile_w1_loss_r3(
        pred, batch.teacher_quantiles, batch.quantile_levels, batch.group_weight
    )
    loss.backward()
    audit = r3.assert_shadow_gradient_ownership_r3(production_model=production, shadow_head=head)
    optimizer.step()

    assert audit["production_student_gradient_parameter_count"] == 0
    assert audit["shadow_gradient_parameter_tensors"] > 0
    for name, tensor in production.state_dict().items():
        assert torch.equal(tensor, production_before[name])


def test_shadow_parameter_report_is_small_and_fp32():
    head = r3.DistributionalStudentShadowHeadR3(shared_dim=256, action_count=9, quantile_count=7)
    report = r3.shadow_parameter_report_r3(head)
    assert report["parameter_count"] == 20_543
    assert report["trainable_parameter_count"] == 20_543
    assert report["parameter_bytes"] == 82_172
    assert report["dtype"] == "float32"
