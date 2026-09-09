from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cb16_local_opt.distributional_student_gradient_stability_r32 import (
    R32_SEEDS,
    shared_core_gradient_geometry_r32,
    summarize_seed_stability_r32,
)
from cb16_local_opt.distributional_student_shared_core_shadow_r31 import (
    DistributionalStudentSharedCoreHeadR31,
    make_disposable_shared_core_student_r31,
)


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.operator_stem = nn.Linear(48, 2)
        self.medium_stem = nn.Linear(48, 2)
        self.account_stem = nn.Linear(6, 2)
        self.shared_core = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 4), nn.Tanh())
        self.direction_head = nn.Linear(4, 3)
        self.sizing_head = nn.Linear(4, 1)

    def forward(self, operator48, medium48, account6):
        x = torch.cat(
            [self.operator_stem(operator48), self.medium_stem(medium48), self.account_stem(account6)],
            dim=-1,
        )
        shared = self.shared_core(x)
        logits = self.direction_head(shared)
        return {
            "shared": shared,
            "direction_logits": logits,
            "direction_probs": torch.softmax(logits, dim=-1),
            "requested_risk_raw": torch.sigmoid(self.sizing_head(shared)).squeeze(-1),
        }


def batches():
    torch.manual_seed(9)
    n = 7
    op = torch.randn(n, 48)
    med = torch.randn(n, 48)
    acc = torch.randn(n, 6)
    target_p = torch.softmax(torch.randn(n, 3), dim=-1)
    risk = torch.sigmoid(torch.randn(n))
    w = torch.ones(n)
    teacher_q = torch.sort(torch.randn(n, 3, 3) * 0.05, dim=-1).values
    dist = SimpleNamespace(
        operator48=op,
        medium48=med,
        account6=acc,
        teacher_quantiles=teacher_q,
        quantile_levels=(0.1, 0.5, 0.9),
        group_weight=w,
    )
    canonical = SimpleNamespace(
        operator48=op,
        medium48=med,
        account6=acc,
        direction_target_probs=target_p,
        requested_risk_target=risk,
        group_weight=w,
    )
    return dist, canonical


def test_seed_set_is_fixed_and_ordered():
    assert R32_SEEDS == (31337, 271828, 314159, 161803, 424242)


def test_gradient_geometry_is_finite_and_does_not_store_parameter_grads():
    torch.manual_seed(3)
    production = TinyStudent()
    disposable = make_disposable_shared_core_student_r31(production)
    head = DistributionalStudentSharedCoreHeadR31(shared_dim=4, action_count=3, quantile_count=3)
    dist, canonical = batches()
    report = shared_core_gradient_geometry_r32(
        disposable_model=disposable,
        distributional_head=head,
        distributional_batch=dist,
        canonical_prepared_batch=canonical,
    )
    assert report["parameter_update_performed"] is False
    assert report["shared_core_parameter_count"] > 0
    for key in (
        "distributional_vs_canonical_total",
        "distributional_vs_direction",
        "distributional_vs_sizing",
    ):
        assert -1.000001 <= report[key]["cosine_similarity"] <= 1.000001
        assert report[key]["distributional_l2_norm"] > 0.0
        assert report[key]["canonical_l2_norm"] > 0.0
    assert all(p.grad is None for p in disposable.parameters())
    assert all(p.grad is None for p in head.parameters())


def row(seed: int, shared_delta=-0.01, decision_delta=-0.02, cosine=0.2):
    return {
        "seed": seed,
        "shared_minus_head_validation_distributional_loss": shared_delta,
        "canonical_decision_total_delta": decision_delta,
        "canonical_direction_loss_delta": decision_delta * 0.8,
        "canonical_sizing_loss_delta": decision_delta * 0.2,
        "gradient_geometry": {
            "distributional_vs_canonical_total": {"cosine_similarity": cosine},
            "distributional_vs_direction": {"cosine_similarity": cosine + 0.1},
            "distributional_vs_sizing": {"cosine_similarity": cosine - 0.1},
        },
        "hypothetical_direction_changed_rows": 5,
        "mean_abs_requested_risk_delta": 0.1,
    }


def test_seed_summary_requires_all_preregistered_seeds_and_reports_pattern():
    rows = [row(seed) for seed in R32_SEEDS]
    out = summarize_seed_stability_r32(rows)
    assert out["seed_count"] == 5
    assert out["shared_core_better_distributional_seed_count"] == 5
    assert out["canonical_total_loss_lower_seed_count"] == 5
    assert out["architecture_pattern"] == "ALL_SEEDS_SHARED_BETTER_DISTRIBUTIONAL_AND_LOWER_CANONICAL_DECISION_LOSS"


def test_seed_summary_fails_closed_on_missing_or_reordered_seed():
    rows = [row(seed) for seed in R32_SEEDS]
    with pytest.raises(RuntimeError, match="SEED_SET_OR_ORDER_DRIFT"):
        summarize_seed_stability_r32(list(reversed(rows)))
    with pytest.raises(RuntimeError, match="SEED_SET_OR_ORDER_DRIFT|SEED_COUNT_DRIFT"):
        summarize_seed_stability_r32(rows[:-1])
