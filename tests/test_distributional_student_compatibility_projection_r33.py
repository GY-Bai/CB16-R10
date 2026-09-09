from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from cb16_local_opt.distributional_student_compatibility_projection_r33 import (
    ManualAdamDirectionR33,
    R33_LR,
    apply_adam_direction_r33,
    compatibility_project_adam_direction_r33,
    summarize_projection_steps_r33,
)


def test_manual_adam_direction_matches_torch_adamw_weight_decay_zero():
    torch.manual_seed(123)
    left = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))
    right = copy.deepcopy(left)
    lp = tuple(left.parameters())
    rp = tuple(right.parameters())
    opt = torch.optim.AdamW(lp, lr=R33_LR, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    manual = ManualAdamDirectionR33(rp)

    for step in range(1, 8):
        torch.manual_seed(1000 + step)
        grads = [torch.randn_like(p) * (0.1 + step / 100.0) for p in lp]
        opt.zero_grad(set_to_none=True)
        for p, g in zip(lp, grads):
            p.grad = g.clone()
        opt.step()

        directions = manual.directions([g.clone() for g in grads])
        apply_adam_direction_r33(rp, directions, lr=R33_LR)
        for a, b in zip(lp, rp):
            assert torch.allclose(a, b, rtol=2e-6, atol=2e-7)


def test_projection_removes_only_conflicting_component():
    raw = (torch.tensor([-2.0, 3.0], dtype=torch.float32),)
    guard = (torch.tensor([4.0, 0.0], dtype=torch.float32),)
    safe, report = compatibility_project_adam_direction_r33(raw, guard)
    assert report["projected"] is True
    assert report["dot_before"] < 0.0
    assert report["dot_after"] == pytest.approx(0.0, abs=1e-6)
    assert safe[0][0].item() == pytest.approx(0.0, abs=1e-6)
    assert safe[0][1].item() == pytest.approx(3.0, abs=1e-6)


def test_projection_is_identity_when_nonconflicting():
    raw = (torch.tensor([2.0, 3.0], dtype=torch.float32),)
    guard = (torch.tensor([4.0, 0.0], dtype=torch.float32),)
    safe, report = compatibility_project_adam_direction_r33(raw, guard)
    assert report["projected"] is False
    assert torch.equal(safe[0], raw[0])
    assert report["dot_before"] > 0.0
    assert report["dot_after"] > 0.0


def test_projection_summary_requires_exact_step_budget():
    rows = [
        {"projected": i % 2 == 0, "dot_before": -1.0 if i % 2 == 0 else 1.0, "dot_after": 0.0 if i % 2 == 0 else 1.0}
        for i in range(32)
    ]
    out = summarize_projection_steps_r33(rows)
    assert out["step_count"] == 32
    assert out["projected_step_count"] == 16
    assert out["all_safe_directions_nonconflicting_within_tolerance"] is True
    with pytest.raises(RuntimeError, match="PROJECTION_STEP_COUNT_DRIFT"):
        summarize_projection_steps_r33(rows[:-1])
