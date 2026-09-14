"""Independent hand-computed V-trace reference (never calls production to derive expected)."""

from __future__ import annotations

import math

import pytest
import torch

from cb16_local_opt.cc_vtrace_r0 import vtrace

REWARDS = [0.0, 0.0, 1.0]
DISCOUNTS = [1.0, 1.0, 0.0]
VALUES = [0.20, 0.10, 0.00]
BOOTSTRAP = 0.0
LOG_RHOS = [math.log(2.0), math.log(0.5), math.log(1.5)]
RHO_BAR = 1.0
C_BAR = 1.0
PG_RHO_BAR = 1.0


def _hand_reference_v1() -> dict[str, list[float]]:
    """Explicit scalar transcription of the frozen TODO equations."""
    count = len(REWARDS)
    values_tp1 = VALUES[1:] + [BOOTSTRAP]
    rhos = [math.exp(item) for item in LOG_RHOS]
    clipped_rhos = [min(RHO_BAR, item) for item in rhos]
    cs = [min(C_BAR, item) for item in rhos]
    deltas = [
        clipped_rhos[index] * (REWARDS[index] + DISCOUNTS[index] * values_tp1[index] - VALUES[index])
        for index in range(count)
    ]
    vs = [0.0] * count
    for index in range(count - 1, -1, -1):
        next_vs = BOOTSTRAP if index == count - 1 else vs[index + 1]
        vs[index] = (
            VALUES[index]
            + deltas[index]
            + DISCOUNTS[index] * cs[index] * (next_vs - values_tp1[index])
        )
    advantages = [
        min(PG_RHO_BAR, rhos[index])
        * (REWARDS[index] + DISCOUNTS[index] * values_tp1[index] - VALUES[index])
        for index in range(count)
    ]
    # pg advantage must use the returned value estimates, not the raw values
    advantages = [
        min(PG_RHO_BAR, rhos[index])
        * (REWARDS[index] + DISCOUNTS[index] * (BOOTSTRAP if index == count - 1 else vs[index + 1]) - VALUES[index])
        for index in range(count)
    ]
    return {"rhos": rhos, "clipped_rhos": clipped_rhos, "cs": cs, "vs": vs, "advantages": advantages}


def test_hand_reference_values_are_the_preregistered_numbers():
    reference = _hand_reference_v1()
    assert reference["rhos"] == pytest.approx([2.0, 0.5, 1.5], abs=1e-12)
    assert reference["clipped_rhos"] == pytest.approx([1.0, 0.5, 1.0], abs=1e-12)
    assert reference["cs"] == pytest.approx([1.0, 0.5, 1.0], abs=1e-12)
    assert reference["vs"] == pytest.approx([0.55, 0.55, 1.0], abs=1e-12)
    assert reference["advantages"] == pytest.approx([0.35, 0.45, 1.0], abs=1e-12)


def test_production_vtrace_matches_hand_reference_within_frozen_tolerance():
    reference = _hand_reference_v1()
    result = vtrace(
        rewards=torch.tensor(REWARDS, dtype=torch.float64),
        values=torch.tensor(VALUES, dtype=torch.float64),
        bootstrap_value=torch.tensor(BOOTSTRAP, dtype=torch.float64),
        log_pi=torch.tensor(LOG_RHOS, dtype=torch.float64),
        log_mu=torch.zeros(3, dtype=torch.float64),
        discounts=torch.tensor(DISCOUNTS, dtype=torch.float64),
        rho_bar=float(RHO_BAR),
        c_bar=float(C_BAR),
        pg_rho_bar=float(PG_RHO_BAR),
    )
    assert torch.allclose(result.rhos, torch.tensor(reference["rhos"], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(result.clipped_rhos, torch.tensor(reference["clipped_rhos"], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(result.cs, torch.tensor(reference["cs"], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(result.vs, torch.tensor(reference["vs"], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(result.pg_advantages, torch.tensor(reference["advantages"], dtype=torch.float64), atol=1e-6)


def test_sign_reversal_sentinel_differs_from_correct_direction():
    """A reversed log_ratio must not silently reproduce the correct answer."""
    correct = vtrace(
        rewards=torch.tensor(REWARDS, dtype=torch.float64),
        values=torch.tensor(VALUES, dtype=torch.float64),
        bootstrap_value=torch.tensor(BOOTSTRAP, dtype=torch.float64),
        log_pi=torch.tensor(LOG_RHOS, dtype=torch.float64),
        log_mu=torch.zeros(3, dtype=torch.float64),
        discounts=torch.tensor(DISCOUNTS, dtype=torch.float64),
    )
    reversed_ratio = vtrace(
        rewards=torch.tensor(REWARDS, dtype=torch.float64),
        values=torch.tensor(VALUES, dtype=torch.float64),
        bootstrap_value=torch.tensor(BOOTSTRAP, dtype=torch.float64),
        log_pi=torch.zeros(3, dtype=torch.float64),
        log_mu=torch.tensor(LOG_RHOS, dtype=torch.float64),
        discounts=torch.tensor(DISCOUNTS, dtype=torch.float64),
    )
    assert not torch.allclose(correct.vs, reversed_ratio.vs)
    assert not torch.allclose(correct.pg_advantages, reversed_ratio.pg_advantages)
