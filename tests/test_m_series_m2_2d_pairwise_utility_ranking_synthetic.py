from __future__ import annotations

import numpy as np
import pytest
import torch

from cb16_local_opt.m_series_m2_2d_pairwise_utility_ranking_synthetic import (
    DIRECTION_PAIRS,
    REGIMES_M22D,
    pairwise_gap_weighted_loss_m22d,
    pairwise_surface_m22d,
)


def test_pairwise_surface_uses_all_three_pairs_with_expected_signs():
    mu=np.asarray([
        [0.006,0.002,-0.001],
        [-0.003,0.004,0.001],
    ],dtype=np.float64)
    signs,weights,receipt=pairwise_surface_m22d(mu)
    assert DIRECTION_PAIRS==((0,1),(0,2),(1,2))
    assert receipt["direction_pairs"]==[[0,1],[0,2],[1,2]]
    assert np.array_equal(signs,np.asarray([[1,1,1],[-1,-1,1]],dtype=np.float32))
    assert weights.shape==(2,3)
    assert receipt["zero_teacher_gap_count"]==0


def test_pairwise_weights_are_absolute_gap_proportional_and_globally_mean_one():
    mu=np.asarray([
        [0.006,0.002,-0.001],
        [-0.003,0.004,0.001],
    ],dtype=np.float64)
    _,weights,receipt=pairwise_surface_m22d(mu)
    gaps=np.asarray([[0.004,0.007,0.003],[0.007,0.004,0.003]],dtype=np.float64)
    expected=gaps/np.mean(gaps)
    assert np.allclose(weights,expected.astype(np.float32),atol=1e-7,rtol=0.0)
    assert abs(float(np.mean(weights))-1.0)<1e-6
    assert abs(float(receipt["mean_pair_weight"])-1.0)<1e-12
    assert receipt["probability_target"]=="NONE"
    assert receipt["temperature_threshold_or_margin_hyperparameter"]=="NONE"


def test_pairwise_loss_rewards_teacher_consistent_logit_ordering():
    signs=torch.tensor([[1.0,1.0,1.0]],dtype=torch.float32)
    weights=torch.ones((1,3),dtype=torch.float32)
    good=torch.tensor([[2.0,1.0,0.0]],dtype=torch.float32)
    bad=torch.tensor([[0.0,1.0,2.0]],dtype=torch.float32)
    assert float(pairwise_gap_weighted_loss_m22d(good,signs,weights)) < float(pairwise_gap_weighted_loss_m22d(bad,signs,weights))


def test_exact_teacher_gap_tie_fails_closed():
    mu=np.asarray([[0.001,0.001,-0.002]],dtype=np.float64)
    with pytest.raises(RuntimeError,match="M22D_EXACT_ZERO_TEACHER_GAP"):
        pairwise_surface_m22d(mu)


def test_new_seed_registry_is_exact_and_disjoint_from_all_prior_primary_sets():
    expected={
        "CLEAN_CONDITIONAL":(4101,4102,4103,4104,4105),
        "NOISY_TEACHER":(4201,4202,4203,4204,4205),
        "WEAK_CONDITIONAL_STRONG_MARGINAL":(4301,4302,4303,4304,4305),
        "NEAR_OPTIMAL_CHAMPION":(4401,4402,4403,4404,4405),
    }
    observed={r.name:r.seeds for r in REGIMES_M22D}
    assert observed==expected
    seeds={s for r in REGIMES_M22D for s in r.seeds}
    old=set()
    for start in (1101,1201,1301,1401,2101,2201,2301,2401,3101,3201,3301,3401):
        old.update(range(start,start+5))
    assert len(seeds)==20
    assert not (seeds&old)
