from __future__ import annotations

import numpy as np
import torch

from cb16_local_opt.m_series_m2_2e_direct_champion_relative_regret_synthetic import (
    REGIMES_M22E,
    direct_expected_advantage_loss_m22e,
    direct_positive_advantage_surface_m22e,
)


def test_positive_advantage_is_measured_against_frozen_champion_action():
    base_logits=np.asarray([[3.0,1.0,0.0],[0.0,2.0,1.0]],dtype=np.float32)
    mu=np.asarray([[0.010,0.014,0.007],[0.009,0.008,0.011]],dtype=np.float64)
    adv,r=direct_positive_advantage_surface_m22e(mu_teacher=mu,base_logits=base_logits)
    # row 0 Champion=0: only action 1 beats it; row 1 Champion=1: actions 0 and 2 beat it.
    assert adv[0,0]==0.0 and adv[0,1]>0.0 and adv[0,2]==0.0
    assert adv[1,0]>0.0 and adv[1,1]==0.0 and adv[1,2]>0.0
    assert r["strictly_positive_action_entries"]==3
    assert r["teacher_probability_target"]=="NONE"
    assert r["pairwise_ranking_target"]=="NONE"


def test_positive_advantage_normalization_has_mean_one_over_positive_entries():
    base_logits=np.asarray([[3.0,1.0,0.0],[0.0,2.0,1.0]],dtype=np.float32)
    mu=np.asarray([[0.010,0.014,0.007],[0.009,0.008,0.011]],dtype=np.float64)
    adv,r=direct_positive_advantage_surface_m22e(mu_teacher=mu,base_logits=base_logits)
    assert abs(float(np.mean(adv[adv>0.0]))-1.0)<1e-6
    assert abs(float(r["mean_normalized_positive_advantage"])-1.0)<1e-12
    assert r["temperature_threshold_margin_or_tunable_scale"]=="NONE"


def test_teacher_champion_agreement_row_has_exact_zero_direct_gradient_in_isolation():
    # Frozen Champion action 0; Teacher also ranks action 0 best, so all A+ are exactly zero.
    base_logits_np=np.asarray([[2.0,1.0,0.0],[0.0,2.0,1.0]],dtype=np.float32)
    mu=np.asarray([[0.010,0.004,-0.002],[0.0,0.002,0.006]],dtype=np.float64)
    adv,_=direct_positive_advantage_surface_m22e(mu_teacher=mu,base_logits=base_logits_np)
    assert np.array_equal(adv[0],np.zeros(3,dtype=np.float32))
    logits=torch.tensor([[2.0,1.0,0.0]],dtype=torch.float32,requires_grad=True)
    a=torch.from_numpy(adv[:1])
    loss=direct_expected_advantage_loss_m22e(logits,a)
    loss.backward()
    assert torch.equal(logits.grad,torch.zeros_like(logits.grad))


def test_direct_loss_rewards_probability_mass_on_higher_positive_champion_relative_advantage():
    adv=torch.tensor([[0.0,0.5,1.5]],dtype=torch.float32)
    favor_high=torch.tensor([[0.0,0.0,2.0]],dtype=torch.float32)
    favor_low=torch.tensor([[0.0,2.0,0.0]],dtype=torch.float32)
    assert float(direct_expected_advantage_loss_m22e(favor_high,adv)) < float(direct_expected_advantage_loss_m22e(favor_low,adv))


def test_new_seed_registry_is_exact_and_disjoint_from_all_prior_primary_sets():
    expected={
        "CLEAN_CONDITIONAL":(5101,5102,5103,5104,5105),
        "NOISY_TEACHER":(5201,5202,5203,5204,5205),
        "WEAK_CONDITIONAL_STRONG_MARGINAL":(5301,5302,5303,5304,5305),
        "NEAR_OPTIMAL_CHAMPION":(5401,5402,5403,5404,5405),
    }
    observed={r.name:r.seeds for r in REGIMES_M22E}
    assert observed==expected
    seeds={s for r in REGIMES_M22E for s in r.seeds}
    old=set()
    for start in (1101,1201,1301,1401,2101,2201,2301,2401,3101,3201,3301,3401,4101,4201,4301,4401):
        old.update(range(start,start+5))
    assert len(seeds)==20
    assert not (seeds&old)
