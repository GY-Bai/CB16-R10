from __future__ import annotations

import numpy as np

from cb16_local_opt.m_series_m2_2b_advantage_weighted_selective_synthetic import target_and_multiplier_m22b
from cb16_local_opt.m_series_m2_2c_balanced_preserve_correction_synthetic import (
    REGIMES_M22C,
    balanced_target_multiplier_m22c,
)


def _fixture():
    p0=np.asarray([
        [0.70,0.20,0.10],
        [0.70,0.20,0.10],
        [0.20,0.70,0.10],
        [0.10,0.20,0.70],
        [0.60,0.30,0.10],
        [0.20,0.60,0.20],
    ],dtype=np.float32)
    mu=np.asarray([
        [0.0,0.002,-0.001],
        [0.0,0.006,-0.002],
        [0.0,0.005,0.001],
        [0.0,0.001,0.008],
        [0.010,0.002,-0.001],
        [0.0,0.009,0.001],
    ],dtype=np.float64)
    return p0,mu


def test_balanced_target_is_identical_to_aws_and_binary_selective():
    p0,mu=_fixture()
    q_bal,_,_=balanced_target_multiplier_m22c(mu_teacher=mu,base_probs=p0)
    q_aws,_,_=target_and_multiplier_m22b(objective="ADVANTAGE_WEIGHTED_SELECTIVE_CE",mu_teacher=mu,base_probs=p0)
    q_sel,_,_=target_and_multiplier_m22b(objective="SELECTIVE_CHAMPION_PRESERVE_CE",mu_teacher=mu,base_probs=p0)
    assert np.array_equal(q_bal,q_aws)
    assert np.array_equal(q_bal,q_sel)


def test_balanced_multiplier_allocates_exact_half_mass_to_each_component():
    p0,mu=_fixture()
    _,w,r=balanced_target_multiplier_m22c(mu_teacher=mu,base_probs=p0)
    g0=np.argmax(p0,axis=1); tb=np.argmax(mu,axis=1); disagreement=tb!=g0; agreement=~disagreement
    n=float(len(w))
    assert abs(float(np.sum(w[agreement])/n)-0.5)<1e-6
    assert abs(float(np.sum(w[disagreement])/n)-0.5)<1e-6
    assert abs(float(np.mean(w))-1.0)<1e-6
    assert r["component_weights"]==[0.5,0.5]
    assert r["difference_from_m2_2b"]=="GLOBAL_COMPONENT_BALANCING_ONLY"


def test_balanced_correction_weights_preserve_advantage_ordering():
    p0,mu=_fixture()
    _,w,_=balanced_target_multiplier_m22c(mu_teacher=mu,base_probs=p0)
    g0=np.argmax(p0,axis=1); tb=np.argmax(mu,axis=1); disagreement=tb!=g0
    rows=np.arange(len(mu)); adv=np.maximum(mu[rows,tb]-mu[rows,g0],0.0)
    order=np.argsort(adv[disagreement])
    assert np.all(np.diff(w[disagreement][order])>=0.0)


def test_new_seed_registry_is_exact_and_disjoint_from_prior_primary_sets():
    expected={
        "CLEAN_CONDITIONAL":(3101,3102,3103,3104,3105),
        "NOISY_TEACHER":(3201,3202,3203,3204,3205),
        "WEAK_CONDITIONAL_STRONG_MARGINAL":(3301,3302,3303,3304,3305),
        "NEAR_OPTIMAL_CHAMPION":(3401,3402,3403,3404,3405),
    }
    observed={r.name:r.seeds for r in REGIMES_M22C}
    assert observed==expected
    seeds={s for r in REGIMES_M22C for s in r.seeds}
    old=set()
    for start in (1101,1201,1301,1401,2101,2201,2301,2401):
        old.update(range(start,start+5))
    assert len(seeds)==20
    assert not (seeds&old)
