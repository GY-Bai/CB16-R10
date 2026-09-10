from __future__ import annotations

import numpy as np

from cb16_local_opt.m_series_m2_2b_advantage_weighted_selective_synthetic import (
    REGIMES_M22B,
    target_and_multiplier_m22b,
)


def test_aws_target_is_byte_identical_to_binary_selective_target():
    p0=np.asarray([
        [0.70,0.20,0.10],
        [0.20,0.60,0.20],
        [0.50,0.30,0.20],
        [0.10,0.20,0.70],
    ],dtype=np.float32)
    mu=np.asarray([
        [0.00,0.004,-0.002],
        [0.00,0.005,0.002],
        [0.001,0.006,0.000],
        [0.008,0.002,0.004],
    ],dtype=np.float64)
    qs,ms,_=target_and_multiplier_m22b(objective="SELECTIVE_CHAMPION_PRESERVE_CE",mu_teacher=mu,base_probs=p0)
    qw,mw,_=target_and_multiplier_m22b(objective="ADVANTAGE_WEIGHTED_SELECTIVE_CE",mu_teacher=mu,base_probs=p0)
    assert np.array_equal(qs,qw)
    assert np.array_equal(ms,np.ones_like(ms))
    assert not np.array_equal(mw,np.ones_like(mw))


def test_aws_disagreement_multiplier_is_positive_and_mean_one():
    p0=np.asarray([
        [0.7,0.2,0.1],
        [0.7,0.2,0.1],
        [0.2,0.7,0.1],
        [0.1,0.2,0.7],
    ],dtype=np.float32)
    mu=np.asarray([
        [0.0,0.002,-0.001],
        [0.0,0.006,-0.002],
        [0.0,0.005,0.001],
        [0.0,0.001,0.008],
    ],dtype=np.float64)
    _,w,r=target_and_multiplier_m22b(objective="ADVANTAGE_WEIGHTED_SELECTIVE_CE",mu_teacher=mu,base_probs=p0)
    g0=np.argmax(p0,axis=1); tb=np.argmax(mu,axis=1); disagree=tb!=g0
    assert np.all(w>0.0)
    assert abs(float(np.mean(w[disagree]))-1.0)<1e-6
    assert np.all(w[~disagree]==1.0)
    # The larger Teacher-implied correction advantage must receive larger weight.
    disagreement_weights=w[disagree]
    assert disagreement_weights[1] > disagreement_weights[0]
    assert r["mean_disagreement_multiplier"]==1.0


def test_aws_has_no_threshold_or_temperature_parameter_in_multiplier_semantics():
    p0=np.asarray([[0.8,0.1,0.1],[0.8,0.1,0.1]],dtype=np.float32)
    mu=np.asarray([[0.0,1e-9,-1.0],[0.0,2e-9,-1.0]],dtype=np.float64)
    _,w,_=target_and_multiplier_m22b(objective="ADVANTAGE_WEIGHTED_SELECTIVE_CE",mu_teacher=mu,base_probs=p0)
    assert np.all(w>0.0)
    assert np.isclose(float(w[1]/w[0]),2.0,rtol=1e-5,atol=0.0)


def test_new_seed_registry_is_exact_disjoint_and_not_m22_primary_registry():
    expected={
        "CLEAN_CONDITIONAL":(2101,2102,2103,2104,2105),
        "NOISY_TEACHER":(2201,2202,2203,2204,2205),
        "WEAK_CONDITIONAL_STRONG_MARGINAL":(2301,2302,2303,2304,2305),
        "NEAR_OPTIMAL_CHAMPION":(2401,2402,2403,2404,2405),
    }
    observed={r.name:r.seeds for r in REGIMES_M22B}
    assert observed==expected
    seeds=[s for r in REGIMES_M22B for s in r.seeds]
    assert len(seeds)==20 and len(set(seeds))==20
    old=set(range(1101,1106))|set(range(1201,1206))|set(range(1301,1306))|set(range(1401,1406))
    assert not (set(seeds)&old)
