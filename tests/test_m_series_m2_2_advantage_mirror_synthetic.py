from __future__ import annotations

import numpy as np

from cb16_local_opt.m_series_m2_2_advantage_mirror_synthetic import (
    REGIMES,
    SCENARIOS,
    TRAIN_GROUPS,
    objective_target_m22,
    rotate_teacher_surface_m22,
)


def test_advantage_mirror_exactly_preserves_g0_when_teacher_best_is_g0():
    p0=np.asarray([[0.2,0.7,0.1],[0.6,0.2,0.2]],dtype=np.float32)
    mu=np.asarray([[0.0,0.03,-0.01],[0.02,0.01,-0.02]],dtype=np.float64)
    q,r=objective_target_m22(objective="POSITIVE_ADVANTAGE_MIRROR_CE",mu_teacher=mu,base_probs=p0)
    assert np.allclose(q,p0,atol=1e-7,rtol=0.0)
    assert r["teacher_g0_agreement_rows"]==2
    assert r["exact_preservation_rows"]==2


def test_advantage_mirror_boosts_only_positive_teacher_advantage_over_g0():
    p0=np.asarray([[0.6,0.3,0.1]],dtype=np.float32)
    mu=np.asarray([[0.0,0.004,-0.002]],dtype=np.float64)
    q,_=objective_target_m22(objective="POSITIVE_ADVANTAGE_MIRROR_CE",mu_teacher=mu,base_probs=p0)
    assert q[0,1] > p0[0,1]
    assert q[0,2] < p0[0,2]


def test_whole_group_rotation_preserves_each_scenario_multiset():
    x=np.arange(TRAIN_GROUPS*SCENARIOS*3,dtype=np.float64).reshape(TRAIN_GROUPS*SCENARIOS,3)
    y=rotate_teacher_surface_m22(x,7)
    assert y.shape==x.shape
    assert not np.array_equal(x,y)
    xc=x.reshape(TRAIN_GROUPS,SCENARIOS,3); yc=y.reshape(TRAIN_GROUPS,SCENARIOS,3)
    for s in range(SCENARIOS):
        assert np.array_equal(np.sort(xc[:,s,:],axis=0),np.sort(yc[:,s,:],axis=0))


def test_regime_seed_registry_is_exact_and_disjoint():
    assert [r.name for r in REGIMES]==[
        "CLEAN_CONDITIONAL","NOISY_TEACHER","WEAK_CONDITIONAL_STRONG_MARGINAL","NEAR_OPTIMAL_CHAMPION"
    ]
    seeds=[s for r in REGIMES for s in r.seeds]
    assert len(seeds)==20 and len(set(seeds))==20
