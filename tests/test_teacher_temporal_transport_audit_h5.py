from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cb16_local_opt.teacher_temporal_transport_audit_h5 import (
    H5_QUANTILES,
    H5_SCENARIOS,
    _aggregate_parent_rows_h5,
    adjudicate_h5,
    pinball_score_h5,
    shuffled_target_feature_index_h5,
)
from cb16_local_opt.teacher_vectorized_r11 import ColumnarTeacherIndexR11


def test_pinball_is_zero_for_exact_quantiles_and_positive_for_error():
    q=np.full(len(H5_QUANTILES),0.25,dtype=np.float64)
    assert pinball_score_h5(0.25,q)==0.0
    assert pinball_score_h5(0.35,q)>0.0


def test_future_group_aggregation_weights_groups_equally_not_parent_rows():
    rows=[]
    for gid,value in (("G0",1.0),("G1",3.0)):
        for i,scenario in enumerate(H5_SCENARIOS):
            rows.append({
                "dependence_group_id":gid,"scenario":scenario,"qscore":value,
                "mean_utility_mse":value,"coverage_10_90":0.5,"coverage_05_95":0.75,
            })
    out=_aggregate_parent_rows_h5(rows)
    assert out["future_dependence_groups"]==2
    assert out["parent_contexts"]==12
    assert out["qscore"]==2.0
    assert out["mean_utility_mse"]==2.0


def _tiny_eval_index():
    parent_ids=[]; student=[]; features=[]; parent_dep=[]; parents={}
    dep_ids=("FUT:A:1","FUT:A:2")
    dep_parent_rows=np.full((2,6),-1,dtype=np.int32)
    for g,gid in enumerate(dep_ids):
        for s,scenario in enumerate(H5_SCENARIOS):
            row=len(parent_ids); pid=f"P:A:{g}:{scenario}"
            parent_ids.append(pid); student.append(f"CTX:{g}:{scenario}")
            features.append([float(g),float(s),float(row)])
            parent_dep.append(g); dep_parent_rows[g,s]=row
            parents[pid]=SimpleNamespace(dependence_group_id=gid,scenario=scenario,decision_time_ms=100+g)
    action_directions=np.asarray([-1,-1,-1,0,1,1,1,1,1],dtype=np.int8)
    action_risks=np.asarray([0.25,0.5,0.75,0.0,0.1,0.25,0.5,0.75,1.0],dtype=np.float64)
    index=ColumnarTeacherIndexR11(
        parent_ids=tuple(parent_ids),student_context_ids=tuple(student),
        parent_timestamps=np.asarray([100]*6+[101]*6,dtype=np.int64),
        parent_dep_index=np.asarray(parent_dep,dtype=np.int32),
        features=np.asarray(features,dtype=np.float64),utilities=np.zeros((12,9),dtype=np.float64),
        action_directions=action_directions,action_risks=action_risks,
        dep_ids=dep_ids,dep_timestamps=np.asarray([100,101],dtype=np.int64),dep_parent_rows=dep_parent_rows,
        parent_row_by_id={p:i for i,p in enumerate(parent_ids)},dep_row_by_id={d:i for i,d in enumerate(dep_ids)},
    )
    return index,parents,tuple(parent_ids)


def test_whole_group_feature_shuffle_preserves_scenario_and_exact_feature_multiset():
    index,parents,pids=_tiny_eval_index()
    shuffled,receipt=shuffled_target_feature_index_h5(index=index,eval_parents=parents,eval_parent_ids=pids,shift=1)
    assert receipt["scenario_identity_preserved"] is True
    assert receipt["target_feature_multiset_preserved"] is True
    assert receipt["fixed_points"]==0
    for scenario in H5_SCENARIOS:
        p0=f"P:A:0:{scenario}"; p1=f"P:A:1:{scenario}"
        r0=index.parent_row_by_id[p0]; r1=index.parent_row_by_id[p1]
        assert np.array_equal(shuffled.features[r0],index.features[r1])
        assert np.array_equal(shuffled.features[r1],index.features[r0])


def _fold(fold:int,aligned:float,clim:float,shuffles:list[float]):
    med=float(np.median(np.asarray(shuffles,dtype=np.float64)))
    return {
        "fold":fold,"aligned":{"qscore":aligned},"climatology":{"qscore":clim},"median_shuffle_qscore":med,
        "aligned_minus_climatology_qscore":aligned-clim,"aligned_minus_median_shuffle_qscore":aligned-med,
        "aligned_lt_climatology":aligned<clim,"aligned_lt_median_shuffle":aligned<med,
        "aligned_lt_each_shuffle_count":sum(aligned<x for x in shuffles),
    }


def test_h5_adjudication_requires_late_folds_even_if_four_of_five_global_counts_pass():
    rows=[
        _fold(1,1.0,2.0,[2.0]*5),
        _fold(2,1.0,2.0,[2.0]*5),
        _fold(3,1.0,2.0,[2.0]*5),
        _fold(4,1.0,2.0,[2.0]*5),
        _fold(5,3.0,2.0,[2.0]*5),
    ]
    out=adjudicate_h5(rows)
    assert out["aligned_lt_climatology_fold_count"]==4
    assert out["aligned_lt_median_shuffle_fold_count"]==4
    assert out["aligned_lt_each_shuffle_pair_count_of_25"]==20
    assert out["folds_4_and_5_both_pass_climatology_and_shuffle"] is False
    assert out["overall_pass"] is False


def test_h5_adjudication_passes_only_when_all_preregistered_rules_pass():
    rows=[_fold(i,1.0,2.0,[2.0]*5) for i in range(1,6)]
    out=adjudicate_h5(rows)
    assert out["overall_pass"] is True
    assert out["aligned_lt_each_shuffle_pair_count_of_25"]==25
