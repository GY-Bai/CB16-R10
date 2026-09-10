from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cb16_local_opt.m_series_m1_direction_relative_sufficiency import (
    GAP_SATURATION,
    TAU,
    adjudicate_m1,
    audit_direction_sufficiency_m1,
)


def _teacher_softmax(means):
    x = np.asarray(means, dtype=np.float64) / TAU
    x = x - np.max(x)
    e = np.exp(np.clip(x, -60.0, 60.0))
    return e / e.sum()


def _evidence(means_rows):
    out=[]
    for means in means_rows:
        probs=_teacher_softmax(means)
        laws=[]
        for cls,direction in enumerate((-1,0,1)):
            risk=(0.25,0.0,0.75)[cls]
            laws.append(SimpleNamespace(direction=direction,requested_risk=risk,mean_utility=float(means[cls])))
            if direction != 0:
                laws.append(SimpleNamespace(direction=direction,requested_risk=min(1.0,risk+0.2),mean_utility=float(means[cls]-0.01)))
        out.append(SimpleNamespace(action_laws=tuple(laws),direction_target_probs=tuple(float(x) for x in probs)))
    return out


def _g0_probs(classes):
    out=[]
    for c in classes:
        p=np.full(3,0.05,dtype=np.float64); p[int(c)]=0.90; out.append(p)
    return np.asarray(out)


def test_unsaturated_reduced_probs_recover_best_vs_g0_mean_advantage():
    means=np.asarray([
        [0.010,0.014,0.012],
        [0.021,0.020,0.027],
        [-0.003,-0.005,-0.004],
    ])
    x=audit_direction_sufficiency_m1(evidence=_evidence(means),g0_direction_probs=_g0_probs([0,1,2]))
    assert x['rich_vs_reduced_best_direction_argmax_mismatch_count']==0
    assert x['best_vs_g0_advantage_saturation_row_count']==0
    assert x['best_vs_g0_advantage_reconstruction_max_abs_error_on_unsaturated_rows'] <= 1e-12
    assert x['actual_best_vs_g0_advantage_max'] < GAP_SATURATION


def test_softmax_clip_is_detected_as_material_margin_compression():
    means=np.asarray([[0.20,0.0,-0.05]])
    x=audit_direction_sufficiency_m1(evidence=_evidence(means),g0_direction_probs=_g0_probs([1]))
    assert x['best_vs_g0_advantage_saturation_row_count']==1
    assert x['actual_best_vs_g0_advantage_max'] > GAP_SATURATION


def _fold(fold, saturation=0, mismatch=0, error=0.0):
    return {
        'fold':fold,
        'rows':10,
        'rich_vs_reduced_best_direction_argmax_mismatch_count':mismatch,
        'best_vs_g0_advantage_saturation_row_count':saturation,
        'best_vs_g0_advantage_reconstruction_max_abs_error_on_unsaturated_rows':error,
        'actual_best_vs_g0_advantage_max':0.01,
        'all_pairwise_gap_saturation_rate':0.0,
    }


def test_m1_adjudication_requires_all_five_folds_exact():
    good=adjudicate_m1([_fold(i) for i in range(1,6)])
    assert good['direction_mean_geometry_retained'] is True
    assert good['conclusion']=='DIRECTION_BEST_MEAN_GEOMETRY_RETAINED_ON_CONSUMED_TRAIN_ONLY_SUPPORT'
    bad=adjudicate_m1([_fold(1),_fold(2),_fold(3,saturation=1),_fold(4),_fold(5)])
    assert bad['direction_mean_geometry_retained'] is False
    assert bad['conclusion']=='DIRECTION_BEST_MEAN_GEOMETRY_PARTIALLY_COMPRESSED_ON_CONSUMED_TRAIN_ONLY_SUPPORT'
