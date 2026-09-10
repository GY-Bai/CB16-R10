from __future__ import annotations

import torch

from cb16_local_opt.m_series_m2_1_correction_quality import adjudicate_m21


def _q(corr: float, capture: float, harm: float, net: float = 0.1, raw: float = 0.5):
    return {
        "advantage_weighted_teacher_correction_coverage": corr,
        "available_advantage_capture_fraction": capture,
        "harmful_cost_mean": harm,
        "net_gain_mean": net,
        "raw_move_to_teacher_rate": raw,
    }


def test_m21_adjudication_separates_correction_quality_from_harm_suppression():
    folds=[]
    for fold in range(1,6):
        arms={
            "SELECTIVE_ALIGNED":{"quality":_q(0.70,0.75,0.01)},
            "ABS_CE_ALIGNED":{"quality":_q(0.65,0.70,0.03)},
        }
        for shift in (1,7,13,23,31):
            arms[f"SELECTIVE_SHUFFLE_{shift}"]={"quality":_q(0.30,0.35,0.08)}
        folds.append({"fold":fold,"arms":arms})
    out=adjudicate_m21(folds)
    assert out["correction_quality_pass"] is True
    assert out["capture_pass"] is True
    assert out["harm_suppression_pass"] is True
    assert out["selective_harmful_cost_lt_abs_ce_aligned_fold_count"]==5
    assert out["m2_retroactive_status_change"] is False


def test_m21_harm_only_interpretation_is_distinct():
    folds=[]
    for fold in range(1,6):
        arms={
            "SELECTIVE_ALIGNED":{"quality":_q(0.20,0.25,0.01)},
            "ABS_CE_ALIGNED":{"quality":_q(0.50,0.55,0.03)},
        }
        for shift in (1,7,13,23,31):
            arms[f"SELECTIVE_SHUFFLE_{shift}"]={"quality":_q(0.30,0.35,0.08)}
        folds.append({"fold":fold,"arms":arms})
    out=adjudicate_m21(folds)
    assert out["correction_quality_pass"] is False
    assert out["capture_pass"] is False
    assert out["harm_suppression_pass"] is True
    assert out["joint_interpretation"]=="M2_SELECTIVE_POSITIVE_GAIN_IS_PRIMARILY_PRESERVATION_HARM_SUPPRESSION_NOT_SUPERIOR_DISAGREEMENT_CORRECTION_SELECTION"
