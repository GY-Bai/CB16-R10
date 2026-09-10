from __future__ import annotations

import pytest
import torch

from cb16_local_opt.canonical_state_alignment_falsification_r41 import (
    R41_SCENARIOS,
    R41_SHIFTS,
    shuffle_reduced_targets_by_future_group_r41,
    summarize_state_alignment_control_r41,
)
from cb16_local_opt.training_runtime_r11 import PreparedEvidenceR11


def _prepared(groups: int = 4) -> PreparedEvidenceR11:
    parent_ids=[]; group_ids=[]; rows=[]
    for g in range(groups):
        gid=f"FUT:BTCUSDT:{1000+g}"
        for s,scenario in enumerate(R41_SCENARIOS):
            row=torch.zeros(107,dtype=torch.float32)
            row[:102]=float(g*10+s)/100.0
            raw=torch.tensor([1.0+g,2.0+s,3.0+g+s],dtype=torch.float32)
            row[102:105]=raw/raw.sum()
            row[105]=0.05+0.01*g+0.001*s
            row[106]=1.0
            rows.append(row); group_ids.append(gid); parent_ids.append(f"R11R21:P:BTCUSDT:{1000+g}:{scenario}")
    packed=torch.stack(rows)
    return PreparedEvidenceR11(tuple(parent_ids),tuple(group_ids),packed,"base",0)


def test_whole_future_shuffle_preserves_inputs_scenarios_and_target_multiset():
    base=_prepared(4); shuffled,receipt=shuffle_reduced_targets_by_future_group_r41(base,shift=1)
    assert torch.equal(base.packed[:,:102],shuffled.packed[:,:102])
    assert torch.equal(base.packed[:,106],shuffled.packed[:,106])
    assert receipt['target_multiset_exactly_preserved'] is True
    assert receipt['scenario_identity_preserved'] is True
    assert receipt['fixed_point_future_groups'] == 0
    assert torch.equal(shuffled.packed[0,102:106],base.packed[6,102:106])
    assert torch.equal(shuffled.packed[0,:102],base.packed[0,:102])


def test_unregistered_identity_shift_fails_closed():
    with pytest.raises(RuntimeError,match="UNREGISTERED_SHIFT"):
        shuffle_reduced_targets_by_future_group_r41(_prepared(),shift=0)


def test_unknown_scenario_fails_closed():
    base=_prepared(); ids=list(base.parent_ids); ids[0]=ids[0].rsplit(':',1)[0]+':UNKNOWN'
    bad=PreparedEvidenceR11(tuple(ids),base.dependence_group_ids,base.packed,base.evidence_hash,0)
    with pytest.raises(RuntimeError,match="UNKNOWN_ACCOUNT_SCENARIO"):
        shuffle_reduced_targets_by_future_group_r41(bad,shift=1)


def test_summary_uses_exact_pre_registered_arm_set_and_no_scientific_verdict():
    aligned={'loss':1.0,'direction_loss':0.8,'sizing_loss':0.2}
    shuffled={s:{'loss':1.1+s*1e-4,'direction_loss':0.9,'sizing_loss':0.21} for s in R41_SHIFTS}
    out=summarize_state_alignment_control_r41(aligned,shuffled)
    assert out['aligned_lower_total_count'] == len(R41_SHIFTS)
    assert out['descriptive_pattern'] == 'ALIGNED_TOTAL_LOWER_THAN_ALL_PRE_REGISTERED_SHUFFLES'
    assert out['scientific_market_information_verdict'] is False
    assert out['canonical_promotion_authorized'] is False
