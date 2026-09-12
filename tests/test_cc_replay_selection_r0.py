import pytest
from cb16_local_opt.cc_replay_selection_r0 import ReplaySelectionMetadata

def test_selection_provenance_is_auditable_and_outcome_keys_forbidden():
    x=ReplaySelectionMetadata('s','rng@7',.2,'CC','G3',{'ess':9.0,'clip_fraction':.1},'sel-v1').validate()
    assert x.sampling_probability_or_weight==.2
    with pytest.raises(ValueError): ReplaySelectionMetadata('s','r',.2,'CC','G3',{'realized_pnl':1},'v').validate()
