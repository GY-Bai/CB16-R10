import pytest
from cb16_local_opt.cc_experience_sequence_r0 import OrderedExperienceSequence
from tests.cc_thread_c_fixtures import tx

def build(ts): return OrderedExperienceSequence.build('s',ts,science_semantic_version='R1',market_lineage_id='m',source_classification='CC',normalizer_identities=('n',))

def test_order_continuity_and_mixed_generation_attribution():
    s=build((tx(0,generation='G3'),tx(1,generation='G4'),tx(2,generation='G5')))
    assert [(x.from_policy_generation,x.to_policy_generation) for x in s.policy_switches]==[('G3','G4'),('G4','G5')]
    assert s.pure_generation() is None

def test_reorder_duplicate_cross_account_and_hash_gap_fail_closed():
    for bad in ((tx(1),tx(0)),(tx(0),tx(0)),(tx(0),tx(1,lineage='other')),(tx(0,post='x'),tx(1,pre='y'))):
        with pytest.raises(ValueError): build(bad)
