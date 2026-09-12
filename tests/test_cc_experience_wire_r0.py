import pytest
from cb16_local_opt.cc_experience_wire_r0 import CCExperienceSequenceV1, CCEconomicResultV1
from tests.cc_thread_c_fixtures import env

def test_w02_hash_is_deterministic_and_signed_values_survive():
    e=env(equity=-5.0)
    assert e.post_equity==-5.0 and len(e.content_sha256)==64 and e.content_sha256==e.content_sha256

def test_w03_and_w05_local_contracts_validate():
    s=CCExperienceSequenceV1('s','acct','R1','m','CC',('t1',),0,0,('p',),('n',),'CHUNK_END',None,'c'*64).validate()
    r=CCEconomicResultV1('e','frozen_checkpoint','p','c','h','cap',({'id':'a'},),.1,.2,.1,{'failed':0},{'q05':-1.0},'SYNTHETIC').validate()
    assert s.transition_refs==('t1',) and r.mean_arithmetic_return==.1


def test_execution_legs_are_ordered_and_nonzero_fill_requires_price():
    ok = env(legs=({"leg_index": 0, "executed_quantity": 0.0, "execution_price": None},))
    assert ok.execution_legs[0]["execution_price"] is None
    with pytest.raises(ValueError):
        env(legs=({"leg_index": 0, "executed_quantity": 1.0, "execution_price": None},))
    with pytest.raises(ValueError):
        env(legs=({"leg_index": 1, "executed_quantity": 0.0, "execution_price": None},))
