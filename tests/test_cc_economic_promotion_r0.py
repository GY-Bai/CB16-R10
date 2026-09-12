from cb16_local_opt.cc_experience_wire_r0 import CCEconomicResultV1
from cb16_local_opt.cc_economic_promotion_r0 import assess, UNRESOLVED_OWNER_DECISION

def result(bh,flat): return CCEconomicResultV1('e','frozen_checkpoint','p','c','T','cap',({'a':1},),.1,bh,flat,{}, {},'SYNTHETIC')

def test_conflicting_baselines_return_owner_decision_not_hidden_metric():
    assert assess(result(-.1,.1)).status==UNRESOLVED_OWNER_DECISION
    assert assess(result(.1,.1)).status=='QUALIFIES_BOTH_BASELINES'
