import pytest
from dataclasses import asdict
from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0
from cb16_local_opt.cc_runtime_generation_switch_r0 import switch_generation_r0
from tests.cc_thread_a_support_r0 import runtime,decision
def test_generation_switch_preserves_account_clocks_lineage_and_attributes_next_transition():
    r=runtime(); before=(asdict(r.account),asdict(r.clocks),r.account_lineage_id,r.predecessor_token); rec=switch_generation_r0(r,new_policy_generation='g2',new_policy_id='p2',new_policy_sha256='c'*64); assert before==(asdict(r.account),asdict(r.clocks),r.account_lineage_id,r.predecessor_token); assert rec.old_policy_id=='p1' and r.policy_id=='p2'
    captured=[]
    def cb(a,c):
        d=decision(a,c,generation='g2',policy='p2',sha='c'*64); captured.append(d); return d
    tr=r.step(CCEnvironmentIntervalR0(101),cb,expected_predecessor_token=r.predecessor_token)
    assert tr.policy_decision_ref==captured[0].ref and r.policy_generation=='g2'
def test_generation_switch_rejects_non_sha_identity():
    r=runtime()
    with pytest.raises(RuntimeError): switch_generation_r0(r,new_policy_generation='g2',new_policy_id='p2',new_policy_sha256='z'*64)
