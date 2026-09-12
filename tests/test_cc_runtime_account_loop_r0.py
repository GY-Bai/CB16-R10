import pytest
from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0
from tests.cc_thread_a_support_r0 import runtime,decision
def test_second_and_third_decisions_see_prior_real_account_consequences():
    r=runtime(); seen=[]
    def cb(a,c): seen.append((a.mark_price,a.equity,a.position_quantity)); return decision(a,c)
    r.step(CCEnvironmentIntervalR0(110),cb,expected_predecessor_token=r.predecessor_token); r.step(CCEnvironmentIntervalR0(120),cb,expected_predecessor_token=r.predecessor_token); r.step(CCEnvironmentIntervalR0(90),cb,expected_predecessor_token=r.predecessor_token); assert seen[1][0]==110 and seen[2][0]==120 and len(set(seen))==3
def test_within_account_predecessor_rejects_duplicate_or_reorder():
    r=runtime(); old=r.predecessor_token; r.step(CCEnvironmentIntervalR0(110),lambda a,c:decision(a,c),expected_predecessor_token=old)
    with pytest.raises(RuntimeError): r.step(CCEnvironmentIntervalR0(120),lambda a,c:decision(a,c),expected_predecessor_token=old)
