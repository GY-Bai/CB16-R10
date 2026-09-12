import pytest
from functools import partial
from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0
from cb16_local_opt.cc_runtime_account_loop_r0 import execute_via_frozen_r1_r0
from tests.cc_thread_a_support_r0 import runtime,decision,acct

def test_second_and_third_decisions_see_prior_real_account_consequences():
    r=runtime(); seen=[]
    def cb(a,c): seen.append((a.mark_price,a.equity,a.position_quantity)); return decision(a,c)
    r.step(CCEnvironmentIntervalR0(110),cb,expected_predecessor_token=r.predecessor_token); r.step(CCEnvironmentIntervalR0(120),cb,expected_predecessor_token=r.predecessor_token); r.step(CCEnvironmentIntervalR0(90),cb,expected_predecessor_token=r.predecessor_token)
    assert seen[1][0]==110 and seen[2][0]==120 and len(set(seen))==3

def test_within_account_predecessor_rejects_duplicate_or_reorder():
    r=runtime(); old=r.predecessor_token; r.step(CCEnvironmentIntervalR0(110),lambda a,c:decision(a,c),expected_predecessor_token=old)
    with pytest.raises(RuntimeError): r.step(CCEnvironmentIntervalR0(120),lambda a,c:decision(a,c),expected_predecessor_token=old)

def test_frozen_r1_permission_sizing_feasibility_execution_bridge_is_live():
    from cb16_local_opt.actor_critic_physics_adapter_r1 import Round2MechanicalExecutionConfigR1
    config=Round2MechanicalExecutionConfigR1(fee_rate=.001,slippage_bps=1.0,initial_margin_rate=.1,maintenance_margin_rate=.05,max_gross_leverage=2.0)
    r=runtime(account=acct(q=0,basis=0,margin=0,cash=1000),executor=partial(execute_via_frozen_r1_r0,config=config))
    tr=r.step(CCEnvironmentIntervalR0(101),lambda a,c:decision(a,c,'LONG',.1),expected_predecessor_token=r.predecessor_token)
    assert tr.permission_status=='ACCEPT'; assert tr.target_quantity>0; assert len(tr.execution_legs)==1; assert r.account.position_quantity>0
