from dataclasses import replace, asdict
from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0,advance_environment_r0
from cb16_local_opt.cc_runtime_wire_r0 import CCExecutionLegV1
from cb16_local_opt.cc_runtime_account_loop_r0 import CCExecutionSummaryR0
from cb16_local_opt.cc_environment_lifecycle_r0 import OBSERVATION_AVAILABLE,DECISION_CAPTURED,EXECUTION_APPLIED
from cb16_local_opt.cc_runtime_boundary_r0 import PROCESS_FAILURE
from cb16_local_opt.cc_account_recovery_r0 import seal_runtime_r0,restore_runtime_r0
from tests.cc_thread_a_support_r0 import acct,runtime,decision,fake_exec

def test_negative_equity_liability_and_liquidation_pending_settlement_preserved():
    a=acct(mark=10,cash=-100,liab=50); assert a.equity<0
    b=advance_environment_r0(a,CCEnvironmentIntervalR0(5,liability_delta=25,external_capital_flow=10,external_capital_flow_ref='flow-1',next_account_lineage_id='line-after-flow',force_liquidate=True,boundary_type='TRADING_DISABLED_PENDING_SETTLEMENT')); assert b.liabilities==75 and b.position_quantity==0 and b.external_capital_flows_cumulative==10

def _scripted_executor(target_q,legs):
    def f(a,d):
        after=replace(a,position_quantity=target_q,position_cost_basis=(a.mark_price if target_q else 0.0),margin_collateral=(abs(target_q)*a.mark_price*.1 if target_q else 0.0))
        return CCExecutionSummaryR0('ACCEPT','HOSTILE_SCRIPT',d.nominal_direction,d.nominal_target_risk,target_q,tuple(legs),after)
    return f

def test_partial_reduce_full_close_and_reversal_second_leg_reject_survive_runtime_transition():
    cases=(
        (.5,(CCExecutionLegV1(0,-.5,'EXECUTED',100),)),
        (0.0,(CCExecutionLegV1(0,-1.0,'EXECUTED',100),)),
        (0.0,(CCExecutionLegV1(0,-1.0,'EXECUTED',100),CCExecutionLegV1(1,0.0,'REJECTED',None))),
    )
    for target_q,legs in cases:
        r=runtime(executor=_scripted_executor(target_q,legs)); tr=r.step(CCEnvironmentIntervalR0(101),lambda a,c:decision(a,c),expected_predecessor_token=r.predecessor_token)
        assert tr.execution_legs==legs and r.account.position_quantity==target_q
        tr.validate()

def test_same_risk_rebalances_after_price_equity_change():
    r=runtime(); qs=[]
    def cb(a,c): qs.append(a.position_quantity); return decision(a,c,'LONG',.5)
    r.step(CCEnvironmentIntervalR0(120),cb,expected_predecessor_token=r.predecessor_token); r.step(CCEnvironmentIntervalR0(80),cb,expected_predecessor_token=r.predecessor_token); assert r.account.position_quantity!=qs[-1]

def test_external_capital_flow_switches_next_lineage_explicitly():
    r=runtime(); old=r.account_lineage_id; r.step(CCEnvironmentIntervalR0(101,external_capital_flow=50,external_capital_flow_ref='deposit-1',next_account_lineage_id='line-2'),lambda a,c:decision(a,c),expected_predecessor_token=r.predecessor_token); assert old=='line' and r.account_lineage_id=='line-2' and r.account.external_capital_flows_cumulative==50

def _resume_hostile_phase(stage):
    r=runtime(); cb=lambda a,c:decision(a,c)
    r.begin_interval(CCEnvironmentIntervalR0(110,funding_cashflow=-1),expected_predecessor_token=r.predecessor_token)
    if stage>=1: r.capture_decision(cb)
    if stage>=2: r.execute_pending()
    if stage>=3: r.advance_pending_environment()
    rr,_=restore_runtime_r0(seal_runtime_r0(r,policy_memory_token='hostile-pause'),executor=fake_exec)
    if rr.phase==OBSERVATION_AVAILABLE: rr.capture_decision(cb)
    if rr.phase==DECISION_CAPTURED: rr.execute_pending()
    if rr.phase==EXECUTION_APPLIED: rr.advance_pending_environment()
    return rr.publish_pending(),rr

def test_pause_at_every_lifecycle_phase_matches_uninterrupted_hostile_reference():
    ref=runtime(); tr0=ref.step(CCEnvironmentIntervalR0(110,funding_cashflow=-1),lambda a,c:decision(a,c),expected_predecessor_token=ref.predecessor_token)
    for stage in range(4):
        tr,r=_resume_hostile_phase(stage)
        assert asdict(tr)==asdict(tr0)
        assert asdict(r.account)==asdict(ref.account)
        assert r.predecessor_token==ref.predecessor_token

def test_process_failure_recovery_preserves_pending_semantics_and_compute_clock():
    r=runtime(); r.clocks=r.clocks.next_chunk(); predecessor=r.predecessor_token
    r.begin_interval(CCEnvironmentIntervalR0(105,boundary_type=PROCESS_FAILURE),expected_predecessor_token=predecessor)
    r.capture_decision(lambda a,c:decision(a,c))
    rr,token=restore_runtime_r0(seal_runtime_r0(r,policy_memory_token='hostile-process'),executor=fake_exec)
    assert token=='hostile-process' and rr.clocks.compute_chunk_index==1 and rr.phase==DECISION_CAPTURED and rr.predecessor_token==predecessor
    rr.execute_pending(); rr.advance_pending_environment(); tr=rr.publish_pending()
    assert tr.boundary_type==PROCESS_FAILURE and rr.clocks.environment_time==1 and rr.clocks.compute_chunk_index==1
