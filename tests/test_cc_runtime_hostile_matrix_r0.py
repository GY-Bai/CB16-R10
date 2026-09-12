from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0,advance_environment_r0
from cb16_local_opt.cc_runtime_wire_r0 import CCExecutionLegV1
from tests.cc_thread_a_support_r0 import acct,runtime,decision
def test_negative_equity_liability_and_liquidation_pending_settlement_preserved():
    a=acct(mark=10,cash=-100,liab=50); assert a.equity<0; b=advance_environment_r0(a,CCEnvironmentIntervalR0(5,liability_delta=25,external_capital_flow=10,external_capital_flow_ref='flow-1',force_liquidate=True,boundary_type='TRADING_DISABLED_PENDING_SETTLEMENT')); assert b.liabilities==75 and b.position_quantity==0 and b.external_capital_flows_cumulative==10
def test_partial_reduce_full_close_reversal_second_leg_reject_are_representable():
    for legs in ((CCExecutionLegV1(0,-.25,'EXECUTED',100),),(CCExecutionLegV1(0,-1,'EXECUTED',100),),(CCExecutionLegV1(0,-1,'EXECUTED',100),CCExecutionLegV1(1,0,'REJECTED',None))):
        for i,l in enumerate(legs): assert l.leg_index==i; l.validate()
def test_same_risk_rebalances_after_price_equity_change():
    r=runtime(); qs=[]
    def cb(a,c): qs.append(a.position_quantity); return decision(a,c,'LONG',.5)
    r.step(CCEnvironmentIntervalR0(120),cb,expected_predecessor_token=r.predecessor_token); r.step(CCEnvironmentIntervalR0(80),cb,expected_predecessor_token=r.predecessor_token); assert r.account.position_quantity!=qs[-1]
