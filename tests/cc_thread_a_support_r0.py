from dataclasses import replace
from cb16_local_opt.account_economics_r0 import AccountEconomicsStateR0
from cb16_local_opt.cc_runtime_wire_r0 import CCPolicyDecisionV1,CCExecutionLegV1
from cb16_local_opt.cc_clock_r0 import CCFourClockR0
from cb16_local_opt.cc_runtime_decision_schedule_r0 import CCDecisionScheduleR0
from cb16_local_opt.cc_runtime_account_loop_r0 import CCContinuousAccountRuntimeR0,CCExecutionSummaryR0
H='a'*64; O='b'*64
def acct(q=1.0,mark=100.0,cash=900.0,basis=100.0,margin=100.0,liab=0.0,account_id='acct'):
    return AccountEconomicsStateR0('CB16_R11_BC_ACCOUNT_ECONOMICS_V1_R0',account_id,cash,q,basis if q else 0.0,mark,0,0,0,margin if q else 0,liab,0,True)
def decision(a,c,direction='LONG',risk=.5,lineage='line',generation='g1',policy='p1',sha=H):
    return CCPolicyDecisionV1('SCI',lineage,c.policy_decision_index,c.environment_time,generation,policy,sha,'obs',O,'norm',direction,risk,-.2,'continuous_density' if direction!='FLAT' else 'point_mass','rng',c.policy_decision_index)
def fake_exec(a,d):
    target=(a.equity/a.mark_price)*d.nominal_target_risk*(1 if d.nominal_direction=='LONG' else -1 if d.nominal_direction=='SHORT' else 0); delta=target-a.position_quantity; after=replace(a,position_quantity=target,position_cost_basis=a.mark_price if target else 0,margin_collateral=abs(target)*a.mark_price*.1); legs=() if abs(delta)<1e-12 else (CCExecutionLegV1(0,delta,'EXECUTED',a.mark_price),); return CCExecutionSummaryR0('ACCEPT','LEGAL_NOMINAL_REQUEST',d.nominal_direction,d.nominal_target_risk,target,legs,after)
def static_exec(status='REJECT',reason='TEST_REJECT'):
    def f(a,d): return CCExecutionSummaryR0(status,reason,d.nominal_direction,d.nominal_target_risk,a.position_quantity,(),a)
    return f
def runtime(every=1,executor=fake_exec,account=None,lineage='line',policy='p1',generation='g1',sha=H):
    return CCContinuousAccountRuntimeR0(account_lineage_id=lineage,account=account or acct(),clocks=CCFourClockR0(0,0,0,'H'),schedule=CCDecisionScheduleR0(every),executor=executor,policy_generation=generation,policy_id=policy,policy_sha256=sha)
