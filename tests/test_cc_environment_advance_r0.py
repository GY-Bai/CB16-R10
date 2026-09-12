from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0,advance_environment_r0
from tests.cc_thread_a_support_r0 import acct,runtime,decision,static_exec
def test_held_position_progresses_without_order():
    a=acct(); b=advance_environment_r0(a,CCEnvironmentIntervalR0(110,funding_cashflow=-2)); assert b.unrealized_pnl==10 and b.funding_cumulative==-2 and b.equity!=a.equity
def test_reject_noop_flat_and_unscheduled_do_not_freeze_world():
    for ex in (static_exec('REJECT','SUPERVISOR_REJECT'),static_exec('NOOP','SAME_TARGET'),static_exec('NOOP','BELOW_MIN_DELTA')):
        r=runtime(executor=ex); tr=r.step(CCEnvironmentIntervalR0(110,funding_cashflow=-1),lambda a,c:decision(a,c),expected_predecessor_token=r.predecessor_token); assert tr.environment_time_after==1 and r.account.mark_price==110 and tr.funding==-1
    r=runtime(every=2); r.step(CCEnvironmentIntervalR0(110),lambda a,c:decision(a,c),expected_predecessor_token=r.predecessor_token); tr=r.step(CCEnvironmentIntervalR0(120),None,expected_predecessor_token=r.predecessor_token); assert tr.policy_decision_ref is None and r.account.mark_price==120
    flat=acct(q=0,margin=0,basis=0); r=runtime(account=flat,executor=static_exec('NOOP','ALREADY_FLAT')); tr=r.step(CCEnvironmentIntervalR0(101),lambda a,c:decision(a,c,'FLAT',0),expected_predecessor_token=r.predecessor_token); assert tr.environment_time_after==1 and r.account.mark_price==101
