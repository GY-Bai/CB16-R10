from cb16_local_opt.cc_runtime_reference_scheduler_r0 import CCReferenceSchedulerR0
from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0
from tests.cc_thread_a_support_r0 import runtime,decision
def _run(order):
    rs={'a':runtime(2),'b':runtime(2)}; s=CCReferenceSchedulerR0(rs); work={k:[CCEnvironmentIntervalR0(101),CCEnvironmentIntervalR0(102)] for k in rs}; out=s.run(work,{k:(lambda a,c:decision(a,c)) for k in rs},order); return {k:[(x.environment_time_before,x.post_equity,x.post_account_truth_hash) for x in v] for k,v in out.items()}
def test_interaccount_order_does_not_change_per_account_semantics(): assert _run(['a','b'])==_run(['b','a'])
