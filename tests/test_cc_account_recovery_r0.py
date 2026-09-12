from dataclasses import asdict
from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0
from cb16_local_opt.cc_environment_lifecycle_r0 import OBSERVATION_AVAILABLE,DECISION_CAPTURED,EXECUTION_APPLIED
from cb16_local_opt.cc_account_recovery_r0 import seal_runtime_r0,restore_runtime_r0
from tests.cc_thread_a_support_r0 import runtime,decision,fake_exec
def _resumed(stage):
    r=runtime(); cb=lambda a,c:decision(a,c); r.begin_interval(CCEnvironmentIntervalR0(110,funding_cashflow=-1),expected_predecessor_token=r.predecessor_token)
    if stage>=1:r.capture_decision(cb)
    if stage>=2:r.execute_pending()
    if stage>=3:r.advance_pending_environment()
    rr,mem=restore_runtime_r0(seal_runtime_r0(r,policy_memory_token='opaque'),executor=fake_exec); assert mem=='opaque'
    if rr.phase==OBSERVATION_AVAILABLE:rr.capture_decision(cb)
    if rr.phase==DECISION_CAPTURED:rr.execute_pending()
    if rr.phase==EXECUTION_APPLIED:rr.advance_pending_environment()
    return rr.publish_pending(),rr
def test_pause_resume_exact_at_all_semantic_phases():
    ref=runtime(); tr0=ref.step(CCEnvironmentIntervalR0(110,funding_cashflow=-1),lambda a,c:decision(a,c),expected_predecessor_token=ref.predecessor_token)
    for stage in range(4):
        tr,r=_resumed(stage); assert asdict(tr)==asdict(tr0); assert asdict(r.account)==asdict(ref.account); assert r.predecessor_token==ref.predecessor_token
