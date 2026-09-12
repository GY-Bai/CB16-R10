import math,pytest
from dataclasses import replace
from cb16_local_opt.cc_runtime_wire_r0 import CCExecutionLegV1,CCEnvironmentTransitionV1
from cb16_local_opt.cc_clock_r0 import CCFourClockR0
from tests.cc_thread_a_support_r0 import acct,decision
def test_w01_validation_and_negative_cases():
    x=decision(acct(),CCFourClockR0(0,0,0,'H')); x.validate(); assert len(x.ref)==64
    for bad in (replace(x,log_mu=math.nan),replace(x,nominal_target_risk=2),replace(x,account_lineage_id=''),replace(x,policy_sha256='x')):
        with pytest.raises(RuntimeError): bad.validate()
def test_w02_ordered_legs_and_hashes():
    leg=CCExecutionLegV1(0,1,'EXECUTED',100); t=CCEnvironmentTransitionV1('line',0,0,1,'a'*64,'b'*64,'ACCEPT','ok','LONG',.5,1,(leg,),1,0,0,0,0,'c'*64,100,'CONTINUE',False,None); t.validate()
    with pytest.raises(RuntimeError): replace(t,execution_legs=(replace(leg,leg_index=1),)).validate()
