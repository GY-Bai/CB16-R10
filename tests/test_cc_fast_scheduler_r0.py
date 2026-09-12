import pytest
from cb16_local_opt.cc_fast_scheduler_r0 import *
def test_same_account_strict_serial_cross_account_parallel():
    s=AccountSerialScheduler(["a","b"]); s.submit(ScheduledAccountTask("a",0,1,0),now_tick=0); s.submit(ScheduledAccountTask("b",0,1,0),now_tick=0)
    with pytest.raises(RuntimeError): s.submit(ScheduledAccountTask("a",1,1,0),now_tick=0)
    with pytest.raises(RuntimeError): s.commit("a",1,now_tick=0)
    s.commit("a",0,now_tick=0); s.submit(ScheduledAccountTask("a",1,1,1),now_tick=1); s.commit("a",1,now_tick=1)
