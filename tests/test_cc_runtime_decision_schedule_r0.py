import pytest
from cb16_local_opt.cc_runtime_decision_schedule_r0 import CCDecisionScheduleR0
def test_every_n_resume_end_data_and_reorder():
    s=CCDecisionScheduleR0(2); assert [s.is_decision_time(i) for i in range(5)]==[True,False,True,False,True]; s=s.record(environment_time=0,decision_index=0); s=s.record(environment_time=2,decision_index=1)
    with pytest.raises(RuntimeError): s.record(environment_time=2,decision_index=2)
    with pytest.raises(RuntimeError): s.record(environment_time=4,decision_index=3)
