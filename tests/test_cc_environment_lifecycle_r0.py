import pytest
from cb16_local_opt.cc_environment_lifecycle_r0 import *
def test_lifecycle_order_versioned():
    assert PHASE_ORDER==(OBSERVATION_AVAILABLE,DECISION_CAPTURED,EXECUTION_APPLIED,ENVIRONMENT_APPLIED,POST_STATE_PUBLISHED); assert next_phase_r0(OBSERVATION_AVAILABLE)==DECISION_CAPTURED
    with pytest.raises(RuntimeError): next_phase_r0(POST_STATE_PUBLISHED)
