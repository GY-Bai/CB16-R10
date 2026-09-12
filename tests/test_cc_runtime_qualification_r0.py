import pytest
from cb16_local_opt.cc_runtime_qualification_r0 import *
def test_thread_a_qualification_scope_is_fail_closed():
    q=compile_thread_a_qualification_r0({k:True for k in REQUIRED},test_command='pytest -q tests/test_cc_*r0.py'); assert q['status']=='PASS' and q['evidence_level']=='THREAD_LOCAL_CLOSED_LOOP' and 'economic_improvement' in q['not_claimed']
    with pytest.raises(RuntimeError): compile_thread_a_qualification_r0({**{k:True for k in REQUIRED},'learner':True},test_command='x')
