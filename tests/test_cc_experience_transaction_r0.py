import pytest
from cb16_local_opt.cc_experience_transaction_r0 import RawFactTransactionHarness

def test_restart_faults_and_retry_yield_one_semantic_fact(tmp_path):
    h=RawFactTransactionHarness(tmp_path); fact={'x':1}
    with pytest.raises(RuntimeError): h.crash_before_write('x',fact)
    assert h.store.count()==0
    tmp=h.partial_temp_write(); assert tmp.exists() and h.store.count()==0
    with pytest.raises(RuntimeError): h.durable_before_ack('x',fact)
    assert h.store.count()==0
    a=h.retry('x',fact); b=h.retry('x',fact)
    assert a==b and h.store.count()==1
