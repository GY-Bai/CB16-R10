import pytest
from cb16_local_opt.cc_fast_fact_queue_r0 import *
def test_byte_bounded_backpressure_never_drops_terminal():
    a=encode_fact({"x":"a"*10},semantic_id="a",terminal_or_failure=True); b=encode_fact({"x":"b"*10},semantic_id="b"); q=FactOutputQueue(max_bytes=a.nbytes+1,max_age_s=10); q.put(a)
    with pytest.raises(BackpressureRequired): q.put(b)
    got=q.get(); assert got.semantic_id=="a" and got.terminal_or_failure; q.put(b); assert q.depth==1
