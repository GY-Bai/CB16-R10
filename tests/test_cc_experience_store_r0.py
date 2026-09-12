import pytest
from cb16_local_opt.cc_experience_store_r0 import RawFactStore, SemanticConflict

def test_content_addressed_exactly_once_and_conflict(tmp_path):
    s=RawFactStore(tmp_path); a={'kind':'failed','equity':-2}
    h=s.put('t1',a); assert s.put('t1',a)==h and s.count()==1 and s.get('t1')==a
    with pytest.raises(SemanticConflict): s.put('t1',{'kind':'success'})

def test_ingest_does_not_filter_failure_or_terminal(tmp_path):
    s=RawFactStore(tmp_path)
    for i,k in enumerate(('successful','ordinary','failed','terminal')): s.put(k,{'kind':k,'i':i})
    assert s.count()==4
