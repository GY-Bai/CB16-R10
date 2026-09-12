import pytest
from cb16_local_opt.cc_learning_qualification_r0 import *
def test_qualification_scope_and_fail_closed():
 q=compile_qualification({k:True for k in REQUIRED}); assert q.status=='PASS' and q.strongest_evidence=='KNOWN_ANSWER' and 'NO_REAL_MARKET_EDGE' in q.claim_scope
 bad=compile_qualification({k:True for k in REQUIRED if k!='final_firewall'}); assert bad.status=='FAIL'
def test_final_and_fresh_firewalls():
 enforce_data_firewall(uses_final=False,uses_fresh_market_data=False)
 with pytest.raises(RuntimeError,match='FINAL'): enforce_data_firewall(uses_final=True,uses_fresh_market_data=False)
 with pytest.raises(RuntimeError,match='FRESH'): enforce_data_firewall(uses_final=False,uses_fresh_market_data=True)
