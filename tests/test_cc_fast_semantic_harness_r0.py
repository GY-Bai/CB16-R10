from cb16_local_opt.cc_fast_semantic_harness_r0 import *
def test_semantic_harness_passes():
    r=run_account_known_answers(); assert r.verdict=="PASS",r.checks
