from dataclasses import replace
from cb16_local_opt.cc_replay_compatibility_r0 import ReplaySemanticIdentity, check_replay_compatibility

def ident(): return ReplaySemanticIdentity('R1','obs','norm','act','exec','env','bound',True)

def test_semantic_mismatch_reason_codes_and_missing_likelihood_fail_closed():
    target=ident(); bad=replace(target,normalizer_id='other',true_behavior_likelihood_available=False)
    r=check_replay_compatibility(bad,target)
    assert not r.compatible and set(r.reason_codes)=={'MISSING_TRUE_BEHAVIOR_LIKELIHOOD','NORMALIZER_MISMATCH'}
