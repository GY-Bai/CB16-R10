from cb16_local_opt.cc_replay_support_r0 import ReplaySupportSample, support_health
from cb16_local_opt.cc_replay_compatibility_r0 import ReplaySemanticIdentity, check_replay_compatibility

def test_support_health_reports_ratios_clipping_ess_and_contributions():
    xs=[ReplaySupportSample(-1,0,'G3','CC'),ReplaySupportSample(0,-4,'G4','CC')]
    h=support_health(xs,rho_clip=1)
    assert h.clipping_fraction==.5 and h.low_support_frequency==.5 and h.effective_support>0
    assert h.generation_contribution=={'G3':1,'G4':1}

def test_age_alone_never_invalidates_compatible_history():
    old=ReplaySemanticIdentity('R1','obs','n','a','e','env','b',True)
    current=ReplaySemanticIdentity('R1','obs','n','a','e','env','b',True)
    # No timestamp/age is an axis in the compatibility contract.
    assert check_replay_compatibility(old,current).compatible
