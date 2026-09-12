from cb16_local_opt.cc_experience_source_r0 import SourceClass, classify

def test_legacy_without_true_log_mu_remains_fact_not_vtrace_replay():
    x=classify(SourceClass.LEGACY_H72_TEACHER_DEMONSTRATION,true_log_mu_available=False)
    assert not x.replay_eligible and x.reason=='MISSING_TRUE_BEHAVIOR_LIKELIHOOD'

def test_cc_stochastic_with_true_log_mu_is_replay_eligible():
    assert classify(SourceClass.CC_STOCHASTIC_TRAJECTORY,true_log_mu_available=True).replay_eligible
