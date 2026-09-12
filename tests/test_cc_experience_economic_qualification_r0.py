import pytest
from cb16_local_opt.cc_experience_economic_qualification_r0 import compile_qualification

def test_thread_c_qualification_compiler_requires_all_evidence():
    keys=['immutable_facts_sequences','failure_retention','four_view_separation','historical_source_classification','replay_compatibility_support','no_age_expiration','generation_attribution','idempotent_persistence','economic_known_answers','synthetic_local_only']
    q=compile_qualification(**{k:True for k in keys}); assert q.passed()
    with pytest.raises(ValueError): compile_qualification(**{k:True for k in keys[:-1]})
