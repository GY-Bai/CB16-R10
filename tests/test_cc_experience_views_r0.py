import pytest
from cb16_local_opt.cc_experience_views_r0 import ExperienceViews

def test_four_views_are_explicit_and_raw_failure_is_never_deleted():
    v=ExperienceViews(); v.ingest_raw('failed'); v.ingest_raw('good')
    v.set_membership('good','demonstration',True); v.set_membership('failed','replay',False)
    v.set_membership('failed','economic',True); v.set_membership('good','economic',True)
    assert 'failed' in v.raw_ids and 'failed' not in v.demonstration_ids
    v.assert_economic_cohort({'failed','good'})
    v.set_membership('failed','economic',False)
    with pytest.raises(ValueError): v.assert_economic_cohort({'failed','good'})
