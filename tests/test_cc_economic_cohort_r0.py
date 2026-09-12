import pytest
from cb16_local_opt.cc_economic_cohort_r0 import EconomicCohort

def test_preregistered_cohort_binds_population_weights_capital_horizon_policy():
    c=EconomicCohort('c',{'a':'s1','b':'s2'},{'a':1,'b':2},'cap','T','frozen_checkpoint','sha','REALIZED').validate()
    assert c.common_horizon_id=='T'
    with pytest.raises(ValueError): EconomicCohort('c',{'a':'s1','b':'s2'},{'a':1},'cap','T','frozen_checkpoint','sha','REALIZED').validate()
