import pytest
from cb16_local_opt.cc_economic_baselines_r0 import compute_baselines

def test_bh_flat_share_identical_denominator_cohort_horizon():
    bh,flat=compute_baselines(cohort_id='c',common_horizon_id='T',capital_denominator_id='cap',weights={'a':1,'dead':1},buy_hold_returns={'a':.2,'dead':.4})
    assert bh.mean_arithmetic_return == pytest.approx(.3) and flat.mean_arithmetic_return == 0
    assert (bh.cohort_id,bh.common_horizon_id,bh.capital_denominator_id)==(flat.cohort_id,flat.common_horizon_id,flat.capital_denominator_id)
    with pytest.raises(ValueError): compute_baselines(cohort_id='c',common_horizon_id='T',capital_denominator_id='cap',weights={'a':1,'dead':1},buy_hold_returns={'a':.2})
