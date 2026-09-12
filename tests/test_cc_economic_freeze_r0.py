import pytest
from dataclasses import replace
from cb16_local_opt.cc_economic_freeze_r0 import EvaluationFreeze, require_new_version_if_changed

def test_post_result_rescue_requires_new_evaluation_version():
    a=EvaluationFreeze('c','T','w','cap','base','p','data','v1')
    with pytest.raises(ValueError): require_new_version_if_changed(a,replace(a,common_horizon_id='T2'))
    require_new_version_if_changed(a,replace(a,common_horizon_id='T2',version='v2'))
