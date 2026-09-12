import pytest
from cb16_local_opt.cc_critic_bootstrap_r0 import *
def test_boundary_aware_bootstrap():
 assert bootstrap_value('ECONOMIC_TERMINAL',9)==0 and bootstrap_value('COMPUTE_CHUNK',9)==9 and bootstrap_value('PENDING_SETTLEMENT',7)==7
 with pytest.raises(ValueError): bootstrap_value('DATA_END_ASSUME_TERMINAL',3)
