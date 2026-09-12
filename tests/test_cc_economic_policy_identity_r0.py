import pytest
from cb16_local_opt.cc_economic_policy_identity_r0 import EconomicPolicyIdentity

def test_mixed_generation_is_chain_not_final_checkpoint():
    c=EconomicPolicyIdentity.generation_chain('deploy',('G3','G4','G5'))
    with pytest.raises(ValueError): c.assert_pure_generation('G5')
    EconomicPolicyIdentity.frozen_checkpoint('sha','G5').assert_pure_generation('G5')
