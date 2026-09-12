import torch
from cb16_local_opt.cc_policy_brain_r0 import *
def test_brain_composition_and_bindings():
 b=CCCentralBrain(2,2,1,4,BrainBindings('o','n','d','s')); b.assert_gradient_ownership(); out=b(torch.ones(3,2),torch.ones(3,2),torch.ones(3,1)); assert out[0].shape==(3,3) and len(b.bindings.semantic_hash())==64
