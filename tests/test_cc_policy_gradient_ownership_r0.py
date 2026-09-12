import torch, pytest
from cb16_local_opt.cc_policy_brain_r0 import *
def test_gradient_ownership_positive_and_negative():
 b=CCCentralBrain(2,2,1,4,BrainBindings('o','n','d','s')); before=[p.detach().clone() for p in b.market_organ.parameters()]
 logits,loc,ls=b(torch.ones(2,2),torch.ones(2,2),torch.ones(2,1)); (logits.sum()+loc.sum()+ls.sum()).backward()
 assert all(p.grad is None for p in b.market_organ.parameters()); assert any(p.grad is not None for p in b.account_stem.parameters()); assert all(torch.equal(x,p) for x,p in zip(before,b.market_organ.parameters()))
 next(b.market_organ.parameters()).requires_grad_(True)
 with pytest.raises(RuntimeError): b.assert_gradient_ownership()
