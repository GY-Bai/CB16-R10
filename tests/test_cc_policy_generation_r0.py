import torch
from torch import nn
from cb16_local_opt.cc_policy_generation_r0 import *
def test_fixed_behavior_generation_immutable_during_child_learning():
 torch.manual_seed(1); m=nn.Linear(2,3); g=freeze_behavior_generation(m,4,'p','c'); child=child_copy(m)
 with torch.no_grad(): next(child.parameters()).add_(1.)
 assert freeze_behavior_generation(m,4,'p','c').state_bytes==g.state_bytes and module_sha256(child)!=module_sha256(m)
