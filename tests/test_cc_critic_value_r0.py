import torch
from torch import nn
from cb16_local_opt.cc_critic_value_r0 import *
def test_critic_separate_and_mean_converges():
 torch.manual_seed(2); actor=nn.Linear(2,3); c=SeparateCritic(2,8); assert_disjoint_parameters(actor,c)
 x=torch.tensor([[-1.,0.],[0.,0.],[1.,0.],[2.,0.]]); y=2*x[:,0]+1; opt=torch.optim.Adam(c.parameters(),lr=.05)
 for _ in range(300):
  loss=mean_value_loss(c(x),y); opt.zero_grad(); loss.backward(); opt.step()
 assert float(mean_value_loss(c(x),y))<1e-3
