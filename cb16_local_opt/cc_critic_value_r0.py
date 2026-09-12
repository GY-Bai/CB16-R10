from __future__ import annotations
import torch
from torch import nn

class SeparateCritic(nn.Module):
    def __init__(self,input_dim:int,hidden:int=16):
        super().__init__(); self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.Tanh(),nn.Linear(hidden,1))
    def forward(self,x:torch.Tensor)->torch.Tensor:return self.net(x).squeeze(-1)

def mean_value_loss(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    return torch.mean((pred-target.detach())**2)

def assert_disjoint_parameters(actor:nn.Module,critic:nn.Module)->None:
    if {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}:
        raise RuntimeError("ACTOR_CRITIC_PARAMETER_ALIAS")
