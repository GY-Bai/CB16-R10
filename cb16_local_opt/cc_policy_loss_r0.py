from __future__ import annotations
import torch

def actor_policy_gradient_loss(log_probs:torch.Tensor,vtrace_advantages:torch.Tensor,mask:torch.Tensor|None=None)->torch.Tensor:
    if mask is None: mask=torch.ones_like(log_probs)
    denom=torch.clamp(mask.sum(),min=1.0)
    return -torch.sum(mask*log_probs*vtrace_advantages.detach())/denom

def entropy_diagnostic(direction_probs:torch.Tensor)->torch.Tensor:
    p=torch.clamp(direction_probs,1e-12,1.0)
    return -(p*torch.log(p)).sum(dim=-1).mean()
