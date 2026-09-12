from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class VTraceReturns:
    vs: torch.Tensor
    pg_advantages: torch.Tensor
    rhos: torch.Tensor
    clipped_rhos: torch.Tensor
    cs: torch.Tensor

def vtrace(rewards:torch.Tensor, values:torch.Tensor, bootstrap_value:torch.Tensor,
           log_pi:torch.Tensor, log_mu:torch.Tensor, discounts:torch.Tensor,
           rho_bar:float=1.0,c_bar:float=1.0,pg_rho_bar:float=1.0)->VTraceReturns:
    if not (rewards.ndim==values.ndim==log_pi.ndim==log_mu.ndim==discounts.ndim==1): raise ValueError("VTRACE_1D_ONLY")
    n=len(rewards)
    if not all(len(x)==n for x in (values,log_pi,log_mu,discounts)): raise ValueError("VTRACE_LENGTH")
    rhos=torch.exp(torch.clamp(log_pi-log_mu,-80,80))
    cr=torch.clamp(rhos,max=rho_bar); cs=torch.clamp(rhos,max=c_bar)
    vals_tp1=torch.cat([values[1:],bootstrap_value.reshape(1)])
    deltas=cr*(rewards+discounts*vals_tp1-values)
    vs=torch.empty_like(values); acc=torch.zeros_like(bootstrap_value)
    for t in range(n-1,-1,-1):
        acc=deltas[t]+discounts[t]*cs[t]*acc
        vs[t]=values[t]+acc
    vs_tp1=torch.cat([vs[1:],bootstrap_value.reshape(1)])
    pg_rhos=torch.clamp(rhos,max=pg_rho_bar)
    adv=pg_rhos*(rewards+discounts*vs_tp1-values)
    return VTraceReturns(vs,adv,rhos,cr,cs)
