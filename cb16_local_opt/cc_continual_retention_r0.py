from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class RetentionResult:
    a_initial:float; b_after_adapt:float; a_after_b_no_replay:float; a_after_b_with_replay:float; a_after_revisit:float

def _acc(model,x,y): return float((torch.argmax(model(x),1)==y).float().mean())
def _train(model,opt,x,y,steps):
    for _ in range(steps):
        loss=nn.functional.cross_entropy(model(x),y); opt.zero_grad(); loss.backward(); opt.step()

def run_retention(seed:int=7)->RetentionResult:
    torch.manual_seed(seed)
    # Context is causal input; no external mode selector. A and B are overlapping classification tasks.
    xa=torch.tensor([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]]) ; ya=torch.tensor([0,0,1,1])
    xb=torch.tensor([[-2.,1.],[-1.,1.],[1.,1.],[2.,1.]]) ; yb=torch.tensor([1,1,0,0])
    base=nn.Linear(2,2); opt=torch.optim.SGD(base.parameters(),lr=.15)
    _train(base,opt,xa,ya,50); a0=_acc(base,xa,ya)
    no=nn.Linear(2,2); no.load_state_dict(base.state_dict()); noopt=torch.optim.SGD(no.parameters(),lr=.15)
    _train(no,noopt,xb,yb,40); b=_acc(no,xb,yb); ano=_acc(no,xa,ya)
    rep=nn.Linear(2,2); rep.load_state_dict(base.state_dict()); ropt=torch.optim.SGD(rep.parameters(),lr=.15)
    xr=torch.cat([xb,xa]); yr=torch.cat([yb,ya]); _train(rep,ropt,xr,yr,40); arep=_acc(rep,xa,ya)
    _train(rep,ropt,xa,ya,12); revisit=_acc(rep,xa,ya)
    return RetentionResult(a0,b,ano,arep,revisit)
