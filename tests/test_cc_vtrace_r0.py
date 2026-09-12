import torch
from cb16_local_opt.cc_vtrace_r0 import vtrace
def test_same_policy_hand_computed_sequence():
 r=torch.tensor([1.,2.]); v=torch.tensor([.5,.25]); lp=torch.tensor([-.2,-.4]); d=torch.tensor([.9,.9]); out=vtrace(r,v,torch.tensor(.1),lp,lp,d)
 # same policy rho=c=1: v1=2+.9*.1=2.09; v0=1+.9*2.09=2.881
 assert torch.allclose(out.vs,torch.tensor([2.881,2.09]),atol=1e-6) and torch.allclose(out.rhos,torch.ones(2))
def test_clipping_boundary():
 out=vtrace(torch.tensor([1.]),torch.tensor([0.]),torch.tensor(0.),torch.tensor([0.]),torch.tensor([-2.]),torch.tensor([0.]),rho_bar=1.,c_bar=.5); assert float(out.clipped_rhos[0])==1 and float(out.cs[0])==.5
