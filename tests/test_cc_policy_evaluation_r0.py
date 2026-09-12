import torch
from cb16_local_opt.cc_policy_evaluation_r0 import *
from cb16_local_opt.cc_policy_rng_r0 import PolicyRNG
def test_eval_identity_and_no_rng_consumption():
 r=PolicyRNG('p','a',1); pos=r.counter; e=DeterministicEvaluationPolicy.from_behavior('p','h'); assert e.action(torch.tensor([0.,2.,1.]),torch.zeros(3))==('FLAT',0.0); assert r.counter==pos and e.evaluation_id!='p'
