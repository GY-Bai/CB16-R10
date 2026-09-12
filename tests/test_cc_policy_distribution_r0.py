import math, torch, pytest
from cb16_local_opt.cc_policy_distribution_r0 import *
from cb16_local_opt.cc_policy_rng_r0 import PolicyRNG
def test_extreme_logits_finite_and_flat_point_mass():
 logits=torch.tensor([1000.,0.,-1000.],dtype=torch.float64); loc=torch.zeros(3,dtype=torch.float64); ls=torch.zeros(3,dtype=torch.float64)
 lp=stable_direction_log_probs(logits); assert torch.isfinite(lp).all() and torch.allclose(torch.exp(lp).sum(),torch.tensor(1.,dtype=torch.float64))
 flat=joint_log_prob(torch.zeros(3,dtype=torch.float64),loc,ls,'FLAT',0.); assert abs(float(flat)+math.log(3))<1e-10
def test_nonflat_joint_known_answer_and_endpoints():
 logits=torch.zeros(3,dtype=torch.float64); loc=torch.zeros(3,dtype=torch.float64); ls=torch.zeros(3,dtype=torch.float64)
 got=float(joint_log_prob(logits,loc,ls,'LONG',.5)); expected=-math.log(3)-.5*math.log(2*math.pi)+math.log(4)
 assert abs(got-expected)<1e-10
 with pytest.raises(ValueError): joint_log_prob(logits,loc,ls,'LONG',0.)
def test_sample_can_rescore_exactly():
 logits=torch.tensor([.2,.1,.7],dtype=torch.float64); loc=torch.tensor([-.3,0.,.4],dtype=torch.float64); ls=torch.tensor([-.2,0.,-.4],dtype=torch.float64); rng=PolicyRNG('p','a',5)
 a=sample_nominal(logits,loc,ls,rng); assert abs(a.log_prob-float(joint_log_prob(logits,loc,ls,a.direction,a.target_risk)))<1e-10
