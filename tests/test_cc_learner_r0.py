import copy, torch
from torch import nn
from cb16_local_opt.cc_learner_r0 import CCLearner
from cb16_local_opt.cc_policy_generation_r0 import freeze_behavior_generation,child_copy
def make():
 torch.manual_seed(3); return nn.Linear(2,3),nn.Sequential(nn.Linear(2,4),nn.Tanh(),nn.Linear(4,1))
def test_deterministic_delta_exactly_once_and_behavior_immutable():
 a,c=make(); behavior=freeze_behavior_generation(a,0,'p','unit'); child=child_copy(a); l=CCLearner(child,c,.05,.05)
 obs=torch.tensor([[1.,0.],[-1.,0.],[.5,0.]]); acts=torch.tensor([2,0,2]); rewards=torch.tensor([1.,1.,.5]); discounts=torch.zeros(3); mu=torch.log_softmax(a(obs).detach(),1).gather(1,acts[:,None]).squeeze(1)
 m=l.update_categorical('u1',obs,obs,acts,rewards,discounts,mu,0.,('s',)); sha=l.checkpoint_sha(); m2=l.update_categorical('u1',obs,obs,acts,rewards,discounts,mu,0.,('s',)); assert l.checkpoint_sha()==sha and m2.optimizer_step==m.optimizer_step==1
 assert freeze_behavior_generation(a,0,'p','unit').state_bytes==behavior.state_bytes
 a2,c2=make(); l2=CCLearner(child_copy(a2),c2,.05,.05); l2.update_categorical('u1',obs,obs,acts,rewards,discounts,mu,0.,('s',)); assert l2.checkpoint_sha()==sha
