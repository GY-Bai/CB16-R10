import random, torch
from torch import nn
from cb16_local_opt.cc_learner_r0 import CCLearner
from cb16_local_opt.cc_learner_checkpoint_r0 import *
from cb16_local_opt.cc_policy_rng_r0 import PolicyRNG
def new():
 torch.manual_seed(9); return CCLearner(nn.Linear(2,3),nn.Linear(2,1),.02,.02)
def test_full_recovery_rng_and_next_transaction():
 l=new(); pr=PolicyRNG('p','a',11); _=pr.random(); rr=random.Random(8); _=rr.random(); b=save_checkpoint(l,pr.state_dict(),rr.getstate(),{'normalizer':'n','science':'s','generation':2})
 expect=pr.random(); expect_lr=rr.random(); clone=new(); ps,ls,ids=restore_checkpoint(clone,b); pr2=PolicyRNG.from_state_dict(ps); rr2=random.Random(); rr2.setstate(ls); assert pr2.random()==expect and rr2.random()==expect_lr and ids['generation']==2
 obs=torch.tensor([[1.,0.]]); act=torch.tensor([2]); rew=torch.tensor([1.]); disc=torch.tensor([0.]); mu=torch.log_softmax(l.actor(obs).detach(),1)[:,2]
 m1=l.update_categorical('next',obs,obs,act,rew,disc,mu,0.,('s',)); m2=clone.update_categorical('next',obs,obs,act,rew,disc,mu,0.,('s',)); assert l.checkpoint_sha()==clone.checkpoint_sha() and m1==m2
