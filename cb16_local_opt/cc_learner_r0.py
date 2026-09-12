from __future__ import annotations
from dataclasses import dataclass
import copy, hashlib, io
import torch
from torch import nn
from .cc_vtrace_r0 import vtrace
from .cc_policy_loss_r0 import actor_policy_gradient_loss
from .cc_critic_value_r0 import mean_value_loss, assert_disjoint_parameters
from .cc_learner_transaction_r0 import ExactlyOnceLedger

@dataclass(frozen=True)
class UpdateMetrics:
    actor_loss:float; critic_loss:float; rho_mean:float; rho_clip_fraction:float; optimizer_step:int

class CCLearner:
    def __init__(self,actor:nn.Module,critic:nn.Module,actor_lr:float=0.03,critic_lr:float=0.03):
        assert_disjoint_parameters(actor,critic)
        self.actor=actor; self.critic=critic
        self.actor_opt=torch.optim.SGD(actor.parameters(),lr=actor_lr)
        self.critic_opt=torch.optim.SGD(critic.parameters(),lr=critic_lr)
        self.step=0; self.ledger=ExactlyOnceLedger()

    def checkpoint_sha(self)->str:
        b=io.BytesIO(); torch.save({"a":self.actor.state_dict(),"c":self.critic.state_dict(),"s":self.step},b)
        return hashlib.sha256(b.getvalue()).hexdigest()

    def update_categorical(self, update_id:str, observations:torch.Tensor, critic_obs:torch.Tensor, actions:torch.Tensor,
                           rewards:torch.Tensor, discounts:torch.Tensor, log_mu:torch.Tensor, bootstrap:float,
                           sequence_ids:tuple[str,...])->UpdateMetrics:
        parent=self.checkpoint_sha(); self.ledger.prepare(update_id,parent,sequence_ids,self.step)
        if update_id in self.ledger.committed:
            rec=self.ledger.committed[update_id]
            return UpdateMetrics(float(rec["actor_loss"]),float(rec["critic_loss"]),float(rec["rho_mean"]),float(rec["rho_clip_fraction"]),int(rec["step_after"]))
        logits=self.actor(observations); logp_all=torch.log_softmax(logits,dim=-1); log_pi=logp_all.gather(1,actions[:,None]).squeeze(1)
        values=self.critic(critic_obs).squeeze(-1)
        vt=vtrace(rewards,values.detach(),torch.tensor(float(bootstrap),dtype=values.dtype),log_pi.detach(),log_mu,discounts)
        aloss=actor_policy_gradient_loss(log_pi,vt.pg_advantages)
        closs=mean_value_loss(values,vt.vs)
        self.actor_opt.zero_grad(); aloss.backward(); self.actor_opt.step()
        self.critic_opt.zero_grad(); closs.backward(); self.critic_opt.step(); self.step+=1
        child=self.checkpoint_sha(); frac=float((vt.rhos>1.0).float().mean())
        committed=self.ledger.commit(update_id,child,self.step)
        if not committed: raise RuntimeError("DUPLICATE_COMMIT_INTERNAL")
        rec=self.ledger.committed[update_id]; rec.update(actor_loss=float(aloss.detach()),critic_loss=float(closs.detach()),rho_mean=float(vt.rhos.mean()),rho_clip_fraction=frac)
        return UpdateMetrics(float(aloss.detach()),float(closs.detach()),float(vt.rhos.mean()),frac,self.step)
