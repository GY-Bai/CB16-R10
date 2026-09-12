from __future__ import annotations
from dataclasses import dataclass


def account_dependent_optimal_action(market:float,position:float,liability:float)->str:
    # same market can rationally demand different action because account exposure differs
    score=market-position-2.0*liability
    return "LONG" if score>0.25 else ("SHORT" if score<-0.25 else "FLAT")

@dataclass(frozen=True)
class DelayedCase:
    short_gain:float; delayed_loss:float
    def short_horizon(self)->float:return self.short_gain
    def long_horizon(self)->float:return self.short_gain+self.delayed_loss

def delayed_consequence_case()->DelayedCase:return DelayedCase(1.0,-3.0)

def horizon_reversal()->dict[str,float]:return {"action_x_h1":1.0,"action_y_h1":0.2,"action_x_h3":-2.0,"action_y_h3":0.6}

def high_bankruptcy_higher_expectation()->dict[str,float]:
    # Risky: 80% -1, 20% +6 => +0.4; safe: always +0.2
    return {"risky_bankruptcy_frequency":.8,"risky_expectation":.4,"safe_bankruptcy_frequency":0.0,"safe_expectation":.2}

def offpolicy_known_answer(mu:float,pi:float,reward:float,value:float,bootstrap:float,gamma:float,rho_bar:float=1.0)->dict[str,float]:
    rho=pi/mu; cr=min(rho,rho_bar); delta=cr*(reward+gamma*bootstrap-value); vs=value+delta
    return {"rho":rho,"clipped_rho":cr,"vs":vs}

def learn_two_action_arithmetic_preference(return_a:float, return_b:float, *, steps:int=80, lr:float=.15, seed:int=13)->dict[str,float]:
    import torch
    torch.manual_seed(seed)
    logits=torch.zeros(2,requires_grad=True)
    opt=torch.optim.SGD([logits],lr=lr)
    returns=torch.tensor([float(return_a),float(return_b)])
    for _ in range(steps):
        probs=torch.softmax(logits,dim=0)
        objective=torch.sum(probs*returns)
        opt.zero_grad(); (-objective).backward(); opt.step()
    p=torch.softmax(logits.detach(),dim=0)
    return {"prob_a":float(p[0]),"prob_b":float(p[1]),"preferred":"A" if p[0]>p[1] else "B"}

def delayed_long_horizon_learning()->dict[str,object]:
    case=delayed_consequence_case()
    short=learn_two_action_arithmetic_preference(case.short_horizon(),0.0)
    long=learn_two_action_arithmetic_preference(case.long_horizon(),0.0)
    return {"short_horizon":short,"long_horizon":long,"analytical_short":case.short_horizon(),"analytical_long":case.long_horizon()}

def high_bankruptcy_learning()->dict[str,object]:
    x=high_bankruptcy_higher_expectation()
    learned=learn_two_action_arithmetic_preference(x["risky_expectation"],x["safe_expectation"])
    return {**x,"learned":learned}

def offpolicy_comparison(mu:float=.8,pi:float=.1,reward:float=1.0,value:float=.1,bootstrap:float=.4,gamma:float=.9)->dict[str,float]:
    ratio=pi/mu
    correct=min(ratio,1.0)*(reward+gamma*bootstrap-value)+value
    same_policy=(reward+gamma*bootstrap-value)+value
    uncorrected=reward+gamma*bootstrap
    return {"ratio":ratio,"correct_vtrace":correct,"same_policy_reduction":same_policy,"uncorrected_replay":uncorrected}
