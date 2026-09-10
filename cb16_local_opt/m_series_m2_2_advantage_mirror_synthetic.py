from __future__ import annotations

"""M-series M2.2 synthetic benchmark for Champion-relative Direction objectives."""

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TAU=0.002
TRAIN_GROUPS=384
EVAL_GROUPS=192
SCENARIOS=6
LATENT_DIM=8
FEATURE_DIM=256
DIRECTIONS=3
SHIFTS=(1,7,13,23,31)
OBJECTIVES=("ABS_CE","SELECTIVE_CHAMPION_PRESERVE_CE","POSITIVE_ADVANTAGE_MIRROR_CE")
EPOCHS=12
BATCH_SIZE=512
LR=3e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=10.0
BASE_SEED=24680
RESIDUAL_PARAMS=16643


@dataclass(frozen=True)
class RegimeM22:
    name:str
    seeds:tuple[int,...]
    conditional_signal_scale:float
    global_action_bias_scale:float
    teacher_noise_std:float
    champion_truth_correlation:float


REGIMES=(
    RegimeM22("CLEAN_CONDITIONAL",(1101,1102,1103,1104,1105),0.004,0.001,0.0005,0.55),
    RegimeM22("NOISY_TEACHER",(1201,1202,1203,1204,1205),0.004,0.001,0.002,0.55),
    RegimeM22("WEAK_CONDITIONAL_STRONG_MARGINAL",(1301,1302,1303,1304,1305),0.0015,0.0025,0.001,0.55),
    RegimeM22("NEAR_OPTIMAL_CHAMPION",(1401,1402,1403,1404,1405),0.004,0.001,0.001,0.90),
)


def require(cond:bool,code:str)->None:
    if not cond: raise RuntimeError(code)


def _norm_cols(x:np.ndarray)->np.ndarray:
    d=np.sqrt(np.sum(x*x,axis=0,keepdims=True))
    return x/np.maximum(d,1e-12)


def _softmax_np(logits:np.ndarray)->np.ndarray:
    z=np.asarray(logits,dtype=np.float64)
    z=z-np.max(z,axis=1,keepdims=True)
    e=np.exp(np.clip(z,-60.0,60.0))
    return e/np.sum(e,axis=1,keepdims=True)


def _scenario_onehot(groups:int)->tuple[np.ndarray,np.ndarray]:
    ids=np.tile(np.arange(SCENARIOS,dtype=np.int64),groups)
    one=np.eye(SCENARIOS,dtype=np.float64)[ids]
    return ids,one


def _make_split(*,rng:np.random.Generator,groups:int,params:Mapping[str,np.ndarray],regime:RegimeM22,teacher_noise:bool)->dict[str,np.ndarray]:
    z_group=rng.standard_normal((groups,LATENT_DIM))
    z=np.repeat(z_group,SCENARIOS,axis=0)
    scenario,one=_scenario_onehot(groups)
    lin=z@params["w_true"]
    non=np.tanh(z@params["w_non"])
    scen=params["scenario_true"][scenario]
    true_score=lin+0.5*non+0.35*scen
    bias=regime.global_action_bias_scale*params["bias"]
    mu_true=bias[None,:]+regime.conditional_signal_scale*true_score

    rand_score=z@params["w_champ"]+0.35*params["scenario_champ"][scenario]
    rho=float(regime.champion_truth_correlation)
    champ_cond=rho*true_score+math.sqrt(max(0.0,1.0-rho*rho))*rand_score
    champ_mu=bias[None,:]+regime.conditional_signal_scale*champ_cond
    base_logits=champ_mu/TAU
    base_probs=_softmax_np(base_logits)

    base14=np.concatenate([z,one],axis=1)
    features=np.tanh(base14@params["feature_proj"])
    out={
        "features":features.astype(np.float32),
        "base_logits":base_logits.astype(np.float32),
        "base_probs":base_probs.astype(np.float32),
        "mu_true":mu_true.astype(np.float64),
        "scenario":scenario,
        "group":np.repeat(np.arange(groups,dtype=np.int64),SCENARIOS),
    }
    if teacher_noise:
        mu_teacher=mu_true+rng.normal(0.0,regime.teacher_noise_std,size=mu_true.shape)
        out["mu_teacher"]=mu_teacher.astype(np.float64)
        out["teacher_probs"]=_softmax_np(mu_teacher/TAU).astype(np.float32)
    return out


def generate_world_m22(seed:int,regime:RegimeM22)->dict[str,Any]:
    rng=np.random.default_rng(int(seed))
    w_true=_norm_cols(rng.standard_normal((LATENT_DIM,DIRECTIONS)))
    w_non=_norm_cols(rng.standard_normal((LATENT_DIM,DIRECTIONS)))
    w_champ=_norm_cols(rng.standard_normal((LATENT_DIM,DIRECTIONS)))
    scenario_true=rng.standard_normal((SCENARIOS,DIRECTIONS)); scenario_true-=scenario_true.mean(axis=0,keepdims=True)
    scenario_champ=rng.standard_normal((SCENARIOS,DIRECTIONS)); scenario_champ-=scenario_champ.mean(axis=0,keepdims=True)
    bias=rng.standard_normal(DIRECTIONS); bias-=bias.mean(); bias/=max(float(np.std(bias)),1e-12)
    feature_proj=rng.standard_normal((LATENT_DIM+SCENARIOS,FEATURE_DIM))/math.sqrt(LATENT_DIM+SCENARIOS)
    params={"w_true":w_true,"w_non":w_non,"w_champ":w_champ,"scenario_true":scenario_true,"scenario_champ":scenario_champ,"bias":bias,"feature_proj":feature_proj}
    train=_make_split(rng=rng,groups=TRAIN_GROUPS,params=params,regime=regime,teacher_noise=True)
    eval_=_make_split(rng=rng,groups=EVAL_GROUPS,params=params,regime=regime,teacher_noise=False)
    return {"seed":int(seed),"regime":regime.name,"train":train,"eval":eval_}


def rotate_teacher_surface_m22(mu_teacher:np.ndarray,shift:int)->np.ndarray:
    require(int(shift) in SHIFTS,f"M22_UNREGISTERED_SHIFT:{shift}")
    x=np.asarray(mu_teacher,dtype=np.float64)
    require(x.shape==(TRAIN_GROUPS*SCENARIOS,DIRECTIONS),"M22_TEACHER_SURFACE_SHAPE")
    cube=x.reshape(TRAIN_GROUPS,SCENARIOS,DIRECTIONS)
    out=np.roll(cube,shift=-int(shift),axis=0).reshape(x.shape)
    # exact scenario-wise multiset preservation
    for s in range(SCENARIOS):
        before=np.sort(cube[:,s,:],axis=0)
        after=np.sort(out.reshape(TRAIN_GROUPS,SCENARIOS,DIRECTIONS)[:,s,:],axis=0)
        require(np.array_equal(before,after),f"M22_SHUFFLE_MULTISET_DRIFT:{shift}:{s}")
    return out.copy()


class ResidualM22(nn.Module):
    def __init__(self)->None:
        super().__init__(); self.h=nn.Linear(FEATURE_DIM,64); self.o=nn.Linear(64,DIRECTIONS); self.act=nn.SiLU()
        with torch.no_grad(): self.o.weight.zero_(); self.o.bias.zero_()
    def forward(self,x:torch.Tensor)->torch.Tensor: return self.o(self.act(self.h(x)))


def build_residual_m22(device:str|torch.device)->ResidualM22:
    torch.manual_seed(BASE_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(BASE_SEED)
    m=ResidualM22().to(device=device,dtype=torch.float32)
    require(sum(p.numel() for p in m.parameters())==RESIDUAL_PARAMS,"M22_RESIDUAL_PARAM_COUNT")
    return m


def objective_target_m22(*,objective:str,mu_teacher:np.ndarray,base_probs:np.ndarray)->tuple[np.ndarray,dict[str,Any]]:
    require(objective in OBJECTIVES,f"M22_UNKNOWN_OBJECTIVE:{objective}")
    mu=np.asarray(mu_teacher,dtype=np.float64); p0=np.asarray(base_probs,dtype=np.float64)
    require(mu.shape==p0.shape and mu.shape[1]==3,"M22_TARGET_SHAPE")
    pt=_softmax_np(mu/TAU)
    g0=np.argmax(p0,axis=1); tb=np.argmax(mu,axis=1); rows=np.arange(len(mu))
    preserve=tb==g0
    if objective=="ABS_CE":
        q=pt
    elif objective=="SELECTIVE_CHAMPION_PRESERVE_CE":
        q=pt.copy(); q[preserve]=p0[preserve]
    else:
        aplus=np.maximum(mu-mu[rows,g0][:,None],0.0)
        logq=np.log(np.clip(p0,np.finfo(np.float64).tiny,1.0))+aplus/TAU
        q=_softmax_np(logq)
        require(np.max(np.abs(q[preserve]-p0[preserve]))<=2e-15,"M22_ADV_MIRROR_EXACT_PRESERVE_DRIFT")
    return q.astype(np.float32),{
        "objective":objective,"rows":int(len(q)),"teacher_g0_agreement_rows":int(np.sum(preserve)),
        "exact_preservation_rows":int(np.sum(np.max(np.abs(q-p0),axis=1)<=2e-15)),
    }


def epoch_permutations_m22(rows:int,device:str|torch.device)->tuple[torch.Tensor,...]:
    out=[]
    for e in range(EPOCHS):
        g=torch.Generator(device="cpu"); g.manual_seed(BASE_SEED+e)
        out.append(torch.randperm(rows,generator=g,dtype=torch.long).to(device))
    return tuple(out)


def train_residual_m22(*,features:torch.Tensor,base_logits:torch.Tensor,target:torch.Tensor,permutations:Sequence[torch.Tensor],device:str)->tuple[ResidualM22,dict[str,Any]]:
    m=build_residual_m22(device); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    with torch.inference_mode():
        d=m(features); require(float(torch.max(torch.abs(d)).cpu())==0.0,"M22_INITIAL_DELTA_NOT_ZERO")
    steps=0
    for perm in permutations:
        for start in range(0,len(features),BATCH_SIZE):
            ids=perm[start:start+BATCH_SIZE]; opt.zero_grad(set_to_none=True)
            logits=base_logits.index_select(0,ids)+m(features.index_select(0,ids))
            q=target.index_select(0,ids).detach(); loss=-(q*F.log_softmax(logits,dim=-1)).sum(dim=-1).mean()
            require(bool(torch.isfinite(loss).item()),"M22_NONFINITE_LOSS"); loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(m.parameters(),GRAD_CLIP); require(bool(torch.isfinite(gn).item()),"M22_NONFINITE_GRAD")
            opt.step(); steps+=1
    return m,{"optimizer":"AdamW_FP32","epochs":EPOCHS,"batch_size":BATCH_SIZE,"lr":LR,"weight_decay":WEIGHT_DECAY,"steps":steps}


def evaluate_true_m22(*,residual:ResidualM22,features:torch.Tensor,base_logits:torch.Tensor,base_probs:np.ndarray,mu_true:np.ndarray)->dict[str,Any]:
    residual.eval()
    with torch.inference_mode(): p=torch.softmax(base_logits+residual(features),dim=-1).cpu().numpy().astype(np.float64)
    p0=np.asarray(base_probs,dtype=np.float64); mu=np.asarray(mu_true,dtype=np.float64); rows=np.arange(len(mu))
    g0=np.argmax(p0,axis=1); new=np.argmax(p,axis=1); best=np.argmax(mu,axis=1)
    gain=mu[rows,new]-mu[rows,g0]; available=np.maximum(mu[rows,best]-mu[rows,g0],0.0); captured=np.maximum(gain,0.0); harmful=np.maximum(-gain,0.0)
    denom=float(np.sum(available)); require(denom>0.0,"M22_NO_TRUE_AVAILABLE_ADVANTAGE")
    return {
        "rows":int(len(mu)),"true_discrete_champion_relative_gain":float(np.mean(gain)),
        "true_available_advantage_capture_fraction":float(np.sum(captured)/denom),
        "true_harmful_cost_mean":float(np.mean(harmful)),"true_negative_gain_rate":float(np.mean(gain<0.0)),
        "true_direction_change_rate":float(np.mean(new!=g0)),"true_move_to_best_rate_on_disagreement":float(np.mean((new[best!=g0]==best[best!=g0]).astype(np.float64))) if np.any(best!=g0) else None,
    }


def run_seed_cell_m22(*,seed:int,regime:RegimeM22,device:str)->dict[str,Any]:
    world=generate_world_m22(seed,regime); tr=world["train"]; ev=world["eval"]
    ft=torch.from_numpy(tr["features"]).to(device); bl=torch.from_numpy(tr["base_logits"]).to(device)
    fe=torch.from_numpy(ev["features"]).to(device); ble=torch.from_numpy(ev["base_logits"]).to(device)
    perms=epoch_permutations_m22(len(ft),device)
    controls={"ALIGNED":tr["mu_teacher"]}
    for s in SHIFTS: controls[f"SHUFFLE_{s}"]=rotate_teacher_surface_m22(tr["mu_teacher"],s)
    arms={}
    for obj in OBJECTIVES:
        for cname,mu_t in controls.items():
            q,trec=objective_target_m22(objective=obj,mu_teacher=mu_t,base_probs=tr["base_probs"])
            target=torch.from_numpy(q).to(device)
            model,trainrec=train_residual_m22(features=ft,base_logits=bl,target=target,permutations=perms,device=device)
            evrec=evaluate_true_m22(residual=model,features=fe,base_logits=ble,base_probs=ev["base_probs"],mu_true=ev["mu_true"])
            arms[f"{obj}__{cname}"]={"target":trec,"training":trainrec,"evaluation":evrec}
            del model,target
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"regime":regime.name,"seed":int(seed),"arms":arms,"train_rows":len(ft),"eval_rows":len(fe)}


def adjudicate_m22(cells:Sequence[Mapping[str,Any]])->dict[str,Any]:
    require(len(cells)==20,f"M22_CELL_COUNT:{len(cells)}")
    per_regime=[]; all_pass=True
    for regime in REGIMES:
        rows=sorted([c for c in cells if c["regime"]==regime.name],key=lambda x:int(x["seed"]))
        require(tuple(int(x["seed"]) for x in rows)==regime.seeds,f"M22_SEED_SET:{regime.name}")
        align_wins=harm_wins=capture_wins=0; per_seed=[]
        for c in rows:
            arms=c["arms"]
            adv=arms["POSITIVE_ADVANTAGE_MIRROR_CE__ALIGNED"]["evaluation"]
            adv_sh=[arms[f"POSITIVE_ADVANTAGE_MIRROR_CE__SHUFFLE_{s}"]["evaluation"] for s in SHIFTS]
            abs_=arms["ABS_CE__ALIGNED"]["evaluation"]; sel=arms["SELECTIVE_CHAMPION_PRESERVE_CE__ALIGNED"]["evaluation"]
            med_gain=float(statistics.median(float(x["true_discrete_champion_relative_gain"]) for x in adv_sh))
            a= float(adv["true_discrete_champion_relative_gain"]); h=float(adv["true_harmful_cost_mean"]); cap=float(adv["true_available_advantage_capture_fraction"])
            wa=a>med_gain; wh=h<float(abs_["true_harmful_cost_mean"]); wc=cap>float(sel["true_available_advantage_capture_fraction"])
            align_wins+=int(wa); harm_wins+=int(wh); capture_wins+=int(wc)
            per_seed.append({"seed":int(c["seed"]),"adv_aligned_gain":a,"median_adv_shuffle_gain":med_gain,"adv_harm":h,"abs_harm":float(abs_["true_harmful_cost_mean"]),"adv_capture":cap,"selective_capture":float(sel["true_available_advantage_capture_fraction"]),"alignment_win":wa,"harm_win":wh,"capture_win":wc})
        rp=align_wins>=4 and harm_wins>=4 and capture_wins>=3; all_pass=all_pass and rp
        per_regime.append({"regime":regime.name,"alignment_specific_gain_win_count":align_wins,"harm_vs_abs_win_count":harm_wins,"capture_vs_selective_win_count":capture_wins,"pass":rp,"per_seed":per_seed})
    return {"schema":"CB16_R11_M2_2_SYNTHETIC_ADJUDICATION_V1","overall_pass":bool(all_pass),"conclusion":"POSITIVE_ADVANTAGE_MIRROR_OBJECTIVE_SYNTHETICALLY_QUALIFIED_FOR_CONSUMED_SUPPORT_SHADOW_SCREEN" if all_pass else "POSITIVE_ADVANTAGE_MIRROR_OBJECTIVE_NOT_SYNTHETICALLY_QUALIFIED","per_regime":per_regime,"market_information_verdict":False,"canonical_change_authorized":False,"m3_sizing_authorized":False}
