from __future__ import annotations

"""M2.2B synthetic benchmark: advantage-weighted state selection with unchanged selective targets.

This module intentionally reuses the already exercised M2.2 synthetic world, residual,
shuffle, and true-evaluation machinery.  The only scientific intervention is the
per-row multiplier for Teacher/G0 disagreement states.
"""

from dataclasses import dataclass
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .m_series_m2_2_advantage_mirror_synthetic import (
    TAU,
    SHIFTS,
    EPOCHS,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    GRAD_CLIP,
    BASE_SEED,
    RegimeM22,
    _softmax_np,
    generate_world_m22,
    rotate_teacher_surface_m22,
    build_residual_m22,
    epoch_permutations_m22,
    evaluate_true_m22,
)

OBJECTIVES_M22B=(
    "ABS_CE",
    "SELECTIVE_CHAMPION_PRESERVE_CE",
    "ADVANTAGE_WEIGHTED_SELECTIVE_CE",
)

REGIMES_M22B=(
    RegimeM22("CLEAN_CONDITIONAL",(2101,2102,2103,2104,2105),0.004,0.001,0.0005,0.55),
    RegimeM22("NOISY_TEACHER",(2201,2202,2203,2204,2205),0.004,0.001,0.002,0.55),
    RegimeM22("WEAK_CONDITIONAL_STRONG_MARGINAL",(2301,2302,2303,2304,2305),0.0015,0.0025,0.001,0.55),
    RegimeM22("NEAR_OPTIMAL_CHAMPION",(2401,2402,2403,2404,2405),0.004,0.001,0.001,0.90),
)


def require(cond:bool,code:str)->None:
    if not cond:
        raise RuntimeError(code)


def target_and_multiplier_m22b(
    *,objective:str,mu_teacher:np.ndarray,base_probs:np.ndarray
)->tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    require(objective in OBJECTIVES_M22B,f"M22B_UNKNOWN_OBJECTIVE:{objective}")
    mu=np.asarray(mu_teacher,dtype=np.float64)
    p0=np.asarray(base_probs,dtype=np.float64)
    require(mu.shape==p0.shape and mu.ndim==2 and mu.shape[1]==3,"M22B_TARGET_SHAPE")
    teacher_probs=_softmax_np(mu/TAU)
    g0=np.argmax(p0,axis=1)
    teacher_best=np.argmax(mu,axis=1)
    rows=np.arange(len(mu))
    disagreement=teacher_best!=g0
    q=teacher_probs.copy()
    multiplier=np.ones(len(mu),dtype=np.float64)

    if objective in ("SELECTIVE_CHAMPION_PRESERVE_CE","ADVANTAGE_WEIGHTED_SELECTIVE_CE"):
        q[~disagreement]=p0[~disagreement]

    advantage=np.maximum(mu[rows,teacher_best]-mu[rows,g0],0.0)
    if objective=="ADVANTAGE_WEIGHTED_SELECTIVE_CE":
        require(np.any(disagreement),"M22B_NO_DISAGREEMENT_ROWS")
        require(np.all(advantage[disagreement]>0.0),"M22B_NONPOSITIVE_DISAGREEMENT_ADVANTAGE")
        mean_adv=float(np.mean(advantage[disagreement]))
        require(np.isfinite(mean_adv) and mean_adv>0.0,"M22B_INVALID_MEAN_DISAGREEMENT_ADVANTAGE")
        multiplier[disagreement]=advantage[disagreement]/mean_adv
        require(abs(float(np.mean(multiplier[disagreement]))-1.0)<=1e-12,"M22B_DISAGREEMENT_WEIGHT_NORMALIZATION_DRIFT")

    q32=q.astype(np.float32)
    m32=multiplier.astype(np.float32)
    require(np.isfinite(q32).all() and np.isfinite(m32).all(),"M22B_NONFINITE_TARGET_OR_WEIGHT")
    require(np.all(m32>0.0),"M22B_NONPOSITIVE_ROW_MULTIPLIER")
    if objective=="ADVANTAGE_WEIGHTED_SELECTIVE_CE":
        # Intervention isolation: AWS changes weights only, never the selective target.
        q_selective=teacher_probs.copy(); q_selective[~disagreement]=p0[~disagreement]
        require(np.array_equal(q32,q_selective.astype(np.float32)),"M22B_AWS_TARGET_DRIFT_FROM_SELECTIVE")
    return q32,m32,{
        "objective":objective,
        "rows":int(len(mu)),
        "agreement_rows":int(np.sum(~disagreement)),
        "disagreement_rows":int(np.sum(disagreement)),
        "mean_disagreement_teacher_advantage":None if not np.any(disagreement) else float(np.mean(advantage[disagreement])),
        "mean_disagreement_multiplier":None if not np.any(disagreement) else float(np.mean(multiplier[disagreement])),
        "overall_multiplier_mean":float(np.mean(multiplier)),
        "target_sha_basis":"SELECTIVE_TARGET_IDENTICAL_FOR_SELECTIVE_AND_AWS" if objective=="ADVANTAGE_WEIGHTED_SELECTIVE_CE" else objective,
    }


def train_residual_weighted_m22b(
    *,features:torch.Tensor,base_logits:torch.Tensor,target:torch.Tensor,
    multiplier:torch.Tensor,permutations:Sequence[torch.Tensor],device:str
):
    require(target.ndim==2 and target.shape[1]==3,"M22B_TRAIN_TARGET_SHAPE")
    require(multiplier.ndim==1 and len(multiplier)==len(target),"M22B_TRAIN_MULTIPLIER_SHAPE")
    model=build_residual_m22(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    with torch.inference_mode():
        delta=model(features)
        require(float(torch.max(torch.abs(delta)).cpu())==0.0,"M22B_INITIAL_DELTA_NOT_ZERO")
    steps=0
    for perm in permutations:
        for start in range(0,len(features),BATCH_SIZE):
            ids=perm[start:start+BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits=base_logits.index_select(0,ids)+model(features.index_select(0,ids))
            q=target.index_select(0,ids).detach()
            w=multiplier.index_select(0,ids).detach()
            row_loss=-(q*F.log_softmax(logits,dim=-1)).sum(dim=-1)
            loss=(row_loss*w).mean()
            require(bool(torch.isfinite(loss).item()),"M22B_NONFINITE_LOSS")
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            require(bool(torch.isfinite(grad_norm).item()),"M22B_NONFINITE_GRAD")
            optimizer.step(); steps+=1
    return model,{
        "optimizer":"AdamW_FP32",
        "epochs":EPOCHS,
        "batch_size":BATCH_SIZE,
        "lr":LR,
        "weight_decay":WEIGHT_DECAY,
        "steps":steps,
        "weighted_row_ce":True,
    }


def run_seed_cell_m22b(*,seed:int,regime:RegimeM22,device:str)->dict[str,Any]:
    world=generate_world_m22(seed,regime)
    train=world["train"]; eval_=world["eval"]
    train_features=torch.from_numpy(train["features"]).to(device)
    train_logits=torch.from_numpy(train["base_logits"]).to(device)
    eval_features=torch.from_numpy(eval_["features"]).to(device)
    eval_logits=torch.from_numpy(eval_["base_logits"]).to(device)
    permutations=epoch_permutations_m22(len(train_features),device)
    controls={"ALIGNED":train["mu_teacher"]}
    for shift in SHIFTS:
        controls[f"SHUFFLE_{shift}"]=rotate_teacher_surface_m22(train["mu_teacher"],shift)

    arms:dict[str,Any]={}
    for objective in OBJECTIVES_M22B:
        for control_name,mu_teacher in controls.items():
            q,mult,objective_receipt=target_and_multiplier_m22b(
                objective=objective,mu_teacher=mu_teacher,base_probs=train["base_probs"]
            )
            target=torch.from_numpy(q).to(device)
            multiplier=torch.from_numpy(mult).to(device)
            model,train_receipt=train_residual_weighted_m22b(
                features=train_features,base_logits=train_logits,target=target,
                multiplier=multiplier,permutations=permutations,device=device
            )
            evaluation=evaluate_true_m22(
                residual=model,features=eval_features,base_logits=eval_logits,
                base_probs=eval_["base_probs"],mu_true=eval_["mu_true"]
            )
            arms[f"{objective}__{control_name}"]={
                "objective":objective_receipt,
                "training":train_receipt,
                "evaluation":evaluation,
            }
            del model,target,multiplier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {
        "regime":regime.name,
        "seed":int(seed),
        "train_rows":int(len(train_features)),
        "eval_rows":int(len(eval_features)),
        "arms":arms,
    }


def adjudicate_m22b(cells:Sequence[Mapping[str,Any]])->dict[str,Any]:
    require(len(cells)==20,f"M22B_CELL_COUNT:{len(cells)}")
    per_regime=[]
    overall=True
    for regime in REGIMES_M22B:
        rows=sorted([c for c in cells if c["regime"]==regime.name],key=lambda c:int(c["seed"]))
        require(tuple(int(c["seed"]) for c in rows)==regime.seeds,f"M22B_SEED_SET:{regime.name}")
        alignment_wins=capture_wins=gain_vs_selective_wins=harm_wins=0
        per_seed=[]
        for cell in rows:
            arms=cell["arms"]
            aws=arms["ADVANTAGE_WEIGHTED_SELECTIVE_CE__ALIGNED"]["evaluation"]
            aws_shuffles=[arms[f"ADVANTAGE_WEIGHTED_SELECTIVE_CE__SHUFFLE_{s}"]["evaluation"] for s in SHIFTS]
            selective=arms["SELECTIVE_CHAMPION_PRESERVE_CE__ALIGNED"]["evaluation"]
            absolute=arms["ABS_CE__ALIGNED"]["evaluation"]
            aws_gain=float(aws["true_discrete_champion_relative_gain"])
            median_shuffle_gain=float(statistics.median(float(v["true_discrete_champion_relative_gain"]) for v in aws_shuffles))
            aws_capture=float(aws["true_available_advantage_capture_fraction"])
            selective_capture=float(selective["true_available_advantage_capture_fraction"])
            selective_gain=float(selective["true_discrete_champion_relative_gain"])
            aws_harm=float(aws["true_harmful_cost_mean"])
            abs_harm=float(absolute["true_harmful_cost_mean"])
            alignment_win=aws_gain>median_shuffle_gain
            capture_win=aws_capture>selective_capture
            gain_win=aws_gain>selective_gain
            harm_win=aws_harm<abs_harm
            alignment_wins+=int(alignment_win); capture_wins+=int(capture_win)
            gain_vs_selective_wins+=int(gain_win); harm_wins+=int(harm_win)
            per_seed.append({
                "seed":int(cell["seed"]),
                "aws_gain":aws_gain,
                "median_aws_shuffle_gain":median_shuffle_gain,
                "aws_capture":aws_capture,
                "selective_capture":selective_capture,
                "selective_gain":selective_gain,
                "aws_harm":aws_harm,
                "abs_harm":abs_harm,
                "alignment_win":alignment_win,
                "capture_win":capture_win,
                "gain_vs_selective_win":gain_win,
                "harm_win":harm_win,
            })
        regime_pass=(alignment_wins>=4 and capture_wins>=3 and gain_vs_selective_wins>=3 and harm_wins>=4)
        overall=overall and regime_pass
        per_regime.append({
            "regime":regime.name,
            "alignment_specific_gain_win_count":alignment_wins,
            "capture_vs_selective_win_count":capture_wins,
            "gain_vs_selective_win_count":gain_vs_selective_wins,
            "harm_vs_abs_win_count":harm_wins,
            "pass":regime_pass,
            "per_seed":per_seed,
        })
    return {
        "schema":"CB16_R11_M2_2B_SYNTHETIC_ADJUDICATION_V1",
        "overall_pass":bool(overall),
        "conclusion":(
            "ADVANTAGE_WEIGHTED_STATE_SELECTION_SYNTHETICALLY_QUALIFIED_FOR_CONSUMED_TRAIN_ONLY_SHADOW_SCREEN"
            if overall else
            "ADVANTAGE_WEIGHTED_STATE_SELECTION_NOT_SYNTHETICALLY_QUALIFIED__DO_NOT_OPEN_MARKET_SHADOW"
        ),
        "per_regime":per_regime,
        "market_information_verdict":False,
        "canonical_change_authorized":False,
        "m3_sizing_authorized":False,
    }
