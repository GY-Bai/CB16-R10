from __future__ import annotations

"""M2.2D synthetic benchmark: all-pairs Teacher utility ranking.

This exits the CE target/reweighting family. The synthetic world, residual model,
optimizer, shuffles, and true-utility evaluator are inherited unchanged. The only
scientific intervention is the Direction loss for the candidate arm: all three
pairwise final-logit margins are trained toward the sign of the corresponding
Teacher mean-utility gaps, weighted by globally normalized absolute gap size.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .m_series_m2_2_advantage_mirror_synthetic import (
    SHIFTS,
    EPOCHS,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    GRAD_CLIP,
    RegimeM22,
    generate_world_m22,
    rotate_teacher_surface_m22,
    build_residual_m22,
    epoch_permutations_m22,
    evaluate_true_m22,
)
from .m_series_m2_2b_advantage_weighted_selective_synthetic import (
    OBJECTIVES_M22B,
    target_and_multiplier_m22b,
    train_residual_weighted_m22b,
)

PAIRWISE_OBJECTIVE="PAIRWISE_GAP_WEIGHTED_UTILITY_RANKING"
OBJECTIVES_M22D=OBJECTIVES_M22B+(PAIRWISE_OBJECTIVE,)
DIRECTION_PAIRS=((0,1),(0,2),(1,2))

REGIMES_M22D=(
    RegimeM22("CLEAN_CONDITIONAL",(4101,4102,4103,4104,4105),0.004,0.001,0.0005,0.55),
    RegimeM22("NOISY_TEACHER",(4201,4202,4203,4204,4205),0.004,0.001,0.002,0.55),
    RegimeM22("WEAK_CONDITIONAL_STRONG_MARGINAL",(4301,4302,4303,4304,4305),0.0015,0.0025,0.001,0.55),
    RegimeM22("NEAR_OPTIMAL_CHAMPION",(4401,4402,4403,4404,4405),0.004,0.001,0.001,0.90),
)


def require(cond:bool,code:str)->None:
    if not cond:
        raise RuntimeError(code)


def pairwise_surface_m22d(mu_teacher:np.ndarray)->tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    mu=np.asarray(mu_teacher,dtype=np.float64)
    require(mu.ndim==2 and mu.shape[1]==3,"M22D_TEACHER_SURFACE_SHAPE")
    deltas=np.stack([mu[:,a]-mu[:,b] for a,b in DIRECTION_PAIRS],axis=1)
    require(np.isfinite(deltas).all(),"M22D_NONFINITE_TEACHER_GAP")
    zero_count=int(np.sum(deltas==0.0))
    require(zero_count==0,"M22D_EXACT_ZERO_TEACHER_GAP")
    signs=np.sign(deltas)
    abs_gap=np.abs(deltas)
    mean_abs=float(np.mean(abs_gap))
    require(np.isfinite(mean_abs) and mean_abs>0.0,"M22D_INVALID_MEAN_ABS_GAP")
    weights=abs_gap/mean_abs
    require(np.all(weights>0.0) and np.isfinite(weights).all(),"M22D_INVALID_PAIR_WEIGHT")
    require(abs(float(np.mean(weights))-1.0)<=1e-12,"M22D_PAIR_WEIGHT_NORMALIZATION_DRIFT")
    return signs.astype(np.float32),weights.astype(np.float32),{
        "objective":PAIRWISE_OBJECTIVE,
        "rows":int(len(mu)),
        "direction_pairs":[list(p) for p in DIRECTION_PAIRS],
        "pair_count_per_row":len(DIRECTION_PAIRS),
        "zero_teacher_gap_count":zero_count,
        "mean_abs_teacher_gap":mean_abs,
        "mean_pair_weight":float(np.mean(weights)),
        "probability_target":"NONE",
        "champion_probability_prior_in_loss":"NONE",
        "temperature_threshold_or_margin_hyperparameter":"NONE",
    }


def pairwise_gap_weighted_loss_m22d(
    logits:torch.Tensor,signs:torch.Tensor,weights:torch.Tensor
)->torch.Tensor:
    require(logits.ndim==2 and logits.shape[1]==3,"M22D_LOGIT_SHAPE")
    require(signs.shape==weights.shape and signs.ndim==2 and signs.shape[1]==3,"M22D_PAIR_TENSOR_SHAPE")
    require(signs.shape[0]==logits.shape[0],"M22D_PAIR_ROW_MISMATCH")
    margins=torch.stack([
        logits[:,0]-logits[:,1],
        logits[:,0]-logits[:,2],
        logits[:,1]-logits[:,2],
    ],dim=1)
    return (weights*F.softplus(-signs*margins)).mean()


def train_residual_pairwise_m22d(
    *,features:torch.Tensor,base_logits:torch.Tensor,signs:torch.Tensor,
    weights:torch.Tensor,permutations:Sequence[torch.Tensor],device:str
):
    require(signs.ndim==2 and signs.shape[1]==3,"M22D_TRAIN_SIGN_SHAPE")
    require(weights.shape==signs.shape,"M22D_TRAIN_WEIGHT_SHAPE")
    require(len(signs)==len(features)==len(base_logits),"M22D_TRAIN_ROW_MISMATCH")
    model=build_residual_m22(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    with torch.inference_mode():
        delta=model(features)
        require(float(torch.max(torch.abs(delta)).cpu())==0.0,"M22D_INITIAL_DELTA_NOT_ZERO")
    steps=0
    for perm in permutations:
        for start in range(0,len(features),BATCH_SIZE):
            ids=perm[start:start+BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits=base_logits.index_select(0,ids)+model(features.index_select(0,ids))
            s=signs.index_select(0,ids).detach()
            w=weights.index_select(0,ids).detach()
            loss=pairwise_gap_weighted_loss_m22d(logits,s,w)
            require(bool(torch.isfinite(loss).item()),"M22D_NONFINITE_PAIRWISE_LOSS")
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            require(bool(torch.isfinite(grad_norm).item()),"M22D_NONFINITE_GRAD")
            optimizer.step(); steps+=1
    return model,{
        "optimizer":"AdamW_FP32",
        "epochs":EPOCHS,
        "batch_size":BATCH_SIZE,
        "lr":LR,
        "weight_decay":WEIGHT_DECAY,
        "steps":steps,
        "objective":"ALL_PAIRS_GAP_WEIGHTED_SOFTPLUS_RANKING",
        "direction_pairs":[list(p) for p in DIRECTION_PAIRS],
    }


def run_seed_cell_m22d(*,seed:int,regime:RegimeM22,device:str)->dict[str,Any]:
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
    for objective in OBJECTIVES_M22D:
        for control_name,mu_teacher in controls.items():
            if objective==PAIRWISE_OBJECTIVE:
                sign_np,weight_np,objective_receipt=pairwise_surface_m22d(mu_teacher)
                signs=torch.from_numpy(sign_np).to(device)
                weights=torch.from_numpy(weight_np).to(device)
                model,training=train_residual_pairwise_m22d(
                    features=train_features,base_logits=train_logits,signs=signs,
                    weights=weights,permutations=permutations,device=device
                )
                del signs,weights
            else:
                target_np,mult_np,objective_receipt=target_and_multiplier_m22b(
                    objective=objective,mu_teacher=mu_teacher,base_probs=train["base_probs"]
                )
                target=torch.from_numpy(target_np).to(device)
                multiplier=torch.from_numpy(mult_np).to(device)
                model,training=train_residual_weighted_m22b(
                    features=train_features,base_logits=train_logits,target=target,
                    multiplier=multiplier,permutations=permutations,device=device
                )
                del target,multiplier
            evaluation=evaluate_true_m22(
                residual=model,features=eval_features,base_logits=eval_logits,
                base_probs=eval_["base_probs"],mu_true=eval_["mu_true"]
            )
            arms[f"{objective}__{control_name}"]={
                "objective":objective_receipt,
                "training":training,
                "evaluation":evaluation,
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {
        "regime":regime.name,
        "seed":int(seed),
        "train_rows":int(len(train_features)),
        "eval_rows":int(len(eval_features)),
        "arms":arms,
    }


def adjudicate_m22d(cells:Sequence[Mapping[str,Any]])->dict[str,Any]:
    require(len(cells)==20,f"M22D_CELL_COUNT:{len(cells)}")
    per_regime=[]; overall=True
    for regime in REGIMES_M22D:
        cells_r=sorted([c for c in cells if c["regime"]==regime.name],key=lambda c:int(c["seed"]))
        require(tuple(int(c["seed"]) for c in cells_r)==regime.seeds,f"M22D_SEED_SET:{regime.name}")
        alignment=capture=gain=harm_abs=harm_aws=0
        per_seed=[]
        for cell in cells_r:
            arms=cell["arms"]
            pair=arms[f"{PAIRWISE_OBJECTIVE}__ALIGNED"]["evaluation"]
            pair_sh=[arms[f"{PAIRWISE_OBJECTIVE}__SHUFFLE_{s}"]["evaluation"] for s in SHIFTS]
            sel=arms["SELECTIVE_CHAMPION_PRESERVE_CE__ALIGNED"]["evaluation"]
            aws=arms["ADVANTAGE_WEIGHTED_SELECTIVE_CE__ALIGNED"]["evaluation"]
            abs_=arms["ABS_CE__ALIGNED"]["evaluation"]
            pair_gain=float(pair["true_discrete_champion_relative_gain"])
            med_shuffle=float(statistics.median(float(a["true_discrete_champion_relative_gain"]) for a in pair_sh))
            pair_capture=float(pair["true_available_advantage_capture_fraction"])
            sel_capture=float(sel["true_available_advantage_capture_fraction"])
            sel_gain=float(sel["true_discrete_champion_relative_gain"])
            pair_harm=float(pair["true_harmful_cost_mean"])
            abs_harm=float(abs_["true_harmful_cost_mean"])
            aws_harm=float(aws["true_harmful_cost_mean"])
            wa=pair_gain>med_shuffle
            wc=pair_capture>sel_capture
            wg=pair_gain>sel_gain
            wha=pair_harm<abs_harm
            whw=pair_harm<aws_harm
            alignment+=int(wa); capture+=int(wc); gain+=int(wg); harm_abs+=int(wha); harm_aws+=int(whw)
            per_seed.append({
                "seed":int(cell["seed"]),
                "pairwise_gain":pair_gain,
                "median_pairwise_shuffle_gain":med_shuffle,
                "pairwise_capture":pair_capture,
                "selective_capture":sel_capture,
                "selective_gain":sel_gain,
                "pairwise_harm":pair_harm,
                "abs_harm":abs_harm,
                "aws_harm":aws_harm,
                "alignment_win":wa,
                "capture_win":wc,
                "gain_vs_selective_win":wg,
                "harm_vs_abs_win":wha,
                "harm_vs_aws_win":whw,
            })
        regime_pass=(alignment>=4 and capture>=3 and gain>=3 and harm_abs>=4 and harm_aws>=3)
        overall=overall and regime_pass
        per_regime.append({
            "regime":regime.name,
            "alignment_specific_gain_win_count":alignment,
            "capture_vs_selective_win_count":capture,
            "gain_vs_selective_win_count":gain,
            "harm_vs_abs_win_count":harm_abs,
            "harm_vs_aws_win_count":harm_aws,
            "pass":regime_pass,
            "per_seed":per_seed,
        })
    return {
        "schema":"CB16_R11_M2_2D_SYNTHETIC_ADJUDICATION_V1",
        "overall_pass":bool(overall),
        "conclusion":(
            "PAIRWISE_UTILITY_RANKING_SYNTHETICALLY_QUALIFIED_FOR_SEPARATELY_PREREGISTERED_CONSUMED_TRAIN_ONLY_SHADOW"
            if overall else
            "PAIRWISE_UTILITY_RANKING_NOT_SYNTHETICALLY_QUALIFIED__DO_NOT_OPEN_MARKET_SHADOW"
        ),
        "per_regime":per_regime,
        "market_information_verdict":False,
        "canonical_change_authorized":False,
        "m3_sizing_authorized":False,
    }
