from __future__ import annotations

"""M2.2E synthetic benchmark: direct positive Champion-relative expected advantage.

This is the terminal planned Direction-objective family screen before broader
mechanistic reassessment.  It does not imitate a Teacher probability target or
ranking target.  The candidate directly rewards Student policy mass assigned to
actions whose Teacher mean utility exceeds the frozen Champion action's Teacher
mean utility.  Synthetic world, residual, optimizer, shuffles, and true evaluator
remain inherited and unchanged.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import torch

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
from .m_series_m2_2d_pairwise_utility_ranking_synthetic import (
    PAIRWISE_OBJECTIVE,
    pairwise_surface_m22d,
    train_residual_pairwise_m22d,
)

DIRECT_REGRET_OBJECTIVE="DIRECT_POSITIVE_CHAMPION_RELATIVE_EXPECTED_ADVANTAGE"
OBJECTIVES_M22E=OBJECTIVES_M22B+(PAIRWISE_OBJECTIVE,DIRECT_REGRET_OBJECTIVE)

REGIMES_M22E=(
    RegimeM22("CLEAN_CONDITIONAL",(5101,5102,5103,5104,5105),0.004,0.001,0.0005,0.55),
    RegimeM22("NOISY_TEACHER",(5201,5202,5203,5204,5205),0.004,0.001,0.002,0.55),
    RegimeM22("WEAK_CONDITIONAL_STRONG_MARGINAL",(5301,5302,5303,5304,5305),0.0015,0.0025,0.001,0.55),
    RegimeM22("NEAR_OPTIMAL_CHAMPION",(5401,5402,5403,5404,5405),0.004,0.001,0.001,0.90),
)


def require(cond:bool,code:str)->None:
    if not cond:
        raise RuntimeError(code)


def direct_positive_advantage_surface_m22e(
    *,mu_teacher:np.ndarray,base_logits:np.ndarray
)->tuple[np.ndarray,dict[str,Any]]:
    mu=np.asarray(mu_teacher,dtype=np.float64)
    z0=np.asarray(base_logits,dtype=np.float64)
    require(mu.shape==z0.shape and mu.ndim==2 and mu.shape[1]==3,"M22E_SURFACE_SHAPE")
    require(np.isfinite(mu).all() and np.isfinite(z0).all(),"M22E_NONFINITE_SURFACE")
    champion=np.argmax(z0,axis=1)
    rows=np.arange(len(mu))
    baseline=mu[rows,champion]
    aplus=np.maximum(mu-baseline[:,None],0.0)
    positive=aplus>0.0
    require(np.any(positive),"M22E_NO_POSITIVE_CHAMPION_RELATIVE_ADVANTAGE")
    mean_positive=float(np.mean(aplus[positive]))
    require(np.isfinite(mean_positive) and mean_positive>0.0,"M22E_INVALID_POSITIVE_ADVANTAGE_MEAN")
    normalized=aplus/mean_positive
    teacher_best=np.argmax(mu,axis=1)
    agreement=teacher_best==champion
    require(np.all(normalized[agreement]==0.0),"M22E_AGREEMENT_ROW_NONZERO_ADVANTAGE")
    require(abs(float(np.mean(normalized[normalized>0.0]))-1.0)<=1e-12,"M22E_POSITIVE_ADVANTAGE_NORMALIZATION_DRIFT")
    return normalized.astype(np.float32),{
        "objective":DIRECT_REGRET_OBJECTIVE,
        "rows":int(len(mu)),
        "champion_teacher_agreement_rows":int(np.sum(agreement)),
        "champion_teacher_disagreement_rows":int(np.sum(~agreement)),
        "strictly_positive_action_entries":int(np.sum(positive)),
        "rows_with_any_positive_action":int(np.sum(np.any(positive,axis=1))),
        "mean_raw_positive_advantage":mean_positive,
        "mean_normalized_positive_advantage":float(np.mean(normalized[normalized>0.0])),
        "agreement_rows_exact_zero_advantage":bool(np.all(normalized[agreement]==0.0)),
        "teacher_probability_target":"NONE",
        "pairwise_ranking_target":"NONE",
        "champion_probability_prior_in_loss":"NONE",
        "preservation_ce_or_kl_term":"NONE",
        "temperature_threshold_margin_or_tunable_scale":"NONE",
    }


def direct_expected_advantage_loss_m22e(
    logits:torch.Tensor,normalized_positive_advantage:torch.Tensor
)->torch.Tensor:
    require(logits.shape==normalized_positive_advantage.shape,"M22E_DIRECT_LOSS_SHAPE")
    require(logits.ndim==2 and logits.shape[1]==3,"M22E_DIRECT_LOGIT_SHAPE")
    probs=torch.softmax(logits,dim=-1)
    row_value=(probs*normalized_positive_advantage).sum(dim=-1)
    return -row_value.mean()


def train_residual_direct_regret_m22e(
    *,features:torch.Tensor,base_logits:torch.Tensor,normalized_positive_advantage:torch.Tensor,
    permutations:Sequence[torch.Tensor],device:str
):
    require(normalized_positive_advantage.shape==base_logits.shape,"M22E_TRAIN_ADVANTAGE_SHAPE")
    require(len(features)==len(base_logits),"M22E_TRAIN_ROW_MISMATCH")
    model=build_residual_m22(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    with torch.inference_mode():
        delta=model(features)
        require(float(torch.max(torch.abs(delta)).cpu())==0.0,"M22E_INITIAL_DELTA_NOT_ZERO")
    steps=0
    for perm in permutations:
        for start in range(0,len(features),BATCH_SIZE):
            ids=perm[start:start+BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits=base_logits.index_select(0,ids)+model(features.index_select(0,ids))
            adv=normalized_positive_advantage.index_select(0,ids).detach()
            loss=direct_expected_advantage_loss_m22e(logits,adv)
            require(bool(torch.isfinite(loss).item()),"M22E_NONFINITE_DIRECT_REGRET_LOSS")
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            require(bool(torch.isfinite(grad_norm).item()),"M22E_NONFINITE_GRAD")
            optimizer.step(); steps+=1
    return model,{
        "optimizer":"AdamW_FP32",
        "epochs":EPOCHS,
        "batch_size":BATCH_SIZE,
        "lr":LR,
        "weight_decay":WEIGHT_DECAY,
        "steps":steps,
        "objective":"MAXIMIZE_NORMALIZED_POSITIVE_TEACHER_ADVANTAGE_OVER_FROZEN_CHAMPION",
        "teacher_probability_target":False,
        "pairwise_ranking_target":False,
    }


def run_seed_cell_m22e(*,seed:int,regime:RegimeM22,device:str)->dict[str,Any]:
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
    for objective in OBJECTIVES_M22E:
        for control_name,mu_teacher in controls.items():
            if objective==DIRECT_REGRET_OBJECTIVE:
                adv_np,objective_receipt=direct_positive_advantage_surface_m22e(
                    mu_teacher=mu_teacher,base_logits=train["base_logits"]
                )
                adv=torch.from_numpy(adv_np).to(device)
                model,training=train_residual_direct_regret_m22e(
                    features=train_features,base_logits=train_logits,
                    normalized_positive_advantage=adv,permutations=permutations,device=device
                )
                del adv
            elif objective==PAIRWISE_OBJECTIVE:
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


def adjudicate_m22e(cells:Sequence[Mapping[str,Any]])->dict[str,Any]:
    require(len(cells)==20,f"M22E_CELL_COUNT:{len(cells)}")
    per_regime=[]; overall=True
    for regime in REGIMES_M22E:
        cells_r=sorted([c for c in cells if c["regime"]==regime.name],key=lambda c:int(c["seed"]))
        require(tuple(int(c["seed"]) for c in cells_r)==regime.seeds,f"M22E_SEED_SET:{regime.name}")
        alignment=capture=gain=harm_abs=harm_aws=0
        per_seed=[]
        for cell in cells_r:
            arms=cell["arms"]
            direct=arms[f"{DIRECT_REGRET_OBJECTIVE}__ALIGNED"]["evaluation"]
            direct_sh=[arms[f"{DIRECT_REGRET_OBJECTIVE}__SHUFFLE_{s}"]["evaluation"] for s in SHIFTS]
            sel=arms["SELECTIVE_CHAMPION_PRESERVE_CE__ALIGNED"]["evaluation"]
            aws=arms["ADVANTAGE_WEIGHTED_SELECTIVE_CE__ALIGNED"]["evaluation"]
            abs_=arms["ABS_CE__ALIGNED"]["evaluation"]
            pair=arms[f"{PAIRWISE_OBJECTIVE}__ALIGNED"]["evaluation"]
            direct_gain=float(direct["true_discrete_champion_relative_gain"])
            med_shuffle=float(statistics.median(float(a["true_discrete_champion_relative_gain"]) for a in direct_sh))
            direct_capture=float(direct["true_available_advantage_capture_fraction"])
            sel_capture=float(sel["true_available_advantage_capture_fraction"])
            sel_gain=float(sel["true_discrete_champion_relative_gain"])
            direct_harm=float(direct["true_harmful_cost_mean"])
            abs_harm=float(abs_["true_harmful_cost_mean"])
            aws_harm=float(aws["true_harmful_cost_mean"])
            wa=direct_gain>med_shuffle
            wc=direct_capture>sel_capture
            wg=direct_gain>sel_gain
            wha=direct_harm<abs_harm
            whw=direct_harm<aws_harm
            alignment+=int(wa); capture+=int(wc); gain+=int(wg); harm_abs+=int(wha); harm_aws+=int(whw)
            per_seed.append({
                "seed":int(cell["seed"]),
                "direct_gain":direct_gain,
                "median_direct_shuffle_gain":med_shuffle,
                "direct_capture":direct_capture,
                "selective_capture":sel_capture,
                "selective_gain":sel_gain,
                "direct_harm":direct_harm,
                "abs_harm":abs_harm,
                "aws_harm":aws_harm,
                "pairwise_gain_descriptive":float(pair["true_discrete_champion_relative_gain"]),
                "pairwise_harm_descriptive":float(pair["true_harmful_cost_mean"]),
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
        "schema":"CB16_R11_M2_2E_SYNTHETIC_ADJUDICATION_V1",
        "overall_pass":bool(overall),
        "conclusion":(
            "DIRECT_CHAMPION_RELATIVE_DECISION_VALUE_SURROGATE_SYNTHETICALLY_QUALIFIED_FOR_SEPARATELY_PREREGISTERED_CONSUMED_TRAIN_ONLY_SHADOW"
            if overall else
            "DIRECT_CHAMPION_RELATIVE_DECISION_VALUE_SURROGATE_NOT_SYNTHETICALLY_QUALIFIED__STOP_DIRECTION_OBJECTIVE_SEARCH_BEFORE_BROADER_MECHANISTIC_REASSESSMENT"
        ),
        "per_regime":per_regime,
        "market_information_verdict":False,
        "canonical_change_authorized":False,
        "m3_sizing_authorized":False,
    }
