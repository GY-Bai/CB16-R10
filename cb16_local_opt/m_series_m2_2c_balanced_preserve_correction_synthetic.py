from __future__ import annotations

"""M2.2C synthetic benchmark: explicit balanced preservation and correction components.

The target distribution, Teacher surface, residual, optimizer, and advantage definition
are inherited from M2.2B.  The only intervention is global row weighting that makes
agreement-preservation and disagreement-correction each contribute exactly half of
the full-data weighted CE objective before stochastic minibatch sampling.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .m_series_m2_2_advantage_mirror_synthetic import (
    SHIFTS,
    RegimeM22,
    generate_world_m22,
    rotate_teacher_surface_m22,
    epoch_permutations_m22,
    evaluate_true_m22,
)
from .m_series_m2_2b_advantage_weighted_selective_synthetic import (
    OBJECTIVES_M22B,
    target_and_multiplier_m22b,
    train_residual_weighted_m22b,
)

BALANCED_OBJECTIVE="BALANCED_PRESERVE_ADVANTAGE_CORRECTION_CE"
OBJECTIVES_M22C=OBJECTIVES_M22B+(BALANCED_OBJECTIVE,)

REGIMES_M22C=(
    RegimeM22("CLEAN_CONDITIONAL",(3101,3102,3103,3104,3105),0.004,0.001,0.0005,0.55),
    RegimeM22("NOISY_TEACHER",(3201,3202,3203,3204,3205),0.004,0.001,0.002,0.55),
    RegimeM22("WEAK_CONDITIONAL_STRONG_MARGINAL",(3301,3302,3303,3304,3305),0.0015,0.0025,0.001,0.55),
    RegimeM22("NEAR_OPTIMAL_CHAMPION",(3401,3402,3403,3404,3405),0.004,0.001,0.001,0.90),
)


def require(cond:bool,code:str)->None:
    if not cond:
        raise RuntimeError(code)


def balanced_target_multiplier_m22c(
    *,mu_teacher:np.ndarray,base_probs:np.ndarray
)->tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    # Start from the exact M2.2B AWS selective target and advantage semantics.
    q,_,base_receipt=target_and_multiplier_m22b(
        objective="ADVANTAGE_WEIGHTED_SELECTIVE_CE",
        mu_teacher=mu_teacher,
        base_probs=base_probs,
    )
    mu=np.asarray(mu_teacher,dtype=np.float64)
    p0=np.asarray(base_probs,dtype=np.float64)
    g0=np.argmax(p0,axis=1)
    teacher_best=np.argmax(mu,axis=1)
    disagreement=teacher_best!=g0
    agreement=~disagreement
    require(np.any(agreement),"M22C_NO_AGREEMENT_ROWS")
    require(np.any(disagreement),"M22C_NO_DISAGREEMENT_ROWS")
    rows=np.arange(len(mu))
    advantage=np.maximum(mu[rows,teacher_best]-mu[rows,g0],0.0)
    require(np.all(advantage[disagreement]>0.0),"M22C_NONPOSITIVE_DISAGREEMENT_ADVANTAGE")
    mean_adv=float(np.mean(advantage[disagreement]))
    require(np.isfinite(mean_adv) and mean_adv>0.0,"M22C_INVALID_MEAN_ADVANTAGE")

    n=float(len(mu)); n_agree=float(np.sum(agreement)); n_dis=float(np.sum(disagreement))
    f_agree=n_agree/n; f_dis=n_dis/n
    multiplier=np.empty(len(mu),dtype=np.float64)
    multiplier[agreement]=0.5/f_agree
    multiplier[disagreement]=(0.5/f_dis)*(advantage[disagreement]/mean_adv)

    # Full-data mean(row_loss * multiplier) == .5*mean_agreement + .5*weighted_mean_disagreement.
    agree_mass=float(np.sum(multiplier[agreement])/n)
    correction_mass=float(np.sum(multiplier[disagreement])/n)
    require(abs(agree_mass-0.5)<=1e-12,"M22C_PRESERVATION_COMPONENT_MASS_DRIFT")
    require(abs(correction_mass-0.5)<=1e-12,"M22C_CORRECTION_COMPONENT_MASS_DRIFT")
    require(abs(float(np.mean(multiplier))-1.0)<=1e-12,"M22C_TOTAL_MULTIPLIER_MEAN_DRIFT")
    require(np.all(multiplier>0.0) and np.isfinite(multiplier).all(),"M22C_INVALID_MULTIPLIER")

    return q,multiplier.astype(np.float32),{
        "objective":BALANCED_OBJECTIVE,
        "rows":int(len(mu)),
        "agreement_rows":int(n_agree),
        "disagreement_rows":int(n_dis),
        "mean_disagreement_teacher_advantage":mean_adv,
        "preservation_component_mass":agree_mass,
        "correction_component_mass":correction_mass,
        "overall_multiplier_mean":float(np.mean(multiplier)),
        "component_weights":[0.5,0.5],
        "target_identity":"BYTE_IDENTICAL_TO_M2_2B_AWS_AND_BINARY_SELECTIVE_TARGET",
        "difference_from_m2_2b":"GLOBAL_COMPONENT_BALANCING_ONLY",
        "inherited_target_receipt":base_receipt,
    }


def run_seed_cell_m22c(*,seed:int,regime:RegimeM22,device:str)->dict[str,Any]:
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
    for objective in OBJECTIVES_M22C:
        for control_name,mu_teacher in controls.items():
            if objective==BALANCED_OBJECTIVE:
                target_np,mult_np,objective_receipt=balanced_target_multiplier_m22c(
                    mu_teacher=mu_teacher,base_probs=train["base_probs"]
                )
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
            evaluation=evaluate_true_m22(
                residual=model,features=eval_features,base_logits=eval_logits,
                base_probs=eval_["base_probs"],mu_true=eval_["mu_true"]
            )
            arms[f"{objective}__{control_name}"]={
                "objective":objective_receipt,
                "training":training,
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


def adjudicate_m22c(cells:Sequence[Mapping[str,Any]])->dict[str,Any]:
    require(len(cells)==20,f"M22C_CELL_COUNT:{len(cells)}")
    per_regime=[]; overall=True
    for regime in REGIMES_M22C:
        cells_r=sorted([c for c in cells if c["regime"]==regime.name],key=lambda c:int(c["seed"]))
        require(tuple(int(c["seed"]) for c in cells_r)==regime.seeds,f"M22C_SEED_SET:{regime.name}")
        alignment=capture=gain=harm_abs=harm_aws=0
        per_seed=[]
        for cell in cells_r:
            arms=cell["arms"]
            bal=arms[f"{BALANCED_OBJECTIVE}__ALIGNED"]["evaluation"]
            bal_sh=[arms[f"{BALANCED_OBJECTIVE}__SHUFFLE_{s}"]["evaluation"] for s in SHIFTS]
            sel=arms["SELECTIVE_CHAMPION_PRESERVE_CE__ALIGNED"]["evaluation"]
            aws=arms["ADVANTAGE_WEIGHTED_SELECTIVE_CE__ALIGNED"]["evaluation"]
            abs_=arms["ABS_CE__ALIGNED"]["evaluation"]
            bal_gain=float(bal["true_discrete_champion_relative_gain"])
            med_shuffle=float(statistics.median(float(a["true_discrete_champion_relative_gain"]) for a in bal_sh))
            bal_capture=float(bal["true_available_advantage_capture_fraction"])
            sel_capture=float(sel["true_available_advantage_capture_fraction"])
            sel_gain=float(sel["true_discrete_champion_relative_gain"])
            bal_harm=float(bal["true_harmful_cost_mean"])
            aws_harm=float(aws["true_harmful_cost_mean"])
            abs_harm=float(abs_["true_harmful_cost_mean"])
            wa=bal_gain>med_shuffle
            wc=bal_capture>sel_capture
            wg=bal_gain>sel_gain
            wha=bal_harm<abs_harm
            whw=bal_harm<aws_harm
            alignment+=int(wa); capture+=int(wc); gain+=int(wg); harm_abs+=int(wha); harm_aws+=int(whw)
            per_seed.append({
                "seed":int(cell["seed"]),
                "balanced_gain":bal_gain,
                "median_balanced_shuffle_gain":med_shuffle,
                "balanced_capture":bal_capture,
                "selective_capture":sel_capture,
                "selective_gain":sel_gain,
                "balanced_harm":bal_harm,
                "aws_harm":aws_harm,
                "abs_harm":abs_harm,
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
        "schema":"CB16_R11_M2_2C_SYNTHETIC_ADJUDICATION_V1",
        "overall_pass":bool(overall),
        "conclusion":(
            "TWO_TERM_PRESERVATION_PLUS_VALUE_SENSITIVE_CORRECTION_SYNTHETICALLY_QUALIFIED_FOR_CONSUMED_TRAIN_ONLY_SHADOW"
            if overall else
            "TWO_TERM_DIRECTION_OBJECTIVE_NOT_SYNTHETICALLY_QUALIFIED__DO_NOT_OPEN_MARKET_SHADOW"
        ),
        "per_regime":per_regime,
        "market_information_verdict":False,
        "canonical_change_authorized":False,
        "m3_sizing_authorized":False,
    }
