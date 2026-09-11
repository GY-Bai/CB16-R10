#!/usr/bin/env python3
from __future__ import annotations

"""CB16 R11 M-series M2.1 — correction-quality attribution on exact M2 arms."""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from cb16_local_opt.m_series_m2_1_correction_quality import (
    M21_RUNTIME,
    adjudicate_m21,
    correction_quality_m21,
)
from cb16_local_opt.m_series_m2_frozen_g0_direction_residual import (
    M2_OBJECTIVES,
    M2_SHIFTS,
    adjudicate_m2,
    best_direction_means_from_evidence_m2,
    build_objective_target_m2,
    cache_frozen_g0_direction_m2,
    evaluate_direction_residual_m2,
    prepare_epoch_permutations_m2,
    train_direction_residual_m2,
)
from cb16_local_opt.training_runtime_r11 import policy_hash_r11
from scripts import r11_m_series_m2_frozen_g0_direction_residual as m2exec
from scripts import r11_science_g0_reduced_teacher_target_information_audit_r6 as r6

SCHEMA="CB16_R11_M_SERIES_M2_1_CORRECTION_QUALITY_ATTRIBUTION_RESULT_V1"
M21_PREREG_COMMIT="d0d02013a0fcdb95c9b01c9477bccc1f8896a396"
M21_GATE_BLOB="9c9488e83e19dfbeaa7264b291c561f2aa87d79b"
M2_ADJUDICATION_BLOB="e459883f2be01ee5240c0105a930ddf7241fed45"
M2_MODULE_BLOB="3c73b9487f1ea3f70b029b8e8636ea6e68dc23d8"
R7_FREEZE_COMMIT="62ad74a0c62ab616dfceaf70c6f7b30f7088c81f"
FROZEN_STATUS="DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
GAIN_TOL=1e-10
RATE_TOL=1e-10


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def git(*args: str) -> str:
    return subprocess.check_output(["git",*args],text=True).strip()


def atomic_json(path: Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def verify_repo_contract_m21()->dict[str,Any]:
    for ancestor,code in ((R7_FREEZE_COMMIT,"M21_NOT_DESCENDED_FROM_R7_FREEZE"),(M21_PREREG_COMMIT,"M21_NOT_DESCENDED_FROM_PREREGISTRATION")):
        require(subprocess.call(["git","merge-base","--is-ancestor",ancestor,"HEAD"])==0,code)
    exact={
        "m21_gate":("authority/rearchitecture_r11/CB16_R11_M_SERIES_M2_1_CORRECTION_QUALITY_ATTRIBUTION_GATE_V1.json",M21_GATE_BLOB),
        "m2_adjudication":("authority/rearchitecture_r11/CB16_R11_M_SERIES_M2_FROZEN_G0_DIRECTION_RESIDUAL_ADJUDICATION_V1.json",M2_ADJUDICATION_BLOB),
        "m2_gate":("authority/rearchitecture_r11/CB16_R11_M_SERIES_M2_FROZEN_G0_DIRECTION_RESIDUAL_GATE_V1.json",m2exec.M2_GATE_BLOB),
        "m2_module":("cb16_local_opt/m_series_m2_frozen_g0_direction_residual.py",M2_MODULE_BLOB),
        "r6_executor":("scripts/r11_science_g0_reduced_teacher_target_information_audit_r6.py",m2exec.R6_EXECUTOR_BLOB),
    }
    observed={}
    for name,(path,expected) in exact.items():
        got=git("rev-parse",f"HEAD:{path}")
        require(got==expected,f"M21_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name]=got
    return {"execution_head":git("rev-parse","HEAD"),"m21_preregistration_commit":M21_PREREG_COMMIT,"immutable_blobs":observed}


def _reference_adjudication()->dict[str,Any]:
    p=Path("authority/rearchitecture_r11/CB16_R11_M_SERIES_M2_FROZEN_G0_DIRECTION_RESIDUAL_ADJUDICATION_V1.json")
    x=json.loads(p.read_text(encoding="utf-8"))
    require(x["source"]["artifact_digest"]=="sha256:585d0f5a6780150f163ca255ebf44dc797f1599638fdfa5a8cc2b4e899323dd7","M21_REFERENCE_ARTIFACT_DRIFT")
    return x


def _arm_name(objective:str,control:str)->str:
    return ("ABS_CE" if objective=="ABS_CE" else "SELECTIVE")+"_"+control


def _reproduction_readout(fold_result:Mapping[str,Any])->dict[str,float]:
    arms=fold_result["arms"]
    s_sh=[arms[f"SELECTIVE_SHUFFLE_{s}"]["evaluation"] for s in M2_SHIFTS]
    a_sh=[arms[f"ABS_CE_SHUFFLE_{s}"]["evaluation"] for s in M2_SHIFTS]
    sa=arms["SELECTIVE_ALIGNED"]["evaluation"]
    aa=arms["ABS_CE_ALIGNED"]["evaluation"]
    return {
        "selective_aligned_gain":float(sa["discrete_champion_relative_gain"]),
        "median_selective_shuffle_gain":float(statistics.median(float(x["discrete_champion_relative_gain"]) for x in s_sh)),
        "selective_aligned_move_to_teacher_rate":float(sa["teacher_g0_disagreement_move_to_teacher_rate"]),
        "median_selective_shuffle_move_to_teacher_rate":float(statistics.median(float(x["teacher_g0_disagreement_move_to_teacher_rate"]) for x in s_sh)),
        "selective_aligned_agreement_change_rate":float(sa["teacher_g0_agreement_change_rate"]),
        "abs_ce_aligned_agreement_change_rate":float(aa["teacher_g0_agreement_change_rate"]),
        "abs_ce_aligned_gain":float(aa["discrete_champion_relative_gain"]),
        "median_abs_ce_shuffle_gain":float(statistics.median(float(x["discrete_champion_relative_gain"]) for x in a_sh)),
    }


def _verify_fold_reproduction(fold_result:Mapping[str,Any],reference:Mapping[str,Any])->dict[str,Any]:
    got=_reproduction_readout(fold_result)
    ref=next(x for x in reference["per_fold_primary_readout"] if int(x["fold"])==int(fold_result["fold"]))
    errors={}
    for k,v in got.items():
        err=abs(float(v)-float(ref[k])); errors[k]=err
        tol=GAIN_TOL if "gain" in k else RATE_TOL
        require(err<=tol,f"M21_M2_REPRODUCTION_DRIFT:FOLD={fold_result['fold']}:{k}:{err}")
    return {"fold":int(fold_result["fold"]),"max_abs_error":max(errors.values()),"errors":errors,"pass":True}


def run_fold_m21(*,fold_spec:Mapping[str,Any],all_train_parents:Mapping[str,Any],all_train_samples:list[Any],g0_root:Path,device:str)->dict[str,Any]:
    fold=int(fold_spec["fold"])
    parents,train_evidence,eval_evidence,teacher_receipt=r6.build_fold_evidence(
        fold_spec=fold_spec,all_train_parents=all_train_parents,all_train_samples=all_train_samples)
    campaign=r6.r4.prepare_evidence_campaign_r11(train_evidence=train_evidence,validation_evidence=eval_evidence,parents=parents,device=device)
    train_rich=best_direction_means_from_evidence_m2(train_evidence,campaign.train.parent_ids)
    eval_rich=best_direction_means_from_evidence_m2(eval_evidence,campaign.validation.parent_ids)
    m2exec._teacher_best_matches_reduced(campaign.train,train_rich,f"TRAIN_FOLD_{fold}")
    m2exec._teacher_best_matches_reduced(campaign.validation,eval_rich,f"EVAL_FOLD_{fold}")
    g0=r6.r4.r1.load_bootstrap_model(g0_root,device)
    g0_hash=policy_hash_r11(g0)
    train_cache=cache_frozen_g0_direction_m2(g0,campaign.train)
    eval_cache=cache_frozen_g0_direction_m2(g0,campaign.validation)
    permutations=prepare_epoch_permutations_m2(campaign.train.rows,device=device)
    surfaces=m2exec._build_train_surfaces(campaign,train_rich)
    arms={}
    controls=("ALIGNED",*[f"SHUFFLE_{s}" for s in M2_SHIFTS])
    for objective in M2_OBJECTIVES:
        for control in controls:
            src=surfaces[control]["prepared"]
            target,target_receipt=build_objective_target_m2(objective=objective,teacher_probs=src.direction_target_probs,g0_probs=train_cache.base_probs)
            residual,training=train_direction_residual_m2(cache=train_cache,target_probs=target,group_weight=campaign.train.group_weight,permutations=permutations,device=device)
            evaluation=evaluate_direction_residual_m2(residual=residual,cache=eval_cache,aligned_teacher_probs=campaign.validation.direction_target_probs,aligned_best_direction_means=eval_rich,group_weight=campaign.validation.group_weight)
            quality=correction_quality_m21(residual=residual,cache=eval_cache,aligned_best_direction_means=eval_rich,group_weight=campaign.validation.group_weight)
            require(abs(float(quality["net_gain_mean"])-float(evaluation["discrete_champion_relative_gain"]))<=1e-12,f"M21_NET_GAIN_IDENTITY_DRIFT:FOLD={fold}:{objective}:{control}")
            require(abs(float(quality["raw_move_to_teacher_rate"])-float(evaluation["teacher_g0_disagreement_move_to_teacher_rate"]))<=1e-12,f"M21_RAW_MOVE_IDENTITY_DRIFT:FOLD={fold}:{objective}:{control}")
            arms[_arm_name(objective,control)]={"objective":objective,"control":control,"target":target_receipt,"training":training,"evaluation":evaluation,"quality":quality,"control_receipt":surfaces[control]["control_receipt"]}
            del residual,target
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    require(policy_hash_r11(g0)==g0_hash,f"M21_G0_MUTATED:FOLD={fold}")
    return {"fold":fold,"outer_support":dict(fold_spec),"teacher":teacher_receipt,"train_rows":campaign.train.rows,"eval_rows":campaign.validation.rows,"arms":arms}


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--g0-root",type=Path,default=Path(os.environ.get("CB16_G0_ROOT","/cb16/g0")))
    ap.add_argument("--package-root",type=Path,default=Path(os.environ.get("CB16_PACKAGE_ROOT","/cb16/package")))
    ap.add_argument("--r104-root",type=Path,default=Path(os.environ.get("CB16_R104_ROOT","/cb16/runtime/r104")))
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()
    g0_root=args.g0_root.resolve(); package_root=args.package_root.resolve(); r104_root=args.r104_root.resolve(); output=args.output.resolve()
    repo=verify_repo_contract_m21(); reference=_reference_adjudication()
    runtime=r6.r4.r1.runtime_identity(args.device)
    require(runtime["python_series"]==[3,10],f"M21_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    _lineage,g0_identity=r6.r4.r1.verify_g0_authority(g0_root)
    frozen_before=r6.frozen_authority_hashes(package_root); g0_before=dict(g0_identity["authority_file_hashes"])
    all_train_parents,all_train_samples,source=r6.load_train_only_support(r104_root)
    folds=r6.build_outer_folds_r6(all_train_parents)
    completed=[]; reproduction=[]
    for spec in folds:
        f=run_fold_m21(fold_spec=spec,all_train_parents=all_train_parents,all_train_samples=all_train_samples,g0_root=g0_root,device=args.device)
        reproduction.append(_verify_fold_reproduction(f,reference)); completed.append(f)
    require(len(completed)==5 and len(reproduction)==5,"M21_INCOMPLETE_FOLDS")
    m2_summary=adjudicate_m2(completed)
    require(m2_summary["selective_alignment_pass"] is False,"M21_M2_PRIMARY_REPRODUCTION_STATUS_DRIFT")
    require(m2_summary["preservation_pass"] is True,"M21_M2_PRESERVATION_REPRODUCTION_STATUS_DRIFT")
    require(m2_summary["absolute_residual_alignment_secondary_pass"] is True,"M21_M2_ABS_REPRODUCTION_STATUS_DRIFT")
    m21_summary=adjudicate_m21(completed)
    require(r6.frozen_authority_hashes(package_root)==frozen_before,"M21_FROZEN_PACKAGE_MUTATED")
    _,g0_after=r6.r4.r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"]==g0_before,"M21_G0_AUTHORITY_MUTATED")
    result={
        "schema":SCHEMA,"status":"M2_1_CORRECTION_QUALITY_ATTRIBUTION_COMPLETE","runtime":M21_RUNTIME,
        "frozen_scientific_status_before":FROZEN_STATUS,"frozen_scientific_status_after":FROZEN_STATUS,
        "repo":repo,"runtime_identity":runtime,"source":source,
        "protocol":{"post_m2_followup":True,"m2_retroactive_pass_forbidden":True,"exact_m2_training_reproduced":True,"new_independent_support":False,"g0_frozen":True,"sizing_gradient":False,"r7_candidate_evaluated":False,"final_holdout_opened":False,"fresh_market_data":False,"non_status_driving":True},
        "m2_reproduction":{"reference_adjudication_blob":M2_ADJUDICATION_BLOB,"gain_abs_tolerance":GAIN_TOL,"rate_abs_tolerance":RATE_TOL,"per_fold":reproduction,"status":"PASS"},
        "fold_results":completed,"reproduced_m2_summary":m2_summary,"m2_1_summary":m21_summary,
        "semantic_guards":{"canonical_generation_advanced":False,"champion_promoted":False,"production_cutover":False,"market_information_verdict_reopened":False,"profitability_or_alpha_claimed":False,"m2_retroactively_changed":False,"r7_candidate_evaluated":False,"final_holdout_payload_opened":False,"fresh_market_data_downloaded":False,"r5_purge_support_reopened":False,"legacy_validation_rows_used":False,"raw_market_payload_used":False,"frozen_package_unchanged":True,"g0_authority_unchanged":True},
        "next_legal_step":(
            "M2_2_DIRECT_CORRECTION_VALUE_OBJECTIVE_SHADOW_DESIGN_MAY_BE_PREREGISTERED__M2_REMAINS_FAILED" if (m21_summary["correction_quality_pass"] and m21_summary["harm_suppression_pass"])
            else "ADVANTAGE_WEIGHTED_DIRECTION_OBJECTIVE_SYNTHETIC_OR_SHADOW_DESIGN_REQUIRED_BEFORE_SIZING" if m21_summary["harm_suppression_pass"]
            else "RETURN_TO_DIRECTION_OBJECTIVE_OR_TEACHER_TRANSPORT_ASSUMPTIONS_BEFORE_EXPANSION"),
    }
    atomic_json(output,result)
    print(json.dumps({"status":result["status"],"m2_reproduction":result["m2_reproduction"]["status"],"correction_quality_conclusion":m21_summary["correction_quality_conclusion"],"harm_suppression_conclusion":m21_summary["harm_suppression_conclusion"],"joint_interpretation":m21_summary["joint_interpretation"],"per_fold":m21_summary["per_fold"],"next_legal_step":result["next_legal_step"]},indent=2,sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
