#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import torch

from cb16_local_opt.m_series_m2_2e_direct_champion_relative_regret_synthetic import (
    REGIMES_M22E,
    adjudicate_m22e,
    run_seed_cell_m22e,
)

SCHEMA="CB16_R11_M_SERIES_M2_2E_DIRECT_CHAMPION_RELATIVE_REGRET_SYNTHETIC_RESULT_V1"
PREREG_COMMIT="e9b7c12d8ad92622d89d7e47d442790d5e486ff1"
GATE_BLOB="638516b1b948dc074d7b90824334c3e9b276d959"
M22D_ADJ_BLOB="e6ce2a9507a1a943b6ca3f3d0a4c1e681276df89"
FROZEN_STATUS="DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"


def require(cond:bool,code:str)->None:
    if not cond: raise RuntimeError(code)


def git(*args:str)->str:
    return subprocess.check_output(["git",*args],text=True).strip()


def atomic_json(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def verify_repo()->dict[str,Any]:
    require(subprocess.call(["git","merge-base","--is-ancestor",PREREG_COMMIT,"HEAD"])==0,"M22E_NOT_DESCENDED_FROM_PREREGISTRATION")
    exact={
        "gate":("authority/rearchitecture_r11/CB16_R11_M_SERIES_M2_2E_DIRECT_CHAMPION_RELATIVE_REGRET_SYNTHETIC_GATE_V1.json",GATE_BLOB),
        "m22d_adjudication":("authority/rearchitecture_r11/CB16_R11_M_SERIES_M2_2D_PAIRWISE_UTILITY_RANKING_SYNTHETIC_ADJUDICATION_V1.json",M22D_ADJ_BLOB),
    }
    observed={}
    for name,(path,expected) in exact.items():
        got=git("rev-parse",f"HEAD:{path}")
        require(got==expected,f"M22E_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name]=got
    return {"execution_head":git("rev-parse","HEAD"),"preregistration_commit":PREREG_COMMIT,"immutable_blobs":observed}


def runtime_identity(device:str)->dict[str,Any]:
    dev=torch.device(device)
    out={"python":platform.python_version(),"torch":torch.__version__,"requested_device":str(dev),"cuda_available":bool(torch.cuda.is_available())}
    if dev.type=="cuda":
        require(torch.cuda.is_available(),"M22E_CUDA_REQUIRED_BUT_UNAVAILABLE")
        out.update({"cuda_device_name":torch.cuda.get_device_name(dev),"cuda_capability":list(torch.cuda.get_device_capability(dev)),"torch_cuda":torch.version.cuda})
    return out


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()
    repo=verify_repo(); runtime=runtime_identity(args.device)
    cells=[]
    for regime in REGIMES_M22E:
        for seed in regime.seeds:
            cells.append(run_seed_cell_m22e(seed=seed,regime=regime,device=args.device))
    summary=adjudicate_m22e(cells)
    result={
        "schema":SCHEMA,
        "status":"M2_2E_DIRECT_CHAMPION_RELATIVE_REGRET_SYNTHETIC_COMPLETE",
        "repo":repo,
        "runtime_identity":runtime,
        "frozen_scientific_status_before":FROZEN_STATUS,
        "frozen_scientific_status_after":FROZEN_STATUS,
        "protocol":{
            "synthetic_only":True,
            "market_data_read":False,
            "market_evidence_read":False,
            "prior_primary_seed_cells_reused":False,
            "cells":20,
            "arms_per_cell":30,
            "total_arms":600,
            "candidate_teacher_probability_target":False,
            "candidate_pairwise_ranking_target":False,
            "candidate_preservation_ce_or_kl":False,
            "candidate_temperature_threshold_margin_or_tunable_scale":False,
            "candidate_positive_advantage_normalization_deterministic":True,
            "new_trainable_parameters":0,
            "final_holdout_opened":False,
            "r7_candidate_evaluated":False,
            "canonical_change":False,
        },
        "cells":cells,
        "m2_2e_summary":summary,
        "semantic_guards":{
            "market_data_used":False,
            "market_evidence_used":False,
            "canonical_generation_advanced":False,
            "champion_promoted":False,
            "production_cutover":False,
            "market_information_verdict_reopened":False,
            "final_holdout_payload_opened":False,
            "r7_candidate_evaluated":False,
            "m3_sizing_authorized":False,
        },
        "next_legal_step":(
            "ADJUDICATE_M2_2E_THEN_PREREGISTER_CONSUMED_R6_TRAIN_ONLY_DIRECT_REGRET_DIRECTION_SHADOW"
            if summary["overall_pass"] else
            "ADJUDICATE_M2_2E_FAILURE_AND_STOP_DIRECTION_OBJECTIVE_SEARCH__RETURN_TO_H5_H2_H1_MECHANISTIC_REASSESSMENT"
        ),
    }
    atomic_json(args.output.resolve(),result)
    print(json.dumps({"status":result["status"],"conclusion":summary["conclusion"],"overall_pass":summary["overall_pass"],"per_regime":summary["per_regime"],"next_legal_step":result["next_legal_step"]},indent=2,sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
