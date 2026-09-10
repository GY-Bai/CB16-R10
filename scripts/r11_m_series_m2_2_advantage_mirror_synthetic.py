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

from cb16_local_opt.m_series_m2_2_advantage_mirror_synthetic import REGIMES,adjudicate_m22,run_seed_cell_m22

SCHEMA="CB16_R11_M_SERIES_M2_2_ADVANTAGE_MIRROR_SYNTHETIC_RESULT_V1"
PREREG_COMMIT="9b48115c080e221efd70df0896347dbb224c4ada"
GATE_BLOB="4a75dcd18eb78fbef4b24a94ccad0bf444e4e2b0"
M21_ADJ_BLOB="f4e2a1ff358f00b42c27656a5c82bbdb751cc391"
FROZEN_STATUS="DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"


def require(c:bool,code:str)->None:
    if not c: raise RuntimeError(code)


def git(*args:str)->str: return subprocess.check_output(["git",*args],text=True).strip()


def atomic_json(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8"); os.replace(tmp,path)


def verify_repo()->dict[str,Any]:
    require(subprocess.call(["git","merge-base","--is-ancestor",PREREG_COMMIT,"HEAD"])==0,"M22_NOT_DESCENDED_FROM_PREREGISTRATION")
    exact={
        "gate":("authority/rearchitecture_r11/CB16_R11_M_SERIES_M2_2_ADVANTAGE_MIRROR_SYNTHETIC_GATE_V1.json",GATE_BLOB),
        "m21_adjudication":("authority/rearchitecture_r11/CB16_R11_M_SERIES_M2_1_CORRECTION_QUALITY_ATTRIBUTION_ADJUDICATION_V1.json",M21_ADJ_BLOB),
    }
    observed={}
    for n,(p,e) in exact.items():
        g=git("rev-parse",f"HEAD:{p}"); require(g==e,f"M22_IMMUTABLE_BLOB_DRIFT:{n}:{g}"); observed[n]=g
    return {"execution_head":git("rev-parse","HEAD"),"preregistration_commit":PREREG_COMMIT,"immutable_blobs":observed}


def runtime_identity(device:str)->dict[str,Any]:
    dev=torch.device(device); out={"python":platform.python_version(),"torch":torch.__version__,"requested_device":str(dev),"cuda_available":bool(torch.cuda.is_available())}
    if dev.type=="cuda":
        require(torch.cuda.is_available(),"M22_CUDA_REQUIRED_BUT_UNAVAILABLE")
        out.update({"cuda_device_name":torch.cuda.get_device_name(dev),"cuda_capability":list(torch.cuda.get_device_capability(dev)),"torch_cuda":torch.version.cuda})
    return out


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--device",default="cuda"); args=ap.parse_args()
    repo=verify_repo(); runtime=runtime_identity(args.device)
    cells=[]
    for regime in REGIMES:
        for seed in regime.seeds:
            cells.append(run_seed_cell_m22(seed=seed,regime=regime,device=args.device))
    summary=adjudicate_m22(cells)
    result={
        "schema":SCHEMA,"status":"M2_2_ADVANTAGE_MIRROR_SYNTHETIC_COMPLETE","repo":repo,"runtime_identity":runtime,
        "frozen_scientific_status_before":FROZEN_STATUS,"frozen_scientific_status_after":FROZEN_STATUS,
        "protocol":{"synthetic_only":True,"market_data_read":False,"market_evidence_read":False,"cells":20,"arms_per_cell":18,"total_arms":360,"ground_truth_known":True,"final_holdout_opened":False,"r7_candidate_evaluated":False,"canonical_change":False},
        "cells":cells,"m2_2_summary":summary,
        "semantic_guards":{"market_data_used":False,"market_evidence_used":False,"canonical_generation_advanced":False,"champion_promoted":False,"production_cutover":False,"market_information_verdict_reopened":False,"final_holdout_payload_opened":False,"r7_candidate_evaluated":False,"m3_sizing_authorized":False},
        "next_legal_step":"PREREGISTER_M2_3_CONSUMED_TRAIN_ONLY_SHADOW_SCREEN" if summary["overall_pass"] else "RETURN_TO_DIRECTION_OBJECTIVE_FAMILY_DESIGN__DO_NOT_OPEN_M2_3_MARKET_SHADOW",
    }
    atomic_json(args.output.resolve(),result)
    print(json.dumps({"status":result["status"],"conclusion":summary["conclusion"],"overall_pass":summary["overall_pass"],"per_regime":summary["per_regime"],"next_legal_step":result["next_legal_step"]},indent=2,sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
