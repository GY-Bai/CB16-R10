#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, platform, subprocess
from pathlib import Path
from typing import Any

from cb16_local_opt.state_utility_geometry_transport_h55 import adjudicate_h55, run_fold_h55
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA="CB16_R11_M_SERIES_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_RESULT_V1"
PREREG_COMMIT="e4a3b661869cd51b3a234fc9e027d15cf2e7289e"
FROZEN_STATUS="DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"


def require(c,code):
    if not c: raise RuntimeError(code)

def git(*args): return subprocess.check_output(["git",*args],text=True).strip()

def atomic_json(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8"); os.replace(tmp,path)


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--r104-root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    parents,samples,support=load_train_only_support(args.r104_root.resolve())
    folds=build_outer_folds_r6(parents); require(len(folds)==5,"H55_OUTER_FOLD_COUNT_DRIFT")
    results=[run_fold_h55(fold_spec=f,all_train_parents=parents,all_train_samples=samples) for f in folds]
    summary=adjudicate_h55(results)
    result={
      "schema":SCHEMA,"status":"H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_COMPLETE",
      "repo":{"execution_head":git("rev-parse","HEAD"),"preregistration_commit":PREREG_COMMIT},
      "runtime_identity":{"python":platform.python_version(),"student_training":False,"student_inference":False,"teacher_compilation":False,"teacher_kernel_used":False,"teacher_support_selection_used":False},
      "support":support,"frozen_scientific_status_before":FROZEN_STATUS,"frozen_scientific_status_after":FROZEN_STATUS,
      "protocol":{"support_classification":"CONSUMED_TRAIN_ONLY_MECHANISTIC_SUPPORT__NOT_INDEPENDENT_MARKET_SUPPORT","outer_folds":5,"shifts":[1,7,13,23,31],"teacher_free":True,"same_scenario_one_parent_per_train_future_group":True,"utility_used_only_after_state_distance":True,"fresh_market_data_downloaded":False,"final_holdout_opened":False,"r7_candidate_evaluated":False},
      "folds":results,"h5_5_summary":summary,
      "semantic_guards":{"teacher_compilation_used":False,"teacher_kernel_used":False,"teacher_support_selection_used":False,"student_trained":False,"student_inference_used":False,"new_direction_loss_created":False,"canonical_change_authorized":False,"fresh_market_data_downloaded":False,"final_holdout_payload_opened":False,"r7_candidate_evaluated":False},
      "next_legal_step":("ADJUDICATE_H5_5_THEN_RETURN_TO_H2_H3_WITH_H5_CAVEAT" if summary["classification"]=="FULL_STATE_GEOMETRY_TRANSPORT_SUPPORTED" else "ADJUDICATE_H5_5_THEN_REASSESS_H1_H2_H3__DO_NOT_TUNE_TEACHER_OR_ORGAN_WEIGHTS")
    }
    atomic_json(args.output.resolve(),result)
    print(json.dumps({"status":result["status"],"classification":summary["classification"],"metrics":summary["metrics"],"all_rotation_identity_guards_pass":summary["all_rotation_identity_guards_pass"],"next_legal_step":result["next_legal_step"]},indent=2,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
