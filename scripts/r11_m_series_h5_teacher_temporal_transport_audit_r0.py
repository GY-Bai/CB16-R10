#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from cb16_local_opt.teacher_temporal_transport_audit_h5 import adjudicate_h5, run_fold_h5
from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

SCHEMA="CB16_R11_M_SERIES_H5_TEACHER_TEMPORAL_TRANSPORT_AUDIT_R0_RESULT_V1"
PREREG_COMMIT="ee1ff7962f816ebdf9034813de5ddd521cf4a654"
GATE_BLOB="e0e0c6422bef5e5731044952a01eb465b85482cd"
M22E_ADJ_BLOB="73bcda067d7576b5a26063073cde7790dd550662"
R6_ADJ_BLOB="5a38e4d33e20c049507f3935b4de6b10c0a12912"
R6_GATE_BLOB="1d1254fb03cc00283423426e07ae240a196c627e"
FROZEN_STATUS="DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

PINNED={
    "gate":("authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_TEACHER_TEMPORAL_TRANSPORT_AUDIT_R0_GATE_V1.json",GATE_BLOB),
    "m22e_adjudication":("authority/rearchitecture_r11/CB16_R11_M_SERIES_M2_2E_DIRECT_CHAMPION_RELATIVE_REGRET_SYNTHETIC_ADJUDICATION_V1.json",M22E_ADJ_BLOB),
    "r6_adjudication":("authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_REDUCED_TEACHER_TARGET_INFORMATION_AUDIT_R6_ADJUDICATION_V1.json",R6_ADJ_BLOB),
    "r6_gate":("authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_REDUCED_TEACHER_TARGET_INFORMATION_AUDIT_R6_GATE_V1.json",R6_GATE_BLOB),
    "teacher_balanced_runtime":("cb16_local_opt/teacher_balanced_runtime_r11.py","e834facdbf2d08de25e3cab917f0faf587f3386e"),
    "teacher_vectorized":("cb16_local_opt/teacher_vectorized_r11.py","083ca6541b45577cfeb6d9a376b25e0eaf8d060a"),
    "teacher_authority_candidate":("cb16_local_opt/r11_teacher_authority_candidate.py","6e77e8f989a6d4358a7feac99c269db7b7d6549e"),
    "r5_target_only_compiler":("cb16_local_opt/independent_purge_alignment_replication_r5.py","96560a80c9793dec8b527b800758547a69ced55a"),
    "probabilistic_teacher":("cb16_local_opt/probabilistic_teacher_r6.py","3de3092d244c3fd315950c9f8b0bf4d78c50ee2b"),
    "evidence_cache_loader":("cb16_local_opt/r102_evidence_cache.py","91fb73565ec60f66bf4232957f9b9b2f88cc3cfe"),
    "r6_helper":("cb16_local_opt/reduced_teacher_target_information_audit_r6.py","bd7a8780e8dcab05cfae92744fde523ce62cc9ae"),
    "r6_executor":("scripts/r11_science_g0_reduced_teacher_target_information_audit_r6.py","72bede8dd09015c702ec26ead639a81f48efe38a"),
}


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
    require(subprocess.call(["git","merge-base","--is-ancestor",PREREG_COMMIT,"HEAD"])==0,"H5_NOT_DESCENDED_FROM_PREREGISTRATION")
    observed={}
    for name,(path,expected) in PINNED.items():
        got=git("rev-parse",f"HEAD:{path}")
        require(got==expected,f"H5_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name]=got
    return {"execution_head":git("rev-parse","HEAD"),"preregistration_commit":PREREG_COMMIT,"immutable_blobs":observed}


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--r104-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    repo=verify_repo()
    parents,samples,support_receipt=load_train_only_support(args.r104_root.resolve())
    folds=build_outer_folds_r6(parents)
    require(len(folds)==5,"H5_OUTER_FOLD_COUNT_DRIFT")
    results=[run_fold_h5(
        fold_spec=spec,all_train_parents=parents,all_train_samples=samples,block_targets=32
    ) for spec in folds]
    summary=adjudicate_h5(results)
    result={
        "schema":SCHEMA,
        "status":"H5_TEACHER_TEMPORAL_TRANSPORT_AUDIT_COMPLETE",
        "repo":repo,
        "runtime_identity":{"python":platform.python_version(),"student_training":False,"student_inference":False,"teacher_only":True},
        "frozen_scientific_status_before":FROZEN_STATUS,
        "frozen_scientific_status_after":FROZEN_STATUS,
        "support":support_receipt,
        "protocol":{
            "support_classification":"CONSUMED_TRAIN_ONLY_MECHANISTIC_SUPPORT__NOT_INDEPENDENT_MARKET_SUPPORT",
            "outer_folds":5,"clock_blocks":6,"train_to_eval_gap_hours_each_fold":256,
            "all_ten_symbols":True,"student_training":False,"student_inference":False,
            "teacher_config_tuning":False,"teacher_semantics_changed":False,
            "primary_score":"DEPENDENCE_GROUP_EQUAL_WEIGHTED_MEAN_QUANTILE_PINBALL_SCORE",
            "quantile_levels":[0.05,0.10,0.25,0.50,0.75,0.90,0.95],
            "climatology_control":"ACTION_RISK_SPECIFIC_EQUAL_WEIGHT_OUTER_TRAIN_CLIMATOLOGY",
            "state_shuffle_shifts":[1,7,13,23,31],
            "raw_market_payload_read":False,"legacy_validation_rows_used":False,"r5_purge_support_used":False,
            "fresh_market_data_downloaded":False,"final_holdout_opened":False,"r7_candidate_evaluated":False,
        },
        "folds":results,
        "h5_summary":summary,
        "semantic_guards":{
            "student_trained":False,"student_inference_used":False,"new_direction_loss_created":False,
            "canonical_generation_advanced":False,"champion_promoted":False,"production_cutover":False,
            "market_information_verdict_reopened":False,"canonical_change_authorized":False,
            "fresh_market_data_downloaded":False,"raw_market_payload_used":False,"legacy_validation_rows_used":False,
            "r5_purge_support_reopened":False,"final_holdout_payload_opened":False,"r7_candidate_evaluated":False,
            "m3_sizing_opened":False,
        },
        "next_legal_step":(
            "ADJUDICATE_H5_THEN_PREREGISTER_H2_STUDENT_UPDATE_INTERFERENCE_AUDIT_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
            if summary["overall_pass"] else
            "ADJUDICATE_H5_THEN_STUDY_FROZEN_TEACHER_SUPPORT_AND_CALIBRATION_FAILURE_MODES_WITHOUT_STUDENT_OR_NEW_DIRECTION_LOSS"
        ),
    }
    atomic_json(args.output.resolve(),result)
    print(json.dumps({"status":result["status"],"conclusion":summary["conclusion"],"overall_pass":summary["overall_pass"],
                      "aligned_lt_climatology_fold_count":summary["aligned_lt_climatology_fold_count"],
                      "aligned_lt_median_shuffle_fold_count":summary["aligned_lt_median_shuffle_fold_count"],
                      "aligned_lt_each_shuffle_pair_count_of_25":summary["aligned_lt_each_shuffle_pair_count_of_25"],
                      "late_transport":summary["folds_4_and_5_both_pass_climatology_and_shuffle"],
                      "per_fold":summary["per_fold"],"next_legal_step":result["next_legal_step"]},indent=2,sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
