#!/usr/bin/env python3
from __future__ import annotations

"""CB16 R11 M-series M1 — read-only Direction relative sufficiency audit."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from cb16_local_opt.m_series_m1_direction_relative_sufficiency import (
    M1_RUNTIME,
    adjudicate_m1,
    audit_direction_sufficiency_m1,
)
from cb16_local_opt.training_runtime_r11 import PreparedEvidenceR11, policy_hash_r11
from scripts import r11_science_g0_reduced_teacher_target_information_audit_r6 as r6

SCHEMA="CB16_R11_M_SERIES_M1_DIRECTION_RELATIVE_SUFFICIENCY_RESULT_V1"
M1_PREREG_COMMIT="b5e0d11e235e6df9c7029dbe05a42a42bcab1872"
M1_GATE_BLOB="71acc8708e03be4e7e5b7c0d7621213e723ce61f"
M0_ADJUDICATION_BLOB="54174f86fee2ae0da5c1813af44fa84fe82962db"
R7_FREEZE_COMMIT="62ad74a0c62ab616dfceaf70c6f7b30f7088c81f"
R6_EXECUTOR_BLOB="72bede8dd09015c702ec26ead639a81f48efe38a"
FROZEN_STATUS="DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def git(*args: str) -> str:
    return subprocess.check_output(["git",*args],text=True).strip()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def verify_repo_contract_m1() -> dict[str,Any]:
    require(subprocess.call(["git","merge-base","--is-ancestor",R7_FREEZE_COMMIT,"HEAD"])==0,"M1_NOT_DESCENDED_FROM_R7_FREEZE")
    require(subprocess.call(["git","merge-base","--is-ancestor",M1_PREREG_COMMIT,"HEAD"])==0,"M1_NOT_DESCENDED_FROM_PREREGISTRATION")
    exact={
        "m1_gate":("authority/rearchitecture_r11/CB16_R11_M_SERIES_M1_DIRECTION_RELATIVE_SUFFICIENCY_GATE_V1.json",M1_GATE_BLOB),
        "m0_adjudication":("authority/rearchitecture_r11/CB16_R11_M_SERIES_M0_POLICY_DRIFT_ATTRIBUTION_ADJUDICATION_V1.json",M0_ADJUDICATION_BLOB),
        "r6_executor":("scripts/r11_science_g0_reduced_teacher_target_information_audit_r6.py",R6_EXECUTOR_BLOB),
    }
    observed={}
    for name,(path,expected) in exact.items():
        got=git("rev-parse",f"HEAD:{path}")
        require(got==expected,f"M1_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name]=got
    return {
        "execution_head":git("rev-parse","HEAD"),
        "m1_preregistration_commit":M1_PREREG_COMMIT,
        "immutable_blobs":observed,
        "r6_immutable_contract":r6.verify_repo_contract(),
    }


def _model_direction_probs(model: torch.nn.Module, prepared: PreparedEvidenceR11) -> np.ndarray:
    was_training=bool(model.training)
    model.eval()
    with torch.no_grad():
        out=model(prepared.operator48,prepared.medium48,prepared.account6)
        probs=out["direction_probs"].detach().cpu().numpy().astype(np.float64,copy=False)
    if was_training:
        model.train()
    return probs


def _ordered_eval_evidence(eval_evidence, parent_ids) -> list[Any]:
    admitted=[e for e in eval_evidence if bool(e.admission.admitted)]
    by_parent={str(e.parent_id):e for e in admitted}
    require(len(by_parent)==len(admitted),"M1_DUPLICATE_EVAL_PARENT")
    require(set(by_parent)==set(str(x) for x in parent_ids),"M1_EVAL_PARENT_SET_DRIFT")
    return [by_parent[str(pid)] for pid in parent_ids]


def run_fold_m1(*,fold_spec:Mapping[str,Any],all_train_parents:Mapping[str,Any],all_train_samples:list[Any],g0_root:Path,device:str)->dict[str,Any]:
    fold=int(fold_spec["fold"])
    parents,train_evidence,eval_evidence,teacher_receipt=r6.build_fold_evidence(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    prepared=PreparedEvidenceR11.from_evidence(eval_evidence,parents,device=device)
    g0=r6.r4.r1.load_bootstrap_model(g0_root,device)
    g0_hash=policy_hash_r11(g0)
    g0_probs=_model_direction_probs(g0,prepared)
    ordered=_ordered_eval_evidence(eval_evidence,prepared.parent_ids)
    audit=audit_direction_sufficiency_m1(evidence=ordered,g0_direction_probs=g0_probs)
    audit["fold"]=fold
    require(int(audit["optimizer_steps"])==0,"M1_OPTIMIZER_STEP_DRIFT")
    require(policy_hash_r11(g0)==g0_hash,f"M1_G0_MUTATED:FOLD={fold}")
    return {
        "fold":fold,
        "outer_support":dict(fold_spec),
        "teacher":teacher_receipt,
        "eval_rows":int(prepared.rows),
        "eval_dependence_groups":int(len(set(prepared.dependence_group_ids))),
        "m1_audit":audit,
        "optimizer_steps":0,
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--g0-root",type=Path,default=Path(os.environ.get("CB16_G0_ROOT","/cb16/g0")))
    ap.add_argument("--package-root",type=Path,default=Path(os.environ.get("CB16_PACKAGE_ROOT","/cb16/package")))
    ap.add_argument("--r104-root",type=Path,default=Path(os.environ.get("CB16_R104_ROOT","/cb16/runtime/r104")))
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()

    g0_root=args.g0_root.resolve(); package_root=args.package_root.resolve(); r104_root=args.r104_root.resolve(); output=args.output.resolve()
    repo=verify_repo_contract_m1()
    runtime=r6.r4.r1.runtime_identity(args.device)
    require(runtime["python_series"]==[3,10],f"M1_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    _lineage,g0_identity=r6.r4.r1.verify_g0_authority(g0_root)
    frozen_before=r6.frozen_authority_hashes(package_root)
    g0_before=dict(g0_identity["authority_file_hashes"])
    all_train_parents,all_train_samples,source=r6.load_train_only_support(r104_root)
    folds=r6.build_outer_folds_r6(all_train_parents)
    completed=[]
    for fold_spec in folds:
        completed.append(run_fold_m1(
            fold_spec=fold_spec,
            all_train_parents=all_train_parents,
            all_train_samples=all_train_samples,
            g0_root=g0_root,
            device=args.device,
        ))
    require(len(completed)==5,"M1_INCOMPLETE_FOLDS")
    summary=adjudicate_m1([x["m1_audit"] for x in completed])
    require(r6.frozen_authority_hashes(package_root)==frozen_before,"M1_FROZEN_PACKAGE_MUTATED")
    _,g0_after=r6.r4.r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"]==g0_before,"M1_G0_AUTHORITY_MUTATED")
    result={
        "schema":SCHEMA,
        "status":"M1_DIRECTION_RELATIVE_SUFFICIENCY_PASS",
        "runtime":M1_RUNTIME,
        "frozen_scientific_status_before":FROZEN_STATUS,
        "frozen_scientific_status_after":FROZEN_STATUS,
        "repo":repo,
        "runtime_identity":runtime,
        "source":source,
        "protocol":{
            "consumed_r6_train_only_outer_eval_support":True,
            "student_training":False,
            "optimizer_steps":0,
            "teacher_semantics_changed":False,
            "g0_direction_only":True,
            "continuous_g0_requested_risk_used":False,
            "r7_candidate_evaluated":False,
            "final_holdout_opened":False,
            "fresh_market_data":False,
            "non_status_driving":True,
        },
        "fold_results":completed,
        "m1_summary":summary,
        "semantic_guards":{
            "canonical_generation_advanced":False,
            "champion_promoted":False,
            "production_cutover":False,
            "market_information_verdict_reopened":False,
            "profitability_or_alpha_claimed":False,
            "r7_candidate_evaluated":False,
            "final_holdout_payload_opened":False,
            "fresh_market_data_downloaded":False,
            "r5_purge_support_reopened":False,
            "legacy_validation_rows_used":False,
            "raw_market_payload_used":False,
            "frozen_package_unchanged":True,
            "g0_authority_unchanged":True,
        },
        "next_legal_step":(
            "M2_BASELINE_RELATIVE_OBJECTIVE_AND_FROZEN_G0_RESIDUAL_MAY_BE_PREREGISTERED"
            if summary["direction_mean_geometry_retained"]
            else "M1_1_DIRECTION_TARGET_COMPRESSION_FOLLOWUP_REQUIRED_BEFORE_M2"
        ),
    }
    atomic_json(output,result)
    print(json.dumps({"status":result["status"],"summary":summary,"next_legal_step":result["next_legal_step"]},indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
