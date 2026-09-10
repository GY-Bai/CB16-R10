#!/usr/bin/env python3
from __future__ import annotations

"""Deterministically materialize R10.4 TRAIN-only raw Teacher truth.

This is an infrastructure/data-layout support step, not a scientific audit. It
mechanically copies only rows whose frozen ParentContext split is TRAIN from the
already-existing R10.4 evidence cache. Validation rows are scanned only to route
bytes and are never written to the output. No Teacher, Student, loss, fold,
promotion, market verdict, raw market archive, R5 purge support, or FINAL is used.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

EXPECTED_SYMBOLS=("BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","SOLUSDT")
EXPECTED_TRAIN_PARENTS=9714
EXPECTED_TRAIN_BRANCHES=87426
EXPECTED_TRAIN_GROUPS=1619
EXPECTED_BRANCHES_PER_PARENT=9


def require(x: bool, code: str) -> None:
    if not x:
        raise RuntimeError(code)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(chunk),b""):
            h.update(b)
    return h.hexdigest()


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")


def deterministic_gzip_writer(path: Path):
    raw=path.open("wb")
    gz=gzip.GzipFile(filename="",mode="wb",compresslevel=6,fileobj=raw,mtime=0)
    return raw,gz


def write_json_atomic(path: Path, obj) -> None:
    tmp=path.with_name(path.name+".tmp")
    tmp.write_bytes(canonical_bytes(obj)+b"\n")
    os.replace(tmp,path)


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--r104-root",type=Path,default=Path("/cb16/runtime/r104"))
    ap.add_argument("--output-root",type=Path,default=Path("/cb16/worker/r6_train_only_r104"))
    a=ap.parse_args()
    r104=a.r104_root.resolve(); out=a.output_root.resolve()
    manifest_path=r104/"REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    require(manifest_path.is_file(),"R6_MATERIALIZE_R104_MANIFEST_MISSING")
    m=json.loads(manifest_path.read_text(encoding="utf-8"))
    require(tuple(m.get("symbols",()))==EXPECTED_SYMBOLS,"R6_MATERIALIZE_SYMBOL_SET_DRIFT")
    require(int(m.get("stride_hours",-1))==256,"R6_MATERIALIZE_STRIDE_DRIFT")
    require(m.get("final_holdout_2025_09_accessed") is False,"R6_MATERIALIZE_FINAL_GUARD_DRIFT")
    cache_root=r104/"evidence_cache"
    parent_src=cache_root/Path(str(m["parents_file"])).name
    branch_src=cache_root/Path(str(m["branches_file"])).name
    require(parent_src.is_file() and branch_src.is_file(),"R6_MATERIALIZE_SOURCE_FILES_MISSING")
    require(sha256_file(parent_src)==str(m["parents_sha256"]),"R6_MATERIALIZE_PARENT_SOURCE_SHA_DRIFT")
    require(sha256_file(branch_src)==str(m["branches_sha256"]),"R6_MATERIALIZE_BRANCH_SOURCE_SHA_DRIFT")

    source_identity={
        "r104_manifest_sha256":sha256_file(manifest_path),
        "parents_sha256":sha256_file(parent_src),
        "branches_sha256":sha256_file(branch_src),
    }
    receipt_path=out/"TRAIN_ONLY_MATERIALIZATION_R6.json"
    if receipt_path.is_file():
        old=json.loads(receipt_path.read_text(encoding="utf-8"))
        if old.get("status")=="R6_TRAIN_ONLY_RAW_TRUTH_MATERIALIZATION_PASS" and old.get("source_identity")==source_identity:
            p=out/old["outputs"]["parents_file"]
            b=out/old["outputs"]["branches_file"]
            if p.is_file() and b.is_file() and sha256_file(p)==old["outputs"]["parents_sha256"] and sha256_file(b)==old["outputs"]["branches_sha256"]:
                print(json.dumps({"status":old["status"],"mode":"REUSED_VERIFIED_EXISTING","receipt":str(receipt_path)},indent=2,sort_keys=True))
                return 0
        raise RuntimeError("R6_MATERIALIZE_EXISTING_OUTPUT_CONFLICT")

    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix="r6_train_only_materialize.",dir=str(out.parent)))
    parent_dst=tmp/"PARENT_CONTEXTS_TRAIN_ONLY_R6.jsonl.gz"
    branch_dst=tmp/"COUNTERFACTUAL_BRANCHES_H72_TRAIN_ONLY_R6.jsonl.gz"
    train_ids:set[str]=set(); validation_ids:set[str]=set(); dep_ids:set[str]=set(); symbols:set[str]=set()
    train_parent_raw_hash=hashlib.sha256(); validation_parent_rows_scanned=0
    min_t=None; max_t=None
    rawf,gzf=deterministic_gzip_writer(parent_dst)
    try:
        with gzip.open(parent_src,"rb") as src:
            for rawline in src:
                obj=json.loads(rawline)
                pid=str(obj["parent_id"]); split=str(obj["split"])
                if split=="TRAIN":
                    require(pid not in train_ids,"R6_MATERIALIZE_DUPLICATE_TRAIN_PARENT")
                    train_ids.add(pid); dep_ids.add(str(obj["dependence_group_id"])); symbols.add(str(obj["symbol"]))
                    t=int(obj["decision_time_ms"]); min_t=t if min_t is None else min(min_t,t); max_t=t if max_t is None else max(max_t,t)
                    line=rawline if rawline.endswith(b"\n") else rawline+b"\n"
                    gzf.write(line); train_parent_raw_hash.update(line)
                elif split=="VALIDATION":
                    validation_ids.add(pid); validation_parent_rows_scanned+=1
                else:
                    raise RuntimeError(f"R6_MATERIALIZE_UNEXPECTED_PARENT_SPLIT:{split}")
    finally:
        gzf.close(); rawf.close()

    require(len(train_ids)==EXPECTED_TRAIN_PARENTS,f"R6_MATERIALIZE_TRAIN_PARENT_COUNT:{len(train_ids)}")
    require(len(dep_ids)==EXPECTED_TRAIN_GROUPS,f"R6_MATERIALIZE_TRAIN_GROUP_COUNT:{len(dep_ids)}")
    require(symbols==set(EXPECTED_SYMBOLS),f"R6_MATERIALIZE_TRAIN_SYMBOLS:{sorted(symbols)}")
    require(train_ids.isdisjoint(validation_ids),"R6_MATERIALIZE_PARENT_SPLIT_OVERLAP")

    branch_counts=Counter(); train_branch_raw_hash=hashlib.sha256(); validation_branch_rows_scanned=0; unknown_branch_rows=0
    rawf,gzf=deterministic_gzip_writer(branch_dst)
    try:
        with gzip.open(branch_src,"rb") as src:
            for rawline in src:
                obj=json.loads(rawline); pid=str(obj["parent_id"])
                if pid in train_ids:
                    branch_counts[pid]+=1
                    line=rawline if rawline.endswith(b"\n") else rawline+b"\n"
                    gzf.write(line); train_branch_raw_hash.update(line)
                elif pid in validation_ids:
                    validation_branch_rows_scanned+=1
                else:
                    unknown_branch_rows+=1
    finally:
        gzf.close(); rawf.close()

    require(unknown_branch_rows==0,f"R6_MATERIALIZE_UNKNOWN_BRANCH_PARENT_ROWS:{unknown_branch_rows}")
    require(set(branch_counts)==train_ids,"R6_MATERIALIZE_TRAIN_PARENT_BRANCH_COVERAGE_DRIFT")
    bad=[(pid,n) for pid,n in branch_counts.items() if n!=EXPECTED_BRANCHES_PER_PARENT]
    require(not bad,f"R6_MATERIALIZE_NON_NINE_BRANCH_PARENT:{bad[:1]}")
    train_branches=sum(branch_counts.values())
    require(train_branches==EXPECTED_TRAIN_BRANCHES,f"R6_MATERIALIZE_TRAIN_BRANCH_COUNT:{train_branches}")

    receipt={
      "schema":"CB16_R11_R6_TRAIN_ONLY_RAW_TRUTH_MATERIALIZATION_V1",
      "status":"R6_TRAIN_ONLY_RAW_TRUTH_MATERIALIZATION_PASS",
      "role":"INFRA_SUPPORT_LANE__DETERMINISTIC_SPLIT_MATERIALIZATION__NOT_SCIENTIFIC_EVIDENCE",
      "source_identity":source_identity,
      "source":{
        "r104_root":str(r104),"manifest":str(manifest_path),"parents_file":str(parent_src),"branches_file":str(branch_src),
        "source_cache_is_mixed_train_validation":True,"final_holdout_present_in_source":False
      },
      "split_contract":{
        "selector":"EXACT_FROZEN_PARENT_CONTEXT_FIELD_split_EQUALS_TRAIN",
        "validation_rows_scanned_only_for_mechanical_routing":True,
        "validation_parent_rows_scanned":validation_parent_rows_scanned,
        "validation_branch_rows_scanned":validation_branch_rows_scanned,
        "validation_rows_written":0,
        "no_numeric_outcome_or_target_used_to_select_rows":True,
        "no_optional_selection":True
      },
      "train_only_integrity":{
        "parent_contexts":len(train_ids),"counterfactual_branch_samples":train_branches,"dependence_groups":len(dep_ids),
        "symbols":sorted(symbols),"branches_per_parent":EXPECTED_BRANCHES_PER_PARENT,
        "decision_time_min_ms":min_t,"decision_time_max_ms":max_t,
        "all_output_parents_split_train":True,"all_output_branch_parent_ids_in_train_set":True,
        "parent_uncompressed_rows_sha256":train_parent_raw_hash.hexdigest(),"branch_uncompressed_rows_sha256":train_branch_raw_hash.hexdigest()
      },
      "outputs":{
        "parents_file":parent_dst.name,"branches_file":branch_dst.name,
        "parents_sha256":sha256_file(parent_dst),"branches_sha256":sha256_file(branch_dst)
      },
      "semantic_guards":{
        "teacher_compiled":False,"student_executed":False,"loss_computed":False,"folds_constructed":False,
        "scientific_verdict_created":False,"canonical_generation_advanced":False,"promotion_or_cutover":False,
        "r5_purge_support_opened":False,"raw_market_payload_opened":False,"final_holdout_payload_opened":False,
        "fresh_market_data_downloaded":False
      },
      "next_legal_step":"R6_MAY_CONSUME_ONLY_THIS_VERIFIED_TRAIN_ONLY_RAW_TRUTH_MATERIALIZATION"
    }
    write_json_atomic(tmp/"TRAIN_ONLY_MATERIALIZATION_R6.json",receipt)
    os.sync()
    if out.exists():
        shutil.rmtree(tmp,ignore_errors=True)
        raise RuntimeError("R6_MATERIALIZE_OUTPUT_APPEARED_CONCURRENTLY")
    os.replace(tmp,out)
    print(json.dumps({"status":receipt["status"],"output_root":str(out),"parent_contexts":len(train_ids),"branches":train_branches,"validation_rows_written":0},indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
