#!/usr/bin/env python3
from __future__ import annotations

"""R11 R5.0 metadata-only inventory for genuinely unconsumed frozen support.

This probe reads authority/manifests/legacy receipts only. It never opens G0
anchor/hourly payloads, raw archives, Teacher outcomes, or final holdout bytes.
"""

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess

SYMBOLS=("BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","SOLUSDT")
HOUR_MS=3_600_000
H72=72
PURGE=128
STRIDE=256
TRAIN_END=datetime(2025,1,1,tzinfo=timezone.utc)
EXPECTED_CANDIDATE=datetime(2024,12,28,8,tzinfo=timezone.utc)
EXPECTED_BLOBS={
    "cb16_local_opt/r102_market.py":"f3ea90f99565a36135da123ef0034057534eb40e",
    "cb16_local_opt/r102_evidence_cache.py":"91fb73565ec60f66bf4232957f9b9b2f88cc3cfe",
    "scripts/run_r104_long_research.py":"b5152ccc33c560edeeff9c2f32bd1d56ee4cfcdb",
}


def require(x:bool, code:str)->None:
    if not x: raise RuntimeError(code)


def load(path:Path):
    require(path.is_file(),f"R5_0_METADATA_FILE_MISSING:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args:str)->str:
    return subprocess.check_output(["git",*args],text=True).strip()


def iso(dt:datetime)->str:
    return dt.isoformat().replace("+00:00","Z")


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--g0-root",type=Path,default=Path("/cb16/g0"))
    ap.add_argument("--r104-root",type=Path,default=Path("/cb16/runtime/r104"))
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    for p,h in EXPECTED_BLOBS.items():
        require(git("rev-parse",f"HEAD:{p}")==h,f"R5_0_SOURCE_BLOB_DRIFT:{p}")

    gap_start=TRAIN_END-timedelta(hours=H72+PURGE)
    epoch=datetime(1970,1,1,tzinfo=timezone.utc)
    lo=int((gap_start-epoch).total_seconds()//3600)+1
    hi=int((TRAIN_END-epoch).total_seconds()//3600)
    phase_hours=[h for h in range(lo,hi) if h%STRIDE==0]
    phase_times=[epoch+timedelta(hours=h) for h in phase_hours]
    require(phase_times==[EXPECTED_CANDIDATE],f"R5_0_PURGE_PHASE_DRIFT:{[iso(x) for x in phase_times]}")

    lineage=load(a.g0_root/"authority/CB16_R11_G0_DERIVED_CACHE_LINEAGE.json")
    require(lineage.get("symbols")==list(SYMBOLS),"R5_0_G0_SYMBOL_SET_DRIFT")
    require(lineage.get("stride_hours")==STRIDE,"R5_0_G0_STRIDE_DRIFT")
    require(lineage.get("final_holdout_payload_opened") is False,"R5_0_G0_HOLDOUT_GUARD_DRIFT")

    legacy_manifest=load(a.r104_root/"REAL_EVIDENCE_CACHE_MANIFEST_R102.json")
    require(legacy_manifest.get("symbols")==list(SYMBOLS),"R5_0_R104_SYMBOL_SET_DRIFT")
    require(legacy_manifest.get("stride_hours")==STRIDE,"R5_0_R104_STRIDE_DRIFT")
    require(legacy_manifest.get("final_holdout_2025_09_accessed") is False,"R5_0_R104_HOLDOUT_GUARD_DRIFT")
    require(legacy_manifest.get("train_boundary")=="H72 maturity + 128h purge <= 2025-01-01T00:00:00Z","R5_0_R104_TRAIN_BOUNDARY_DRIFT")
    require(legacy_manifest.get("validation_boundary")=="2025-01-01 <= decision_time and H72 maturity < 2025-09-01","R5_0_R104_VALIDATION_BOUNDARY_DRIFT")

    g66=load(a.r104_root/"generations/G66/GENERATION_RESULT.json")
    require(g66.get("generation_attempt")==66,"R5_0_G66_IDENTITY_DRIFT")
    g66_consumption=load(a.r104_root/"generations/G66/SNAPSHOT_CONSUMPTION_G66.json")
    require(g66_consumption.get("generation")==66 and g66_consumption.get("status")=="CONSUMED_EXACTLY_ONCE","R5_0_G66_CONSUMPTION_DRIFT")

    per_symbol=[]
    legacy_symbol_manifests=legacy_manifest.get("symbol_market_manifests",{})
    for symbol in SYMBOLS:
        gm=load(a.g0_root/f"market_cache/{symbol}.manifest_r102.json")
        lm=legacy_symbol_manifests.get(symbol)
        require(isinstance(lm,dict),f"R5_0_R104_SYMBOL_MANIFEST_MISSING:{symbol}")
        for label,m in (("G0",gm),("R104",lm)):
            require(m.get("stride_hours")==STRIDE,f"R5_0_{label}_STRIDE_DRIFT:{symbol}")
            require(m.get("prehistory_hours")==96,f"R5_0_{label}_PREHISTORY_DRIFT:{symbol}")
            require(m.get("horizon_hours")==H72,f"R5_0_{label}_HORIZON_DRIFT:{symbol}")
            require(m.get("purged_boundary_anchors")==1,f"R5_0_{label}_PURGE_ANCHOR_COUNT_NOT_ONE:{symbol}:{m.get('purged_boundary_anchors')}")
            require(m.get("forbidden_month_opened") is False,f"R5_0_{label}_HOLDOUT_OPENED:{symbol}")
        # The same globally phased grid and exactly one purged anchor imply the
        # sole candidate is the mathematically pre-registered 2024-12-28 08:00Z.
        per_symbol.append({
            "symbol":symbol,
            "candidate_decision_time":iso(EXPECTED_CANDIDATE),
            "g0_purged_boundary_anchors":gm["purged_boundary_anchors"],
            "legacy_r104_purged_boundary_anchors":lm["purged_boundary_anchors"],
            "legacy_train_or_validation_consumed":False,
            "eligibility_basis":"GLOBAL_256H_PHASE_PLUS_EXPLICIT_PURGED_BOUNDARY_COUNT_AND_FROZEN_R104_SPLIT_RULE",
        })

    result={
        "schema":"CB16_R11_SCIENCE_G0_INDEPENDENT_SUPPORT_INVENTORY_R5_0_V1",
        "status":"INDEPENDENT_PURGE_GAP_SUPPORT_METADATA_QUALIFIED",
        "scientific_evidence":False,
        "scientific_verdict_created":False,
        "market_payload_files_opened":0,
        "teacher_outcomes_compiled":0,
        "final_holdout_payload_opened":False,
        "fresh_market_data_downloaded":False,
        "network_reads_by_probe":0,
        "legacy_g66_generation_verified":True,
        "legacy_campaign_fixed_evidence_cache":True,
        "legacy_train_end":iso(TRAIN_END),
        "legacy_train_last_legal_decision_boundary":iso(gap_start),
        "purge_interval":"decision_time > 2024-12-23T16:00:00Z and decision_time < 2025-01-01T00:00:00Z",
        "global_anchor_stride_hours":STRIDE,
        "mathematically_possible_purge_anchor_times":[iso(x) for x in phase_times],
        "candidate_timestamp":iso(EXPECTED_CANDIDATE),
        "candidate_symbols":list(SYMBOLS),
        "candidate_future_group_count":len(SYMBOLS),
        "per_symbol":per_symbol,
        "legacy_consumption_proof":{
            "r104_cache_includes_all_ten_symbols":True,
            "r104_builder_excludes_purge_anchors_from_both_train_and_validation":True,
            "g66_consumed_snapshot_from_fixed_r104_evidence_cache":True,
            "candidate_was_not_legacy_train_or_validation":True,
        },
        "caution":{
            "ten_symbol_groups_share_one_clock_timestamp":True,
            "cross_symbol_macro_correlation_possible":True,
            "nominal_future_groups":10,
            "effective_independent_information_may_be_less_than_10":True,
            "support_is_small":True,
        },
        "next_legal_step":"PRE_REGISTER_R5_INDEPENDENT_PURGE_GAP_REPLICATION_BEFORE_OPENING_CANDIDATE_MARKET_OR_TEACHER_OUTCOMES",
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="per_symbol"},indent=2,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
