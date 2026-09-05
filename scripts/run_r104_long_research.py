#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from cb16_local_opt.r102_campaign import run_campaign
from cb16_local_opt.r102_common import ALL_SUPPORTED_SYMBOLS_R102

def _write_failure_receipt(run_root, exc):
    rr=Path(run_root); rr.mkdir(parents=True,exist_ok=True)
    out={
        "schema":"CB16_R10_2_1_EXECUTION_BLOCKER_RECEIPT_V1",
        "status":"ENGINEERING_EXECUTION_BLOCKED",
        "phase":"R10_4",
        "exception_type":type(exc).__name__,
        "exception":str(exc),
        "traceback":traceback.format_exc(),
        "scientific_verdict_changed":False,
        "final_holdout_2025_09_accessed":False,
        "instruction":"Do not convert this exception into scientific FAIL; repair only the engineering compatibility cause and rerun from the same frozen authority.",
    }
    tmp=rr/'EXECUTION_BLOCKER_RECEIPT_R1021.json.tmp'; final=rr/'EXECUTION_BLOCKER_RECEIPT_R1021.json'
    tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(final)


def main():
    ap=argparse.ArgumentParser(description='R10.4 optional 100-generation long research run; only after R10.3 PASS; FINAL remains locked')
    ap.add_argument('--package-root', default=os.environ.get('CB16_R10_PACKAGE_ROOT', '/home/bgy/m3-infra/CB16_SHANXI_R10_2_REAL_HISTORICAL_G0_LEARNING_V1'))
    ap.add_argument('--data-root',default='/data/cb16_hdd/binance_usdm_1m_funding_2020_2026')
    ap.add_argument('--r103-root',default='/home/bgy/cb16_ssd/runtime/R10_3')
    ap.add_argument('--run-root',default='/data/cb16_hdd/cb16_runtime/R10_4')
    ap.add_argument('--parent-r101-root',default='/home/bgy/m3-infra/CB16_SHANXI_FROZEN_BODY_G0_BRAIN_R10_1_THIN_V1')
    ap.add_argument('--parent-g0',default='/home/bgy/cb16_ssd/runtime/R10_1/G0/central_brain_g0_r10_1.pt')
    ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    prev=Path(a.r103_root)/'FINAL_RESULT_R102.json'; start=Path(a.r103_root)/'generations/G19/champion_after.pt'
    r=run_campaign(package_root=a.package_root,data_root=a.data_root,run_root=a.run_root,parent_r101_root=a.parent_r101_root,parent_g0=a.parent_g0,
        device=a.device,symbols=ALL_SUPPORTED_SYMBOLS_R102,attempts=100,stride_hours=256,prehistory_hours=96,epochs=12,batch_size=512,lr=3e-4,
        profile_name='R10_4_LONG_100GEN_RESEARCH',prerequisite_result=prev,start_checkpoint=start)
    print(json.dumps({'final_status':r['final_status'],'return_bundle':r.get('return_bundle')},indent=2))
if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        _write_failure_receipt('/data/cb16_hdd/cb16_runtime/R10_4', exc)
        raise

