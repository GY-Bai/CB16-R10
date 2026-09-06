#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, sqlite3, sys, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from cb16_local_opt.r102_campaign import run_campaign
from cb16_local_opt.r102_common import ALL_SUPPORTED_SYMBOLS_R102
from cb16_local_opt.r10_ci_reporting import append_scientific_summary, status_is_pass
from cb16_local_opt.r10_host_bindings import host_binding

DEFAULT_PACKAGE_ROOT='/home/bgy/m3-infra/CB16_SHANXI_R10_2_REAL_HISTORICAL_G0_LEARNING_V1'
DEFAULT_DATA_ROOT='/data/cb16_hdd/binance_usdm_1m_funding_2020_2026'
DEFAULT_R103_ROOT='/home/bgy/cb16_ssd/runtime/R10_3'
DEFAULT_RUN_ROOT='/data/cb16_hdd/cb16_runtime/R10_4'
DEFAULT_PARENT_R101='/home/bgy/m3-infra/CB16_SHANXI_FROZEN_BODY_G0_BRAIN_R10_1_THIN_V1'
DEFAULT_PARENT_G0='/home/bgy/cb16_ssd/runtime/R10_1/G0/central_brain_g0_r10_1.pt'


def _sqlite_category(exc: sqlite3.Error):
    name=getattr(exc,'sqlite_errorname',None)
    code=getattr(exc,'sqlite_errorcode',None)
    msg=str(exc).lower()
    if name in {'SQLITE_BUSY','SQLITE_LOCKED'} or 'database is locked' in msg or 'database table is locked' in msg:
        cat='SQLITE_BUSY_OR_LOCKED'
    elif name == 'SQLITE_FULL' or 'database or disk is full' in msg:
        cat='SQLITE_FULL'
    elif (name and name.startswith('SQLITE_IOERR')) or 'disk i/o error' in msg:
        cat='SQLITE_IOERR'
    elif name == 'SQLITE_CORRUPT' or 'database disk image is malformed' in msg:
        cat='SQLITE_CORRUPT'
    elif 'readonly database' in msg:
        cat='SQLITE_READONLY'
    else:
        cat='SQLITE_OTHER_OPERATIONAL_ERROR'
    return cat,name,code


def _safe_summary(exc):
    typ=type(exc).__name__; code=typ.upper(); detail=None
    out={"exception_type":typ,"error_code":code,"detail":detail,"scientific_verdict_changed":False,"final_holdout_2025_09_accessed":False}
    if isinstance(exc,PermissionError):
        out['error_code']='HOST_PATH_PERMISSION'; out['detail']=Path(getattr(exc,'filename','') or '').name or None
    elif isinstance(exc,FileNotFoundError):
        out['error_code']='HOST_FILE_OR_ASSET_MISSING'; out['detail']=Path(getattr(exc,'filename','') or '').name or None
    elif isinstance(exc,ModuleNotFoundError):
        out['error_code']='MISSING_PYTHON_MODULE'; out['detail']=getattr(exc,'name',None)
    elif isinstance(exc,sqlite3.OperationalError):
        cat,name,sqlcode=_sqlite_category(exc)
        out['error_code']=cat; out['detail']=name; out['sqlite_errorcode']=sqlcode
    elif isinstance(exc,RuntimeError):
        m=re.match(r'^([A-Z][A-Z0-9_]{2,80})(?::|$)',str(exc)); out['error_code']=m.group(1) if m else 'RUNTIME_ERROR'
    return out


def _append_blocker(exc):
    if not os.environ.get('CI_OUT'): return
    s=_safe_summary(exc)
    with (Path(os.environ['CI_OUT'])/'REPORT.md').open('a',encoding='utf-8') as f:
        f.write('\n## R10 engineering blocker\n\n')
        for k,v in s.items(): f.write(f'- {k}: {json.dumps(v,ensure_ascii=True)}\n')


def _local_receipt(run_root,exc):
    rr=Path(run_root); rr.mkdir(parents=True,exist_ok=True)
    (rr/'EXECUTION_BLOCKER_RECEIPT_R104.json').write_text(json.dumps({"schema":"CB16_R10_ENGINEERING_BLOCKER_V1","phase":"R10_4","exception_type":type(exc).__name__,"exception":str(exc),"sqlite_errorname":getattr(exc,'sqlite_errorname',None),"sqlite_errorcode":getattr(exc,'sqlite_errorcode',None),"traceback":traceback.format_exc(),"scientific_verdict_changed":False,"final_holdout_2025_09_accessed":False},indent=2,sort_keys=True)+'\n')


def main() -> int:
    ap=argparse.ArgumentParser(description='R10.4 optional 100-generation long research run; only after R10.3 PASS; FINAL remains locked')
    ap.add_argument('--package-root',default=host_binding('CB16_R10_PACKAGE_ROOT',DEFAULT_PACKAGE_ROOT))
    ap.add_argument('--data-root',default=host_binding('CB16_R10_DATA_ROOT',DEFAULT_DATA_ROOT))
    ap.add_argument('--r103-root',default=host_binding('CB16_R10_R103_ROOT',DEFAULT_R103_ROOT))
    ap.add_argument('--run-root',default=host_binding('CB16_R10_R104_ROOT',DEFAULT_RUN_ROOT))
    ap.add_argument('--parent-r101-root',default=host_binding('CB16_R10_PARENT_R101_ROOT',DEFAULT_PARENT_R101))
    ap.add_argument('--parent-g0',default=host_binding('CB16_R10_PARENT_G0',DEFAULT_PARENT_G0))
    ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    prev=Path(a.r103_root)/'FINAL_RESULT_R102.json'; start=Path(a.r103_root)/'generations/G19/champion_after.pt'
    # run_campaign is recovery-first: existing R10.4 evidence cache and complete
    # generation receipts are reused. Runtime scheduling changes do not request a
    # scientific cache rebuild.
    r=run_campaign(package_root=a.package_root,data_root=a.data_root,run_root=a.run_root,parent_r101_root=a.parent_r101_root,parent_g0=a.parent_g0,device=a.device,symbols=ALL_SUPPORTED_SYMBOLS_R102,attempts=100,stride_hours=256,prehistory_hours=96,epochs=12,batch_size=512,lr=3e-4,profile_name='R10_4_LONG_100GEN_RESEARCH',prerequisite_result=prev,start_checkpoint=start)
    append_scientific_summary(r,'R10_4')
    print(json.dumps({'final_status':r.get('final_status'),'return_bundle':r.get('return_bundle')},indent=2))
    return 0 if status_is_pass(r,'R10_4') else 2

if __name__=='__main__':
    try: rc=main()
    except Exception as exc:
        try: _append_blocker(exc)
        except Exception: pass
        try: _local_receipt(host_binding('CB16_R10_R104_ROOT',DEFAULT_RUN_ROOT),exc)
        except Exception: pass
        raise
    else: raise SystemExit(rc)
