#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cb16_local_opt.r102_campaign import run_campaign


def _write_failure_receipt(run_root, exc):
    rr=Path(run_root); rr.mkdir(parents=True,exist_ok=True)
    out={
        "schema":"CB16_R10_2_1_EXECUTION_BLOCKER_RECEIPT_V1",
        "status":"ENGINEERING_EXECUTION_BLOCKED",
        "phase":"R10_2",
        "exception_type":type(exc).__name__,
        "exception":str(exc),
        "traceback":traceback.format_exc(),
        "scientific_verdict_changed":False,
        "final_holdout_2025_09_accessed":False,
        "instruction":"Do not convert this exception into scientific FAIL; repair only the engineering compatibility cause and rerun from the same frozen authority.",
    }
    tmp=rr/'EXECUTION_BLOCKER_RECEIPT_R1021.json.tmp'; final=rr/'EXECUTION_BLOCKER_RECEIPT_R1021.json'
    tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(final)


def _safe_blocker_summary(exc):
    typ=type(exc).__name__
    code=typ.upper()
    detail=None
    if isinstance(exc, ModuleNotFoundError):
        code='MISSING_PYTHON_MODULE'
        detail=getattr(exc,'name',None)
    elif isinstance(exc, PermissionError):
        code='HOST_PATH_PERMISSION'
        fn=getattr(exc,'filename',None)
        detail=Path(fn).name if fn else None
    elif isinstance(exc, FileNotFoundError):
        code='HOST_FILE_OR_ASSET_MISSING'
        fn=getattr(exc,'filename',None)
        detail=Path(fn).name if fn else None
    elif isinstance(exc, RuntimeError):
        m=re.match(r'^([A-Z][A-Z0-9_]{2,80})(?::|$)',str(exc))
        code=m.group(1) if m else 'RUNTIME_ERROR'
    return {"exception_type":typ,"error_code":code,"detail":detail,"scientific_verdict_changed":False,"final_holdout_2025_09_accessed":False}


def _append_ci_blocker_summary(exc):
    ci_out=os.environ.get('CI_OUT')
    if not ci_out:
        return
    report=Path(ci_out)/'REPORT.md'
    summary=_safe_blocker_summary(exc)
    with report.open('a',encoding='utf-8') as f:
        f.write('\n## R10 engineering blocker\n\n')
        for k in ('exception_type','error_code','detail','scientific_verdict_changed','final_holdout_2025_09_accessed'):
            f.write(f'- {k}: {json.dumps(summary[k],ensure_ascii=True)}\n')


def main():
    ap=argparse.ArgumentParser(description="CB16 R10.2 real historical G0->G1.. 5-generation qualification")
    ap.add_argument('--package-root', default=os.environ.get('CB16_R10_PACKAGE_ROOT', '/home/bgy/m3-infra/CB16_SHANXI_R10_2_REAL_HISTORICAL_G0_LEARNING_V1'))
    ap.add_argument('--data-root', default='/data/cb16_hdd/binance_usdm_1m_funding_2020_2026')
    ap.add_argument('--run-root', default='/home/bgy/cb16_ssd/runtime/R10_2')
    ap.add_argument('--parent-r101-root', default='/home/bgy/m3-infra/CB16_SHANXI_FROZEN_BODY_G0_BRAIN_R10_1_THIN_V1')
    ap.add_argument('--parent-g0', default='/home/bgy/cb16_ssd/runtime/R10_1/G0/central_brain_g0_r10_1.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--verify-all-cache-checksums', action='store_true')
    args=ap.parse_args()
    result=run_campaign(
        package_root=args.package_root, data_root=args.data_root, run_root=args.run_root,
        parent_r101_root=args.parent_r101_root, parent_g0=args.parent_g0,
        device=args.device, attempts=5, stride_hours=512, prehistory_hours=96,
        epochs=12, batch_size=512, lr=3e-4,
        verify_checksum_samples=True, verify_all_cache_checksums=args.verify_all_cache_checksums,
        profile_name='R10_2_5GEN_QUALIFICATION',
    )
    print(json.dumps({k: result.get(k) for k in ['final_status','attempts_completed','promotions','rejections','final_champion_semantic_sha256','return_bundle']}, indent=2))

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        _write_failure_receipt('/home/bgy/cb16_ssd/runtime/R10_2', exc)
        _append_ci_blocker_summary(exc)
        raise

