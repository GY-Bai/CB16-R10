#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from cb16_local_opt.r102_campaign import run_campaign
from cb16_local_opt.r102_common import ALL_SUPPORTED_SYMBOLS_R102

def main():
    ap=argparse.ArgumentParser(description='R10.3 optional 10-asset / 20-generation expansion; only legal after R10.2 PASS')
    ap.add_argument('--data-root',default='/data/cb16_hdd/binance_usdm_1m_funding_2020_2026')
    ap.add_argument('--r102-root',default='/home/bgy/cb16_ssd/runtime/R10_2')
    ap.add_argument('--run-root',default='/home/bgy/cb16_ssd/runtime/R10_3')
    ap.add_argument('--parent-r101-root',default='/home/bgy/m3-infra/CB16_SHANXI_FROZEN_BODY_G0_BRAIN_R10_1_THIN_V1')
    ap.add_argument('--parent-g0',default='/home/bgy/cb16_ssd/runtime/R10_1/G0/central_brain_g0_r10_1.pt')
    ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    prev=Path(a.r102_root)/'FINAL_RESULT_R102.json'; start=Path(a.r102_root)/'generations/G04/champion_after.pt'
    r=run_campaign(package_root=ROOT,data_root=a.data_root,run_root=a.run_root,parent_r101_root=a.parent_r101_root,parent_g0=a.parent_g0,
        device=a.device,symbols=ALL_SUPPORTED_SYMBOLS_R102,attempts=20,stride_hours=512,prehistory_hours=96,epochs=12,batch_size=512,lr=3e-4,
        profile_name='R10_3_10ASSET_20GEN_EXPANSION',prerequisite_result=prev,start_checkpoint=start)
    print(json.dumps({'final_status':r['final_status'],'return_bundle':r.get('return_bundle')},indent=2))
if __name__=='__main__': main()
