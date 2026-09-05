#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cb16_local_opt.r102_campaign import run_campaign


def main():
    ap=argparse.ArgumentParser(description="CB16 R10.2 real historical G0->G1.. 5-generation qualification")
    ap.add_argument('--data-root', default='/data/cb16_hdd/binance_usdm_1m_funding_2020_2026')
    ap.add_argument('--run-root', default='/home/bgy/cb16_ssd/runtime/R10_2')
    ap.add_argument('--parent-r101-root', default='/home/bgy/m3-infra/CB16_SHANXI_FROZEN_BODY_G0_BRAIN_R10_1_THIN_V1')
    ap.add_argument('--parent-g0', default='/home/bgy/cb16_ssd/runtime/R10_1/G0/central_brain_g0_r10_1.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--verify-all-cache-checksums', action='store_true')
    args=ap.parse_args()
    result=run_campaign(
        package_root=ROOT, data_root=args.data_root, run_root=args.run_root,
        parent_r101_root=args.parent_r101_root, parent_g0=args.parent_g0,
        device=args.device, attempts=5, stride_hours=512, prehistory_hours=96,
        epochs=12, batch_size=512, lr=3e-4,
        verify_checksum_samples=True, verify_all_cache_checksums=args.verify_all_cache_checksums,
        profile_name='R10_2_5GEN_QUALIFICATION',
    )
    print(json.dumps({k: result.get(k) for k in ['final_status','attempts_completed','promotions','rejections','final_champion_semantic_sha256','return_bundle']}, indent=2))

if __name__=='__main__': main()
