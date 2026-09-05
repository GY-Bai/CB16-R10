#!/usr/bin/env python3
"""Optional local orchestration through the user-authorized research ladder.

This never opens FINAL. It executes R10.2, then only on PASS R10.3, then only on PASS R10.4.
Use --through r102/r103/r104 to choose the stopping point. Default r102 is conservative.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(script):
    print(f"\n=== EXEC {script} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, check=True)

def read_result(root):
    p=Path(root)/'FINAL_RESULT_R102.json'
    if not p.is_file(): raise SystemExit(f'MISSING_RESULT:{p}')
    return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--through',choices=['r102','r103','r104'],default='r102'); a=ap.parse_args()
    run('run_r102_pipeline.py')
    r=read_result('/home/bgy/cb16_ssd/runtime/R10_2')
    if not r.get('final_status','').endswith('PASS'): raise SystemExit('STOP_AFTER_R102_NONPASS')
    if a.through=='r102': return
    run('run_r103_expansion.py')
    r=read_result('/home/bgy/cb16_ssd/runtime/R10_3')
    if not r.get('final_status','').endswith('PASS'): raise SystemExit('STOP_AFTER_R103_NONPASS')
    if a.through=='r103': return
    run('run_r104_long_research.py')
    r=read_result('/data/cb16_hdd/cb16_runtime/R10_4')
    if not r.get('final_status','').endswith('PASS'): raise SystemExit('STOP_AFTER_R104_NONPASS')
    print('AUTHORIZED_RESEARCH_LADDER_COMPLETE__FINAL_STILL_LOCKED')

if __name__=='__main__': main()
