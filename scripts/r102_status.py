#!/usr/bin/env python3
from pathlib import Path
import argparse, json
ap=argparse.ArgumentParser(); ap.add_argument('--run-root',default='/home/bgy/cb16_ssd/runtime/R10_2'); a=ap.parse_args(); r=Path(a.run_root)
for name in ['PARENT_ADOPTION_RECEIPT_R102.json','TEN_SYMBOL_DATA_PREFLIGHT_R102.json','REAL_EVIDENCE_CACHE_MANIFEST_R102.json','TEACHER_EVIDENCE_SUMMARY_R102.json','F0_F1_F2_F3_CONTROLS_R102.json','FINAL_RESULT_R102.json']:
 p=r/name
 print(name, 'FOUND' if p.is_file() else 'PENDING')
 if p.is_file() and name=='FINAL_RESULT_R102.json':
  x=json.loads(p.read_text()); print(json.dumps({'final_status':x.get('final_status'),'attempts_completed':x.get('attempts_completed'),'promotions':x.get('promotions'),'rejections':x.get('rejections'),'return_bundle':x.get('return_bundle')},indent=2))
