#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 mf=ROOT/'SHA256SUMS_R10_2.txt'
 if not mf.is_file(): raise SystemExit('MISSING_SHA256SUMS_R10_2')
 errors=[]; n=0
 for line in mf.read_text().splitlines():
  if not line.strip(): continue
  digest, rel=line.split(None,1); rel=rel.strip().lstrip('*')
  p=ROOT/rel; n+=1
  if not p.is_file(): errors.append(f'MISSING:{rel}'); continue
  a=sha(p)
  if a!=digest: errors.append(f'HASH:{rel}:{a}!={digest}')
 physics=ROOT/'authority/CB16_ACCOUNT_PHYSICS_STATE_V1_R0.tar.gz'
 if physics.is_file() and sha(physics)!='19f89018ef9b7c7301fe13c57e6b2abb512ffc770b29a7b3fb9df9a1be9f47de': errors.append('PHYSICS_AUTHORITY_TAR_SHA')
 out={'schema':'CB16_R10_2_STATIC_PACKAGE_VERIFICATION_V1','status':'PASS' if not errors else 'FAIL','files_verified':n,'errors':errors}
 print(json.dumps(out,indent=2))
 if errors: raise SystemExit(2)
if __name__=='__main__': main()
