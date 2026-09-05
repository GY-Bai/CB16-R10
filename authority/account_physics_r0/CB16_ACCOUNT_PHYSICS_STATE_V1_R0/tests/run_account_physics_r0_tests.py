#!/usr/bin/env python3
"""Self-contained durable reload/behavior canary for CB16 Account Physics R0."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from account_physics_runtime_r0 import (
    canonical_json_bytes, sha256_obj, snapshot_from_portable_npz_bytes,
    step_account, project_observation,
)

def main():
    contract=json.loads((ROOT/'ACCOUNT_PHYSICS_CANARY_CONTRACT_V1.json').read_text())
    native=json.loads((ROOT/'canaries/SIMULATOR_SNAPSHOT_NATIVE.json').read_text())
    portable=snapshot_from_portable_npz_bytes((ROOT/'canaries/SIMULATOR_SNAPSHOT_PORTABLE.npz').read_bytes())
    canary=json.loads((ROOT/'canaries/SNAPSHOT_BEHAVIOR_CANARY.json').read_text())
    inp=json.loads((ROOT/'canaries/SNAPSHOT_BEHAVIOR_CANARY_INPUT.json').read_text())
    assert sha256_obj(inp)==canary['input_sha256']
    assert sha256_obj(native)==canary['representation_sha256']
    assert canonical_json_bytes(native)==canonical_json_bytes(portable)
    before=canonical_json_bytes(native)
    out=step_account(native,inp['action'],inp['market_execution_input'],contract)
    assert sha256_obj(out)==canary['prediction_sha256']
    mark=inp['market_execution_input']['bar']['mark_price']
    p1=project_observation(native,mark,contract); p2=project_observation(native,mark,contract)
    assert canonical_json_bytes(p1)==canonical_json_bytes(p2)
    assert canonical_json_bytes(native)==before
    assert len(p1['payload'])==6 and len(p1['validity_flags'])==7
    print(json.dumps({'schema':'CB16_ACCOUNT_PHYSICS_BUNDLE_CANARY_RESULT_V1','status':'PASS','representation_sha256':sha256_obj(native),'prediction_sha256':sha256_obj(out)},sort_keys=True))
if __name__=='__main__': main()
