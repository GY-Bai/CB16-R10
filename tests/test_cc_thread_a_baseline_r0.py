import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BASELINE=ROOT/'authority/rearchitecture_r11/CB16_R11_CC_THREAD_A_BASELINE_V1.json'
def _git_blob_sha(path):
    data=path.read_bytes(); return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
def test_frozen_base_inventory_and_wire_authority_are_exact():
    p=json.loads(BASELINE.read_text()); assert p['implementation_base']=='89d62bf966f476e598f0e2f5c5e8e03c15a8db51'; assert p['planning_authority']['blob_sha']=='b84ad753a2400062e1f3f785f511105b9d25b66a'; assert p['thread_authority']['blob_sha']=='88beeceb5b81a29a28d5a8d59ffee70e0de96442'
    for rel,expected in p['consumed_frozen_blobs'].items(): assert _git_blob_sha(ROOT/rel)==expected,rel
def test_no_sibling_cc_module_dependency():
    forbidden=('cc_policy_','cc_critic_','cc_learner_','cc_experience_','cc_replay_','cc_economic_','cc_fast_')
    for path in (ROOT/'cb16_local_opt').glob('cc_*r0.py'):
        text=path.read_text(); assert not any(f'from .{x}' in text or f'import {x}' in text for x in forbidden),path.name
