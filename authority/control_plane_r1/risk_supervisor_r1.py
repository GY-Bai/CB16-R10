from __future__ import annotations
import copy, hashlib, json, math
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from account_physics_runtime_r0 import (canonical_json_bytes as physics_canonical_json_bytes, sha256_obj as physics_sha256_obj,
    validate_physics_contract, project_observation, step_account, snapshot_ref, ContentAddressedSnapshotStore)
from central_brain_hard_isolation_runtime import consume as hard_isolated_consume

SHORT, FLAT, LONG = 0, 1, 2
DECISIONS=("ACCEPT","CLAMP","REJECT","FORCED_NOOP")
class ControlPlaneIntegrityError(RuntimeError): pass

def canonical_json_bytes(obj:Any)->bytes:
    return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def sha256_obj(obj:Any)->str: return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()
def sha256_file(path:str|Path)->str:
    h=hashlib.sha256();
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def validate_risk_authority(r:Mapping[str,Any], account_id:str|None=None)->None:
    if r.get('schema')!='EXTERNAL_RISK_BUDGET_AUTHORITY_V1' or r.get('authority')!='CONTROL_PLANE_EXTERNAL_STATE': raise ControlPlaneIntegrityError('RISK_AUTHORITY_SCHEMA_OR_OWNER_MISMATCH')
    if r.get('update_rule_status')!='UPDATE_RULE_UNRESOLVED' or r.get('physics_mutation_policy')!='PRESERVE_UNCHANGED': raise ControlPlaneIntegrityError('RISK_AUTHORITY_POLICY_MISMATCH')
    if account_id is not None and r.get('account_id')!=account_id: raise ControlPlaneIntegrityError('RISK_AUTHORITY_ACCOUNT_MISMATCH')
    rem=float(r['risk_budget_remaining']); cap=float(r['risk_budget_capacity'])
    if not(math.isfinite(rem) and math.isfinite(cap)) or cap<=0 or rem<0 or rem>cap: raise ControlPlaneIntegrityError('RISK_AUTHORITY_VALUE_INVALID')

def validate_snapshot_risk_sync(snapshot:Mapping[str,Any], r:Mapping[str,Any])->None:
    validate_risk_authority(r)
    s=snapshot.get('observation_support_state',{})
    if s.get('authority')!='EXTERNAL_AUTHORITATIVE_RISK_STATE' or s.get('physics_mutation_policy')!='PRESERVE_UNCHANGED': raise ControlPlaneIntegrityError('SNAPSHOT_RISK_SUPPORT_OWNER_MISMATCH')
    if float(s.get('risk_budget_remaining',float('nan'))) != float(r['risk_budget_remaining']) or float(s.get('risk_budget_capacity',float('nan'))) != float(r['risk_budget_capacity']): raise ControlPlaneIntegrityError('SNAPSHOT_RISK_AUTHORITY_DESYNCHRONIZED')

def _noop(decision:str, intent:Mapping[str,Any], snapshot:Mapping[str,Any], reasons:list[str], contract:Mapping[str,Any])->dict[str,Any]:
    return _decision(decision,intent,snapshot,FLAT,0.0,reasons,contract)
def _decision(kind:str,intent:Mapping[str,Any],snapshot:Mapping[str,Any],direction:int,risk:float,reasons:list[str],contract:Mapping[str,Any])->dict[str,Any]:
    base={'schema':'RiskSupervisorDecisionV1','decision':kind,'requested_direction':intent.get('direction'),'requested_risk_multiplier':intent.get('requested_risk_multiplier'),
          'executable_direction':int(direction),'executable_risk_multiplier':float(risk),'reason_codes':sorted(set(reasons)),
          'intent_sha256':sha256_obj(intent),'snapshot_sha256':sha256_obj(snapshot),'physics_contract_sha256':contract['contract_sha256']}
    base['decision_sha256']=sha256_obj(base)
    return base

def supervise(intent:Mapping[str,Any], snapshot:Mapping[str,Any], risk_authority:Mapping[str,Any], physics_contract:Mapping[str,Any])->dict[str,Any]:
    validate_physics_contract(physics_contract)
    if snapshot.get('schema')!='SimulatorStateSnapshotV1' or snapshot.get('physics_contract_sha256')!=physics_contract.get('contract_sha256'): raise ControlPlaneIntegrityError('SNAPSHOT_PHYSICS_LINEAGE_MISMATCH')
    validate_snapshot_risk_sync(snapshot,risk_authority)
    # malformed ActionIntent is rejected into explicit no-op, not repaired into a valid intent
    if intent.get('schema')!='ActionIntentV1': return _noop('REJECT',intent,snapshot,['INTENT_SCHEMA_INVALID'],physics_contract)
    d=intent.get('direction'); rr=intent.get('requested_risk_multiplier')
    if isinstance(d,bool) or not isinstance(d,int) or d not in (SHORT,FLAT,LONG): return _noop('REJECT',intent,snapshot,['DIRECTION_OUT_OF_CONTRACT'],physics_contract)
    try: x=float(rr)
    except Exception: return _noop('REJECT',intent,snapshot,['RISK_NOT_NUMERIC'],physics_contract)
    if not math.isfinite(x): return _noop('REJECT',intent,snapshot,['RISK_NOT_FINITE'],physics_contract)
    if x<0.0 or x>1.0: return _noop('REJECT',intent,snapshot,['RISK_OUT_OF_HARD_BOUND'],physics_contract)
    term=snapshot.get('termination_state',{})
    if term.get('terminated'): return _noop('FORCED_NOOP',intent,snapshot,['TERMINAL_SNAPSHOT'],physics_contract)
    if term.get('truncated'): return _noop('FORCED_NOOP',intent,snapshot,['TRUNCATED_SNAPSHOT'],physics_contract)
    st=snapshot['kernel_state']; pos=float(st['position'])
    if abs(pos)>=1e-12:
        reasons=['POSITION_ALREADY_OPEN','ENVIRONMENT_OWNS_EXIT']
        if int(st.get('position_age_bars',0))>=int(physics_contract['sim_config']['max_holding_bars']): reasons.append('ENVIRONMENT_FORCED_EXIT_PENDING_MAX_HOLDING')
        return _noop('FORCED_NOOP',intent,snapshot,reasons,physics_contract)
    if d==FLAT: return _decision('ACCEPT',intent,snapshot,FLAT,0.0,['FLAT_INTENT_NO_ENTRY'],physics_contract)
    until=int(st.get('stop_cooldown_until',-1)); survived=int(st.get('steps_survived',0))
    if until>=0 and survived+1<=until: return _noop('REJECT',intent,snapshot,['COOLDOWN_ENTRY_BLOCKED'],physics_contract)
    # Flat V5.5 account: cash is the available collateral before a new entry; exact order feasibility remains physics-owned.
    cash=float(st.get('cash',float('nan')))
    if not math.isfinite(cash): raise ControlPlaneIntegrityError('NONFINITE_MARGIN_STATE')
    if cash<=0.0: return _noop('REJECT',intent,snapshot,['MARGIN_INSUFFICIENT'],physics_contract)
    budget_frac=float(risk_authority['risk_budget_remaining'])/float(risk_authority['risk_budget_capacity'])
    if x>budget_frac:
        return _decision('CLAMP',intent,snapshot,d,max(0.0,budget_frac),['RISK_ABOVE_REMAINING_BUDGET_ENVELOPE'],physics_contract)
    return _decision('ACCEPT',intent,snapshot,d,x,['WITHIN_ALL_HARD_ENVELOPES'],physics_contract)

def executable_action(decision:Mapping[str,Any], physics_contract:Mapping[str,Any])->dict[str,Any]:
    if decision.get('schema')!='RiskSupervisorDecisionV1' or decision.get('decision') not in DECISIONS: raise ControlPlaneIntegrityError('SUPERVISOR_DECISION_INVALID')
    expected=dict(decision); got=expected.pop('decision_sha256',None)
    if got!=sha256_obj(expected): raise ControlPlaneIntegrityError('SUPERVISOR_DECISION_HASH_MISMATCH')
    if decision.get('physics_contract_sha256')!=physics_contract.get('contract_sha256'): raise ControlPlaneIntegrityError('SUPERVISOR_PHYSICS_BINDING_MISMATCH')
    return {'schema':'ExecutableActionV1','direction':int(decision['executable_direction']),'risk_multiplier':float(decision['executable_risk_multiplier']),
            'semantics':'V5_5_ENTRY_OR_NOOP_ONLY','supervisor_decision_sha256':decision['decision_sha256'],'physics_contract_sha256':physics_contract['contract_sha256']}

def execute_physics(snapshot:Mapping[str,Any], action:Mapping[str,Any], market_execution_input:Mapping[str,Any], physics_contract:Mapping[str,Any])->dict[str,Any]:
    if action.get('schema')!='ExecutableActionV1' or action.get('semantics')!='V5_5_ENTRY_OR_NOOP_ONLY': raise ControlPlaneIntegrityError('PHYSICS_REQUIRES_EXECUTABLE_ACTION_V1')
    if action.get('physics_contract_sha256')!=physics_contract.get('contract_sha256'): raise ControlPlaneIntegrityError('EXECUTABLE_ACTION_PHYSICS_MISMATCH')
    d=action.get('direction'); r=action.get('risk_multiplier')
    if d not in (SHORT,FLAT,LONG) or not isinstance(r,(int,float)) or isinstance(r,bool) or not math.isfinite(float(r)) or not 0<=float(r)<=1: raise ControlPlaneIntegrityError('EXECUTABLE_ACTION_OUT_OF_DOMAIN')
    return step_account(snapshot,{'decision':int(d),'risk_multiplier':float(r)},market_execution_input,physics_contract)

def transition_record_v2(*,market_packet_key:Mapping[str,Any],snapshot_t:Mapping[str,Any],account_observation_t:Mapping[str,Any],intent:Mapping[str,Any],supervisor_decision:Mapping[str,Any],action:Mapping[str,Any],step_result:Mapping[str,Any],lineage:Mapping[str,Any],snapshot_store:ContentAddressedSnapshotStore)->dict[str,Any]:
    return {
       'schema':'TransitionRecordV2',
       'market_packet_key_t':dict(market_packet_key),
       'simulator_snapshot_ref_t':snapshot_store.put(snapshot_t),
       'account_observation_t':copy.deepcopy(account_observation_t),
       'action':{'intent':copy.deepcopy(intent),'supervisor_decision':copy.deepcopy(supervisor_decision),'executable_action':copy.deepcopy(action)},
       'execution_metadata':copy.deepcopy(step_result['execution_metadata']),
       'simulator_snapshot_ref_t1':snapshot_store.put(step_result['snapshot_t1']),
       'account_observation_t1':copy.deepcopy(step_result['account_observation_t1']),
       'termination_type':step_result['termination_type'],
       'lineage':dict(lineage),
    }

def validate_binding_manifest(manifest:Mapping[str,Any], runtime_root:str|Path)->None:
    rr=Path(runtime_root); files=manifest.get('runtime_file_bindings',{})
    for rel,expected in files.items():
        p=rr/rel
        if not p.is_file() or sha256_file(p)!=expected: raise ControlPlaneIntegrityError('RUNTIME_FILE_BINDING_MISMATCH:'+rel)
    for k in ('frozen_trading_kernel_sha256','sim_config_sha256','account_schema_sha256','risk_supervisor_sha256','risk_authority_contract_sha256','replay_schema_sha256'):
        v=manifest.get(k,'')
        if not isinstance(v,str) or len(v)!=64: raise ControlPlaneIntegrityError('LINEAGE_BINDING_MISSING:'+k)
