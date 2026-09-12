from __future__ import annotations
from dataclasses import dataclass
import copy, hashlib, io, torch

@dataclass(frozen=True)
class BehaviorGeneration:
    generation:int; policy_id:str; policy_sha256:str; collection_unit_id:str; state_bytes:bytes

def module_sha256(module:torch.nn.Module)->str:
    b=io.BytesIO(); torch.save(module.state_dict(),b); return hashlib.sha256(b.getvalue()).hexdigest()

def freeze_behavior_generation(module:torch.nn.Module,generation:int,policy_id:str,collection_unit_id:str)->BehaviorGeneration:
    b=io.BytesIO(); torch.save(copy.deepcopy(module.state_dict()),b)
    raw=b.getvalue(); return BehaviorGeneration(generation,policy_id,hashlib.sha256(raw).hexdigest(),collection_unit_id,raw)

def child_copy(module:torch.nn.Module)->torch.nn.Module:return copy.deepcopy(module)
