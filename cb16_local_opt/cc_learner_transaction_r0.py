from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json

@dataclass
class ExactlyOnceLedger:
    committed: dict[str,dict]=field(default_factory=dict)
    prepared: dict[str,dict]=field(default_factory=dict)

    def prepare(self,update_id:str,parent_checkpoint:str,sequence_ids:tuple[str,...],step_before:int)->dict:
        payload={"update_id":update_id,"parent_checkpoint":parent_checkpoint,"sequence_ids":list(sequence_ids),"step_before":step_before}
        if update_id in self.committed:
            return self.committed[update_id]
        prior=self.prepared.get(update_id)
        if prior is not None and prior!=payload: raise RuntimeError("UPDATE_ID_REBOUND")
        self.prepared[update_id]=payload; return payload

    def commit(self,update_id:str,child_checkpoint:str,step_after:int)->bool:
        if update_id in self.committed:return False
        if update_id not in self.prepared:raise RuntimeError("UPDATE_NOT_PREPARED")
        rec={**self.prepared.pop(update_id),"child_checkpoint":child_checkpoint,"step_after":step_after,"status":"COMMITTED"}
        rec["receipt_sha256"]=hashlib.sha256(json.dumps(rec,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        self.committed[update_id]=rec; return True
