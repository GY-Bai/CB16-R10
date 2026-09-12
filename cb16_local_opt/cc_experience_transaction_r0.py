from dataclasses import dataclass
from typing import Any,Mapping
from .cc_experience_store_r0 import ContentAddressedFactStore,canonical_bytes
class InjectedCrash(RuntimeError): pass
@dataclass(frozen=True)
class TransactionReceipt: object_id:str; content_hash:str; idempotent:bool
class ExperienceTransaction:
    def __init__(self,store:ContentAddressedFactStore): self.store=store
    def commit(self,object_id:str,payload:Mapping[str,Any],*,fault:str|None=None):
        if fault=="before_write": raise InjectedCrash(fault)
        d=self.store.content_hash(payload)
        if fault=="partial_temporary_write":
            raw=canonical_bytes(payload); (self.store.objects/f"{d}.json.tmp.injected").write_bytes(raw[:max(1,len(raw)//2)]); raise InjectedCrash(fault)
        pre=object_id in self.store.identities(); written=self.store.write_blob(payload)
        if fault=="after_durable_object_before_index": raise InjectedCrash(fault)
        self.store.bind_object(object_id,written); return TransactionReceipt(object_id,written,pre)
