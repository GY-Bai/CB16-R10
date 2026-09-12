from __future__ import annotations
from pathlib import Path
from typing import Mapping, Any
from .cc_experience_store_r0 import RawFactStore

class RawFactTransactionHarness:
    def __init__(self, root: str | Path): self.store=RawFactStore(root)
    def crash_before_write(self, logical_id: str, fact: Mapping[str, Any]) -> None:
        raise RuntimeError("FAULT_BEFORE_WRITE")
    def partial_temp_write(self, bytes_: bytes = b"partial") -> Path:
        p=self.store.objects/"zz"; p.mkdir(parents=True, exist_ok=True)
        tmp=p/".cc-tmp-partial"; tmp.write_bytes(bytes_); return tmp
    def durable_before_ack(self, logical_id: str, fact: Mapping[str, Any]) -> None:
        self.store.put(logical_id, fact, crash_after_object=True)
    def retry(self, logical_id: str, fact: Mapping[str, Any]) -> str:
        return self.store.put(logical_id, fact)
