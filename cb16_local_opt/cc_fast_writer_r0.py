from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,os,tempfile
from pathlib import Path
from typing import Sequence
from .cc_fast_fact_queue_r0 import FactEnvelope
WRITER_SCHEMA="CB16_R11_CC_FAST_CHUNK_WRITER_V1"
@dataclass(frozen=True)
class ChunkReceipt:
    schema_version:str; chunk_id:str; path:str; fact_ids:tuple[str,...]; bytes_written:int; content_sha256:str; durable:bool
    def validate(self):
        if self.schema_version!=WRITER_SCHEMA or not self.chunk_id or not self.path: raise ValueError("CHUNK_RECEIPT_INVALID")
        if not self.fact_ids or len(set(self.fact_ids))!=len(self.fact_ids): raise ValueError("CHUNK_FACT_IDS_INVALID")
        if self.bytes_written<=0 or len(self.content_sha256)!=64 or not self.durable: raise ValueError("CHUNK_DURABILITY_INVALID")
        return self
class ContiguousChunkWriter:
    def __init__(self,root): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def write_chunk(self,facts:Sequence[FactEnvelope],*,chunk_id:str):
        if not facts or not chunk_id: raise ValueError("CHUNK_INPUT_INVALID")
        ids=[x.semantic_id for x in facts]
        if len(set(ids))!=len(ids): raise ValueError("DUPLICATE_FACT_ID_IN_CHUNK")
        raw=b"".join(json.dumps({"semantic_id":f.semantic_id,"terminal_or_failure":f.terminal_or_failure,"payload_sha256":hashlib.sha256(f.payload).hexdigest(),"payload":f.payload.decode("utf-8")},sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")+b"\n" for f in facts); digest=hashlib.sha256(raw).hexdigest(); final_path=self.root/f"{chunk_id}.ccchunk"; fd,tmp_name=tempfile.mkstemp(prefix=f".{chunk_id}.",suffix=".tmp",dir=self.root)
        try:
            with os.fdopen(fd,"wb") as fh: fh.write(raw); fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp_name,final_path); dir_fd=os.open(self.root,os.O_RDONLY)
            try: os.fsync(dir_fd)
            finally: os.close(dir_fd)
        finally:
            if os.path.exists(tmp_name): os.unlink(tmp_name)
        return ChunkReceipt(WRITER_SCHEMA,chunk_id,str(final_path),tuple(ids),len(raw),digest,True).validate()
    @staticmethod
    def verify(receipt:ChunkReceipt):
        receipt.validate(); p=Path(receipt.path); return p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==receipt.content_sha256
