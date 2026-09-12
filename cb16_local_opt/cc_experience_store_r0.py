from __future__ import annotations
import hashlib,json,os
from pathlib import Path
from typing import Any,Mapping
class FactStoreConflict(RuntimeError): pass
class FactStoreCorruption(RuntimeError): pass
def canonical_bytes(p:Mapping[str,Any])->bytes: return json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
class ContentAddressedFactStore:
    def __init__(self,root:str|Path):
        self.root=Path(root); self.objects=self.root/"objects"; self.objects.mkdir(parents=True,exist_ok=True); self.index_path=self.root/"index.json"
        if not self.index_path.exists(): self._write_index({})
    def _read_index(self):
        try:return json.loads(self.index_path.read_text())
        except Exception as e: raise FactStoreCorruption("unreadable index") from e
    def _write_index(self,index):
        tmp=self.index_path.with_suffix(".tmp"); tmp.write_text(json.dumps(dict(index),sort_keys=True,separators=(",",":")))
        with tmp.open("rb") as f: os.fsync(f.fileno())
        os.replace(tmp,self.index_path)
    def content_hash(self,p): return hashlib.sha256(canonical_bytes(p)).hexdigest()
    def write_blob(self,p):
        raw=canonical_bytes(p); d=hashlib.sha256(raw).hexdigest(); path=self.objects/f"{d}.json"
        if not path.exists():
            tmp=path.with_suffix(".tmp"); tmp.write_bytes(raw)
            with tmp.open("rb") as f: os.fsync(f.fileno())
            os.replace(tmp,path)
        elif path.read_bytes()!=raw: raise FactStoreCorruption("hash collision/corruption")
        return d
    def bind_object(self,object_id,d):
        if not object_id: raise ValueError("object_id required")
        if not (self.objects/f"{d}.json").exists(): raise FactStoreCorruption("missing blob")
        idx=self._read_index(); existing=idx.get(object_id)
        if existing is not None and existing!=d: raise FactStoreConflict("same ID/different content")
        if existing==d:return d
        idx[object_id]=d; self._write_index(idx); return d
    def put(self,object_id,p): return self.bind_object(object_id,self.write_blob(p))
    def get(self,object_id):
        d=self._read_index()[object_id]; raw=(self.objects/f"{d}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest()!=d: raise FactStoreCorruption("blob hash mismatch")
        return json.loads(raw)
    def identities(self): return tuple(sorted(self._read_index()))
