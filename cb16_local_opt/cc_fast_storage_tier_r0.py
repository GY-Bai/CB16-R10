from __future__ import annotations
from dataclasses import dataclass
import hashlib,os,shutil,tempfile
from pathlib import Path

@dataclass(frozen=True)
class TierConfig:
    ram_buffer_bytes:int; ssd_capacity_bytes:int; hdd_capacity_bytes:int
    def validate(self):
        if min(self.ram_buffer_bytes,self.ssd_capacity_bytes,self.hdd_capacity_bytes)<=0: raise ValueError("TIER_CAPACITY_INVALID")
        return self
@dataclass(frozen=True)
class ArchiveReceipt:
    source_path:str; archive_path:str; content_sha256:str; bytes_archived:int; checksum_verified:bool; source_reclaimed:bool
class StorageTierManager:
    def __init__(self,*,ssd_root:str,hdd_root:str,config:TierConfig):
        self.ssd_root=Path(ssd_root); self.hdd_root=Path(hdd_root); self.config=config.validate(); self.ssd_root.mkdir(parents=True,exist_ok=True); self.hdd_root.mkdir(parents=True,exist_ok=True)
    @staticmethod
    def sha256_file(path:Path):
        h=hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda:fh.read(1<<20),b""): h.update(block)
        return h.hexdigest()
    def archive(self,source,*,reclaim_hot:bool=True):
        src=Path(source)
        if not src.is_file(): raise FileNotFoundError(src)
        size=src.stat().st_size
        if size>self.config.hdd_capacity_bytes: raise RuntimeError("HDD_CAPACITY_BUDGET_EXCEEDED")
        digest=self.sha256_file(src); dest=self.hdd_root/src.name; fd,tmp_name=tempfile.mkstemp(prefix=f".{src.name}.",suffix=".partial",dir=self.hdd_root); os.close(fd)
        try:
            shutil.copyfile(src,tmp_name)
            if self.sha256_file(Path(tmp_name))!=digest: raise RuntimeError("ARCHIVE_CHECKSUM_MISMATCH")
            os.replace(tmp_name,dest)
            if self.sha256_file(dest)!=digest: raise RuntimeError("ARCHIVE_POST_RENAME_CHECKSUM_MISMATCH")
        finally:
            if os.path.exists(tmp_name): os.unlink(tmp_name)
        reclaimed=False
        if reclaim_hot: src.unlink(); reclaimed=True
        return ArchiveReceipt(str(src),str(dest),digest,size,True,reclaimed)
