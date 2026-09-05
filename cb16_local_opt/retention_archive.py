from __future__ import annotations

"""
SSD retention and HDD archival manager.

Safety rule: referenced Champion checkpoints, active snapshots and manifests are pinned.
The manager may move/archive only objects explicitly classified as cold/eligible.
Deletion is never implicit: the default action is verified move to HDD.
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path:str|Path,chunk:int=8<<20)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(chunk),b""): h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class RetentionPolicy:
    ssd_high_watermark: float=0.80
    ssd_low_watermark: float=0.65
    keep_recent_generations: int=3
    keep_recent_checkpoints_per_generation: int=2
    min_age_seconds_before_archive: float=3600.0

    def validate(self):
        if not 0<self.ssd_low_watermark<self.ssd_high_watermark<1:
            raise ValueError("bad watermarks")
        if self.keep_recent_generations<1:
            raise ValueError("must keep recent generations")


@dataclass(frozen=True)
class ArchiveCandidate:
    artifact_id: str
    path: str
    role: str
    generation: int|None
    created_at_unix: float
    pinned: bool=False


@dataclass(frozen=True)
class ArchiveReceipt:
    artifact_id: str
    source_path: str
    archive_path: str
    sha256: str
    bytes: int
    moved: bool
    reason: str


class RetentionArchiveManager:
    def __init__(
        self,
        *,
        ssd_root:str|Path,
        hdd_root:str|Path,
        policy:RetentionPolicy|None=None,
    ):
        self.ssd_root=Path(ssd_root).resolve()
        self.hdd_root=Path(hdd_root).resolve()
        self.hdd_root.mkdir(parents=True,exist_ok=True)
        self.policy=policy or RetentionPolicy()
        self.policy.validate()

    def ssd_usage_fraction(self)->float:
        u=shutil.disk_usage(self.ssd_root)
        return 0.0 if u.total<=0 else (u.total-u.free)/u.total

    def should_archive(self)->bool:
        return self.ssd_usage_fraction()>=self.policy.ssd_high_watermark

    def _relative(self,p:Path)->Path:
        p=p.resolve()
        try:return p.relative_to(self.ssd_root)
        except ValueError as exc:
            raise RuntimeError("ARCHIVE_SOURCE_OUTSIDE_SSD_ROOT") from exc

    def select(
        self,
        candidates:Iterable[ArchiveCandidate],
        *,
        current_generation:int,
        now:float|None=None,
    )->list[ArchiveCandidate]:
        now=time.time() if now is None else float(now)
        floor=current_generation-self.policy.keep_recent_generations+1
        out=[]
        for c in candidates:
            p=Path(c.path)
            if c.pinned or not p.is_file():
                continue
            if now-c.created_at_unix<self.policy.min_age_seconds_before_archive:
                continue
            if c.generation is not None and c.generation>=floor:
                continue
            out.append(c)
        out.sort(key=lambda c:(c.created_at_unix,c.artifact_id))
        return out

    def archive_one(self,c:ArchiveCandidate)->ArchiveReceipt:
        src=Path(c.path).resolve()
        rel=self._relative(src)
        dst=(self.hdd_root/rel).resolve()
        dst.parent.mkdir(parents=True,exist_ok=True)
        src_hash=sha256_file(src)
        size=src.stat().st_size

        if dst.exists():
            if sha256_file(dst)!=src_hash:
                raise RuntimeError("ARCHIVE_DESTINATION_CONFLICT")
            src.unlink()
            return ArchiveReceipt(c.artifact_id,str(src),str(dst),src_hash,size,True,"DEST_ALREADY_VERIFIED")

        tmp=dst.with_name(dst.name+f".{os.getpid()}.partial")
        tmp.unlink(missing_ok=True)
        shutil.copy2(src,tmp)
        if sha256_file(tmp)!=src_hash:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("ARCHIVE_COPY_HASH_MISMATCH")
        with tmp.open("rb") as f: os.fsync(f.fileno())
        os.replace(tmp,dst)
        if sha256_file(dst)!=src_hash:
            raise RuntimeError("ARCHIVE_FINAL_HASH_MISMATCH")
        src.unlink()
        return ArchiveReceipt(c.artifact_id,str(src),str(dst),src_hash,size,True,"VERIFIED_MOVE")

    def relieve_pressure(
        self,
        candidates:Iterable[ArchiveCandidate],
        *,
        current_generation:int,
    )->list[ArchiveReceipt]:
        if not self.should_archive():
            return []
        receipts=[]
        for c in self.select(candidates,current_generation=current_generation):
            receipts.append(self.archive_one(c))
            if self.ssd_usage_fraction()<=self.policy.ssd_low_watermark:
                break
        return receipts
