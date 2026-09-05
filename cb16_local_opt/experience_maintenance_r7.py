from __future__ import annotations

"""
Long-campaign Experience Lake maintenance.

Designed for 100+ generations:
- rebuild a compact global campaign index from sharded metadata;
- verify every frozen snapshot still resolves to immutable object identities;
- WAL TRUNCATE / ANALYZE / optional VACUUM under an exclusive maintenance lock;
- identify CAS payload files not referenced by any metadata row;
- orphan deletion is opt-in and age-gated.

No referenced Experience object is rewritten or recompressed.
"""

import contextlib
import dataclasses
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Any,Iterator,Mapping


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class GenerationIndexRowR7:
    generation:int
    object_type:str
    object_count:int
    bytes_raw:int
    bytes_stored:int
    first_created_at:float
    last_created_at:float


@dataclass(frozen=True)
class SnapshotAuditRowR7:
    snapshot_id:str
    parent_generation:int
    parent_policy_hash:str
    object_count:int
    resolved_objects:int
    missing_objects:int
    identity_mismatches:int
    content_hash:str
    status:str


@dataclass(frozen=True)
class LakeMaintenanceReceiptR7:
    maintenance_version:str
    shards:int
    objects:int
    snapshots:int
    broken_snapshots:int
    orphan_payload_files:int
    orphan_payload_bytes:int
    deleted_orphan_files:int
    wal_checkpointed:bool
    analyzed:bool
    vacuumed:bool
    global_index_path:str
    global_index_sha256:str
    receipt_hash_material:str

    @property
    def content_hash(self):return canonical_hash(self)


class MaintenanceLockR7:
    def __init__(self,root:str|Path):
        self.path=Path(root)/".r7_maintenance.lock"
        self.fd=None

    def __enter__(self):
        try:
            self.fd=os.open(
                self.path,
                os.O_CREAT|os.O_EXCL|os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise RuntimeError("EXPERIENCE_LAKE_MAINTENANCE_ALREADY_RUNNING") from exc
        os.write(self.fd,f"pid={os.getpid()} time={time.time()}\n".encode())
        os.fsync(self.fd)
        return self

    def __exit__(self,exc_type,exc,tb):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


class ExperienceLakeMaintenanceR7:
    def __init__(self,root:str|Path):
        self.root=Path(root)
        self.metadata_root=self.root/"metadata"
        self.objects_root=self.root/"objects"
        if not self.metadata_root.is_dir():
            raise FileNotFoundError(self.metadata_root)
        self.db_paths=sorted(self.metadata_root.glob("experience_*.sqlite"))
        if not self.db_paths:
            raise RuntimeError("NO_EXPERIENCE_METADATA_SHARDS")

    def _connections(self):
        return [
            sqlite3.connect(p,timeout=30,isolation_level=None)
            for p in self.db_paths
        ]

    def rebuild_global_index(self,path:str|Path)->Path:
        out=Path(path)
        out.parent.mkdir(parents=True,exist_ok=True)
        tmp=out.with_name(out.name+f".{os.getpid()}.partial")
        tmp.unlink(missing_ok=True)
        db=sqlite3.connect(tmp)
        db.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE generation_summary(
            generation INTEGER NOT NULL,
            object_type TEXT NOT NULL,
            object_count INTEGER NOT NULL,
            bytes_raw INTEGER NOT NULL,
            bytes_stored INTEGER NOT NULL,
            first_created_at REAL NOT NULL,
            last_created_at REAL NOT NULL,
            PRIMARY KEY(generation,object_type)
        );
        CREATE TABLE snapshots(
            snapshot_id TEXT PRIMARY KEY,
            parent_generation INTEGER NOT NULL,
            parent_policy_hash TEXT NOT NULL,
            object_count INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            missing_objects INTEGER NOT NULL,
            identity_mismatches INTEGER NOT NULL
        );
        CREATE TABLE metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)
        aggregate={}
        object_identity={}
        snapshots=[]
        for shard,p in enumerate(self.db_paths):
            src=sqlite3.connect(p)
            for row in src.execute("""
                SELECT object_id,identity_hash,generation,object_type,
                       bytes_raw,bytes_stored,created_at
                FROM objects
            """):
                oid,ih,gen,typ,br,bs,created=row
                old=object_identity.get(oid)
                if old is not None and old!=ih:
                    src.close();db.close();tmp.unlink(missing_ok=True)
                    raise RuntimeError("GLOBAL_OBJECT_ID_IDENTITY_CONFLICT")
                object_identity[oid]=ih
                key=(int(gen),str(typ))
                a=aggregate.setdefault(
                    key,[0,0,0,float(created),float(created)]
                )
                a[0]+=1;a[1]+=int(br);a[2]+=int(bs)
                a[3]=min(a[3],float(created));a[4]=max(a[4],float(created))
            for row in src.execute("""
                SELECT snapshot_id,content_hash,parent_generation,
                       parent_policy_hash,object_count,payload_json
                FROM snapshots
            """):
                snapshots.append(row)
            src.close()

        for (gen,typ),a in sorted(aggregate.items()):
            db.execute(
                "INSERT INTO generation_summary VALUES(?,?,?,?,?,?,?)",
                (gen,typ,a[0],a[1],a[2],a[3],a[4])
            )

        broken=0
        snapshot_audits=[]
        for sid,ch,gen,ph,count,payload in snapshots:
            obj=json.loads(payload)
            ids=obj["object_ids"]
            ihs=obj["object_identity_hashes"]
            missing=0;mismatch=0;resolved=0
            for oid,expected in zip(ids,ihs):
                actual=object_identity.get(oid)
                if actual is None:missing+=1
                elif actual!=expected:mismatch+=1
                else:resolved+=1
            status="PASS" if missing==0 and mismatch==0 else "BROKEN"
            broken+=status!="PASS"
            audit=SnapshotAuditRowR7(
                snapshot_id=sid,
                parent_generation=int(gen),
                parent_policy_hash=ph,
                object_count=int(count),
                resolved_objects=resolved,
                missing_objects=missing,
                identity_mismatches=mismatch,
                content_hash=ch,
                status=status,
            )
            snapshot_audits.append(audit)
            db.execute(
                "INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,?)",
                (
                    sid,int(gen),ph,int(count),ch,status,missing,mismatch
                )
            )
        db.execute(
            "INSERT INTO metadata VALUES('schema','CB16_EXPERIENCE_GLOBAL_INDEX_R7')"
        )
        db.execute(
            "INSERT INTO metadata VALUES('objects',?)",(str(len(object_identity)),)
        )
        db.execute(
            "INSERT INTO metadata VALUES('broken_snapshots',?)",(str(broken),)
        )
        db.commit()
        db.execute("VACUUM")
        db.close()
        os.replace(tmp,out)
        return out

    def referenced_payload_paths(self)->set[str]:
        paths=set()
        for p in self.db_paths:
            db=sqlite3.connect(p)
            paths.update(
                str(Path(row[0]).resolve())
                for row in db.execute("SELECT payload_path FROM objects")
            )
            db.close()
        return paths

    def orphan_payloads(
        self,
        *,
        grace_seconds:float=86400.0,
        now:float|None=None,
    )->list[Path]:
        now=time.time() if now is None else float(now)
        refs=self.referenced_payload_paths()
        out=[]
        if not self.objects_root.exists():return out
        for p in self.objects_root.rglob("*.json.zlib"):
            if str(p.resolve()) in refs:continue
            if now-p.stat().st_mtime<grace_seconds:continue
            out.append(p)
        out.sort()
        return out

    def run(
        self,
        *,
        global_index_path:str|Path|None=None,
        checkpoint_wal:bool=True,
        analyze:bool=True,
        vacuum:bool=False,
        delete_orphans:bool=False,
        orphan_grace_seconds:float=86400.0,
    )->LakeMaintenanceReceiptR7:
        index=Path(global_index_path or (self.root/"campaign_index_r7.sqlite"))
        with MaintenanceLockR7(self.root):
            index=self.rebuild_global_index(index)
            idxdb=sqlite3.connect(index)
            objects=int(idxdb.execute(
                "SELECT value FROM metadata WHERE key='objects'"
            ).fetchone()[0])
            broken=int(idxdb.execute(
                "SELECT value FROM metadata WHERE key='broken_snapshots'"
            ).fetchone()[0])
            snapshots=int(idxdb.execute(
                "SELECT COUNT(*) FROM snapshots"
            ).fetchone()[0])
            idxdb.close()

            conns=self._connections()
            try:
                for db in conns:
                    if checkpoint_wal:
                        db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                    if analyze:
                        db.execute("ANALYZE")
                    if vacuum:
                        db.execute("VACUUM")
            finally:
                for db in conns:db.close()

            orphans=self.orphan_payloads(
                grace_seconds=orphan_grace_seconds
            )
            orphan_bytes=sum(p.stat().st_size for p in orphans if p.exists())
            deleted=0
            if delete_orphans:
                for p in orphans:
                    p.unlink(missing_ok=True);deleted+=1

            material=canonical_hash({
                "objects":objects,
                "snapshots":snapshots,
                "broken_snapshots":broken,
                "orphan_paths":[str(x) for x in orphans],
                "checkpoint_wal":checkpoint_wal,
                "analyze":analyze,
                "vacuum":vacuum,
                "delete_orphans":delete_orphans,
            })
            receipt=LakeMaintenanceReceiptR7(
                maintenance_version="CB16_EXPERIENCE_MAINTENANCE_R7",
                shards=len(self.db_paths),
                objects=objects,
                snapshots=snapshots,
                broken_snapshots=broken,
                orphan_payload_files=len(orphans),
                orphan_payload_bytes=orphan_bytes,
                deleted_orphan_files=deleted,
                wal_checkpointed=checkpoint_wal,
                analyzed=analyze,
                vacuumed=vacuum,
                global_index_path=str(index),
                global_index_sha256=self._sha(index),
                receipt_hash_material=material,
            )
            if broken:
                raise RuntimeError(
                    f"BROKEN_EXPERIENCE_SNAPSHOTS:{broken}"
                )
            return receipt

    @staticmethod
    def _sha(path:Path):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda:f.read(1<<20),b""):h.update(b)
        return h.hexdigest()
