from __future__ import annotations

"""Continuous immutable dataset catalog and frozen snapshot builder.

Accepted transport bundles may arrive continuously. Campaigns never train from the mutable
catalog head; they train from a frozen CatalogSnapshot whose scientific hash binds the exact
set/order of accepted bundle and shard identities.
"""

import dataclasses, hashlib, json, sqlite3, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .historical_shard_reader import DatasetManifest, ShardSpec
from .portable_dataset_r8 import load_portable_manifest_r8


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class CatalogBundleR8:
    bundle_sha256:str
    accepted_root:str
    manifest_path:str
    portable_dataset_hash:str
    dataset_id:str
    timeframe:str
    symbols:tuple[str,...]
    registered_at:float


@dataclass(frozen=True)
class CatalogSnapshotR8:
    snapshot_id:str
    timeframe:str
    symbols:tuple[str,...]
    bundle_sha256s:tuple[str,...]
    portable_dataset_hashes:tuple[str,...]
    shard_identity_hashes:tuple[str,...]
    scientific_dataset_hash:str
    created_at:float

    @property
    def content_hash(self)->str:
        d=asdict(self);d.pop('created_at',None);return canonical_hash(d)


class DatasetCatalogR8:
    def __init__(self,path:str|Path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,isolation_level=None,timeout=30);self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS bundles(
            bundle_sha256 TEXT PRIMARY KEY,
            accepted_root TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            portable_dataset_hash TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            symbols_json BLOB NOT NULL,
            registered_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snapshots(
            snapshot_id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            payload_json BLOB NOT NULL,
            created_at REAL NOT NULL
        );
        ''')
    def close(self):self.db.close()

    def register_accepted_bundle(self,receipt)->CatalogBundleR8:
        pm=load_portable_manifest_r8(receipt.manifest_path);now=time.time();row=CatalogBundleR8(receipt.bundle_sha256,receipt.accepted_root,receipt.manifest_path,pm.content_hash,pm.dataset_id,pm.timeframe,pm.symbols,now)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old=self.db.execute('SELECT portable_dataset_hash,manifest_path,accepted_root FROM bundles WHERE bundle_sha256=?',(row.bundle_sha256,)).fetchone()
            if old:
                if old!=(row.portable_dataset_hash,row.manifest_path,row.accepted_root):raise RuntimeError('CATALOG_BUNDLE_IDENTITY_CONFLICT')
                self.db.execute('COMMIT');return row
            self.db.execute('INSERT INTO bundles VALUES(?,?,?,?,?,?,?,?)',(row.bundle_sha256,row.accepted_root,row.manifest_path,row.portable_dataset_hash,row.dataset_id,row.timeframe,json.dumps(list(row.symbols)),row.registered_at));self.db.execute('COMMIT');return row
        except Exception:self.db.execute('ROLLBACK');raise

    def list_bundles(self)->list[CatalogBundleR8]:
        return [CatalogBundleR8(r[0],r[1],r[2],r[3],r[4],r[5],tuple(json.loads(r[6])),float(r[7])) for r in self.db.execute('SELECT bundle_sha256,accepted_root,manifest_path,portable_dataset_hash,dataset_id,timeframe,symbols_json,registered_at FROM bundles ORDER BY registered_at,bundle_sha256')]

    def create_snapshot(self,snapshot_id:str,*,bundle_sha256s:Sequence[str]|None=None)->CatalogSnapshotR8:
        bundles=self.list_bundles();wanted=None if bundle_sha256s is None else set(bundle_sha256s);selected=[b for b in bundles if wanted is None or b.bundle_sha256 in wanted]
        if not selected:raise RuntimeError('CATALOG_SNAPSHOT_EMPTY')
        if wanted is not None and {b.bundle_sha256 for b in selected}!=wanted:raise RuntimeError('CATALOG_SNAPSHOT_UNKNOWN_BUNDLE')
        timeframes={b.timeframe for b in selected}
        if len(timeframes)!=1:raise RuntimeError('CATALOG_SNAPSHOT_TIMEFRAME_MISMATCH')
        # Verify no chronology overlap per symbol across every mounted portable shard.
        shard_rows=[];symbols=set()
        for b in selected:
            pm=load_portable_manifest_r8(b.manifest_path);symbols.update(pm.symbols)
            for s in pm.shards:
                shard_rows.append((s.symbol,s.start_timestamp,s.end_timestamp,b.bundle_sha256,s.relative_path,s.sha256,s.rows,s.bytes,s.format,s.shard_id,b.accepted_root))
        prev={}
        for r in sorted(shard_rows,key=lambda x:(x[0],x[1],x[3],x[4])):
            old=prev.get(r[0])
            if old is not None and r[1]<=old:raise RuntimeError('CATALOG_SNAPSHOT_OVERLAPPING_CHRONOLOGY:'+r[0])
            prev[r[0]]=r[2]
        shard_hashes=tuple(canonical_hash({'symbol':r[0],'start':r[1],'end':r[2],'bundle':r[3],'relative_path':r[4],'sha256':r[5],'rows':r[6],'bytes':r[7],'format':r[8],'shard_id':r[9]}) for r in sorted(shard_rows,key=lambda x:(x[0],x[1],x[3],x[4])))
        scientific=canonical_hash({'schema':'CB16_CATALOG_SCIENTIFIC_DATASET_R8','timeframe':next(iter(timeframes)),'symbols':sorted(symbols),'bundles':[b.bundle_sha256 for b in selected],'portable_hashes':[b.portable_dataset_hash for b in selected],'shards':shard_hashes})
        snap=CatalogSnapshotR8(snapshot_id,next(iter(timeframes)),tuple(sorted(symbols)),tuple(b.bundle_sha256 for b in selected),tuple(b.portable_dataset_hash for b in selected),shard_hashes,scientific,time.time())
        payload=asdict(snap);self.db.execute('BEGIN IMMEDIATE')
        try:
            old=self.db.execute('SELECT content_hash FROM snapshots WHERE snapshot_id=?',(snapshot_id,)).fetchone()
            if old:
                if old[0]!=snap.content_hash:raise RuntimeError('CATALOG_SNAPSHOT_IDENTITY_CONFLICT')
                self.db.execute('COMMIT');return snap
            self.db.execute('INSERT INTO snapshots VALUES(?,?,?,?)',(snapshot_id,snap.content_hash,json.dumps(payload,sort_keys=True),snap.created_at));self.db.execute('COMMIT');return snap
        except Exception:self.db.execute('ROLLBACK');raise


    def get_snapshot(self,snapshot_id:str)->CatalogSnapshotR8:
        row=self.db.execute('SELECT payload_json FROM snapshots WHERE snapshot_id=?',(snapshot_id,)).fetchone()
        if row is None:raise KeyError(snapshot_id)
        x=json.loads(row[0]);x['symbols']=tuple(x['symbols']);x['bundle_sha256s']=tuple(x['bundle_sha256s']);x['portable_dataset_hashes']=tuple(x['portable_dataset_hashes']);x['shard_identity_hashes']=tuple(x['shard_identity_hashes']);return CatalogSnapshotR8(**x)

    def mount_snapshot(self,snapshot:CatalogSnapshotR8,*,verify_files:bool=True)->DatasetManifest:
        bundles={b.bundle_sha256:b for b in self.list_bundles()};shards=[]
        for bh in snapshot.bundle_sha256s:
            b=bundles.get(bh)
            if b is None:raise RuntimeError('CATALOG_SNAPSHOT_BUNDLE_DISAPPEARED')
            pm=load_portable_manifest_r8(b.manifest_path);mounted=pm.mount(b.accepted_root,verify_files=verify_files);shards.extend(mounted.shards)
        return DatasetManifest(snapshot.snapshot_id,snapshot.symbols,snapshot.timeframe,tuple(shards),source='CATALOG_SNAPSHOT_R8')
