from __future__ import annotations

"""Location-independent immutable dataset manifest for OCI -> Shanxi transport.

R7 DatasetManifest includes physical shard paths in its content hash.  R8 separates:
- scientific/transport identity: relative path + bytes + SHA256 + chronology metadata;
- local mount path: supplied after transfer and excluded from the scientific hash.
"""

import dataclasses, hashlib, json, os, tarfile, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from .historical_shard_reader import DatasetManifest, ShardSpec, sha256_file


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class PortableShardSpecR8:
    shard_id:str
    relative_path:str
    symbol:str
    timeframe:str
    format:str
    start_timestamp:int
    end_timestamp:int
    rows:int
    sha256:str
    bytes:int

    def validate(self):
        p=PurePosixPath(self.relative_path)
        if p.is_absolute() or '..' in p.parts:raise ValueError('portable relative path invalid')
        if self.format not in {'csv','parquet'}:raise ValueError('format')
        if self.rows<=0 or self.start_timestamp>self.end_timestamp:raise ValueError('chronology/rows')
        if len(self.sha256)!=64:raise ValueError('sha256')


@dataclass(frozen=True)
class PortableDatasetManifestR8:
    dataset_id:str
    symbols:tuple[str,...]
    timeframe:str
    shards:tuple[PortableShardSpecR8,...]
    source:str='HISTORICAL_REPLAY'
    schema:str='CB16_PORTABLE_DATASET_MANIFEST_R8'

    def validate(self):
        if not self.dataset_id or not self.shards:raise ValueError('dataset id/shards')
        ids=set();paths=set();prev={}
        for s in sorted(self.shards,key=lambda x:(x.symbol,x.start_timestamp,x.shard_id)):
            s.validate()
            if s.shard_id in ids:raise ValueError('duplicate shard id')
            if s.relative_path in paths:raise ValueError('duplicate relative path')
            ids.add(s.shard_id);paths.add(s.relative_path)
            if s.symbol not in self.symbols:raise ValueError('shard symbol missing from symbols')
            if s.timeframe!=self.timeframe:raise ValueError('timeframe mismatch')
            old=prev.get(s.symbol)
            if old is not None and s.start_timestamp<=old:raise ValueError('overlapping shard chronology')
            prev[s.symbol]=s.end_timestamp

    @property
    def content_hash(self)->str:
        self.validate()
        return canonical_hash({
            'schema':self.schema,'dataset_id':self.dataset_id,'symbols':self.symbols,'timeframe':self.timeframe,
            'shards':[asdict(s) for s in self.shards],'source':self.source,
        })

    def mount(self,root:str|Path,*,verify_files:bool=True)->DatasetManifest:
        root=Path(root);shards=[]
        for s in self.shards:
            p=(root/s.relative_path).resolve()
            try:p.relative_to(root.resolve())
            except ValueError:raise RuntimeError('PORTABLE_SHARD_ESCAPES_ROOT')
            if verify_files:
                if not p.is_file():raise FileNotFoundError(p)
                if p.stat().st_size!=s.bytes:raise RuntimeError('PORTABLE_SHARD_SIZE_MISMATCH:'+s.shard_id)
                if sha256_file(p)!=s.sha256:raise RuntimeError('PORTABLE_SHARD_HASH_MISMATCH:'+s.shard_id)
            shards.append(ShardSpec(s.shard_id,str(p),s.symbol,s.timeframe,s.format,s.start_timestamp,s.end_timestamp,s.rows,s.sha256,s.bytes))
        return DatasetManifest(self.dataset_id,self.symbols,self.timeframe,tuple(shards),source=self.source)


def save_portable_manifest_r8(m:PortableDatasetManifestR8,path:str|Path)->Path:
    m.validate();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);payload=asdict(m);payload['symbols']=list(m.symbols);payload['shards']=[asdict(x) for x in m.shards];payload['content_hash']=m.content_hash;path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n');return path


def load_portable_manifest_r8(path:str|Path)->PortableDatasetManifestR8:
    x=json.loads(Path(path).read_text());m=PortableDatasetManifestR8(dataset_id=x['dataset_id'],symbols=tuple(x['symbols']),timeframe=x['timeframe'],shards=tuple(PortableShardSpecR8(**s) for s in x['shards']),source=x.get('source','HISTORICAL_REPLAY'),schema=x.get('schema','CB16_PORTABLE_DATASET_MANIFEST_R8'))
    m.validate()
    if x.get('content_hash') and x['content_hash']!=m.content_hash:raise RuntimeError('PORTABLE_MANIFEST_CONTENT_HASH_MISMATCH')
    return m


def portable_from_dataset_manifest_r8(manifest:DatasetManifest,*,common_root:str|Path)->PortableDatasetManifestR8:
    root=Path(common_root).resolve();shards=[]
    for s in manifest.shards:
        p=Path(s.path).resolve()
        try:rel=p.relative_to(root)
        except ValueError as exc:raise RuntimeError('SHARD_NOT_UNDER_COMMON_ROOT:'+str(p)) from exc
        shards.append(PortableShardSpecR8(s.shard_id,rel.as_posix(),s.symbol,s.timeframe,s.format,s.start_timestamp,s.end_timestamp,s.rows,s.sha256,s.bytes))
    return PortableDatasetManifestR8(manifest.dataset_id,manifest.symbols,manifest.timeframe,tuple(shards),source=manifest.source)
