from __future__ import annotations

"""Immutable Japan-OCI -> Shanxi bundle receiver.

The watcher never trains from a partially copied file. A bundle becomes visible to the
factory only after size/mtime settlement, SHA256 sidecar verification and safe extraction.
"""

import dataclasses, hashlib, json, os, re, shutil, tarfile, tempfile, time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


def sha256_file(path:str|Path,chunk:int=8<<20)->str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(chunk),b''):h.update(b)
    return h.hexdigest()


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class IncomingBundlePolicyR8:
    settle_seconds: float=30.0
    max_bundle_gib: float=200.0
    required_manifest_name: str='PORTABLE_DATASET_MANIFEST_R8.json'


@dataclass(frozen=True)
class AcceptedBundleReceiptR8:
    bundle_name:str
    bundle_sha256:str
    bundle_bytes:int
    accepted_root:str
    manifest_path:str
    manifest_sha256:str
    member_count:int
    status:str
    receipt_hash:str


def _sidecar_expected(path:Path)->str:
    side=Path(str(path)+'.sha256')
    if not side.is_file():raise RuntimeError('BUNDLE_SHA256_SIDECAR_MISSING')
    token=side.read_text().strip().split()[0].lower()
    if not re.fullmatch(r'[0-9a-f]{64}',token):raise RuntimeError('BUNDLE_SHA256_SIDECAR_INVALID')
    return token


def _safe_members(tf:tarfile.TarFile):
    members=tf.getmembers()
    for m in members:
        p=PurePosixPath(m.name)
        if p.is_absolute() or '..' in p.parts:raise RuntimeError('TAR_PATH_TRAVERSAL')
        if m.issym() or m.islnk():raise RuntimeError('TAR_LINK_MEMBER_FORBIDDEN')
        if m.isdev():raise RuntimeError('TAR_DEVICE_MEMBER_FORBIDDEN')
    return members


class IncomingBundleWatcherR8:
    def __init__(self,*,inbox:str|Path,accepted:str|Path,rejected:str|Path,policy:IncomingBundlePolicyR8|None=None):
        self.inbox=Path(inbox);self.accepted=Path(accepted);self.rejected=Path(rejected);self.policy=policy or IncomingBundlePolicyR8()
        for p in (self.inbox,self.accepted,self.rejected):p.mkdir(parents=True,exist_ok=True)

    def candidates(self)->Iterator[Path]:
        now=time.time()
        for p in sorted(self.inbox.glob('*.tar.gz')):
            try:
                if p.stat().st_size>self.policy.max_bundle_gib*1024**3:continue
                if now-p.stat().st_mtime<self.policy.settle_seconds:continue
                side=Path(str(p)+'.sha256')
                if not side.exists() or now-side.stat().st_mtime<self.policy.settle_seconds:continue
                # A successfully accepted transport object is consumed logically even if
                # the original inbox file is retained for audit/transport evidence.
                try:
                    expected=_sidecar_expected(p)
                    if (self.accepted/expected/'ACCEPTED_BUNDLE_RECEIPT_R8.json').exists():
                        continue
                except Exception:
                    pass
                yield p
            except FileNotFoundError:continue

    def accept(self,bundle:Path)->AcceptedBundleReceiptR8:
        expected=_sidecar_expected(bundle);actual=sha256_file(bundle)
        if actual!=expected:raise RuntimeError('BUNDLE_SHA256_MISMATCH')
        final=self.accepted/actual
        receipt_path=final/'ACCEPTED_BUNDLE_RECEIPT_R8.json'
        if receipt_path.exists():
            obj=json.loads(receipt_path.read_text());return AcceptedBundleReceiptR8(**obj)
        staging=Path(tempfile.mkdtemp(prefix='.accept_',dir=self.accepted))
        try:
            with tarfile.open(bundle,'r:gz') as tf:
                members=_safe_members(tf)
                try:
                    tf.extractall(staging,members=members,filter='data')
                except TypeError:  # Python < 3.12
                    tf.extractall(staging,members=members)
            manifests=list(staging.rglob(self.policy.required_manifest_name))
            if len(manifests)!=1:raise RuntimeError(f'REQUIRED_MANIFEST_COUNT:{len(manifests)}')
            manifest=manifests[0];msha=sha256_file(manifest)
            if final.exists():raise RuntimeError('ACCEPTED_BUNDLE_DESTINATION_CONFLICT')
            os.replace(staging,final)
            rel_manifest=str(manifest.relative_to(staging))
            material={
                'bundle_name':bundle.name,'bundle_sha256':actual,'bundle_bytes':bundle.stat().st_size,
                'accepted_root':str(final),'manifest_path':str(final/rel_manifest),'manifest_sha256':msha,
                'member_count':len(members),'status':'ACCEPTED',
            }
            receipt=AcceptedBundleReceiptR8(**material,receipt_hash=canonical_hash(material))
            receipt_path.write_text(json.dumps(asdict(receipt),indent=2)+'\n')
            # Preserve original transport object as cold evidence; never silently delete source.
            return receipt
        finally:
            if staging.exists():shutil.rmtree(staging,ignore_errors=True)

    def scan_once(self)->list[AcceptedBundleReceiptR8]:
        out=[]
        for p in self.candidates():
            try:out.append(self.accept(p))
            except Exception as exc:
                # Do not move a still-copying file; only settled candidates reach here.
                fail=self.rejected/(p.name+'.reject.json')
                fail.write_text(json.dumps({'bundle':str(p),'error':repr(exc),'time':time.time()},indent=2)+'\n')
        return out
