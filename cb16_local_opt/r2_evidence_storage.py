from __future__ import annotations

"""CB16 R2 immutable evidence storage.

Payload authority:
- generation-independent evidence content;
- append-only framed pack files, intended for sequential HDD writes.

Metadata authority:
- immutable evidence-set manifests and tiny generation snapshots, intended for SSD;
- SQLite is a rebuildable locator/catalog index, not the only copy of payload bytes.

This changes storage complexity from O(generations * evidence) re-materialization to
O(unique evidence content + generation snapshots).
"""

import hashlib
import json
import os
import sqlite3
import struct
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

PACK_MAGIC = b"CB16R2P1"
PACK_HEADER = struct.Struct(">8s32sIII")  # magic, sha256, raw_len, stored_len, crc32
FAIL_AFTER_PACK_FSYNC_BEFORE_INDEX = "AFTER_PACK_FSYNC_BEFORE_INDEX"
FailHook = Callable[[str], None]


def canonical_json_bytes(obj: Any) -> bytes:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def stable_lane(key: str, lanes: int) -> int:
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % lanes


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical_json_bytes(obj))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    finally:
        tmp.unlink(missing_ok=True)


def _compress(raw: bytes, codec: str) -> tuple[str, bytes]:
    codec = codec.lower()
    if codec == "zstd":
        try:
            import zstandard as zstd  # type: ignore
        except Exception:
            codec = "zlib"
        else:
            return "zstd", zstd.ZstdCompressor(level=3).compress(raw)
    if codec == "zlib":
        return "zlib", zlib.compress(raw, 3)
    if codec == "none":
        return "none", raw
    raise ValueError(f"unsupported codec:{codec}")


def _decompress(data: bytes, codec: str) -> bytes:
    if codec == "zstd":
        import zstandard as zstd  # type: ignore
        return zstd.ZstdDecompressor().decompress(data)
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "none":
        return data
    raise ValueError(f"unsupported codec:{codec}")


@dataclass(frozen=True)
class R2EvidenceItem:
    evidence_id: str
    parent_snapshot_hash: str
    lineage_hash: str
    teacher_protocol_hash: str
    payload: Mapping[str, Any]

    @property
    def payload_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload)

    @property
    def content_hash(self) -> str:
        return sha256_bytes(self.payload_bytes)

    @property
    def identity_hash(self) -> str:
        return sha256_obj({
            "evidence_id": self.evidence_id,
            "content_hash": self.content_hash,
            "parent_snapshot_hash": self.parent_snapshot_hash,
            "lineage_hash": self.lineage_hash,
            "teacher_protocol_hash": self.teacher_protocol_hash,
        })


@dataclass(frozen=True)
class R2PayloadRef:
    content_hash: str
    lane: int
    segment_path: str
    offset: int
    raw_bytes: int
    stored_bytes: int
    codec: str
    crc32: int


@dataclass(frozen=True)
class R2EvidenceSetRef:
    evidence_set_id: str
    content_hash: str
    object_count: int
    manifest_path: str


@dataclass(frozen=True)
class R2GenerationSnapshot:
    snapshot_id: str
    generation: int
    parent_policy_hash: str
    evidence_set_hash: str
    object_count: int
    manifest_path: str

    @property
    def content_hash(self) -> str:
        return sha256_obj({
            "schema": "CB16_R2_GENERATION_SNAPSHOT_V1",
            "snapshot_id": self.snapshot_id,
            "generation": self.generation,
            "parent_policy_hash": self.parent_policy_hash,
            "evidence_set_hash": self.evidence_set_hash,
            "object_count": self.object_count,
        })


@dataclass(frozen=True)
class R2MaterializeReceipt:
    schema: str
    evidence_set_hash: str
    object_count: int
    unique_payload_count: int
    created_payload_count: int
    reused_payload_count: int
    raw_bytes: int
    stored_bytes: int
    pack_write_seconds: float
    index_commit_seconds: float
    total_seconds: float
    storage_amplification_model: str


class R2EvidenceStore:
    """Topology-aware store.

    metadata_root: SSD/NVMe preferred (SQLite WAL, manifests, generation snapshots).
    payload_roots: one path per physical payload device. For one HDD, pass ONE root:
    arbitrary multi-sharding on one spindle would destroy sequential-write locality.
    """

    def __init__(
        self,
        *,
        metadata_root: str | Path,
        payload_roots: Sequence[str | Path],
        segment_target_bytes: int = 256 * 1024 * 1024,
        codec: str = "zstd",
        sqlite_synchronous: str = "FULL",
        recover_on_open: bool = True,
    ):
        if not payload_roots:
            raise ValueError("payload_roots required")
        sync = sqlite_synchronous.upper()
        if sync not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError("invalid sqlite_synchronous")
        self.metadata_root = Path(metadata_root)
        self.payload_roots = tuple(Path(x) for x in payload_roots)
        self.segment_target_bytes = int(segment_target_bytes)
        self.codec = codec
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        for root in self.payload_roots:
            root.mkdir(parents=True, exist_ok=True)
        self.manifest_root = self.metadata_root / "evidence_sets"
        self.snapshot_root = self.metadata_root / "generation_snapshots"
        self.manifest_root.mkdir(exist_ok=True)
        self.snapshot_root.mkdir(exist_ok=True)

        self.conn = sqlite3.connect(
            self.metadata_root / "r2_index.sqlite",
            isolation_level=None,
            timeout=30.0,
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA synchronous={sync}")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS payloads(
          content_hash TEXT PRIMARY KEY,lane INTEGER NOT NULL,segment_path TEXT NOT NULL,
          offset INTEGER NOT NULL,raw_bytes INTEGER NOT NULL,stored_bytes INTEGER NOT NULL,
          codec TEXT NOT NULL,crc32 INTEGER NOT NULL,created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS evidence_catalog(
          evidence_id TEXT PRIMARY KEY,content_hash TEXT NOT NULL,identity_hash TEXT NOT NULL,
          parent_snapshot_hash TEXT NOT NULL,lineage_hash TEXT NOT NULL,
          teacher_protocol_hash TEXT NOT NULL,created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS evidence_sets(
          evidence_set_hash TEXT PRIMARY KEY,evidence_set_id TEXT NOT NULL,
          object_count INTEGER NOT NULL,manifest_path TEXT NOT NULL,created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS generation_snapshots(
          snapshot_id TEXT PRIMARY KEY,content_hash TEXT NOT NULL,generation INTEGER NOT NULL,
          parent_policy_hash TEXT NOT NULL,evidence_set_hash TEXT NOT NULL,
          object_count INTEGER NOT NULL,manifest_path TEXT NOT NULL,created_at REAL NOT NULL);
        """)
        if recover_on_open:
            self.recover_payload_index()

    @property
    def lane_count(self) -> int:
        return len(self.payload_roots)

    def close(self) -> None:
        self.conn.close()

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        mode = mode.upper()
        if mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError("invalid checkpoint mode")
        return tuple(int(x) for x in self.conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone())

    def _lane_dir(self, lane: int) -> Path:
        path = self.payload_roots[lane] / f"cb16_r2_lane_{lane:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _segments(self, lane: int) -> list[Path]:
        return sorted(self._lane_dir(lane).glob("segment_*.pack"))

    def _active_segment(self, lane: int, incoming: int) -> Path:
        segs = self._segments(lane)
        if not segs:
            return self._lane_dir(lane) / "segment_00000000.pack"
        last = segs[-1]
        if last.stat().st_size + incoming <= self.segment_target_bytes:
            return last
        n = int(last.stem.rsplit("_", 1)[1]) + 1
        return self._lane_dir(lane) / f"segment_{n:08d}.pack"

    def _encode(self, raw: bytes) -> tuple[str, bytes, int]:
        h = sha256_bytes(raw)
        codec, stored = _compress(raw, self.codec)
        crc = zlib.crc32(stored) & 0xFFFFFFFF
        tag = codec.encode("ascii")
        if len(tag) > 15:
            raise ValueError("codec tag too long")
        record = (
            bytes([len(tag)]) + tag
            + PACK_HEADER.pack(PACK_MAGIC, bytes.fromhex(h), len(raw), len(stored), crc)
            + stored
        )
        return h, record, crc

    @staticmethod
    def _decode(path: Path, offset: int) -> tuple[R2PayloadRef, bytes, int] | None:
        with path.open("rb") as f:
            f.seek(offset)
            one = f.read(1)
            if not one:
                return None
            tag_len = one[0]
            tag = f.read(tag_len)
            if len(tag) != tag_len:
                return None
            header = f.read(PACK_HEADER.size)
            if len(header) != PACK_HEADER.size:
                return None
            magic, hash_b, raw_len, stored_len, crc = PACK_HEADER.unpack(header)
            if magic != PACK_MAGIC:
                raise RuntimeError(f"R2_PACK_MAGIC_MISMATCH:{path}:{offset}")
            stored = f.read(stored_len)
            if len(stored) != stored_len:
                return None
            if zlib.crc32(stored) & 0xFFFFFFFF != crc:
                raise RuntimeError(f"R2_PACK_CRC_MISMATCH:{path}:{offset}")
            codec = tag.decode("ascii")
            raw = _decompress(stored, codec)
            h = hash_b.hex()
            if len(raw) != raw_len or sha256_bytes(raw) != h:
                raise RuntimeError(f"R2_PACK_CONTENT_MISMATCH:{path}:{offset}")
            end = f.tell()
        return R2PayloadRef(h, -1, str(path), offset, raw_len, stored_len, codec, crc), raw, end

    def recover_payload_index(self) -> dict[str, Any]:
        """Recover valid fsync'd orphan records. Only incomplete trailing record is truncated."""
        discovered, truncated = 0, []
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for lane in range(self.lane_count):
                for path in self._segments(lane):
                    offset, size = 0, path.stat().st_size
                    while offset < size:
                        decoded = self._decode(path, offset)
                        if decoded is None:
                            with path.open("r+b") as f:
                                f.truncate(offset); f.flush(); os.fsync(f.fileno())
                            truncated.append({"path": str(path), "from": size, "to": offset})
                            break
                        ref, _raw, end = decoded
                        old = self.conn.execute(
                            "SELECT 1 FROM payloads WHERE content_hash=?", (ref.content_hash,)
                        ).fetchone()
                        if old is None:
                            self.conn.execute(
                                "INSERT INTO payloads VALUES(?,?,?,?,?,?,?,?,?)",
                                (ref.content_hash,lane,str(path),offset,ref.raw_bytes,
                                 ref.stored_bytes,ref.codec,ref.crc32,time.time()),
                            )
                            discovered += 1
                        offset = end
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK"); raise
        return {"schema":"CB16_R2_PAYLOAD_RECOVERY_V1",
                "discovered_payloads":discovered,"truncated_tails":truncated}

    def _payload_ref(self, h: str) -> R2PayloadRef | None:
        row = self.conn.execute(
            "SELECT content_hash,lane,segment_path,offset,raw_bytes,stored_bytes,codec,crc32 "
            "FROM payloads WHERE content_hash=?", (h,)
        ).fetchone()
        return None if row is None else R2PayloadRef(
            row[0],int(row[1]),row[2],int(row[3]),int(row[4]),int(row[5]),row[6],int(row[7])
        )

    def get_payload(self, h: str) -> dict[str, Any] | None:
        ref = self._payload_ref(h)
        if ref is None:
            return None
        decoded = self._decode(Path(ref.segment_path), ref.offset)
        if decoded is None or decoded[0].content_hash != h:
            raise RuntimeError(f"R2_PAYLOAD_LOCATOR_MISMATCH:{h}")
        return json.loads(decoded[1])

    def _append_missing(
        self, items: Sequence[R2EvidenceItem], fail_hook: FailHook | None
    ) -> tuple[dict[str,R2PayloadRef],int,float,float]:
        unique = {x.content_hash:x.payload_bytes for x in items}
        refs, missing = {}, []
        for h, raw in sorted(unique.items()):
            ref = self._payload_ref(h)
            if ref is None: missing.append((h,raw))
            else: refs[h] = ref

        by_lane: dict[int,list[tuple[str,bytes]]] = {}
        for h, raw in missing:
            by_lane.setdefault(stable_lane(h,self.lane_count),[]).append((h,raw))

        pack_start, appended = time.perf_counter(), []
        for lane, rows in sorted(by_lane.items()):
            encoded, total = [], 0
            for expected, raw in rows:
                h, record, crc = self._encode(raw)
                if h != expected: raise RuntimeError("R2_HASH_DRIFT")
                codec_len = record[0]
                codec = record[1:1+codec_len].decode("ascii")
                stored_len = len(record) - (1+codec_len+PACK_HEADER.size)
                encoded.append((h,record,len(raw),stored_len,codec,crc))
                total += len(record)
            path = self._active_segment(lane,total)
            base = path.stat().st_size if path.exists() else 0
            blob = b"".join(x[1] for x in encoded)
            with path.open("ab",buffering=0) as f:
                f.write(blob); f.flush(); os.fsync(f.fileno())
            offset = base
            for h,record,raw_len,stored_len,codec,crc in encoded:
                appended.append(R2PayloadRef(h,lane,str(path),offset,raw_len,stored_len,codec,crc))
                offset += len(record)
        pack_seconds = time.perf_counter() - pack_start

        if appended and fail_hook is not None:
            fail_hook(FAIL_AFTER_PACK_FSYNC_BEFORE_INDEX)

        index_start = time.perf_counter()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for ref in appended:
                self.conn.execute(
                    "INSERT OR IGNORE INTO payloads VALUES(?,?,?,?,?,?,?,?,?)",
                    (ref.content_hash,ref.lane,ref.segment_path,ref.offset,ref.raw_bytes,
                     ref.stored_bytes,ref.codec,ref.crc32,time.time()),
                )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK"); raise
        index_seconds = time.perf_counter() - index_start
        for ref in appended: refs.setdefault(ref.content_hash,ref)
        return refs,len(appended),pack_seconds,index_seconds

    def materialize_evidence_set(
        self, *, evidence_set_id: str, items: Iterable[R2EvidenceItem],
        fail_hook: FailHook | None = None,
    ) -> tuple[R2EvidenceSetRef,R2MaterializeReceipt]:
        rows = sorted(list(items),key=lambda x:x.evidence_id)
        if not rows: raise RuntimeError("R2_EVIDENCE_SET_EMPTY")
        if len({x.evidence_id for x in rows}) != len(rows):
            raise RuntimeError("R2_EVIDENCE_SET_DUPLICATE_ID")

        entries = [{
            "evidence_id":x.evidence_id,"content_hash":x.content_hash,
            "identity_hash":x.identity_hash,"parent_snapshot_hash":x.parent_snapshot_hash,
            "lineage_hash":x.lineage_hash,"teacher_protocol_hash":x.teacher_protocol_hash,
        } for x in rows]
        manifest = {"schema":"CB16_R2_IMMUTABLE_EVIDENCE_SET_V1",
                    "evidence_set_id":evidence_set_id,"object_count":len(rows),"entries":entries}
        set_hash = sha256_obj(manifest)
        path = self.manifest_root / f"{set_hash}.json"

        old = self.conn.execute(
            "SELECT evidence_set_id,object_count,manifest_path FROM evidence_sets "
            "WHERE evidence_set_hash=?", (set_hash,)
        ).fetchone()
        unique_count = len({x.content_hash for x in rows})
        raw_sum = sum(len(x.payload_bytes) for x in rows)
        if old is not None and path.is_file():
            return (
                R2EvidenceSetRef(old[0],set_hash,int(old[1]),old[2]),
                R2MaterializeReceipt("CB16_R2_MATERIALIZE_RECEIPT_V1",set_hash,len(rows),
                    unique_count,0,unique_count,raw_sum,0,0.0,0.0,0.0,
                    "O(N_CONTENT_ONCE + G_TINY_SNAPSHOTS)"),
            )

        for x in rows:
            old_e = self.conn.execute(
                "SELECT identity_hash FROM evidence_catalog WHERE evidence_id=?", (x.evidence_id,)
            ).fetchone()
            if old_e is not None and old_e[0] != x.identity_hash:
                raise RuntimeError(f"R2_EVIDENCE_ID_CONTENT_CONFLICT:{x.evidence_id}")

        total_start = time.perf_counter()
        refs,created,pack_s,payload_index_s = self._append_missing(rows,fail_hook)
        if not path.exists(): _atomic_json(path,manifest)
        elif sha256_obj(json.loads(path.read_text())) != set_hash:
            raise RuntimeError("R2_EVIDENCE_SET_MANIFEST_CONFLICT")

        catalog_start = time.perf_counter()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for x in rows:
                old_e = self.conn.execute(
                    "SELECT identity_hash FROM evidence_catalog WHERE evidence_id=?", (x.evidence_id,)
                ).fetchone()
                if old_e is None:
                    self.conn.execute(
                        "INSERT INTO evidence_catalog VALUES(?,?,?,?,?,?,?)",
                        (x.evidence_id,x.content_hash,x.identity_hash,x.parent_snapshot_hash,
                         x.lineage_hash,x.teacher_protocol_hash,time.time()),
                    )
                elif old_e[0] != x.identity_hash:
                    raise RuntimeError(f"R2_EVIDENCE_ID_CONTENT_CONFLICT:{x.evidence_id}")
            self.conn.execute(
                "INSERT OR IGNORE INTO evidence_sets VALUES(?,?,?,?,?)",
                (set_hash,evidence_set_id,len(rows),str(path),time.time()),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK"); raise
        catalog_s = time.perf_counter() - catalog_start
        receipt = R2MaterializeReceipt(
            "CB16_R2_MATERIALIZE_RECEIPT_V1",set_hash,len(rows),unique_count,created,
            unique_count-created,raw_sum,sum(r.stored_bytes for r in refs.values()),
            pack_s,payload_index_s+catalog_s,time.perf_counter()-total_start,
            "O(N_CONTENT_ONCE + G_TINY_SNAPSHOTS)",
        )
        return R2EvidenceSetRef(evidence_set_id,set_hash,len(rows),str(path)),receipt

    def seal_generation_snapshot(
        self, *, snapshot_id: str, generation: int, parent_policy_hash: str,
        evidence_set: R2EvidenceSetRef,
    ) -> R2GenerationSnapshot:
        payload = {"schema":"CB16_R2_GENERATION_SNAPSHOT_V1","snapshot_id":snapshot_id,
                   "generation":int(generation),"parent_policy_hash":parent_policy_hash,
                   "evidence_set_hash":evidence_set.content_hash,
                   "object_count":evidence_set.object_count}
        h = sha256_obj(payload)
        path = self.snapshot_root / f"{snapshot_id}.json"
        old = self.conn.execute(
            "SELECT content_hash,manifest_path FROM generation_snapshots WHERE snapshot_id=?",
            (snapshot_id,)
        ).fetchone()
        if old is not None:
            if old[0] != h: raise RuntimeError(f"R2_SNAPSHOT_ID_CONTENT_CONFLICT:{snapshot_id}")
            return R2GenerationSnapshot(snapshot_id,generation,parent_policy_hash,
                                        evidence_set.content_hash,evidence_set.object_count,old[1])
        _atomic_json(path,payload)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "INSERT INTO generation_snapshots VALUES(?,?,?,?,?,?,?,?)",
                (snapshot_id,h,generation,parent_policy_hash,evidence_set.content_hash,
                 evidence_set.object_count,str(path),time.time()),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK"); raise
        return R2GenerationSnapshot(snapshot_id,generation,parent_policy_hash,
                                    evidence_set.content_hash,evidence_set.object_count,str(path))

    def stats(self) -> dict[str,Any]:
        n,raw,stored = self.conn.execute(
            "SELECT COUNT(*),COALESCE(SUM(raw_bytes),0),COALESCE(SUM(stored_bytes),0) FROM payloads"
        ).fetchone()
        count = lambda table:int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return {"schema":"CB16_R2_EVIDENCE_STORE_STATS_V1","payload_objects":int(n),
                "raw_bytes":int(raw),"stored_bytes":int(stored),
                "evidence_catalog_objects":count("evidence_catalog"),
                "evidence_sets":count("evidence_sets"),
                "generation_snapshots":count("generation_snapshots")}

    def audit(self, *, verify_payloads: bool = True) -> dict[str,Any]:
        errors = []
        payloads = self.conn.execute(
            "SELECT content_hash,lane,segment_path,offset FROM payloads ORDER BY content_hash"
        ).fetchall()
        if verify_payloads:
            for h,lane,path,offset in payloads:
                try:
                    got = self._decode(Path(path),int(offset))
                    if got is None or got[0].content_hash != h: raise RuntimeError("locator mismatch")
                except Exception as exc:
                    errors.append({"content_hash":h,"lane":lane,"error":repr(exc)})
        sets = self.conn.execute(
            "SELECT evidence_set_hash,manifest_path,object_count FROM evidence_sets"
        ).fetchall()
        for h,path,n in sets:
            try:
                obj=json.loads(Path(path).read_text())
                if sha256_obj(obj)!=h or int(obj["object_count"])!=int(n):
                    raise RuntimeError("evidence-set manifest mismatch")
            except Exception as exc: errors.append({"evidence_set_hash":h,"error":repr(exc)})
        snaps = self.conn.execute(
            "SELECT snapshot_id,content_hash,manifest_path,evidence_set_hash FROM generation_snapshots"
        ).fetchall()
        for sid,h,path,set_h in snaps:
            try:
                obj=json.loads(Path(path).read_text())
                if sha256_obj(obj)!=h or obj["evidence_set_hash"]!=set_h:
                    raise RuntimeError("generation snapshot mismatch")
            except Exception as exc: errors.append({"snapshot_id":sid,"error":repr(exc)})
        return {"schema":"CB16_R2_EVIDENCE_STORE_AUDIT_V1","payload_lanes":self.lane_count,
                "payload_objects":len(payloads),"evidence_sets":len(sets),
                "generation_snapshots":len(snaps),"errors":errors,"pass":not errors}
