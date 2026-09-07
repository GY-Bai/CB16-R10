from __future__ import annotations

"""R11 immutable evidence payload store.

Scientific evidence meaning is deliberately outside this module.  The store only owns
content-addressed persistence, segment lifecycle and recovery.  Sealed segments are
immutable; only the active tail is scanned during normal startup.
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

PACK_MAGIC = b"CB16R11P"
PACK_HEADER = struct.Struct(">8s32sIII")
FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX = "AFTER_PAYLOAD_FSYNC_BEFORE_INDEX"
FAIL_AFTER_SEAL_MANIFEST_BEFORE_COMMIT = "AFTER_SEAL_MANIFEST_BEFORE_COMMIT"
FailHook = Callable[[str], None]


def canonical_json_bytes(obj: Any) -> bytes:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def stable_lane(key: str, lanes: int) -> int:
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % lanes


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(obj) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
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


def _decompress(stored: bytes, codec: str) -> bytes:
    if codec == "zstd":
        import zstandard as zstd  # type: ignore
        return zstd.ZstdDecompressor().decompress(stored)
    if codec == "zlib":
        return zlib.decompress(stored)
    if codec == "none":
        return stored
    raise ValueError(f"unsupported codec:{codec}")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _overlap(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


@dataclass(frozen=True)
class EvidenceItemR11:
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
class PayloadRefR11:
    content_hash: str
    lane: int
    segment_id: str
    segment_path: str
    offset: int
    end_offset: int
    raw_bytes: int
    stored_bytes: int
    codec: str
    crc32: int


@dataclass(frozen=True)
class EvidenceMaterializeReceiptR11:
    input_count: int
    unique_payload_count: int
    created_payload_count: int
    reused_payload_count: int
    created_evidence_count: int
    reused_evidence_count: int


class EvidenceStoreR11:
    """Immutable content-addressed payloads with sealed segments + active tail.

    ``metadata_root`` should live on SSD/NVMe.  ``payload_roots`` may live on HDDs.
    ``read_only_source_roots`` can point at market/cache authority roots; any overlap
    with a writable store root is rejected before a file is created.
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
        read_only_source_roots: Sequence[str | Path] = (),
    ):
        if not payload_roots:
            raise ValueError("R11_PAYLOAD_ROOT_REQUIRED")
        self.metadata_root = _resolved(metadata_root)
        self.payload_roots = tuple(_resolved(x) for x in payload_roots)
        self.read_only_source_roots = tuple(_resolved(x) for x in read_only_source_roots)
        writable = (self.metadata_root,) + self.payload_roots
        for source in self.read_only_source_roots:
            for target in writable:
                if _overlap(source, target):
                    raise RuntimeError(f"R11_READ_ONLY_SOURCE_ROOT_OVERLAP:{source}:{target}")
        self.segment_target_bytes = int(segment_target_bytes)
        if self.segment_target_bytes <= PACK_HEADER.size + 32:
            raise ValueError("segment_target_bytes too small")
        self.codec = str(codec)
        sync = sqlite_synchronous.upper()
        if sync not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError("invalid sqlite_synchronous")

        self.metadata_root.mkdir(parents=True, exist_ok=True)
        for root in self.payload_roots:
            root.mkdir(parents=True, exist_ok=True)
        self.segment_manifest_root = self.metadata_root / "sealed_segments"
        self.evidence_set_root = self.metadata_root / "evidence_sets"
        self.segment_manifest_root.mkdir(exist_ok=True)
        self.evidence_set_root.mkdir(exist_ok=True)
        self.startup_stats: dict[str, int] = {
            "sealed_segments": 0,
            "sealed_payload_bytes_read": 0,
            "active_tail_bytes_scanned": 0,
            "recovered_payloads": 0,
            "truncated_tail_bytes": 0,
        }

        self.conn = sqlite3.connect(
            self.metadata_root / "r11_evidence.sqlite",
            isolation_level=None,
            timeout=30.0,
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA synchronous={sync}")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS segments(
          segment_id TEXT PRIMARY KEY,lane INTEGER NOT NULL,path TEXT UNIQUE NOT NULL,
          state TEXT NOT NULL,committed_end INTEGER NOT NULL,payload_count INTEGER NOT NULL,
          segment_sha256 TEXT,manifest_path TEXT,created_at REAL NOT NULL,sealed_at REAL);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_r11_one_active_lane
          ON segments(lane) WHERE state='ACTIVE';
        CREATE TABLE IF NOT EXISTS payloads(
          content_hash TEXT PRIMARY KEY,lane INTEGER NOT NULL,segment_id TEXT NOT NULL,
          segment_path TEXT NOT NULL,offset INTEGER NOT NULL,end_offset INTEGER NOT NULL,
          raw_bytes INTEGER NOT NULL,stored_bytes INTEGER NOT NULL,codec TEXT NOT NULL,
          crc32 INTEGER NOT NULL,created_at REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_r11_payload_segment_offset
          ON payloads(segment_id,offset);
        CREATE TABLE IF NOT EXISTS evidence_catalog(
          evidence_id TEXT PRIMARY KEY,content_hash TEXT NOT NULL,identity_hash TEXT NOT NULL,
          parent_snapshot_hash TEXT NOT NULL,lineage_hash TEXT NOT NULL,
          teacher_protocol_hash TEXT NOT NULL,created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS evidence_sets(
          evidence_set_hash TEXT PRIMARY KEY,evidence_set_id TEXT NOT NULL,
          object_count INTEGER NOT NULL,manifest_path TEXT NOT NULL,created_at REAL NOT NULL);
        """)
        try:
            self._ensure_active_segments()
            self.fast_startup_validation()
            if recover_on_open:
                self.recover_active_tails()
        except BaseException:
            self.conn.close()
            raise

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
        path = self.payload_roots[lane] / f"cb16_r11_lane_{lane:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _segment_path(self, lane: int, ordinal: int) -> Path:
        return self._lane_dir(lane) / f"segment_{ordinal:08d}.pack"

    def _next_segment_ordinal(self, lane: int) -> int:
        rows = self.conn.execute("SELECT segment_id FROM segments WHERE lane=?", (lane,)).fetchall()
        ordinals = [int(str(x[0]).rsplit("-", 1)[1]) for x in rows]
        return 0 if not ordinals else max(ordinals) + 1

    def _ensure_active_segments(self) -> None:
        now = time.time()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for lane in range(self.lane_count):
                active = self.conn.execute(
                    "SELECT 1 FROM segments WHERE lane=? AND state='ACTIVE'", (lane,)
                ).fetchone()
                if active is None:
                    ordinal = self._next_segment_ordinal(lane)
                    segment_id = f"L{lane:02d}-{ordinal:08d}"
                    path = self._segment_path(lane, ordinal)
                    self.conn.execute(
                        "INSERT INTO segments VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (segment_id, lane, str(path), "ACTIVE", 0, 0, None, None, now, None),
                    )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def _active_row(self, lane: int) -> tuple[str, Path, int, int]:
        row = self.conn.execute(
            "SELECT segment_id,path,committed_end,payload_count FROM segments "
            "WHERE lane=? AND state='ACTIVE'", (lane,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"R11_ACTIVE_SEGMENT_MISSING:{lane}")
        return str(row[0]), Path(row[1]), int(row[2]), int(row[3])

    def _encode(self, raw: bytes) -> tuple[str, bytes, int, str, int]:
        h = sha256_bytes(raw)
        codec, stored = _compress(raw, self.codec)
        crc = zlib.crc32(stored) & 0xFFFFFFFF
        tag = codec.encode("ascii")
        if not 0 < len(tag) <= 15:
            raise ValueError("invalid codec tag")
        record = bytes([len(tag)]) + tag + PACK_HEADER.pack(
            PACK_MAGIC, bytes.fromhex(h), len(raw), len(stored), crc
        ) + stored
        return h, record, crc, codec, len(stored)

    @staticmethod
    def _decode(path: Path, offset: int, *, lane: int = -1, segment_id: str = "") -> tuple[PayloadRefR11, bytes, int] | None:
        with path.open("rb") as handle:
            handle.seek(offset)
            one = handle.read(1)
            if not one:
                return None
            tag_len = one[0]
            tag = handle.read(tag_len)
            if len(tag) != tag_len:
                return None
            header = handle.read(PACK_HEADER.size)
            if len(header) != PACK_HEADER.size:
                return None
            magic, hash_b, raw_len, stored_len, crc = PACK_HEADER.unpack(header)
            if magic != PACK_MAGIC:
                raise RuntimeError(f"R11_PACK_MAGIC_MISMATCH:{path}:{offset}")
            stored = handle.read(stored_len)
            if len(stored) != stored_len:
                return None
            if zlib.crc32(stored) & 0xFFFFFFFF != crc:
                raise RuntimeError(f"R11_PACK_CRC_MISMATCH:{path}:{offset}")
            codec = tag.decode("ascii")
            raw = _decompress(stored, codec)
            h = hash_b.hex()
            if len(raw) != raw_len or sha256_bytes(raw) != h:
                raise RuntimeError(f"R11_PACK_CONTENT_MISMATCH:{path}:{offset}")
            end = handle.tell()
        return PayloadRefR11(
            h, lane, segment_id, str(path), int(offset), int(end), int(raw_len),
            int(stored_len), codec, int(crc)
        ), raw, int(end)

    def _payload_ref(self, content_hash: str) -> PayloadRefR11 | None:
        row = self.conn.execute(
            "SELECT content_hash,lane,segment_id,segment_path,offset,end_offset,raw_bytes,"
            "stored_bytes,codec,crc32 FROM payloads WHERE content_hash=?", (content_hash,)
        ).fetchone()
        if row is None:
            return None
        return PayloadRefR11(
            str(row[0]), int(row[1]), str(row[2]), str(row[3]), int(row[4]), int(row[5]),
            int(row[6]), int(row[7]), str(row[8]), int(row[9])
        )

    def fast_startup_validation(self) -> dict[str, Any]:
        """Validate metadata/manifests and file sizes without reading sealed payload bytes."""
        sealed = self.conn.execute(
            "SELECT segment_id,path,committed_end,payload_count,segment_sha256,manifest_path "
            "FROM segments WHERE state='SEALED' ORDER BY lane,segment_id"
        ).fetchall()
        for segment_id, path_s, committed_end, payload_count, segment_sha, manifest_path in sealed:
            path = Path(path_s)
            if not path.is_file():
                raise RuntimeError(f"R11_SEALED_SEGMENT_MISSING:{segment_id}")
            if path.stat().st_size != int(committed_end):
                raise RuntimeError(f"R11_SEALED_SEGMENT_SIZE_MISMATCH:{segment_id}")
            if not manifest_path or not Path(manifest_path).is_file():
                raise RuntimeError(f"R11_SEALED_MANIFEST_MISSING:{segment_id}")
            manifest = json.loads(Path(manifest_path).read_text())
            if manifest.get("segment_id") != segment_id:
                raise RuntimeError(f"R11_SEALED_MANIFEST_ID_MISMATCH:{segment_id}")
            if manifest.get("segment_sha256") != segment_sha:
                raise RuntimeError(f"R11_SEALED_MANIFEST_SHA_BINDING_MISMATCH:{segment_id}")
            if int(manifest.get("file_bytes", -1)) != int(committed_end):
                raise RuntimeError(f"R11_SEALED_MANIFEST_SIZE_MISMATCH:{segment_id}")
            if int(manifest.get("payload_count", -1)) != int(payload_count):
                raise RuntimeError(f"R11_SEALED_MANIFEST_COUNT_MISMATCH:{segment_id}")
        active = self.conn.execute(
            "SELECT segment_id FROM segments WHERE state='ACTIVE' ORDER BY lane,segment_id"
        ).fetchall()
        for (segment_id,) in active:
            orphan_manifest = self.segment_manifest_root / f"{segment_id}.json"
            if orphan_manifest.exists():
                raise RuntimeError(f"R11_ORPHAN_SEAL_MANIFEST_FOR_ACTIVE_SEGMENT:{segment_id}")
        self.startup_stats["sealed_segments"] = len(sealed)
        self.startup_stats["sealed_payload_bytes_read"] = 0
        return {
            "schema": "CB16_R11_EVIDENCE_FAST_STARTUP_VALIDATION_V1",
            "sealed_segments": len(sealed),
            "sealed_payload_bytes_read": 0,
            "mode": "MANIFEST_AND_STAT_ONLY_FOR_SEALED_SEGMENTS",
            "pass": True,
        }

    def recover_active_tails(self) -> dict[str, Any]:
        recovered = 0
        scanned = 0
        truncated = 0
        receipts: list[dict[str, Any]] = []
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for lane in range(self.lane_count):
                segment_id, path, committed_end, _ = self._active_row(lane)
                size = path.stat().st_size if path.exists() else 0
                if committed_end > size:
                    raise RuntimeError(f"R11_ACTIVE_COMMITTED_END_PAST_EOF:{segment_id}")
                if committed_end:
                    row = self.conn.execute(
                        "SELECT content_hash,offset,end_offset FROM payloads WHERE segment_id=? "
                        "ORDER BY offset DESC LIMIT 1", (segment_id,)
                    ).fetchone()
                    if row is None or int(row[2]) != committed_end:
                        raise RuntimeError(f"R11_ACTIVE_INDEX_BOUNDARY_MISMATCH:{segment_id}")
                    decoded = self._decode(path, int(row[1]), lane=lane, segment_id=segment_id)
                    if decoded is None or decoded[0].content_hash != str(row[0]) or decoded[2] != committed_end:
                        raise RuntimeError(f"R11_ACTIVE_BOUNDARY_CANARY_FAILED:{segment_id}")
                offset = committed_end
                recovered_here = 0
                original_size = size
                while offset < size:
                    record_offset = offset
                    decoded = self._decode(path, offset, lane=lane, segment_id=segment_id)
                    if decoded is None:
                        with path.open("r+b") as handle:
                            handle.truncate(offset)
                            handle.flush()
                            os.fsync(handle.fileno())
                        truncated += size - offset
                        size = offset
                        break
                    ref, _raw, end = decoded
                    old = self._payload_ref(ref.content_hash)
                    if old is None:
                        self.conn.execute(
                            "INSERT INTO payloads VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (ref.content_hash, lane, segment_id, str(path), record_offset, end,
                             ref.raw_bytes, ref.stored_bytes, ref.codec, ref.crc32, time.time()),
                        )
                        recovered += 1
                        recovered_here += 1
                    elif old.segment_id != segment_id or old.offset != record_offset:
                        raise RuntimeError(f"R11_DUPLICATE_CONTENT_LOCATOR_CONFLICT:{ref.content_hash}")
                    offset = end
                count = int(self.conn.execute(
                    "SELECT COUNT(*) FROM payloads WHERE segment_id=?", (segment_id,)
                ).fetchone()[0])
                self.conn.execute(
                    "UPDATE segments SET committed_end=?,payload_count=? WHERE segment_id=?",
                    (size, count, segment_id),
                )
                scanned_here = max(0, original_size - committed_end)
                scanned += scanned_here
                receipts.append({
                    "lane": lane, "segment_id": segment_id, "start_offset": committed_end,
                    "file_bytes_before": original_size, "file_bytes_after": size,
                    "tail_bytes_scanned": scanned_here, "recovered_payloads": recovered_here,
                })
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.startup_stats["active_tail_bytes_scanned"] += scanned
        self.startup_stats["recovered_payloads"] += recovered
        self.startup_stats["truncated_tail_bytes"] += truncated
        return {
            "schema": "CB16_R11_ACTIVE_TAIL_RECOVERY_V1",
            "segments": receipts,
            "recovered_payloads": recovered,
            "tail_bytes_scanned": scanned,
            "truncated_tail_bytes": truncated,
            "sealed_payload_bytes_read": 0,
            "pass": True,
        }

    def _seal_active(self, lane: int, *, fail_hook: FailHook | None = None) -> None:
        segment_id, path, committed_end, payload_count = self._active_row(lane)
        if committed_end == 0:
            return
        if not path.is_file() or path.stat().st_size != committed_end:
            raise RuntimeError(f"R11_CANNOT_SEAL_UNSTABLE_ACTIVE_SEGMENT:{segment_id}")
        rows = self.conn.execute(
            "SELECT content_hash,offset,end_offset,raw_bytes,stored_bytes,codec,crc32 "
            "FROM payloads WHERE segment_id=? ORDER BY offset", (segment_id,)
        ).fetchall()
        if len(rows) != payload_count:
            raise RuntimeError(f"R11_SEGMENT_INDEX_COUNT_MISMATCH:{segment_id}")
        segment_sha = sha256_file(path)
        manifest = {
            "schema": "CB16_R11_SEALED_PAYLOAD_SEGMENT_V1",
            "segment_id": segment_id,
            "lane": lane,
            "segment_path": str(path),
            "file_bytes": committed_end,
            "payload_count": payload_count,
            "segment_sha256": segment_sha,
            "index_sha256": sha256_obj(rows),
            "entries": [list(row) for row in rows],
            "immutable": True,
        }
        manifest_path = self.segment_manifest_root / f"{segment_id}.json"
        _atomic_json(manifest_path, manifest)
        if fail_hook is not None:
            fail_hook(FAIL_AFTER_SEAL_MANIFEST_BEFORE_COMMIT)
        ordinal = self._next_segment_ordinal(lane)
        next_id = f"L{lane:02d}-{ordinal:08d}"
        next_path = self._segment_path(lane, ordinal)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "UPDATE segments SET state='SEALED',segment_sha256=?,manifest_path=?,sealed_at=? "
                "WHERE segment_id=? AND state='ACTIVE'",
                (segment_sha, str(manifest_path), time.time(), segment_id),
            )
            self.conn.execute(
                "INSERT INTO segments VALUES(?,?,?,?,?,?,?,?,?,?)",
                (next_id, lane, str(next_path), "ACTIVE", 0, 0, None, None, time.time(), None),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def _append_payload(self, raw: bytes, *, fail_hook: FailHook | None = None) -> tuple[PayloadRefR11, bool]:
        content_hash = sha256_bytes(raw)
        old = self._payload_ref(content_hash)
        if old is not None:
            return old, False
        h, record, crc, codec, stored_len = self._encode(raw)
        if h != content_hash:
            raise RuntimeError("R11_CONTENT_HASH_DRIFT")
        lane = stable_lane(h, self.lane_count)
        segment_id, path, committed_end, payload_count = self._active_row(lane)
        if committed_end and committed_end + len(record) > self.segment_target_bytes:
            self._seal_active(lane, fail_hook=fail_hook)
            segment_id, path, committed_end, payload_count = self._active_row(lane)
        path.parent.mkdir(parents=True, exist_ok=True)
        actual_size = path.stat().st_size if path.exists() else 0
        if actual_size != committed_end:
            raise RuntimeError(f"R11_ACTIVE_TAIL_NOT_RECOVERED:{segment_id}:{actual_size}:{committed_end}")
        with path.open("ab", buffering=0) as handle:
            handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())
        if fail_hook is not None:
            fail_hook(FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX)
        end = committed_end + len(record)
        ref = PayloadRefR11(
            h, lane, segment_id, str(path), committed_end, end, len(raw), stored_len, codec, crc
        )
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            conflict = self._payload_ref(h)
            if conflict is not None:
                if conflict != ref:
                    raise RuntimeError(f"R11_CONTENT_ADDRESS_COLLISION_OR_RACE:{h}")
                self.conn.execute("ROLLBACK")
                return conflict, False
            self.conn.execute(
                "INSERT INTO payloads VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (h, lane, segment_id, str(path), committed_end, end, len(raw), stored_len,
                 codec, crc, time.time()),
            )
            self.conn.execute(
                "UPDATE segments SET committed_end=?,payload_count=? WHERE segment_id=?",
                (end, payload_count + 1, segment_id),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        return ref, True

    def put_evidence(
        self, items: Iterable[EvidenceItemR11], *, fail_hook: FailHook | None = None
    ) -> tuple[list[PayloadRefR11], EvidenceMaterializeReceiptR11]:
        rows = list(items)
        identities: dict[str, str] = {}
        for item in rows:
            old = identities.get(item.evidence_id)
            if old is not None and old != item.identity_hash:
                raise RuntimeError(f"R11_EVIDENCE_ID_CONTENT_CONFLICT:{item.evidence_id}")
            identities[item.evidence_id] = item.identity_hash
        unique_payloads = {item.content_hash: item.payload_bytes for item in rows}
        refs: dict[str, PayloadRefR11] = {}
        created_payloads = 0
        for content_hash, raw in sorted(unique_payloads.items()):
            ref, created = self._append_payload(raw, fail_hook=fail_hook)
            if ref.content_hash != content_hash:
                raise RuntimeError("R11_PAYLOAD_REF_HASH_MISMATCH")
            refs[content_hash] = ref
            created_payloads += int(created)

        created_evidence = 0
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for item in rows:
                old = self.conn.execute(
                    "SELECT identity_hash FROM evidence_catalog WHERE evidence_id=?", (item.evidence_id,)
                ).fetchone()
                if old is not None:
                    if str(old[0]) != item.identity_hash:
                        raise RuntimeError(f"R11_EVIDENCE_ID_CONTENT_CONFLICT:{item.evidence_id}")
                    continue
                self.conn.execute(
                    "INSERT INTO evidence_catalog VALUES(?,?,?,?,?,?,?)",
                    (item.evidence_id, item.content_hash, item.identity_hash,
                     item.parent_snapshot_hash, item.lineage_hash, item.teacher_protocol_hash,
                     time.time()),
                )
                created_evidence += 1
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return [refs[item.content_hash] for item in rows], EvidenceMaterializeReceiptR11(
            input_count=len(rows), unique_payload_count=len(unique_payloads),
            created_payload_count=created_payloads,
            reused_payload_count=len(unique_payloads) - created_payloads,
            created_evidence_count=created_evidence,
            reused_evidence_count=len(rows) - created_evidence,
        )

    def get_payload(self, content_hash: str) -> Mapping[str, Any] | None:
        ref = self._payload_ref(content_hash)
        if ref is None:
            return None
        decoded = self._decode(Path(ref.segment_path), ref.offset, lane=ref.lane, segment_id=ref.segment_id)
        if decoded is None or decoded[0].content_hash != content_hash or decoded[2] != ref.end_offset:
            raise RuntimeError(f"R11_PAYLOAD_LOCATOR_MISMATCH:{content_hash}")
        return json.loads(decoded[1])

    def seal_evidence_set(self, evidence_set_id: str, evidence_ids: Sequence[str]) -> dict[str, Any]:
        unique_ids = sorted(set(str(x) for x in evidence_ids))
        rows = []
        for evidence_id in unique_ids:
            row = self.conn.execute(
                "SELECT content_hash,identity_hash,parent_snapshot_hash,lineage_hash,teacher_protocol_hash "
                "FROM evidence_catalog WHERE evidence_id=?", (evidence_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"R11_EVIDENCE_NOT_FOUND:{evidence_id}")
            rows.append([evidence_id] + [str(x) for x in row])
        body = {
            "schema": "CB16_R11_EVIDENCE_SET_V1",
            "evidence_set_id": str(evidence_set_id),
            "object_count": len(rows),
            "objects": rows,
            "generation_independent": True,
        }
        evidence_set_hash = sha256_obj(body)
        path = self.evidence_set_root / f"{evidence_set_hash}.json"
        if not path.exists():
            _atomic_json(path, body)
        self.conn.execute(
            "INSERT OR IGNORE INTO evidence_sets VALUES(?,?,?,?,?)",
            (evidence_set_hash, str(evidence_set_id), len(rows), str(path), time.time()),
        )
        return {
            "evidence_set_hash": evidence_set_hash,
            "evidence_set_id": str(evidence_set_id),
            "object_count": len(rows),
            "manifest_path": str(path),
        }

    def seal_all_active_segments(self) -> None:
        for lane in range(self.lane_count):
            self._seal_active(lane)

    def periodic_audit(self) -> dict[str, Any]:
        """Cheap audit: manifests + first/last record of each sealed segment + full active tails."""
        problems: list[dict[str, Any]] = []
        self.fast_startup_validation()
        for segment_id, lane, path_s in self.conn.execute(
            "SELECT segment_id,lane,path FROM segments WHERE state='SEALED' ORDER BY lane,segment_id"
        ):
            rows = self.conn.execute(
                "SELECT content_hash,offset,end_offset FROM payloads WHERE segment_id=? ORDER BY offset",
                (segment_id,),
            ).fetchall()
            for row in (rows[:1] + rows[-1:] if rows else []):
                try:
                    decoded = self._decode(Path(path_s), int(row[1]), lane=int(lane), segment_id=str(segment_id))
                    if decoded is None or decoded[0].content_hash != str(row[0]) or decoded[2] != int(row[2]):
                        raise RuntimeError("BOUNDARY_MISMATCH")
                except Exception as exc:
                    problems.append({"segment_id": segment_id, "error": repr(exc)})
        try:
            self.recover_active_tails()
        except Exception as exc:
            problems.append({"segment_id": "ACTIVE", "error": repr(exc)})
        return {
            "schema": "CB16_R11_EVIDENCE_PERIODIC_AUDIT_V1",
            "problems": problems,
            "pass": not problems,
        }

    def full_forensic_audit(self) -> dict[str, Any]:
        """Read/decompress/rehash every payload and verify sealed whole-file hashes."""
        problems: list[dict[str, Any]] = []
        payloads_checked = 0
        bytes_read = 0
        segments = self.conn.execute(
            "SELECT segment_id,lane,path,state,committed_end,segment_sha256,manifest_path "
            "FROM segments ORDER BY lane,segment_id"
        ).fetchall()
        for segment_id, lane, path_s, state, committed_end, segment_sha, manifest_path in segments:
            path = Path(path_s)
            try:
                if not path.exists():
                    if state == "ACTIVE" and int(committed_end) == 0:
                        continue
                    raise RuntimeError("SEGMENT_MISSING")
                size = path.stat().st_size
                bytes_read += size
                if state == "SEALED":
                    if size != int(committed_end):
                        raise RuntimeError("SEALED_SIZE_MISMATCH")
                    if sha256_file(path) != str(segment_sha):
                        raise RuntimeError("SEALED_SEGMENT_SHA256_MISMATCH")
                    manifest = json.loads(Path(manifest_path).read_text())
                    if manifest.get("index_sha256") != sha256_obj(manifest.get("entries", [])):
                        raise RuntimeError("SEALED_MANIFEST_INDEX_HASH_MISMATCH")
                offset = 0
                seen = []
                while offset < size:
                    decoded = self._decode(path, offset, lane=int(lane), segment_id=str(segment_id))
                    if decoded is None:
                        raise RuntimeError(f"TRUNCATED_RECORD:{offset}")
                    ref, _raw, end = decoded
                    seen.append((ref.content_hash, ref.offset, ref.end_offset))
                    payloads_checked += 1
                    offset = end
                indexed = self.conn.execute(
                    "SELECT content_hash,offset,end_offset FROM payloads WHERE segment_id=? ORDER BY offset",
                    (segment_id,),
                ).fetchall()
                expected = [(str(a), int(b), int(c)) for a, b, c in indexed]
                if seen != expected:
                    raise RuntimeError("SEGMENT_INDEX_CONTENT_MISMATCH")
            except Exception as exc:
                problems.append({"segment_id": str(segment_id), "path": str(path), "error": repr(exc)})
        return {
            "schema": "CB16_R11_EVIDENCE_FULL_FORENSIC_AUDIT_V1",
            "segments": len(segments),
            "payloads_checked": payloads_checked,
            "payload_bytes_read": bytes_read,
            "problems": problems,
            "pass": not problems,
        }
