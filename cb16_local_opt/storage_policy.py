from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class StorageRoots:
    hot: Path       # SSD: active shards / snapshots / indexes
    cold: Path      # HDD: raw history / old trajectories / old generations
    temp: Path      # preferably SSD
    run_id: str

    @classmethod
    def build(
        cls,
        *,
        ssd_root: str | Path,
        hdd_root: str | Path,
        run_id: str,
    ) -> "StorageRoots":
        ssd = Path(ssd_root)
        hdd = Path(hdd_root)
        hot = ssd / "cb16" / "hot" / run_id
        temp = ssd / "cb16" / "tmp" / run_id
        cold = hdd / "cb16" / "cold" / run_id
        for p in (hot, temp, cold):
            p.mkdir(parents=True, exist_ok=True)
        return cls(hot=hot, cold=cold, temp=temp, run_id=run_id)


@dataclass(frozen=True)
class DatasetShard:
    shard_id: str
    path: str
    symbol: str
    timeframe: str
    start_ts: str
    end_ts: str
    rows: int
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    role: str
    path: str
    sha256: str
    bytes: int
    storage_class: str
    created_at_unix: float


class AtomicArtifactWriter:
    """Crash-safe, content-address-aware writer.

    A partially written temp file is never the authoritative target.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_bytes(self, relative_path: str, data: bytes, *, fsync: bool = True) -> Path:
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".partial", dir=str(target.parent))
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                if fsync:
                    os.fsync(f.fileno())
            os.replace(tmp, target)
            if fsync and hasattr(os, "O_DIRECTORY"):
                dfd = os.open(target.parent, os.O_DIRECTORY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            return target
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def write_json(self, relative_path: str, obj: Any) -> Path:
        return self.write_bytes(relative_path, canonical_json_bytes(obj) + b"\n")


class ArtifactManifest:
    """Append-only logical manifest with conflict detection."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ArtifactRecord] = {}
        if self.path.exists():
            obj = json.loads(self.path.read_text())
            for rec in obj.get("artifacts", []):
                r = ArtifactRecord(**rec)
                self._records[r.artifact_id] = r

    def register(self, record: ArtifactRecord) -> None:
        old = self._records.get(record.artifact_id)
        if old is not None and old.sha256 != record.sha256:
            raise RuntimeError(f"ARTIFACT_ID_CONTENT_CONFLICT:{record.artifact_id}")
        self._records[record.artifact_id] = record
        self.flush()

    def flush(self) -> None:
        obj = {
            "schema": "CB16_LOCAL_ARTIFACT_MANIFEST_R2",
            "artifacts": [asdict(v) for _, v in sorted(self._records.items())],
        }
        AtomicArtifactWriter(self.path.parent).write_json(self.path.name, obj)

    def records(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._records[k] for k in sorted(self._records))


class TieredStoragePolicy:
    HOT_ROLES = {
        "ACTIVE_DATASET_SHARD",
        "FROZEN_TRAINING_SNAPSHOT",
        "EXPERIENCE_INDEX",
        "CURRENT_CHAMPION",
        "CURRENT_CHALLENGER",
        "PREPROCESS_CACHE",
        "QUEUE_SPOOL",
    }

    COLD_ROLES = {
        "RAW_HISTORY",
        "OLD_TRAJECTORY_SHARD",
        "OLD_GENERATION",
        "OLD_CHECKPOINT",
        "ARCHIVE",
    }

    def __init__(self, roots: StorageRoots):
        self.roots = roots
        self.hot_writer = AtomicArtifactWriter(roots.hot)
        self.cold_writer = AtomicArtifactWriter(roots.cold)
        self.manifest = ArtifactManifest(roots.hot / "ARTIFACT_MANIFEST.json")

    def _writer_and_class(self, role: str):
        if role in self.COLD_ROLES:
            return self.cold_writer, "COLD"
        return self.hot_writer, "HOT"

    def put_bytes(self, *, artifact_id: str, role: str, relative_path: str, data: bytes) -> ArtifactRecord:
        writer, storage_class = self._writer_and_class(role)
        path = writer.write_bytes(relative_path, data)
        rec = ArtifactRecord(
            artifact_id=artifact_id,
            role=role,
            path=str(path),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
            storage_class=storage_class,
            created_at_unix=time.time(),
        )
        self.manifest.register(rec)
        return rec

    def put_json(self, *, artifact_id: str, role: str, relative_path: str, obj: Any) -> ArtifactRecord:
        return self.put_bytes(
            artifact_id=artifact_id,
            role=role,
            relative_path=relative_path,
            data=canonical_json_bytes(obj) + b"\n",
        )


def estimate_resident_bytes(
    *,
    rows: int,
    feature_dim: int,
    dtype_bytes: int = 4,
    copies: float = 2.5,
) -> int:
    """Approximate live memory including tensor/loader/transformation copies."""
    return int(rows * feature_dim * dtype_bytes * copies)


def assert_memory_budget(
    *,
    expected_bytes: int,
    total_ram_bytes: int,
    fraction: float = 0.35,
) -> None:
    limit = int(total_ram_bytes * fraction)
    if expected_bytes > limit:
        raise MemoryError(
            f"RESIDENT_DATASET_BUDGET_EXCEEDED expected={expected_bytes} limit={limit}; "
            "use smaller shards/streaming/mmap"
        )
