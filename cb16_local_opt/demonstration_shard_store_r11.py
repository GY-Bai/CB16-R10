from __future__ import annotations

"""Deterministic durable storage for R11 demonstration H72 shards."""

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def rows_semantic_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(b"CB16_R11_DEMONSTRATION_ROWS_V1\0")
    for row in rows:
        raw = canonical_json_bytes(row)
        h.update(len(raw).to_bytes(8, "big"))
        h.update(raw)
    return h.hexdigest()


def write_deterministic_jsonl_gz(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    """Atomic gzip JSONL with mtime=0 so identical rows imply identical file bytes."""

    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".tmp", dir=dst.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=4, mtime=0) as gz:
                for row in materialized:
                    gz.write(canonical_json_bytes(row))
                    gz.write(b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return len(materialized), sha256_file(dst)


def read_jsonl_gz(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def shard_identity_r11(
    *,
    manifest_blob: str,
    manifest_sha256: str,
    symbol: str,
    shard_size: int,
    shard_index: int,
    decision_times_ms: Sequence[int],
) -> dict[str, Any]:
    if not decision_times_ms:
        raise ValueError("empty decision_times_ms")
    if int(shard_size) <= 0 or int(shard_index) < 0:
        raise ValueError("invalid shard geometry")
    times = [int(x) for x in decision_times_ms]
    payload = {
        "schema": "CB16_R11_DEMONSTRATION_H72_SHARD_IDENTITY_V1",
        "manifest_blob": str(manifest_blob),
        "manifest_sha256": str(manifest_sha256),
        "symbol": str(symbol),
        "shard_size": int(shard_size),
        "shard_index": int(shard_index),
        "parent_count": len(times),
        "first_decision_ms": times[0],
        "last_decision_ms": times[-1],
        "decision_times_sha256": canonical_json_sha256(times),
    }
    return {**payload, "shard_identity_sha256": canonical_json_sha256(payload)}


def combine_rows_semantic_sha256(parts: Sequence[Sequence[Mapping[str, Any]]]) -> str:
    rows: list[Mapping[str, Any]] = []
    for part in parts:
        rows.extend(part)
    return rows_semantic_sha256(rows)


__all__ = [
    "canonical_json_bytes",
    "canonical_json_sha256",
    "sha256_file",
    "rows_semantic_sha256",
    "write_deterministic_jsonl_gz",
    "read_jsonl_gz",
    "shard_identity_r11",
    "combine_rows_semantic_sha256",
]
