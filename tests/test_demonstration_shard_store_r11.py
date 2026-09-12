from __future__ import annotations

from pathlib import Path

from cb16_local_opt.demonstration_shard_store_r11 import (
    read_jsonl_gz,
    rows_semantic_sha256,
    shard_identity_r11,
    write_deterministic_jsonl_gz,
)


def test_deterministic_gzip_roundtrip_and_hash(tmp_path: Path) -> None:
    rows = [
        {"parent_id": "p0", "value": 1.25},
        {"parent_id": "p1", "value": -2.0},
    ]
    a = tmp_path / "a.jsonl.gz"
    b = tmp_path / "b.jsonl.gz"
    count_a, sha_a = write_deterministic_jsonl_gz(a, rows)
    count_b, sha_b = write_deterministic_jsonl_gz(b, rows)
    assert count_a == count_b == 2
    assert sha_a == sha_b
    assert a.read_bytes() == b.read_bytes()
    assert list(read_jsonl_gz(a)) == rows
    assert rows_semantic_sha256(rows) == rows_semantic_sha256(list(read_jsonl_gz(a)))


def test_shard_identity_commits_geometry_and_times() -> None:
    base = shard_identity_r11(
        manifest_blob="blob",
        manifest_sha256="a" * 64,
        symbol="BTCUSDT",
        shard_size=64,
        shard_index=0,
        decision_times_ms=[1000, 2000, 3000],
    )
    changed = shard_identity_r11(
        manifest_blob="blob",
        manifest_sha256="a" * 64,
        symbol="BTCUSDT",
        shard_size=64,
        shard_index=1,
        decision_times_ms=[1000, 2000, 3000],
    )
    assert base["shard_identity_sha256"] != changed["shard_identity_sha256"]
    assert base["parent_count"] == 3
