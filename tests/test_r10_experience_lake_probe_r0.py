from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from cb16_diagnostics.experience_lake_probe import probe_experience_lake


def _init_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE objects(
            object_id TEXT PRIMARY KEY,
            object_type TEXT,
            generation INTEGER,
            policy_weight_hash TEXT,
            snapshot_hash TEXT,
            lineage_hash TEXT,
            identity_hash TEXT,
            payload_hash TEXT,
            payload_path TEXT,
            bytes_raw INTEGER,
            bytes_stored INTEGER,
            created_at REAL
        );
        CREATE TABLE snapshots(
            snapshot_id TEXT PRIMARY KEY,
            content_hash TEXT,
            parent_generation INTEGER,
            parent_policy_hash TEXT,
            object_count INTEGER,
            payload_json BLOB,
            created_at REAL
        );
        """
    )
    conn.commit()
    conn.close()


def test_probe_recovers_per_generation_lake_timing(tmp_path: Path) -> None:
    rr = tmp_path / "R10_4"
    metadata = rr / "experience_lake" / "metadata"
    metadata.mkdir(parents=True)
    for i in range(4):
        _init_db(metadata / f"experience_{i:02d}.sqlite")

    for j in range(8):
        db = metadata / f"experience_{j % 4:02d}.sqlite"
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO objects VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"o{j}", "EVIDENCE_PACKAGE", 59, "p", "s", "l", f"ih{j}", f"ph{j}", "x", 100, 50, 100 + j),
        )
        conn.commit()
        conn.close()

    conn = sqlite3.connect(metadata / "experience_00.sqlite")
    conn.execute(
        "INSERT INTO snapshots VALUES(?,?,?,?,?,?,?)",
        ("R102_G59_TRAINING_SNAPSHOT", "snap-hash", 59, "p", 8, b"{}", 108),
    )
    conn.commit()
    conn.close()

    gd = rr / "generations" / "G59"
    gd.mkdir(parents=True)
    for name, ts in (
        ("ON_POLICY_REAL_TRACE_RECEIPT.json", 99),
        ("CHALLENGER_TRAINING_RECEIPT_G59.json", 120),
        ("challenger.pt", 122),
        ("GENERATION_RESULT.json", 124),
    ):
        p = gd / name
        p.write_text("{}", encoding="utf-8")
        os.utime(p, (ts, ts))

    result = probe_experience_lake(run_root=rr)
    assert result["status"] == "PASS"
    assert result["shards_found"] == 4
    g = result["generations"][0]
    assert g["generation"] == 59
    assert g["object_count"] == 8
    assert g["object_insert_span_seconds"] == 7.0
    assert g["first_object_to_snapshot_seal_seconds"] == 8.0
    assert g["snapshot_seal_to_training_receipt_seconds"] == 12.0
    assert g["training_receipt_to_challenger_seconds"] == 2.0
    assert g["challenger_to_generation_result_seconds"] == 2.0
    assert g["snapshot_object_count_matches_metadata"] is True
    assert result["safety"]["writes_to_canonical_run_root"] is False
    assert result["safety"]["scientific_semantics_changed"] is False
    assert result["safety"]["final_holdout_2025_09_accessed"] is False


def test_probe_handles_active_unsealed_generation(tmp_path: Path) -> None:
    rr = tmp_path / "R10_4"
    metadata = rr / "experience_lake" / "metadata"
    metadata.mkdir(parents=True)
    for i in range(4):
        _init_db(metadata / f"experience_{i:02d}.sqlite")
    conn = sqlite3.connect(metadata / "experience_02.sqlite")
    for j in range(3):
        conn.execute(
            "INSERT INTO objects VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"g60-{j}", "EVIDENCE_PACKAGE", 60, "p", "s", "l", f"ih{j}", f"ph{j}", "x", 100, 40, 200 + j),
        )
    conn.commit()
    conn.close()

    result = probe_experience_lake(run_root=rr)
    g = result["generations"][0]
    assert g["generation"] == 60
    assert g["object_count"] == 3
    assert g["snapshot_sealed"] is False
    assert g["snapshot_object_count_matches_metadata"] is False
    assert g["object_insert_span_seconds"] == 2.0
