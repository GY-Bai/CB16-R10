from __future__ import annotations

import os
import sqlite3

from cb16_diagnostics.experience_lake_probe import probe_experience_lake


def _init_db(path):
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE objects(
          object_id TEXT PRIMARY KEY, object_type TEXT, generation INTEGER,
          policy_weight_hash TEXT, snapshot_hash TEXT, lineage_hash TEXT,
          identity_hash TEXT, payload_hash TEXT, payload_path TEXT,
          bytes_raw INTEGER, bytes_stored INTEGER, created_at REAL
        );
        CREATE TABLE snapshots(
          snapshot_id TEXT PRIMARY KEY, content_hash TEXT, parent_generation INTEGER,
          parent_policy_hash TEXT, object_count INTEGER, payload_json BLOB, created_at REAL
        );
        """
    )
    return con


def _put(con, oid, typ, gen, created):
    con.execute(
        "INSERT INTO objects VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (oid, typ, gen, "p", "s", "l", "i", "ph", "x", 100, 50, float(created)),
    )


def test_type_aware_snapshot_count_and_recovery_span(tmp_path):
    rr = tmp_path / "R10_4"
    metadata = rr / "experience_lake" / "metadata"
    metadata.mkdir(parents=True)

    for shard in range(4):
        con = _init_db(metadata / f"experience_{shard:02d}.sqlite")
        # Trace objects predate the current retry by a long interval.
        _put(con, f"d{shard}", "DECISION_EVENT", 59, 1000 + shard)
        _put(con, f"o{shard}", "OUTCOME_SAMPLE", 59, 1001 + shard)
        # Training evidence belongs to the current attempt.
        _put(con, f"e{shard}a", "EVIDENCE_PACKAGE", 59, 2000 + shard * 2)
        _put(con, f"e{shard}b", "EVIDENCE_PACKAGE", 59, 2001 + shard * 2)
        if shard == 0:
            con.execute(
                "INSERT INTO snapshots VALUES(?,?,?,?,?,?,?)",
                ("R102_G59_TRAINING_SNAPSHOT", "h", 59, "p", 8, b"{}", 2010.0),
            )
        con.commit()
        con.close()

    gd = rr / "generations" / "G59"
    gd.mkdir(parents=True)
    artifacts = {
        "ON_POLICY_REAL_TRACE_RECEIPT.json": 1999.0,
        "CHALLENGER_TRAINING_RECEIPT_G59.json": 2012.0,
        "challenger.pt": 2012.1,
        "GENERATION_RESULT.json": 2012.2,
    }
    for name, ts in artifacts.items():
        p = gd / name
        p.write_text("x")
        os.utime(p, (ts, ts))

    result = probe_experience_lake(run_root=rr)
    g = result["generations"][0]
    assert result["status"] == "PASS"
    assert g["all_object_count"] == 16
    assert g["training_evidence_object_count"] == 8
    assert g["decision_event_object_count"] == 4
    assert g["outcome_sample_object_count"] == 4
    assert g["snapshot_object_count_matches_training_evidence"] is True
    assert g["historical_span_crosses_prior_attempt_trace"] is True
    assert g["training_evidence_span_crosses_prior_attempt"] is False
    assert g["training_evidence_insert_span_seconds"] == 7.0
    assert g["snapshot_seal_to_training_receipt_seconds"] == 2.0
    assert result["safety"]["writes_to_canonical_run_root"] is False
    assert result["safety"]["scientific_semantics_changed"] is False
    assert result["safety"]["final_holdout_2025_09_accessed"] is False
