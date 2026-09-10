#!/usr/bin/env python3
from __future__ import annotations

"""R11 R6 TRAIN-only provenance preflight.

Reads R10.4 Experience Lake metadata in SQLite read-only mode and, at most, one
object that is already sealed into the G66 TRAINING_SNAPSHOT. It never opens the
mixed legacy parent/branch gzip, R10.4 validation payload, R5 purge support,
market anchors, raw market archives, or FINAL.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

SNAPSHOT_ID = "R102_G66_TRAINING_SNAPSHOT"
GENERATION = 66
SHARDS = 4


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def stable_shard(object_id: str) -> int:
    return int(hashlib.sha256(object_id.encode()).hexdigest()[:16], 16) % SHARDS


def ro_connect(path: Path) -> sqlite3.Connection:
    require(path.is_file(), f"R6_PROVENANCE_DB_MISSING:{path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def current_payload_path(lake_root: Path, shard: int, payload_hash: str) -> Path:
    return lake_root / "objects" / f"shard_{shard:02d}" / payload_hash[:2] / payload_hash[2:4] / f"{payload_hash}.json.zlib"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake-root", type=Path, default=Path("/cb16/runtime/r104/experience_lake"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    root = a.lake_root.resolve()
    dbs = [root / "metadata" / f"experience_{i:02d}.sqlite" for i in range(SHARDS)]

    global_types: Counter[str] = Counter()
    snapshot_rows = []
    for db in dbs:
        con = ro_connect(db)
        try:
            for typ, n in con.execute(
                "SELECT object_type,COUNT(*) FROM objects WHERE generation=? GROUP BY object_type ORDER BY object_type",
                (GENERATION,),
            ):
                global_types[str(typ)] += int(n)
            row = con.execute(
                "SELECT content_hash,parent_generation,parent_policy_hash,object_count,payload_json FROM snapshots WHERE snapshot_id=?",
                (SNAPSHOT_ID,),
            ).fetchone()
            if row is not None:
                snapshot_rows.append((db, row))
        finally:
            con.close()

    require(len(snapshot_rows) == 1, f"R6_PROVENANCE_SNAPSHOT_COUNT:{len(snapshot_rows)}")
    _, row = snapshot_rows[0]
    content_hash, parent_generation, parent_policy_hash, object_count, payload_json = row
    if isinstance(payload_json, memoryview):
        payload_json = payload_json.tobytes()
    if isinstance(payload_json, str):
        payload_json = payload_json.encode("utf-8")
    snap = json.loads(bytes(payload_json))
    require(snap.get("snapshot_id") == SNAPSHOT_ID, "R6_PROVENANCE_SNAPSHOT_ID_DRIFT")
    require(int(parent_generation) == GENERATION and int(snap.get("parent_generation", -1)) == GENERATION,
            "R6_PROVENANCE_SNAPSHOT_GENERATION_DRIFT")
    object_ids = list(snap.get("object_ids", []))
    require(len(object_ids) == int(object_count) == int(snap.get("object_count", -1)),
            "R6_PROVENANCE_SNAPSHOT_OBJECT_COUNT_DRIFT")
    require(len(object_ids) == len(set(object_ids)), "R6_PROVENANCE_SNAPSHOT_DUPLICATE_OBJECT")

    snapshot_types: Counter[str] = Counter()
    referenced_rows = []
    for oid in object_ids:
        si = stable_shard(str(oid))
        con = ro_connect(dbs[si])
        try:
            r = con.execute(
                "SELECT object_type,generation,identity_hash,payload_hash,payload_path FROM objects WHERE object_id=?",
                (str(oid),),
            ).fetchone()
        finally:
            con.close()
        require(r is not None, f"R6_PROVENANCE_SNAPSHOT_OBJECT_MISSING:{oid}")
        typ, gen, identity_hash, payload_hash, stored_path = r
        require(int(gen) == GENERATION, f"R6_PROVENANCE_SNAPSHOT_OBJECT_GENERATION_DRIFT:{oid}:{gen}")
        snapshot_types[str(typ)] += 1
        referenced_rows.append((str(oid), si, str(typ), str(identity_hash), str(payload_hash), str(stored_path)))

    # Read one already-sealed TRAIN EVIDENCE_PACKAGE to establish payload schema.
    evidence_rows = [r for r in referenced_rows if r[2] == "EVIDENCE_PACKAGE"]
    require(evidence_rows, "R6_PROVENANCE_NO_EVIDENCE_PACKAGE_IN_TRAIN_SNAPSHOT")
    sample_oid, sample_shard, _, _, sample_payload_hash, _ = evidence_rows[0]
    pp = current_payload_path(root, sample_shard, sample_payload_hash)
    require(pp.is_file(), f"R6_PROVENANCE_TRAIN_PAYLOAD_MISSING:{pp}")
    raw = zlib.decompress(pp.read_bytes())
    require(hashlib.sha256(raw).hexdigest() == sample_payload_hash, "R6_PROVENANCE_SAMPLE_PAYLOAD_HASH_MISMATCH")
    payload = json.loads(raw)
    keys = sorted(payload)
    expected_reduced = {"operator48","medium48","account6","direction_target_probs","requested_risk_target","action_laws"}
    require(expected_reduced.issubset(payload), f"R6_PROVENANCE_EVIDENCE_SCHEMA_DRIFT:{keys}")

    raw_truth_keys = {
        "realized_utility", "counterfactual_branches", "branch_samples", "realized_utilities",
        "action_realized_utilities", "h72_counterfactual_truth"
    }
    sample_has_raw_truth = any(k in payload for k in raw_truth_keys)
    snapshot_only_compiled_evidence = set(snapshot_types) == {"EVIDENCE_PACKAGE"}
    suspicious_global_types = sorted(
        typ for typ in global_types
        if any(token in typ.upper() for token in ("COUNTERFACTUAL", "BRANCH", "TEACHER_TRUTH"))
    )
    on_policy_outcomes = int(global_types.get("OUTCOME_SAMPLE", 0))

    raw_truth_materialized = bool(sample_has_raw_truth or suspicious_global_types)
    nested_teacher_ready = bool(raw_truth_materialized)
    status = (
        "TRAIN_ONLY_RAW_COUNTERFACTUAL_TRUTH_CANDIDATE_PRESENT_REQUIRES_SCHEMA_AUDIT"
        if raw_truth_materialized
        else "TRAIN_ONLY_NESTED_TEACHER_RAW_TRUTH_NOT_MATERIALIZED"
    )
    result = {
        "schema": "CB16_R11_SCIENCE_G0_R6_TRAIN_ONLY_PROVENANCE_PREFLIGHT_V1",
        "status": status,
        "lake_root": str(root),
        "generation": GENERATION,
        "snapshot": {
            "snapshot_id": SNAPSHOT_ID,
            "content_hash": str(content_hash),
            "parent_generation": int(parent_generation),
            "parent_policy_hash": str(parent_policy_hash),
            "object_count": int(object_count),
            "object_types": dict(sorted(snapshot_types.items())),
            "snapshot_contains_only_compiled_evidence_packages": snapshot_only_compiled_evidence,
        },
        "generation_66_global_object_types": dict(sorted(global_types.items())),
        "sample_train_evidence_package": {
            "object_id": sample_oid,
            "payload_hash": sample_payload_hash,
            "keys": keys,
            "contains_reduced_teacher_targets": True,
            "contains_action_laws": True,
            "contains_raw_nine_branch_realized_utilities": sample_has_raw_truth,
        },
        "raw_counterfactual_candidate_object_types": suspicious_global_types,
        "on_policy_outcome_sample_count": on_policy_outcomes,
        "on_policy_outcomes_are_not_nine_action_counterfactual_grid": True,
        "nested_teacher_recompile_from_outer_train_raw_truth_ready": nested_teacher_ready,
        "mixed_legacy_parent_branch_gzip_opened": False,
        "legacy_validation_payload_opened": False,
        "r5_purge_support_opened": False,
        "market_payload_opened": False,
        "final_holdout_payload_opened": False,
        "fresh_market_data_downloaded": False,
        "network_reads_by_probe": 0,
        "scientific_verdict_created": False,
        "canonical_generation_advanced": False,
        "next_legal_step": (
            "AUDIT_CANDIDATE_RAW_TRUTH_SCHEMA_WITHOUT_OPENING_FORBIDDEN_SUPPORT"
            if nested_teacher_ready
            else "R6_SUPPORT_NOT_READY__PRESERVE_PREREGISTERED_GATE__DO_NOT_READ_MIXED_VALIDATION_GZIP"
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
