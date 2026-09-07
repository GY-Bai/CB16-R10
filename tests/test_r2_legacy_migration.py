from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from cb16_local_opt.r2_evidence_storage import R2EvidenceStore, canonical_json_bytes, sha256_bytes
from cb16_local_opt.r2_legacy_migration import qualify_legacy_generations


def _legacy_root(root: Path, generations=(60,61), changed_generation=None):
    lake = root / "legacy"
    meta = lake / "metadata"
    objroot = lake / "objects" / "shard_00"
    meta.mkdir(parents=True)
    objroot.mkdir(parents=True)
    db = sqlite3.connect(meta / "experience_00.sqlite")
    db.executescript("""
    CREATE TABLE objects(
      object_id TEXT PRIMARY KEY,object_type TEXT NOT NULL,generation INTEGER NOT NULL,
      policy_weight_hash TEXT NOT NULL,snapshot_hash TEXT NOT NULL,lineage_hash TEXT NOT NULL,
      identity_hash TEXT NOT NULL,payload_hash TEXT NOT NULL,payload_path TEXT NOT NULL,
      bytes_raw INTEGER NOT NULL,bytes_stored INTEGER NOT NULL,created_at REAL NOT NULL);
    """)
    for g in generations:
        for i in range(5):
            payload = {
                "schema":"CB16_R10_2_EVIDENCE_PACKAGE_V1",
                "generation":g,
                "parent_id":f"P{i}",
                "dependence_group_id":f"D{i}",
                "student_context_object_id":f"C{i}",
                "operator48":[float(i),1.0],
                "medium48":[2.0],
                "account6":[3.0],
                "direction_target_probs":[0.2,0.3,0.5],
                "requested_risk_target":0.25,
                "action_laws":[],
                "admission":{"admitted":True},
                "teacher_protocol_hash":"TEACHER",
            }
            if changed_generation == g and i == 2:
                payload["requested_risk_target"] = 0.75
            raw = canonical_json_bytes(payload)
            h = sha256_bytes(raw)
            stored = zlib.compress(raw,3)
            p = objroot / f"{g}_{i}.zlib"
            p.write_bytes(stored)
            oid = f"R102:G{g}:E{i}"
            db.execute(
                "INSERT INTO objects VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (oid,"EVIDENCE_PACKAGE",g,f"POLICY{g}",f"SNAP{i}",f"LINEAGE{i}",
                 f"IDENT{g}_{i}",h,str(p),len(raw),len(stored),float(g)),
            )
    db.commit(); db.close()
    return lake


class R2LegacyMigrationTests(unittest.TestCase):
    def test_two_legacy_generations_collapse_to_one_r2_evidence_set(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            legacy = _legacy_root(root)
            store = R2EvidenceStore(
                metadata_root=root/"r2meta",payload_roots=[root/"r2payload"],codec="zlib"
            )
            result = qualify_legacy_generations(
                legacy_lake_root=legacy,generations=[60,61],store=store
            )
            self.assertTrue(result["all_projection_hashes_equal"])
            self.assertTrue(result["all_materialized_evidence_set_hashes_equal"])
            self.assertTrue(result["all_projection_mismatches_zero"])
            self.assertTrue(result["all_metadata_mismatches_zero"])
            self.assertEqual(result["generations"][0]["created_payload_count"],5)
            self.assertEqual(result["generations"][1]["created_payload_count"],0)
            self.assertEqual(store.stats()["payload_objects"],5)
            store.close()

    def test_semantic_change_prevents_cross_generation_projection_equality(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            legacy = _legacy_root(root, changed_generation=61)
            result = qualify_legacy_generations(
                legacy_lake_root=legacy,generations=[60,61],store=None
            )
            self.assertFalse(result["all_projection_hashes_equal"])
            self.assertTrue(result["all_projection_mismatches_zero"])


if __name__ == "__main__":
    unittest.main()
