"""Static, hostile and exact-count tests for the R21 RC2 R4 measurement layer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from ci.r4_write_path_instrument.cb16_r4_write_path_instrument import classify_path

from scripts.measure_r21_rc2_r4_write_path_v1 import (
    REQUIRED_SURFACES,
    _classify_access,
    _count_ops,
    _find_committed_updates,
    _run_instrumentation_canary,
    _surface_report,
    measurement_status,
    INSTRUMENT_DIR,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R4_WRITE_PATH_MEASUREMENT_SPEC_V2.json"
INVENTORY_V2_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R4_WRITE_PATH_INVENTORY_V2.json"
RECEIPT_V2_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R4_WRITE_PATH_MEASUREMENT_RECEIPT_V2.json"
AUDIT_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R4_UPSTREAM_REUSE_AUDIT_V1.json"


class WritePathInventoryV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_spec_identity_and_required_surfaces(self) -> None:
        self.assertEqual(self.spec["schema"], "CB16_R21_RC2_R4_WRITE_PATH_MEASUREMENT_SPEC_V2")
        self.assertEqual(self.spec["status"], "FROZEN_FOR_EXECUTION")
        self.assertEqual(self.spec["s1_runtime_identity"]["authorization_head_sha"], "a974e2803ccc2693d67a0375e460da35837636a4")
        self.assertEqual(tuple(self.spec["required_surfaces"]), REQUIRED_SURFACES)
        self.assertTrue(self.spec["measurement_integrity_requirements"]["exact_count_canary"])
        self.assertIn("sqlite_page_growth_estimate_bytes", self.spec["measurement_integrity_requirements"]["estimated_metrics_separate"])
        self.assertIn("device_io_deltas.sda.sectors_written_delta", self.spec["measurement_integrity_requirements"]["device_metrics_separate"])

    def test_upstream_reuse_audit_classifies_existing_runtime(self) -> None:
        self.assertEqual(self.audit["schema"], "CB16_R21_RC2_UPSTREAM_REUSE_AUDIT_V1")
        classifications = {entry["path"]: entry["classification"] for entry in self.audit["existing_implementation"]}
        self.assertEqual(classifications["scripts/run_r11_post_cc_s1_learnability.py"], "REUSE_AS_IS")
        self.assertIn("ci/r4_write_path_instrument/", classifications)
        self.assertIn("DO_NOT_TOUCH", classifications.values())
        forbidden = " ".join(self.audit["forbidden"])
        self.assertIn("S1 runtime code", forbidden)

    def test_exact_count_instrumentation_canary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_instrumentation_canary(sys.executable, INSTRUMENT_DIR, Path(tmp) / "canary")
        self.assertEqual(result["status"], "PASS", result)
        self.assertTrue(result["exact_count_match"])
        self.assertTrue(result["sqlite_estimate_activity"])
        self.assertTrue(result["metrics_self_exclusion"])
        expected = result["expected_exact"]
        observed = result["observed_exact"]
        for category, fields in expected.items():
            for field, value in fields.items():
                self.assertEqual(observed[category][field], value, (category, field))

    def test_path_classification(self) -> None:
        cases = {
            "/cb16/fast_hot/r4/x/scratch/run/observations/fact.json": "observation_store",
            "/cb16/fast_hot/r4/x/scratch/run/index.sqlite3-wal": "sqlite_index",
            "/cb16/fast_hot/r4/x/scratch/run/replay/material.json": "replay_materialization",
            "/cb16/fast_hot/r4/x/scratch/run/updates/" + "a" * 64 + ".json": "update_journal",
            "/cb16/fast_hot/r4/x/scratch/run/updates/checkpoints/" + "b" * 64 + ".json": "checkpoint_store",
            "/cb16/fast_hot/r4/x/scratch/run/updates/generation_switch_receipts/" + "b" * 64 + ".json": "generation_continuity",
            "/cb16/fast_hot/r4/x/output/provenance_staged/index.json": "provenance",
            "/cb16/fast_hot/r4/x/output/artifacts/pack.json": "artifact_staging",
        }
        for path, expected in cases.items():
            self.assertEqual(classify_path(path), expected, path)

    def test_surface_report_splits_direct_and_estimated_metrics(self) -> None:
        summary = {
            "update_journal": {
                "write": {"count": 2, "bytes": 20, "duration_ns": 0, "extra": {}},
                "fsync": {"count": 1, "bytes": 0, "duration_ns": 1_000_000, "extra": {}},
                "replace": {"count": 1, "bytes": 0, "duration_ns": 0, "extra": {}},
            },
            "sqlite_index": {
                "sqlite_commit": {"count": 3, "bytes": 0, "duration_ns": 3_000_000, "extra": {}},
                "sqlite_page_growth_estimate": {"count": 2, "bytes": 4096, "duration_ns": 0, "extra": {}},
                "sqlite_wal_growth_estimate": {"count": 2, "bytes": 512, "duration_ns": 0, "extra": {}},
            },
        }
        stats = _count_ops(summary, "update_journal")
        self.assertEqual(stats["application_direct_write_calls"], 2)
        self.assertEqual(stats["application_direct_write_bytes"], 20)
        report = _surface_report("update_journal", summary, updates=2, mount={"source": "/dev/sdb"})
        self.assertEqual(report["application_direct_write_bytes_per_update"], 10)
        self.assertEqual(report["application_direct_writes_per_update"], 1)
        sqlite = _surface_report("sqlite_index", summary, updates=2, mount={"source": "/dev/sdb"})
        self.assertEqual(sqlite["application_direct_write_bytes"], 0)
        self.assertEqual(sqlite["sqlite_page_growth_estimate_bytes"], 4096)
        self.assertEqual(sqlite["sqlite_wal_growth_estimate_bytes"], 512)
        self.assertIn("ESTIMATE", sqlite["bytes_measurement_method"])

    def test_access_classification_uses_direct_writes_only(self) -> None:
        self.assertEqual(_classify_access({"application_direct_write_calls": 0, "application_direct_write_bytes": 0, "fsync_calls": 0}), "NO_DIRECT_WRITES_OBSERVED")
        self.assertEqual(
            _classify_access({"application_direct_write_calls": 10, "application_direct_write_bytes": 10 * 1024, "fsync_calls": 10}),
            "RANDOM_OR_SMALL_FSYNC_HEAVY",
        )
        self.assertEqual(
            _classify_access({"application_direct_write_calls": 1, "application_direct_write_bytes": 8 * 1024 * 1024, "fsync_calls": 0}),
            "SEQUENTIAL_LARGE",
        )

    def test_measurement_status_fails_closed(self) -> None:
        self.assertEqual(measurement_status(0, 10, [], "PASS"), "PASS")
        self.assertEqual(measurement_status(1, 10, [], "PASS"), "EVIDENCE_INSUFFICIENT")
        self.assertEqual(measurement_status(0, 0, [], "PASS"), "EVIDENCE_INSUFFICIENT")
        self.assertEqual(measurement_status(0, 10, ["generation_continuity"], "PASS"), "EVIDENCE_INSUFFICIENT")
        self.assertEqual(measurement_status(0, 10, [], "FAIL"), "EVIDENCE_INSUFFICIENT")

    def test_committed_update_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance = root / "provenance"
            provenance.mkdir()
            (provenance / "artifact_only_update_trace_audit.json").write_text(
                json.dumps({"traced_update_count": 7, "all_checks_pass": True}), encoding="utf-8"
            )
            updates, source = _find_committed_updates(root, set())
            self.assertEqual(updates, 7)
            self.assertIn("traced_update_count", source)

    def test_v2_inventory_and_receipt_when_present(self) -> None:
        if not INVENTORY_V2_PATH.exists() or not RECEIPT_V2_PATH.exists():
            self.skipTest("V2 inventory/receipt not generated yet")
        inventory = json.loads(INVENTORY_V2_PATH.read_text(encoding="utf-8"))
        receipt = json.loads(RECEIPT_V2_PATH.read_text(encoding="utf-8"))
        self.assertEqual(inventory["status"], "PASS")
        self.assertEqual(inventory["missing_required_surfaces"], [])
        self.assertTrue(inventory["measurement_integrity"]["canary_status"] == "PASS")
        for surface in REQUIRED_SURFACES:
            self.assertIn(surface, inventory["observed_surfaces"])
        sqlite = {entry["surface"]: entry for entry in inventory["surfaces"]}["sqlite_index"]
        self.assertIn("ESTIMATE", sqlite["bytes_measurement_method"])
        self.assertEqual(receipt["evidence"]["run_id"], inventory["evidence_binding"]["run_id"])
        self.assertFalse(receipt["scientific_boundary"]["s1_runtime_code_changed"])


if __name__ == "__main__":
    unittest.main()
