"""Static and hostile tests for the R21 RC2 R4 write-path measurement."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ci.r4_write_path_instrument.cb16_r4_write_path_instrument import classify_path

from scripts.measure_r21_rc2_r4_write_path_v1 import (
    REQUIRED_SURFACES,
    _classify_access,
    _count_ops,
    _find_committed_updates,
    _surface_report,
    measurement_status,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R4_WRITE_PATH_MEASUREMENT_SPEC_V1.json"
INSTRUMENT_DIR = REPO_ROOT / "ci" / "r4_write_path_instrument"


class WritePathInventoryV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_spec_identity_and_required_surfaces(self) -> None:
        self.assertEqual(self.spec["schema"], "CB16_R21_RC2_R4_WRITE_PATH_MEASUREMENT_SPEC_V1")
        self.assertEqual(self.spec["status"], "FROZEN_FOR_EXECUTION")
        self.assertEqual(self.spec["s1_runtime_identity"]["authorization_head_sha"], "a974e2803ccc2693d67a0375e460da35837636a4")
        self.assertEqual(
            self.spec["s1_runtime_identity"]["reviewed_implementation_tree_sha"],
            "8bffb90d635e11f0e2fdbefff5cf6e1cdba2f66d",
        )
        self.assertEqual(tuple(self.spec["required_surfaces"]), REQUIRED_SURFACES)
        self.assertFalse(self.spec["scope"]["host_changes_allowed"])
        self.assertFalse(self.spec["scope"]["s1_runtime_code_changes_allowed"])
        self.assertFalse(self.spec["scope"]["scientific_manifest_changes_allowed"])
        for field in self.spec["measurement_fields"]:
            self.assertIsInstance(field, str)

    def test_instrument_files_exist_and_are_measurement_only(self) -> None:
        sitecustomize = (INSTRUMENT_DIR / "sitecustomize.py").read_text(encoding="utf-8")
        module = (INSTRUMENT_DIR / "cb16_r4_write_path_instrument.py").read_text(encoding="utf-8")
        self.assertIn("CB16_R4_METRICS_DIR", sitecustomize)
        self.assertIn("install()", sitecustomize)
        self.assertIn("classify_path", module)
        self.assertIn("_record(\"write\"", module)
        self.assertIn("sqlite3.connect", module)
        self.assertIn("fsync", module)
        self.assertIn("os.fdopen", module)

    def test_path_classification(self) -> None:
        cases = {
            "/cb16/fast_hot/r4/x/scratch/run/observations/fact.json": "observation_store",
            "/cb16/fast_hot/r4/x/scratch/run/index.sqlite3-wal": "sqlite_index",
            "/cb16/fast_hot/r4/x/scratch/run/replay/material.json": "replay_materialization",
            "/cb16/fast_hot/r4/x/scratch/run/updates/" + "a" * 64 + ".json": "update_journal",
            "/cb16/fast_hot/r4/x/scratch/run/updates/checkpoints/" + "b" * 64 + ".json": "checkpoint_store",
            "/cb16/fast_hot/r4/x/scratch/run/updates/generation_switch_receipts/" + "b" * 64 + ".json": "generation_continuity",
            "/cb16/fast_hot/r4/x/output/provenance/index.json": "provenance",
            "/cb16/fast_hot/r4/x/output/artifacts/pack.json": "artifact_staging",
        }
        for path, expected in cases.items():
            self.assertEqual(classify_path(path), expected, path)

    def test_surface_report_arithmetic_and_access_classification(self) -> None:
        summary = {
            "update_journal": {
                "write": {"count": 20, "bytes": 2000, "duration_ns": 0, "extra": {}},
                "fsync": {"count": 10, "bytes": 0, "duration_ns": 5_000_000, "extra": {}},
                "replace": {"count": 10, "bytes": 0, "duration_ns": 0, "extra": {}},
            }
        }
        stats = _count_ops(summary, "update_journal")
        self.assertEqual(stats["write_calls"], 20)
        self.assertEqual(stats["bytes_written"], 2000)
        self.assertEqual(stats["fsync_calls"], 10)
        self.assertEqual(stats["replace_calls"], 10)
        report = _surface_report("update_journal", summary, updates=10, mount={"source": "/dev/sdb"})
        self.assertEqual(report["bytes_per_update"], 200)
        self.assertEqual(report["writes_per_update"], 2)
        self.assertEqual(report["durable_sync_frequency_per_update"], 1.0)
        self.assertEqual(report["physical_device"]["source"], "/dev/sdb")
        self.assertIn(report["random_versus_sequential"], {"RANDOM_OR_SMALL_FSYNC_HEAVY", "MIXED"})

    def test_access_classification_is_independent(self) -> None:
        self.assertEqual(_classify_access({"write_calls": 0, "bytes_written": 0, "fsync_calls": 0}), "NO_WRITES_OBSERVED")
        self.assertEqual(
            _classify_access({"write_calls": 10, "bytes_written": 10 * 1024, "fsync_calls": 10}),
            "RANDOM_OR_SMALL_FSYNC_HEAVY",
        )
        self.assertEqual(
            _classify_access({"write_calls": 1, "bytes_written": 8 * 1024 * 1024, "fsync_calls": 0}),
            "SEQUENTIAL_LARGE",
        )

    def test_measurement_status_fails_closed(self) -> None:
        self.assertEqual(measurement_status(0, 10, []), "PASS")
        self.assertEqual(measurement_status(1, 10, []), "EVIDENCE_INSUFFICIENT")
        self.assertEqual(measurement_status(0, 0, []), "EVIDENCE_INSUFFICIENT")
        self.assertEqual(measurement_status(0, 10, ["generation_continuity"]), "EVIDENCE_INSUFFICIENT")

    def test_committed_update_resolution_from_provenance_artifact(self) -> None:
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

    def test_committed_update_resolution_from_observed_paths(self) -> None:
        paths = {f"other|/run/updates/{'c' * 64}.json", f"other|/run/updates/{'d' * 64}.json"}
        with tempfile.TemporaryDirectory() as tmp:
            updates, source = _find_committed_updates(Path(tmp), paths)
        self.assertEqual(updates, 2)
        self.assertEqual(source, "observed_update_journal_ids")


if __name__ == "__main__":
    unittest.main()
