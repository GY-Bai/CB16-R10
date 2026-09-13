"""Static and hostile tests for the R21 RC2 R5 placement canary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_r21_rc2_r5_placement_canary_v1 import normalize_canary_outputs

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R5_PLACEMENT_CANARY_SPEC_V1.json"
REFERENCE_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R5_ACCEPTED_CANARY_REFERENCE_V1.json"


def _write_synthetic_canary(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SMOKE_RESULT.json").write_text(
        json.dumps(
            {
                "schema": "CB16_R11_POST_CC_S1_SMOKE_RESULT_V1",
                "mode": "smoke",
                "status": "SMOKE_ONLY_NOT_SCIENTIFIC_QUALIFICATION",
                "evidence_class": "SMOKE_ONLY_ENGINEERING_EVIDENCE",
                "scientific_verdict_allowed": False,
                "executed_jobs": 1,
                "positive_runs": 1,
                "control_runs": 0,
                "integrity_attack_suite_all_rejected": True,
                "objective_firewall_audit_pass": True,
                "fabricated_log_mu_audit_pass": True,
                "high_bankruptcy_failure_fact_audit_pass": True,
                "artifact_only_update_trace_all_checks_pass": True,
            }
        ),
        encoding="utf-8",
    )
    (root / "execution_manifest.json").write_text(json.dumps({"manifest_sha256": "a" * 64}), encoding="utf-8")
    provenance = root / "provenance"
    provenance.mkdir()
    (provenance / "artifact_only_update_trace_audit.json").write_text(
        json.dumps(
            {
                "schema": "CB16_R11_POST_CC_S1_ARTIFACT_ONLY_UPDATE_TRACE_AUDIT_V1",
                "all_checks_pass": True,
                "checks": {"checkpoint_chain_verified": True},
                "traced_update_count": 1,
                "runs_audited": 1,
                "failures": [],
                "details": {"runs": [{"committed_updates": 1, "failures": []}]},
            }
        ),
        encoding="utf-8",
    )
    (provenance / "run_index.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "task_id": "ACCOUNT_DEPENDENT_ACTION",
                        "seed": 1701,
                        "control_id": None,
                        "decisions_consumed": 16,
                        "evidence_class": "SMOKE_ONLY_ENGINEERING_EVIDENCE",
                        "final_child_checkpoint_sha256": "b" * 64,
                        "final_score": 1.0,
                        "initial_score": 0.0,
                        "oracle_score": 2.0,
                        "unit_evidence_sha256s": ["c" * 64],
                        "run_root": "/path/that/must/not/matter",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (provenance / "task_specs.json").write_text(json.dumps({"task_specs": {"ACCOUNT_DEPENDENT_ACTION": {}}}), encoding="utf-8")


class PlacementCanaryV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

    def test_spec_identity_and_decision_rules(self) -> None:
        self.assertEqual(self.spec["schema"], "CB16_R21_RC2_R5_PLACEMENT_CANARY_SPEC_V1")
        self.assertEqual(self.spec["s1_runtime_identity"]["authorization_head_sha"], "a974e2803ccc2693d67a0375e460da35837636a4")
        self.assertEqual(self.spec["execution"]["workers"], 8)
        self.assertFalse(self.spec["execution"]["s1_runtime_changes_allowed"])
        self.assertFalse(self.spec["execution"]["scientific_manifest_changes_allowed"])
        rules = self.spec["decision_rule"]
        self.assertEqual(rules["semantic_mismatch"], "CONTRACT_MISMATCH")
        self.assertIn("SOL_REVIEW_REQUIRED", rules["semantic_match_but_perf_gate_not_measured"])

    def test_accepted_reference_matches_baseline_identity(self) -> None:
        reference = self.reference
        self.assertEqual(reference["schema"], "CB16_R21_RC2_R5_ACCEPTED_CANARY_REFERENCE_V1")
        self.assertEqual(reference["normalized"]["manifest_sha256"], "b5f2bb868870aa392e9006f58520aad63606c5971fa4e9144050326d96d3c532")
        self.assertTrue(reference["normalized"]["smoke_result"]["artifact_only_update_trace_all_checks_pass"])
        self.assertEqual(reference["normalized"]["artifact_trace"]["traced_update_count"], 52)
        self.assertEqual(len(reference["normalized"]["run_index"]), 13)
        self.assertEqual(len(reference["semantic_digest"]), 64)

    def test_normalizer_is_path_independent_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a"
            second = Path(tmp) / "b"
            _write_synthetic_canary(first)
            _write_synthetic_canary(second)
            normalized_a, digest_a = normalize_canary_outputs(first)
            normalized_b, digest_b = normalize_canary_outputs(second)
            self.assertEqual(normalized_a, normalized_b)
            self.assertEqual(digest_a, digest_b)
            self.assertNotIn("run_root", json.dumps(normalized_a))

    def test_hostile_semantic_mutation_changes_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "a"
            _write_synthetic_canary(root)
            _, digest_a = normalize_canary_outputs(root)
            smoke_path = root / "SMOKE_RESULT.json"
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            smoke["artifact_only_update_trace_all_checks_pass"] = False
            smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
            _, digest_b = normalize_canary_outputs(root)
            self.assertNotEqual(digest_a, digest_b)

    def test_missing_perf_gate_is_declared(self) -> None:
        gates = {gate["gate_id"]: gate for gate in self.spec["gates"]}
        self.assertEqual(gates["NO_SUSTAINED_BACKLOG_GROWTH"]["measurement_status"], "NOT_MEASURED_IN_BOUNDED_SMOKE")
        self.assertEqual(gates["HOT_DURABLE_BOUNDARY_LATENCY_P50_P99"]["verdict"], "EVIDENCE_INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
