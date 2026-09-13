"""Static contract tests for the R21 RC2 Recovery spec.

These tests are deliberately repository-only and do not perform host access,
runtime execution or scientific qualification. They bind the machine-readable
Recovery contract to the frozen values already defined by the R21 RC2 TODO and
the shared qualification verdict taxonomy.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from cb16_local_opt.scientific_qualification_contract_v1 import ALLOWED_VERDICTS_V1

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_RECOVERY_SPEC_V1.json"


class RecoverySpecV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.tasks = {task["task_id"]: task for task in cls.spec["task_registry"]}

    def test_schema_and_pending_review_status(self) -> None:
        self.assertEqual(self.spec["schema"], "CB16_R21_RC2_RECOVERY_SPEC_V1")
        self.assertEqual(self.spec["spec_version"], "V1")
        self.assertEqual(self.spec["status"], "DS_PROPOSED_PENDING_SOL_REVIEW")
        self.assertEqual(self.spec["authored_by_role"], "DS_FLASH")
        self.assertEqual(self.spec["review_owner_role"], "SOL_REVIEWER")

    def test_recovery_sequence_and_task_registry_are_complete(self) -> None:
        expected = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
        self.assertEqual(self.spec["scope"]["recovery_sequence"], expected)
        self.assertEqual(list(self.tasks), expected)
        for task in self.tasks.values():
            for field in (
                "title",
                "purpose",
                "suggested_branch",
                "depends_on",
                "blocking_gate_ids",
                "github_actions_required",
                "host_session",
                "host_container_change",
                "owner_approval",
                "new_runtime_implementation_identity",
                "required_artifact_ids",
                "gate",
                "scope_boundary",
            ):
                self.assertIn(field, task, f"{task['task_id']} missing {field}")

    def test_task_dependencies_follow_authorized_sequence(self) -> None:
        self.assertEqual(self.tasks["R0"]["depends_on"], [])
        self.assertEqual(self.tasks["R1"]["depends_on"], ["R0"])
        self.assertEqual(self.tasks["R2"]["depends_on"], ["R1"])
        self.assertEqual(self.tasks["R3"]["depends_on"], ["R0", "R1", "R2"])
        self.assertEqual(self.tasks["R4"]["depends_on"], ["R3"])
        self.assertEqual(self.tasks["R5"]["depends_on"], ["R4"])
        self.assertEqual(self.tasks["R6"]["depends_on"], ["R5"])
        self.assertEqual(self.tasks["R7"]["depends_on"], ["R5"])
        self.assertEqual(self.tasks["R7"]["conditional_dependency"], "R6 if R6 is opened")
        self.assertEqual(self.tasks["R8"]["depends_on"], ["R7"])

    def test_blocking_gate_references_resolve(self) -> None:
        gates = {gate["gate_id"]: gate for gate in self.spec["blocking_gates"]}
        self.assertEqual(
            self.tasks["R1"]["blocking_gate_ids"],
            ["R1_SOL_ACCEPTANCE_OF_R0"],
        )
        self.assertEqual(self.tasks["R3"]["blocking_gate_ids"], ["R2_OWNER_APPROVAL"])
        self.assertEqual(self.tasks["R6"]["blocking_gate_ids"], ["R5_TARGET_MISS_SOL_REVIEW"])
        self.assertEqual(
            self.tasks["R7"]["blocking_gate_ids"],
            ["R6_SOL_ACCEPTANCE", "R7_HOST_AUTHORIZATION"],
        )
        for task in self.tasks.values():
            for gate_id in task["blocking_gate_ids"]:
                self.assertIn(gate_id, gates, f"{task['task_id']} references {gate_id}")
                self.assertTrue(gates[gate_id]["condition"])

    def test_authority_matrix_values_match_todo_v2(self) -> None:
        expected = {
            "R0": (False, "NONE", False, False, False, False),
            "R1": (True, "READ_ONLY", False, False, False, False),
            "R2": (False, "READ_ONLY", False, False, True, False),
            "R3": (True, "CHANGE_REQUIRED", True, True, False, False),
            "R4": (True, "READ_ONLY_IF_HOST_TELEMETRY_REQUIRED", False, False, False, False),
            "R5": (True, "READ_ONLY_IF_HOST_TELEMETRY_REQUIRED", False, False, False, False),
            "R6": (True, "NONE_NORMALLY", False, False, False, True),
            "R7": (
                True,
                "TASK_LIMITED_CHANGE_ONLY_FOR_CONTAINER_LEVEL_RECOVERY",
                True,
                False,
                False,
                "DEPENDS_ON_R6",
            ),
            "R8": (False, "EVIDENCE_READ", False, False, False, False),
        }
        for task_id, values in expected.items():
            task = self.tasks[task_id]
            github_actions, host_session, host_change, owner_before, owner_before_dependent, new_identity = values
            self.assertEqual(task["github_actions_required"], github_actions)
            self.assertEqual(task["host_session"], host_session)
            self.assertEqual(task["host_container_change"], host_change)
            self.assertEqual(task["owner_approval"]["before_execution"], owner_before)
            self.assertEqual(task["owner_approval"]["before_dependent_change"], owner_before_dependent)
            self.assertEqual(task["new_runtime_implementation_identity"], new_identity)

    def test_r0_r1_r2_authorized_before_later_host_change(self) -> None:
        self.assertEqual(self.spec["immediate_authorized_order"], ["R0", "R1", "R2"])
        self.assertFalse(self.tasks["R0"]["host_container_change"])
        self.assertFalse(self.tasks["R1"]["host_container_change"])
        self.assertFalse(self.tasks["R2"]["host_container_change"])
        self.assertTrue(self.tasks["R3"]["host_container_change"])
        self.assertTrue(self.tasks["R3"]["owner_approval"]["before_execution"])
        self.assertTrue(self.tasks["R2"]["owner_approval"]["before_dependent_change"])

    def test_performance_target_set_has_frozen_values(self) -> None:
        perf = self.spec["performance_targets"]
        self.assertEqual(perf["target_set_id"], "PERF-RC2-V1")
        self.assertTrue(perf["frozen_before_storage_benchmarking"])
        self.assertEqual(perf["classification"], "ENGINEERING_TARGETS_NOT_SCIENTIFIC_GATES")
        self.assertEqual(perf["unmeasurable_metric_verdict"], "EVIDENCE_INSUFFICIENT")
        by_id = {target["metric_id"]: target for target in perf["targets"]}
        expected = {
            "FULL_S1_CI_C_WALL_TIME": ("<=", 240, "minutes"),
            "EIGHT_WORKER_STEADY_STATE_BACKLOG_GROWTH": (
                "NO_SUSTAINED_POSITIVE_GROWTH",
                True,
                "boolean",
            ),
            "POST_PRODUCER_BACKLOG_DRAIN_TIME": ("<=", 60, "seconds"),
            "HOT_DURABLE_BOUNDARY_LATENCY_P50": ("<=", 50, "milliseconds"),
            "HOT_DURABLE_BOUNDARY_LATENCY_P99": ("<=", 250, "milliseconds"),
            "PROCESS_LEVEL_RECOVERY_TIME": ("<=", 60, "seconds"),
            "CONTAINER_LEVEL_RECOVERY_TIME": ("<=", 180, "seconds"),
            "FAST_HOT_NON_ROTATIONAL_PLACEMENT": ("MUST_HOLD", True, "boolean"),
            "ACKNOWLEDGED_LOGICAL_EVENT_LOSSES": ("==", 0, "events"),
            "DUPLICATE_EFFECTIVE_LEARNER_UPDATES": ("==", 0, "updates"),
            "CHECKPOINT_PROVENANCE_EQUIVALENCE": ("REQUIRED", True, "boolean"),
        }
        self.assertEqual(set(by_id), set(expected))
        for metric_id, (constraint, value, unit) in expected.items():
            self.assertEqual(by_id[metric_id]["constraint"], constraint)
            self.assertEqual(by_id[metric_id]["value"], value)
            self.assertEqual(by_id[metric_id]["unit"], unit)
        self.assertEqual(
            by_id["FAST_HOT_NON_ROTATIONAL_PLACEMENT"]["failure_verdict"],
            "CONTRACT_MISMATCH",
        )

    def test_fault_class_taxonomy_and_promotion_rule(self) -> None:
        fault_classes = self.spec["fault_classes"]
        self.assertEqual(
            [entry["fault_class"] for entry in fault_classes["taxonomy"]],
            ["PROCESS_CRASH", "CONTAINER_RESTART", "HOST_REBOOT", "POWER_LOSS"],
        )
        default_claims = fault_classes["recovery_default_testable_classes"]
        self.assertEqual(default_claims, ["PROCESS_CRASH", "CONTAINER_RESTART"])
        not_implied = fault_classes["stronger_classes_not_implied_by_default"]
        self.assertEqual(not_implied, ["HOST_REBOOT", "POWER_LOSS"])
        self.assertIn("must not be promoted", fault_classes["promotion_rule"])
        by_class = {entry["fault_class"]: entry for entry in fault_classes["taxonomy"]}
        self.assertTrue(by_class["PROCESS_CRASH"]["recovery_pass_claim_allowed_by_default"])
        self.assertTrue(by_class["CONTAINER_RESTART"]["recovery_pass_claim_allowed_by_default"])
        self.assertFalse(by_class["HOST_REBOOT"]["recovery_pass_claim_allowed_by_default"])
        self.assertFalse(by_class["POWER_LOSS"]["recovery_pass_claim_allowed_by_default"])
        self.assertTrue(by_class["HOST_REBOOT"]["separate_operational_authorization_required"])
        self.assertTrue(by_class["POWER_LOSS"]["separate_operational_authorization_required"])

    def test_identity_rules_preserve_scientific_identity_and_bound_implementation_identity(self) -> None:
        rules = self.spec["identity_rules"]
        self.assertFalse(rules["scientific_identity_frozen"]["changed_by_recovery"])
        self.assertEqual(
            rules["new_implementation_identity_triggers"],
            [
                "storage-engine replacement",
                "commit-protocol rewrite",
                "SQLite batch-writer rewrite",
                "replay-visibility code change",
            ],
        )
        for surface in ("mount placement", "shared-memory sizing", "resource profile", "runner profile"):
            self.assertIn(surface, rules["execution_surface_changes_preserving_runtime_code_identity"])
        self.assertTrue(rules["inheritance_rule"]["scientific_manifest_remains_frozen_in_all_cases"])
        self.assertIn("R6", rules["inheritance_rule"]["runtime_code_acceptance_not_inherited_when"])

    def test_verdict_taxonomy_reuses_shared_primitive(self) -> None:
        verdicts = self.spec["verdict_taxonomy"]["allowed_verdicts"]
        self.assertEqual(tuple(verdicts), ALLOWED_VERDICTS_V1)
        self.assertTrue(self.spec["verdict_taxonomy"]["no_partial_pass"])
        rules = {entry["verdict"]: entry["rule"] for entry in self.spec["verdict_taxonomy"]["classification_rules"]}
        self.assertIn("never be relabeled SCIENTIFIC_FAIL", rules["SCIENTIFIC_FAIL"])
        self.assertIn("EVIDENCE_INSUFFICIENT", self.spec["performance_targets"]["unmeasurable_metric_verdict"])

    def test_artifact_references_resolve_and_cover_all_tasks(self) -> None:
        registry = self.spec["artifact_registry"]
        per_task = self.spec["required_receipts_and_artifacts"]["per_task_required_artifact_ids"]
        self.assertEqual(list(per_task), ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"])
        for task_id, artifact_ids in per_task.items():
            self.assertTrue(artifact_ids, task_id)
            self.assertEqual(set(artifact_ids), set(self.tasks[task_id]["required_artifact_ids"]))
            for artifact_id in artifact_ids:
                self.assertIn(artifact_id, registry, artifact_id)
                self.assertEqual(registry[artifact_id]["task_id"], task_id)
                self.assertTrue(registry[artifact_id]["required"])

    def test_receipt_fields_and_host_change_receipt_rules(self) -> None:
        receipt_contract = self.spec["required_receipts_and_artifacts"]
        for field in (
            "task_id",
            "head_sha",
            "tree_sha",
            "host_session_mode",
            "host_changes_applied",
            "github_actions_run_id",
            "github_actions_job_id",
            "artifact_ids",
            "verdict",
            "scientific_constants_changed",
        ):
            self.assertIn(field, receipt_contract["common_receipt_fields"])
        rules = " ".join(receipt_contract["rules"])
        self.assertIn("host change", rules)
        self.assertIn("before/after", rules)
        self.assertIn("scientific_constants_changed", rules)

    def test_r6_is_closed_by_default_and_requires_sol_trigger(self) -> None:
        task = self.tasks["R6"]
        self.assertTrue(task["closed_by_default"])
        self.assertTrue(task["new_runtime_implementation_identity"])
        self.assertIn("Sol explicitly opens R6", task["open_trigger"])
        self.assertIn("R6_SOL_IMPLEMENTATION_REVIEW_V1", task["required_artifact_ids"])

    def test_r7_does_not_claim_stronger_fault_classes(self) -> None:
        task = self.tasks["R7"]
        self.assertEqual(task["testable_fault_classes"], ["PROCESS_CRASH", "CONTAINER_RESTART"])
        self.assertEqual(task["not_claimed_without_separate_authorization"], ["HOST_REBOOT", "POWER_LOSS"])
        self.assertIn("task_local_authorization_required", task["owner_approval"])
        self.assertTrue(task["owner_approval"]["task_local_authorization_required"])

    def test_s1_relationship_keeps_pr_102_frozen_until_r8(self) -> None:
        relationship = self.spec["scope"]["s1_relationship"]
        self.assertTrue(relationship["recovery_may_not_change_s1_scientific_constants"])
        self.assertEqual(relationship["s1_ci_c_allowed_after"], "R8")
        self.assertIn("FROZEN", relationship["pr_102_state"])
        self.assertFalse(self.spec["frozen_scientific_identity"]["changed_by_recovery"])

    def test_stop_conditions_cover_fail_closed_paths(self) -> None:
        joined = " ".join(self.spec["stop_conditions"])
        for token in (
            "EXECUTION_BLOCKED",
            "EVIDENCE_INSUFFICIENT",
            "CONTRACT_MISMATCH",
            "SCIENTIFIC_FAIL",
            "READY_FOR_SOL_REVIEW",
        ):
            self.assertIn(token, joined)


if __name__ == "__main__":
    unittest.main()
