"""Static and hostile tests for the R21 RC2 R3 Recovery runner plan V2."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R3_TASK_LOCAL_CHANGE_PLAN_V2.json"
PROFILE_PATH = REPO_ROOT / "infra" / "shanxi_runner" / "runner_launch_spec_r21_v2.json"
APPROVAL_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R2_OWNER_APPROVAL_V2.json"
DIFF_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R2_EXPECTED_VS_ACTUAL_V1.json"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "cb16-r21-rc2-r3-recovery-runner-gate.yml"


def validate_v2_binding(plan: dict, profile: dict, approval: dict) -> None:
    if plan["owner_approval"] != "authority/infra/R21_RC2_R2_OWNER_APPROVAL_V2.json":
        raise AssertionError("V2 plan does not bind owner approval V2")
    if approval["selected_option_id"] != "D_COMBINED_RECLAIM_AND_MIGRATION":
        raise AssertionError("owner approval is not option D")
    revised = plan["revised_frozen_values"]
    for plan_key, approval_key in (
        ("fast_hot_quota_bytes", "fast_hot_quota_bytes"),
        ("reserve_floor_bytes", "reserve_floor_bytes"),
        ("fast_hot_host_path", None),
        ("fast_hot_container_path", None),
    ):
        if approval_key is None:
            continue
        if revised[plan_key] != approval[approval_key]:
            raise AssertionError(f"{plan_key} does not match owner approval V2")
    if revised["fast_hot_quota_bytes"] != 100000000000:
        raise AssertionError("V2 quota is not 100 GB")
    if revised["reserve_floor_bytes"] != 41785114624:
        raise AssertionError("V2 reserve floor changed")
    if profile["fast_hot"]["quota_bytes"] != revised["fast_hot_quota_bytes"]:
        raise AssertionError("profile quota does not match V2 plan")
    if profile["fast_hot"]["reserve_floor_bytes"] != revised["reserve_floor_bytes"]:
        raise AssertionError("profile reserve does not match V2 plan")
    if profile["run"]["shm_size_bytes"] != 2147483648:
        raise AssertionError("profile shm size is not 2 GiB")
    if profile["identity"]["s1_scientific_manifest_changed"]:
        raise AssertionError("profile claims an S1 scientific manifest change")
    if profile["identity"]["s1_runtime_code_changed"]:
        raise AssertionError("profile claims an S1 runtime code change")
    if profile["identity"]["runtime_acceptance_inherited"]:
        raise AssertionError("R3 profile must not inherit old runtime acceptance")


class RecoveryRunnerPlanV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.approval = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
        cls.diff_record = json.loads(DIFF_PATH.read_text(encoding="utf-8"))
        cls.workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_v2_schemas_and_supersession(self) -> None:
        self.assertEqual(self.plan["schema"], "CB16_R21_RC2_R3_TASK_LOCAL_CHANGE_PLAN_V2")
        self.assertEqual(self.plan["status"], "FROZEN_FOR_EXECUTION_OWNER_APPROVED")
        self.assertEqual(self.plan["supersedes"], "authority/infra/R21_RC2_R3_TASK_LOCAL_CHANGE_PLAN_V1.json")
        self.assertEqual(self.profile["schema"], "CB16_SHANXI_RUNNER_LAUNCH_SPEC_R21_V2")
        self.assertEqual(self.profile["profile_id"], "R21_RC2_FAST_HOT_RUNNER_V2")
        self.assertEqual(self.profile["supersedes"], "infra/shanxi_runner/runner_launch_spec_r21_v1.json")

    def test_owner_approval_v2_and_plan_are_bound(self) -> None:
        validate_v2_binding(self.plan, self.profile, self.approval)
        self.assertEqual(self.approval["approved_by_owner"], "bgy")
        self.assertEqual(
            self.approval["projected_free_after_quota_and_reserve_bytes"],
            self.approval["projected_free_after_remaining_migration_bytes"]
            - self.approval["fast_hot_quota_bytes"]
            - self.approval["reserve_floor_bytes"],
        )

    def test_v2_operations_cover_migration_fast_hot_and_runner(self) -> None:
        self.assertEqual(
            [operation["operation_id"] for operation in self.plan["operations"]],
            [
                "MIG-03",
                "MIG-04",
                "MIG-05",
                "FH-01",
                "FH-02",
                "RUN-01",
                "RUN-02",
                "RUN-03",
                "RUN-04",
                "RUN-05",
                "GATE-01",
                "EVID-01",
            ],
        )
        for operation in self.plan["operations"]:
            if operation["operation_id"] == "MIG-04":
                self.assertEqual(operation.get("same_rule_as"), "MIG-03")
            if operation["operation_id"].startswith("MIG-"):
                self.assertIn("SHA256", json.dumps(operation).upper())

    def test_profile_v2_identity_fast_hot_and_protected_inputs(self) -> None:
        container = self.profile["container"]
        self.assertEqual(container["name"], "cb16-runner-r21")
        self.assertEqual(container["hostname"], "cb16-runner-r21")
        self.assertEqual(container["dedicated_github_runner_name"], "shanxi-docker-r21")
        self.assertEqual(
            container["dedicated_github_runner_labels"],
            ["self-hosted", "Linux", "X64", "shanxi-docker-r21"],
        )
        self.assertEqual(self.profile["fast_hot"]["quota_bytes"], 100000000000)
        self.assertEqual(self.profile["fast_hot"]["reserve_floor_bytes"], 41785114624)
        self.assertFalse(self.profile["fast_hot"]["hdd_fallback_allowed"])
        mounts = {entry["container_path"]: entry for entry in self.profile["mounts"]}
        fast_hot = mounts["/cb16/fast_hot"]
        self.assertEqual(fast_hot["host_path"], "/srv/cb16_fast_hot")
        self.assertTrue(fast_hot["mount_rw"])
        self.assertEqual(fast_hot["storage_class"], "SSD_NON_ROTATIONAL")
        for path in ("/cb16/raw", "/cb16/runtime/r104", "/cb16/package", "/cb16/parent-r101", "/cb16/parents", "/cb16/r2-authority", "/run/secrets"):
            self.assertTrue(mounts[path]["protected_input"], path)
            self.assertFalse(mounts[path]["mount_rw"], path)

    def test_profile_v2_resolves_r1_divergence_without_touching_r11(self) -> None:
        env = {entry["name"]: entry for entry in self.profile["environment"]}
        self.assertEqual(env["CB16_PROVISION_ENV"]["value"], "/cb16/worker/provision.env")
        self.assertEqual(env["CB16_FAST_HOT_ROOT"]["value"], "/cb16/fast_hot")
        self.assertEqual(env["CB16_RUNNER_PROFILE_ID"]["value"], "R21_RC2_FAST_HOT_RUNNER_V2")
        self.assertIn("without modifying", env["CB16_PROVISION_ENV"]["note"])

    def test_v2_plan_forbids_unsafe_operations_and_retains_deviations(self) -> None:
        forbidden = " ".join(self.plan["scope"]["forbidden"])
        for token in ("formatting", "cb16-runner-r11", "HDD as a FAST_HOT fallback", "reserve floor", "before SHA256 verification passes"):
            self.assertIn(token, forbidden)
        deviations = {entry["operation_id"]: entry for entry in self.plan["known_deviations_retained"]}
        self.assertIn("CONTRACT_MISMATCH", deviations["CLN-03"]["classification"])
        self.assertEqual(deviations["MIG-02"]["classification"], "EXECUTOR_CHANGED_TO_ROOT_HELPER")

    def test_expected_vs_actual_v2_target_is_met(self) -> None:
        self.assertFalse(self.diff_record["classification"]["option_d_v1_target_met"])
        self.assertTrue(self.diff_record["classification"]["revised_v2_target_met_if_remaining_migration_completes"])
        self.assertEqual(self.diff_record["owner_revision"]["fast_hot_quota_bytes"], 100000000000)
        self.assertEqual(
            self.diff_record["deltas"]["projected_free_after_revised_quota_and_reserve_bytes"],
            self.approval["projected_free_after_quota_and_reserve_bytes"],
        )

    def test_v2_gate_workflow_targets_new_runner_and_uses_v2_verifier(self) -> None:
        self.assertIn("self-hosted, shanxi-docker-r21", self.workflow_text)
        self.assertIn("uses: ./.github/workflows/_cb16-shanxi-preflight.yml", self.workflow_text)
        self.assertIn("needs: shanxi-preflight", self.workflow_text)
        self.assertIn("verify_r21_rc2_r3_recovery_runner_v2.py", self.workflow_text)
        self.assertIn("tests/test_r21_rc2_r3_recovery_runner_v2.py", self.workflow_text)
        self.assertIn("cb16-r21-rc2-r3-recovery-runner-gate", self.workflow_text)

    def test_hostile_v2_quota_mutation_is_rejected(self) -> None:
        mutated = json.loads(json.dumps(self.plan))
        mutated["revised_frozen_values"]["fast_hot_quota_bytes"] += 1
        with self.assertRaises(AssertionError):
            validate_v2_binding(mutated, self.profile, self.approval)


if __name__ == "__main__":
    unittest.main()
