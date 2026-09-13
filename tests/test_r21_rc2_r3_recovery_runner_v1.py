"""Static and hostile tests for the R21 RC2 R3 Recovery runner plan."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R3_TASK_LOCAL_CHANGE_PLAN_V1.json"
PROFILE_PATH = REPO_ROOT / "infra" / "shanxi_runner" / "runner_launch_spec_r21_v1.json"
APPROVAL_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R2_OWNER_APPROVAL_V1.json"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "cb16-r21-rc2-r3-recovery-runner-gate.yml"


def validate_plan_binding(plan: dict, profile: dict, approval: dict) -> None:
    if plan["owner_approval"] != "authority/infra/R21_RC2_R2_OWNER_APPROVAL_V1.json":
        raise AssertionError("plan does not bind the owner approval path")
    if approval["selected_option_id"] != "D_COMBINED_RECLAIM_AND_MIGRATION":
        raise AssertionError("owner approval is not option D")
    frozen = plan["frozen_values"]
    if frozen["fast_hot_quota_bytes"] != approval["fast_hot_quota_bytes"]:
        raise AssertionError("plan quota does not match owner approval")
    if frozen["reserve_floor_bytes"] != approval["reserve_floor_bytes"]:
        raise AssertionError("plan reserve does not match owner approval")
    if frozen["fast_hot_host_path"] != profile["fast_hot"]["host_path"]:
        raise AssertionError("profile host path does not match plan")
    if frozen["fast_hot_container_path"] != profile["fast_hot"]["container_path"]:
        raise AssertionError("profile container path does not match plan")
    if profile["fast_hot"]["quota_bytes"] != approval["fast_hot_quota_bytes"]:
        raise AssertionError("profile quota does not match owner approval")
    if profile["run"]["shm_size_bytes"] != 2147483648:
        raise AssertionError("profile shm size is not 2 GiB")
    if profile["identity"]["s1_scientific_manifest_changed"]:
        raise AssertionError("profile claims an S1 scientific manifest change")
    if profile["identity"]["s1_runtime_code_changed"]:
        raise AssertionError("profile claims an S1 runtime code change")
    if profile["identity"]["runtime_acceptance_inherited"]:
        raise AssertionError("R3 profile must not inherit old runtime acceptance")


class RecoveryRunnerPlanV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.approval = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
        cls.workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_schemas_and_frozen_execution_state(self) -> None:
        self.assertEqual(self.plan["schema"], "CB16_R21_RC2_R3_TASK_LOCAL_CHANGE_PLAN_V1")
        self.assertEqual(self.plan["status"], "FROZEN_FOR_EXECUTION_OWNER_APPROVED")
        self.assertEqual(self.profile["schema"], "CB16_SHANXI_RUNNER_LAUNCH_SPEC_R21_V1")
        self.assertEqual(self.profile["profile_id"], "R21_RC2_FAST_HOT_RUNNER_V1")

    def test_owner_approval_and_plan_are_bound(self) -> None:
        validate_plan_binding(self.plan, self.profile, self.approval)

    def test_operations_cover_reclaim_migration_fast_hot_and_runner(self) -> None:
        operation_ids = [operation["operation_id"] for operation in self.plan["operations"]]
        self.assertEqual(
            operation_ids,
            [
                "CLN-01",
                "CLN-02",
                "CLN-03",
                "CLN-04",
                "CLN-05",
                "MIG-01",
                "MIG-02",
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
                "EVID-01",
            ],
        )

    def test_profile_uses_dedicated_r21_runner_identity(self) -> None:
        container = self.profile["container"]
        self.assertEqual(container["name"], "cb16-runner-r21")
        self.assertEqual(container["hostname"], "cb16-runner-r21")
        self.assertEqual(container["dedicated_github_runner_name"], "shanxi-docker-r21")
        self.assertEqual(
            container["dedicated_github_runner_labels"],
            ["self-hosted", "Linux", "X64", "shanxi-docker-r21"],
        )
        self.assertEqual(self.profile["run"]["shm_size_bytes"], 2147483648)
        self.assertEqual(self.profile["fast_hot"]["hdd_fallback_allowed"], False)

    def test_profile_has_fast_hot_mount_and_protected_inputs_read_only(self) -> None:
        mounts = {entry["container_path"]: entry for entry in self.profile["mounts"]}
        fast_hot = mounts["/cb16/fast_hot"]
        self.assertEqual(fast_hot["host_path"], "/srv/cb16_fast_hot")
        self.assertTrue(fast_hot["mount_rw"])
        self.assertEqual(fast_hot["storage_class"], "SSD_NON_ROTATIONAL")
        for path in ("/cb16/raw", "/cb16/runtime/r104", "/cb16/package", "/cb16/parent-r101", "/cb16/parents", "/cb16/r2-authority", "/run/secrets"):
            self.assertTrue(mounts[path]["protected_input"], path)
            self.assertFalse(mounts[path]["mount_rw"], path)

    def test_profile_resolves_r1_divergence_without_touching_r11(self) -> None:
        env = {entry["name"]: entry for entry in self.profile["environment"]}
        self.assertEqual(env["CB16_PROVISION_ENV"]["value"], "/cb16/worker/provision.env")
        resolution = self.plan["frozen_values"]["provision_env_resolution"]
        self.assertEqual(resolution["live_r11_value"], "/run/secrets/cb16-provision.env")
        self.assertIn("does not exist", resolution["reason"])
        self.assertIn("not modified", resolution["reason"])
        self.assertEqual(env["CB16_FAST_HOT_ROOT"]["value"], "/cb16/fast_hot")
        self.assertEqual(env["CB16_RUNNER_PROFILE_ID"]["value"], "R21_RC2_FAST_HOT_RUNNER_V1")

    def test_plan_forbids_unsafe_operations(self) -> None:
        forbidden = " ".join(self.plan["scope"]["forbidden"])
        for token in ("formatting", "cb16-runner-r11", "HDD as a FAST_HOT fallback", "scientific payloads", "before copy and integrity verification"):
            self.assertIn(token, forbidden)
        self.assertEqual(self.plan["scope"]["host_root_access"]["sudo_nopasswd"], False)
        self.assertIn("docker group", self.plan["scope"]["host_root_access"]["method"])

    def test_migration_operations_require_copy_verify_then_remove(self) -> None:
        for operation in self.plan["operations"]:
            if operation["operation_id"].startswith("MIG-"):
                if operation["operation_id"] == "MIG-01":
                    self.assertIn("checksum", json.dumps(operation).lower())
                else:
                    self.assertEqual(operation.get("same_rule_as"), "MIG-01")

    def test_gate_workflow_targets_dedicated_runner_and_has_preflight(self) -> None:
        self.assertIn("self-hosted, shanxi-docker-r21", self.workflow_text)
        self.assertIn("uses: ./.github/workflows/_cb16-shanxi-preflight.yml", self.workflow_text)
        self.assertIn("needs: shanxi-preflight", self.workflow_text)
        self.assertIn("verify_r21_rc2_r3_recovery_runner_v1.py", self.workflow_text)
        self.assertIn("cb16-r21-rc2-r3-recovery-runner-gate", self.workflow_text)

    def test_hostile_quota_mutation_is_rejected(self) -> None:
        mutated = json.loads(json.dumps(self.plan))
        mutated["frozen_values"]["fast_hot_quota_bytes"] += 1
        with self.assertRaises(AssertionError):
            validate_plan_binding(mutated, self.profile, self.approval)


if __name__ == "__main__":
    unittest.main()
