"""Static and hostile tests for the R21 RC2 R2 SSD plan."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R2_SSD_CAPACITY_INVENTORY_V1.json"
PROPOSAL_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R2_FAST_HOT_PROVISION_PROPOSAL_V1.json"
APPROVAL_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R2_OWNER_APPROVAL_V1.json"
APPROVAL_V2_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R2_OWNER_APPROVAL_V2.json"
DIFF_PATH = REPO_ROOT / "authority" / "infra" / "R21_RC2_R2_EXPECTED_VS_ACTUAL_V1.json"
DECISION_PATH = REPO_ROOT / "docs" / "infra" / "R21_RC2_R2_OWNER_DECISION_REQUEST.md"

NEVER_DELETE_VOLUMES = {
    "cb16-raw",
    "cb16-runtime-r104",
    "cb16-package",
    "cb16-parent-r101",
    "cb16-parent-g0",
    "cb16-r2-authority",
    "cb16-venv",
    "cb16-config",
    "cb16-runner-docker",
    "cb16-worker",
}


def validate_proposal_arithmetic(inventory: dict, proposal: dict) -> None:
    current_free = inventory["filesystems"]["ssd_root_lv"]["free_bytes"]
    if proposal["current_state"]["ssd_root_lv_free_bytes"] != current_free:
        raise AssertionError("current_state free bytes do not match inventory")
    candidates = {entry["candidate_id"]: entry for entry in inventory["reclaim_candidates"]}
    for option in proposal["options"]:
        candidate_ids = option["candidate_ids"]
        selected = option["selected_bytes_by_candidate"]
        if set(selected) != set(candidate_ids):
            raise AssertionError(f"{option['option_id']} selected map does not match candidate ids")
        for candidate_id in candidate_ids:
            if candidate_id not in candidates:
                raise AssertionError(f"{option['option_id']} references unknown {candidate_id}")
            if selected[candidate_id] > candidates[candidate_id]["used_bytes"]:
                raise AssertionError(f"{option['option_id']} selects more bytes than recorded for {candidate_id}")
        if sum(selected.values()) != option["reclaimed_or_migrated_bytes"]:
            raise AssertionError(f"{option['option_id']} reclaimed bytes do not sum")
        expected_free = current_free + option["reclaimed_or_migrated_bytes"]
        if expected_free != option["projected_root_free_bytes"]:
            raise AssertionError(f"{option['option_id']} projected free does not match arithmetic")
        if option.get("new_hardware_required"):
            if option["reserve_floor_bytes"] != option["projected_root_free_bytes"]:
                raise AssertionError(f"{option['option_id']} root reserve does not match projected root free")
        elif option["proposed_fast_hot_quota_bytes"] + option["reserve_floor_bytes"] != option["projected_root_free_bytes"]:
            raise AssertionError(f"{option['option_id']} quota plus reserve does not equal projected free")
        if option["reserve_floor_bytes"] < 0 or option["proposed_fast_hot_quota_bytes"] <= 0:
            raise AssertionError(f"{option['option_id']} has invalid negative/zero allocation")


class SSDPlanV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        cls.proposal = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
        cls.approval = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
        cls.approval_v2 = json.loads(APPROVAL_V2_PATH.read_text(encoding="utf-8"))
        cls.expected_vs_actual = json.loads(DIFF_PATH.read_text(encoding="utf-8"))
        cls.decision_text = DECISION_PATH.read_text(encoding="utf-8")

    def test_schemas_and_pending_owner_state(self) -> None:
        self.assertEqual(self.inventory["schema"], "CB16_R21_RC2_R2_SSD_CAPACITY_INVENTORY_V1")
        self.assertEqual(self.proposal["schema"], "CB16_R21_RC2_R2_FAST_HOT_PROVISION_PROPOSAL_V1")
        self.assertEqual(self.inventory["status"], "DS_PROPOSED_PENDING_OWNER_APPROVAL")
        self.assertEqual(self.proposal["status"], "DS_PROPOSED_PENDING_OWNER_APPROVAL")
        self.assertTrue(self.inventory["owner_approval"]["required_before_r3"])
        self.assertEqual(self.inventory["owner_approval"]["status"], "PENDING_OWNER_DECISION")
        self.assertEqual(self.proposal["owner_approval_record_template"]["status"], "NOT_YET_APPROVED")

    def test_r2_is_read_only_and_did_not_cleanup(self) -> None:
        collection = self.inventory["collection"]
        self.assertTrue(collection["read_only"])
        self.assertFalse(collection["host_changes_applied"])
        self.assertFalse(collection["cleanup_executed"])
        self.assertFalse(collection["secrets_read"])
        self.assertFalse(self.proposal["current_state"]["fast_hot_created"])
        self.assertFalse(self.proposal["current_state"]["cleanup_executed"])
        self.assertFalse(self.proposal["current_state"]["host_change_executed"])

    def test_inventory_filesystem_facts_are_internally_consistent(self) -> None:
        ssd = self.inventory["filesystems"]["ssd_root_lv"]
        self.assertEqual(ssd["free_bytes"], 34029350912)
        self.assertEqual(ssd["used_bytes"], 400243847168)
        self.assertEqual(ssd["used_bytes"] + ssd["free_bytes"] + ssd["reserved_bytes"], ssd["total_bytes"])
        self.assertEqual(ssd["reserved_bytes"], 19580002304)
        self.assertEqual(ssd["use_percent"], 93)
        hdd = self.inventory["filesystems"]["hdd_data"]
        self.assertEqual(hdd["free_bytes"], 695038337024)
        self.assertEqual(hdd["used_bytes"] + hdd["free_bytes"] + hdd["reserved_bytes"], hdd["total_bytes"])
        self.assertTrue(self.inventory["block_devices"]["hdd_sda"]["rotational"])
        self.assertFalse(self.inventory["block_devices"]["ssd_sdb"]["rotational"])

    def test_fast_hot_target_is_non_rotational_and_hdd_fallback_forbidden(self) -> None:
        target = self.proposal["target"]
        self.assertEqual(target["host_path"], "/srv/cb16_fast_hot")
        self.assertEqual(target["container_path"], "/cb16/fast_hot")
        self.assertEqual(target["storage_class"], "SSD_NON_ROTATIONAL")
        self.assertFalse(target["hdd_fallback_allowed"])
        self.assertEqual(target["owner_uid"], 1001)
        self.assertEqual(target["owner_gid"], 1001)
        self.assertEqual(target["creation_task"], "R3")

    def test_proposal_arithmetic_is_consistent(self) -> None:
        validate_proposal_arithmetic(self.inventory, self.proposal)

    def test_all_options_and_recommendation_are_present(self) -> None:
        options = {option["option_id"]: option for option in self.proposal["options"]}
        self.assertEqual(
            set(options),
            {
                "A_NO_CLEANUP_EXISTING_FREE",
                "B_SAFE_RECLAIM_THEN_ALLOCATE",
                "C_MIGRATE_COLD_USER_DATA",
                "D_COMBINED_RECLAIM_AND_MIGRATION",
                "E_NEW_SSD_HARDWARE",
            },
        )
        self.assertEqual(
            self.proposal["recommendation"]["default_option_id"],
            "B_SAFE_RECLAIM_THEN_ALLOCATE",
        )
        recommended = [option for option in options.values() if option["recommended"]]
        self.assertEqual([option["option_id"] for option in recommended], ["B_SAFE_RECLAIM_THEN_ALLOCATE"])

    def test_protected_and_never_delete_lists_are_preserved(self) -> None:
        protected = self.inventory["protect_and_keep"]
        self.assertEqual(set(protected["docker_never_delete_volumes"]), NEVER_DELETE_VOLUMES)
        host_paths = {entry["path"] for entry in protected["host_paths"]}
        for required in ("/swapfile", "/home/bgy/.ssh", "/home/bgy/Downloads/CB16"):
            self.assertIn(required, host_paths)
        selected_candidates = {
            candidate_id
            for option in self.proposal["options"]
            for candidate_id in option["selected_bytes_by_candidate"]
        }
        self.assertNotIn("CACHE_HUGGINGFACE", selected_candidates)
        self.assertFalse(any("swapfile" in entry["path"] for entry in self.inventory["reclaim_candidates"]))

    def test_migration_candidates_have_verified_migration_rule(self) -> None:
        candidates = {entry["candidate_id"]: entry for entry in self.inventory["reclaim_candidates"]}
        for candidate_id, entry in candidates.items():
            if entry["recommended_method"].startswith("MIGRATE"):
                self.assertIn(entry["path"], {
                    "/home/bgy/audiotransfer",
                    "/home/bgy/photo_timeline",
                    "/home/bgy/recovered_photos",
                    "/home/bgy/photo_packs",
                    "/home/bgy/photo_packs.tar.zst",
                    "/home/bgy/Codes_and_Datas",
                    "/home/bgy/m3-infra",
                    "/home/bgy/stage1-nvidia-ref",
                    "/home/bgy/editor_DLW",
                    "/home/bgy/vibevoice-podcast",
                    "/home/bgy/venvs",
                    "/home/bgy/miniforge3/envs",
                })
        option_c = {option["option_id"]: option for option in self.proposal["options"]}["C_MIGRATE_COLD_USER_DATA"]
        self.assertEqual(option_c["migration_integrity_requirement"], "COPY_THEN_SHA256_VERIFY_THEN_REMOVE_SOURCE")
        self.assertTrue(option_c["migration_destination_root"].startswith("/data/"))

    def test_owner_decision_request_mentions_every_option_and_required_fields(self) -> None:
        for option_id in ("A", "B", "C", "D", "E"):
            self.assertIn(f"**{option_id}**", self.decision_text)
        for field in (
            "FAST_HOT_QUOTA_BYTES",
            "RESERVE_FLOOR_BYTES",
            "SELECTED_CANDIDATE_IDS",
            "AUTHORIZE_R3_HOST_CHANGE_PLAN",
        ):
            self.assertIn(field, self.decision_text)

    def test_no_scientific_constants_are_changed(self) -> None:
        rules = " ".join(self.proposal["decision_rules"])
        self.assertIn("S1 scientific manifest", rules)
        self.assertIn("unchanged", rules)
        self.assertFalse(any("reward" in entry["path"].lower() for entry in self.inventory["reclaim_candidates"]))

    def test_owner_approval_binds_option_d_exactly(self) -> None:
        option_d = {option["option_id"]: option for option in self.proposal["options"]}["D_COMBINED_RECLAIM_AND_MIGRATION"]
        approval = self.approval
        self.assertEqual(approval["schema"], "CB16_R21_RC2_R2_OWNER_APPROVAL_V1")
        self.assertEqual(approval["status"], "APPROVED")
        self.assertEqual(approval["approved_by_owner"], "bgy")
        self.assertEqual(approval["selected_option_id"], "D_COMBINED_RECLAIM_AND_MIGRATION")
        self.assertEqual(approval["selected_candidate_ids"], option_d["candidate_ids"])
        self.assertEqual(approval["reclaimed_or_migrated_bytes"], option_d["reclaimed_or_migrated_bytes"])
        self.assertEqual(approval["fast_hot_quota_bytes"], option_d["proposed_fast_hot_quota_bytes"])
        self.assertEqual(approval["reserve_floor_bytes"], option_d["reserve_floor_bytes"])
        self.assertEqual(approval["migration_destination_root"], option_d["migration_destination_root"])
        self.assertEqual(approval["migration_integrity_requirement"], option_d["migration_integrity_requirement"])
        self.assertTrue(approval["authorize_r3_host_change_plan"])
        self.assertIn("scientific constants", " ".join(approval["constraints"]))

    def test_owner_approval_v2_lowers_quota_and_preserves_reserve(self) -> None:
        approval_v2 = self.approval_v2
        self.assertEqual(approval_v2["schema"], "CB16_R21_RC2_R2_OWNER_APPROVAL_V2")
        self.assertEqual(approval_v2["status"], "APPROVED")
        self.assertEqual(approval_v2["supersedes"], "authority/infra/R21_RC2_R2_OWNER_APPROVAL_V1.json")
        self.assertEqual(approval_v2["fast_hot_quota_bytes"], 100000000000)
        self.assertEqual(approval_v2["reserve_floor_bytes"], self.approval["reserve_floor_bytes"])
        self.assertTrue(approval_v2["authorize_r3_host_change_plan"])
        self.assertEqual(
            approval_v2["projected_free_after_remaining_migration_bytes"],
            approval_v2["measured_actual_free_bytes_at_revision"] + approval_v2["remaining_planned_migration_source_bytes"],
        )
        self.assertEqual(
            approval_v2["projected_free_after_quota_and_reserve_bytes"],
            approval_v2["projected_free_after_remaining_migration_bytes"]
            - approval_v2["fast_hot_quota_bytes"]
            - approval_v2["reserve_floor_bytes"],
        )

    def test_expected_vs_actual_record_is_consistent(self) -> None:
        record = self.expected_vs_actual
        expected = record["expected_under_option_d"]
        actual = record["actual_at_stop"]
        self.assertFalse(record["classification"]["option_d_v1_target_met"])
        self.assertFalse(record["classification"]["scientific_constants_changed"])
        self.assertFalse(record["classification"]["hdd_fast_hot_fallback_used"])
        self.assertEqual(
            record["deltas"]["projected_free_shortfall_vs_option_d_bytes"],
            expected["projected_free_after_cleanup_and_migration_bytes"]
            - actual["projected_free_after_remaining_migration_bytes"],
        )
        self.assertEqual(
            record["deltas"]["reclaim_shortfall_bytes"],
            expected["planned_reclaim_bytes"] - actual["actual_reclaim_free_delta_bytes"],
        )
        self.assertEqual(
            actual["projected_free_after_remaining_migration_bytes"],
            actual["actual_free_bytes"] + actual["remaining_migration_source_bytes"],
        )
        self.assertEqual(record["owner_revision"]["fast_hot_quota_bytes"], 100000000000)
        causes = {cause["cause_id"] for cause in record["causes"]}
        self.assertEqual(causes, {"HARDLINK_ACCOUNTING", "CRASH_VERIFICATION_GAP", "ROOT_OWNED_MIGRATION_SOURCE"})

    def test_hostile_arithmetic_mutation_is_rejected(self) -> None:
        mutated = json.loads(json.dumps(self.proposal))
        mutated["options"][0]["reserve_floor_bytes"] += 1
        with self.assertRaises(AssertionError):
            validate_proposal_arithmetic(self.inventory, mutated)


if __name__ == "__main__":
    unittest.main()
