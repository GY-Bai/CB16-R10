"""Static and hostile tests for the R21 RC2 R1 runner definition."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.generate_shanxi_runner_snapshot_v1 import (
    SENSITIVE_NAME_PATTERN,
    sanitize_environment,
    sanitize_inspect_v1,
)
from scripts.verify_r21_rc2_runner_definition_v1 import (
    BUILD_CONTEXT_PATH,
    DOCKERFILE_PATH,
    SPEC_PATH,
    static_check,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class RunnerDefinitionV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads((REPO_ROOT / SPEC_PATH.relative_to(REPO_ROOT)).read_text(encoding="utf-8"))
        cls.build_context = json.loads(
            (REPO_ROOT / BUILD_CONTEXT_PATH.relative_to(REPO_ROOT)).read_text(encoding="utf-8")
        )
        cls.dockerfile = (REPO_ROOT / DOCKERFILE_PATH.relative_to(REPO_ROOT)).read_text(encoding="utf-8")

    def test_static_checks_pass(self) -> None:
        checks = static_check(REPO_ROOT)
        failures = [entry for entry in checks if not entry["ok"]]
        self.assertEqual(failures, [])
        self.assertGreaterEqual(len(checks), 60)

    def test_launch_spec_matches_frozen_snapshot_core(self) -> None:
        snapshot = json.loads(
            (REPO_ROOT / "authority/infra/SHANXI_DOCKER_RUNNER_SNAPSHOT_V1.json").read_text(encoding="utf-8")
        )
        container = snapshot["container"]
        self.assertEqual(self.spec["container"]["id"], container["id"])
        self.assertEqual(self.spec["container"]["name"], container["name"])
        self.assertEqual(self.spec["image"]["live_image_id"], container["image_id"])
        self.assertEqual(self.spec["image"]["base_image"], "python:3.10-slim-bookworm")
        self.assertEqual(self.spec["run"]["shm_size_bytes"], container["shm_size_bytes"])
        self.assertEqual(self.spec["run"]["restart_policy"], container["restart_policy"])
        self.assertEqual(self.spec["run"]["network_mode"], container["network_mode"])
        self.assertEqual(len(self.spec["mounts"]), container["mount_count"])
        spec_mounts = {entry["container_path"]: entry for entry in self.spec["mounts"]}
        snapshot_mounts = {entry["container_path"]: entry for entry in snapshot["docker_mount_contract"]}
        self.assertEqual(set(spec_mounts), set(snapshot_mounts))
        for path, entry in spec_mounts.items():
            self.assertEqual(entry["volume"], snapshot_mounts[path]["volume"])
            self.assertEqual(entry["mount_rw"], snapshot_mounts[path]["rw"])

    def test_launch_spec_environment_records_snapshot_coverage(self) -> None:
        snapshot = json.loads(
            (REPO_ROOT / "authority/infra/SHANXI_DOCKER_RUNNER_SNAPSHOT_V1.json").read_text(encoding="utf-8")
        )
        spec_env = {entry["name"]: entry for entry in self.spec["environment"]}
        snapshot_env = snapshot["environment"]["values"]
        for name, value in snapshot_env.items():
            self.assertIn(name, spec_env)
            self.assertEqual(spec_env[name]["value"], value)
        self.assertEqual(spec_env["CB16_PROVISION_ENV"]["snapshot_status"], "OMITTED_FROM_SNAPSHOT_DIVERGENCE")
        self.assertEqual(spec_env["GPG_KEY"]["snapshot_status"], "OMITTED_NON_SECRET_SNAPSHOT_FILTER")
        self.assertEqual(spec_env["PYTHON_SHA256"]["snapshot_status"], "OMITTED_NON_SECRET_SNAPSHOT_FILTER")

    def test_build_context_pins_single_uv_artifact(self) -> None:
        entries = self.build_context["required_build_context_files"]
        self.assertEqual(len(entries), 1)
        uv = entries[0]
        self.assertEqual(uv["version"], "0.12.6")
        self.assertEqual(uv["sha256"], "d381f11517c66523211b0876552ff7dea5c1b4b0f13800571b35225761302fba")
        self.assertEqual(uv["size_bytes"], 51102256)
        self.assertEqual(uv["container_install_path"], "/usr/local/bin/uv")
        self.assertTrue(uv["observed_in_live_image"])

    def test_dockerfile_contains_reconstructed_contract(self) -> None:
        expected_fragments = (
            "FROM python:3.10-slim-bookworm",
            "COPY uv /usr/local/bin/uv",
            "addgroup --gid 1001 cb16-runner",
            "useradd -m -u 1001 -g 1001 cb16-runner",
            "ENV UV_CACHE_DIR=/cb16/uv-cache",
            "ENV PYTHONUNBUFFERED=1",
            "USER cb16-runner",
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, self.dockerfile)

    def test_divergence_is_explicit_and_host_was_not_changed(self) -> None:
        divergences = {entry["divergence_id"]: entry for entry in self.spec["divergences"]}
        div = divergences["R1-DIV-001"]
        self.assertEqual(div["classification"], "CONTRACT_MISMATCH")
        self.assertFalse(div["auto_fix_applied"])
        self.assertFalse(div["host_changed"])
        self.assertIn("CB16_PROVISION_ENV", json.dumps(div))
        launch = self.spec["reproduction_status"]["disposable_runner_launch"]
        self.assertEqual(launch["status"], "EXECUTION_BLOCKED")
        self.assertFalse(launch["host_changes_applied"])

    def test_sanitizer_filters_secret_like_values_and_keeps_non_secret_identity(self) -> None:
        inspect_obj = {
            "Id": "abc123",
            "Name": "/cb16-runner-r11",
            "Image": "sha256:deadbeef",
            "Size": 123,
            "Created": "2026-09-09T08:36:54Z",
            "Config": {
                "Image": "cb16-runner-r11:latest",
                "Cmd": ["./run.sh"],
                "Entrypoint": None,
                "WorkingDir": "/home/cb16-runner/actions-runner",
                "Labels": {},
                "Env": [
                    "CB16_RAW_ROOT=/cb16/raw",
                    "GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D",
                    "PYTHON_SHA256=a0da1e72132e950154eca0f6f47d5db828454700de20e5113667940d81e0db04",
                    "MY_SECRET_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789",
                    "SERVICE_PASSWORD=hunter2",
                ],
            },
            "HostConfig": {
                "ShmSize": 67108864,
                "NetworkMode": "cb16-net",
                "RestartPolicy": {"Name": "unless-stopped"},
                "Memory": 0,
                "NanoCpus": 0,
                "Privileged": False,
                "Runtime": "runc",
                "DeviceRequests": [],
            },
            "State": {"StartedAt": "2026-09-09T13:21:57Z", "Running": True},
            "Mounts": [
                {
                    "Type": "volume",
                    "Name": "cb16-raw",
                    "Source": "/data/cb16_docker/volumes/cb16-raw/_data",
                    "Destination": "/cb16/raw",
                    "RW": False,
                }
            ],
        }
        snapshot = sanitize_inspect_v1(inspect_obj, collected_at_utc="2026-09-13T00:00:00Z")
        self.assertEqual(snapshot["schema"], "CB16_SHANXI_RUNNER_SANITIZED_CONTAINER_SNAPSHOT_V1")
        values = snapshot["environment"]["values"]
        self.assertIn("GPG_KEY", values)
        self.assertIn("PYTHON_SHA256", values)
        self.assertNotIn("MY_SECRET_TOKEN", values)
        self.assertNotIn("SERVICE_PASSWORD", values)
        self.assertEqual(
            set(snapshot["environment"]["filtered_keys"]),
            {"MY_SECRET_TOKEN", "SERVICE_PASSWORD"},
        )
        self.assertEqual(snapshot["container"]["name"], "cb16-runner-r11")
        self.assertEqual(snapshot["container"]["shm_size_bytes"], 67108864)
        self.assertEqual(snapshot["docker_mount_contract"][0]["container_path"], "/cb16/raw")
        self.assertFalse(snapshot["docker_mount_contract"][0]["rw"])
        self.assertEqual(len(snapshot["collection"]["source_inspect_sha256"]), 64)
        self.assertFalse(snapshot["scientific_boundary"]["final_holdout_accessed"])

    def test_sanitizer_sensitive_name_policy(self) -> None:
        self.assertTrue(SENSITIVE_NAME_PATTERN.search("MY_SECRET_TOKEN"))
        self.assertTrue(SENSITIVE_NAME_PATTERN.search("SERVICE_PASSWORD"))
        self.assertFalse(SENSITIVE_NAME_PATTERN.search("GPG_KEY"))
        self.assertFalse(SENSITIVE_NAME_PATTERN.search("PYTHON_SHA256"))

    def test_sanitize_environment_ignores_malformed_entries(self) -> None:
        snapshot = sanitize_environment({"Config": {"Env": ["NO_EQUALS_SIGN", "A=1"]}})
        self.assertEqual(snapshot["values"], {"A": "1"})

    def test_static_check_fails_on_mutated_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            for relative in (
                SPEC_PATH.relative_to(REPO_ROOT),
                BUILD_CONTEXT_PATH.relative_to(REPO_ROOT),
                DOCKERFILE_PATH.relative_to(REPO_ROOT),
            ):
                target = temp_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / relative, target)
            mutated_path = temp_root / SPEC_PATH.relative_to(REPO_ROOT)
            mutated = json.loads(mutated_path.read_text(encoding="utf-8"))
            mutated["run"]["shm_size_bytes"] = 33554432
            mutated_path.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
            failures = [entry for entry in static_check(temp_root) if not entry["ok"]]
            self.assertTrue(any(entry["check"] == "run_shm_size" for entry in failures), failures)


if __name__ == "__main__":
    unittest.main()
