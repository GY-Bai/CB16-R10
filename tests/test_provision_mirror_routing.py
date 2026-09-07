from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "provision" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import provision_python as pp


class ProvisionUVHostEnvRoutingTests(unittest.TestCase):
    def test_install_env_does_not_invent_or_rewrite_channels(self):
        src = {
            "UV_INDEX_URL": "https://mirror.example/simple/",
            "UV_EXTRA_INDEX_URL": "https://mirror.example/cu126/",
            "HTTPS_PROXY": "http://127.0.0.1:33128",
        }
        out = pp._install_env(src)
        for k, v in src.items():
            self.assertEqual(out[k], v)
        self.assertNotIn("UV_FIND_LINKS", out)

    def test_uv_install_command_contains_no_channel_arguments(self):
        with tempfile.TemporaryDirectory() as td:
            req = Path(td) / "requirements.txt"
            req.write_text("torch==2.8.0+cu126\nnumpy>=2.2\n")
            cmd = pp._uv_install_command(Path(td) / "venv/bin/python", [req])
            pp._assert_no_channel_args(cmd)
            self.assertNotIn("--index-url", cmd)
            self.assertNotIn("--extra-index-url", cmd)
            self.assertNotIn("--find-links", cmd)
            self.assertFalse(any("://" in token for token in cmd))

    def test_host_env_policy_accepts_uv_routes(self):
        with tempfile.TemporaryDirectory() as td:
            req = Path(td) / "requirements.txt"
            req.write_text("torch==2.8.0+cu126\nnumpy>=2.2\n")
            cfg = {"index_policy": "HOST_ENV_REQUIRED", "installer_policy": "UV_REQUIRED"}
            env = {
                "UV_INDEX_URL": "https://mirror.example/simple/",
                "UV_EXTRA_INDEX_URL": "https://mirror.example/cu126/",
                "HTTPS_PROXY": "http://127.0.0.1:33128",
            }
            summary = pp._validate_host_env_policy(cfg, [req], env)
            self.assertEqual(summary["routing_owner"], "HOST_ENVIRONMENT_AND_LOCAL_PROXY")
            self.assertTrue(summary["uv_primary_route_present"])
            self.assertTrue(summary["uv_extra_route_present"])
            self.assertEqual(summary["repository_channel_directive_count"], 0)

    def test_host_env_policy_rejects_embedded_index_or_find_links(self):
        cfg = {"index_policy": "HOST_ENV_REQUIRED", "installer_policy": "UV_REQUIRED"}
        env = {"UV_INDEX_URL": "https://mirror.example/simple/", "UV_EXTRA_INDEX_URL": "https://mirror.example/cu126/"}
        cases = [
            "--extra-index-url https://download.example/cu126\ntorch==2.8.0+cu126\n",
            "--find-links https://download.example/wheels\ntorch==2.8.0+cu126\n",
            "pkg @ https://download.example/pkg.whl\n",
        ]
        for text in cases:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as td:
                req = Path(td) / "requirements.txt"
                req.write_text(text)
                with self.assertRaisesRegex(RuntimeError, "CHANNEL_FORBIDDEN"):
                    pp._validate_host_env_policy(cfg, [req], env)

    def test_host_env_policy_requires_uv_accelerator_route_for_cuda_build(self):
        with tempfile.TemporaryDirectory() as td:
            req = Path(td) / "requirements.txt"
            req.write_text("torch==2.8.0+cu126\n")
            cfg = {"index_policy": "HOST_ENV_REQUIRED", "installer_policy": "UV_REQUIRED"}
            with self.assertRaisesRegex(RuntimeError, "UV_ACCELERATOR_ROUTE_MISSING"):
                pp._validate_host_env_policy(cfg, [req], {"UV_INDEX_URL": "https://mirror.example/simple/"})

    def test_r104_policy_and_requirements_have_no_repository_download_channel(self):
        manifest = json.loads((ROOT / "provision" / "environments" / "r104.json").read_text())
        py = manifest["python"]
        self.assertEqual(py["installer_policy"], "UV_REQUIRED")
        self.assertEqual(py["index_policy"], "HOST_ENV_REQUIRED")
        self.assertEqual(py["package_identity_policy"], "DIRECT_REQUIREMENT_VERSION_EQUIVALENCE")
        self.assertFalse(py["allow_embedded_index_directives"])
        self.assertFalse(py["allow_embedded_find_links"])
        self.assertFalse(py["allow_public_index_fallback"])
        for name in ("requirements-shanxi-pascal.txt", "requirements-ci-runtime.txt"):
            text = (ROOT / name).read_text()
            self.assertNotIn("--index-url", text)
            self.assertNotIn("--extra-index-url", text)
            self.assertNotIn("--find-links", text)
            self.assertNotIn("://", text)


if __name__ == "__main__":
    unittest.main()
