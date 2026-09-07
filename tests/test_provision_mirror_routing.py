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


class ProvisionMirrorRoutingTests(unittest.TestCase):
    def test_pip_routes_bridge_to_uv(self):
        env = pp._bridge_installer_routes({
            "PIP_INDEX_URL": "https://mirror.example/simple/",
            "PIP_EXTRA_INDEX_URL": "https://mirror.example/cu126/",
            "PIP_FIND_LINKS": "https://mirror.example/wheels/",
        })
        self.assertEqual(env["UV_INDEX_URL"], env["PIP_INDEX_URL"])
        self.assertEqual(env["UV_EXTRA_INDEX_URL"], env["PIP_EXTRA_INDEX_URL"])
        self.assertEqual(env["UV_FIND_LINKS"], env["PIP_FIND_LINKS"])

    def test_uv_routes_bridge_to_pip(self):
        env = pp._bridge_installer_routes({
            "UV_INDEX_URL": "https://mirror.example/simple/",
            "UV_EXTRA_INDEX_URL": "https://mirror.example/cu126/",
            "UV_FIND_LINKS": "https://mirror.example/wheels/",
        })
        self.assertEqual(env["PIP_INDEX_URL"], env["UV_INDEX_URL"])
        self.assertEqual(env["PIP_EXTRA_INDEX_URL"], env["UV_EXTRA_INDEX_URL"])
        self.assertEqual(env["PIP_FIND_LINKS"], env["UV_FIND_LINKS"])

    def test_direct_host_mirror_proxy_policy_only_adds_configured_mirror_hosts(self):
        cfg = {"mirror_proxy_policy": "DIRECT_FOR_HOST_MIRRORS"}
        env = pp._bridge_installer_routes({
            "PIP_INDEX_URL": "https://mirrors.volces.com/pypi/simple/",
            "PIP_EXTRA_INDEX_URL": "https://mirrors.aliyun.com/pytorch-wheels/cu126/",
            "UV_EXTRA_INDEX_URL": "https://mirrors.cloud.tencent.com/pypi/simple/",
            "NO_PROXY": "localhost,127.0.0.1",
            "HTTPS_PROXY": "http://127.0.0.1:33128",
        })
        out = pp._apply_mirror_proxy_policy(env, cfg)
        tokens = set(out["NO_PROXY"].split(","))
        self.assertIn("localhost", tokens)
        self.assertIn("127.0.0.1", tokens)
        self.assertIn("mirrors.volces.com", tokens)
        self.assertIn("mirrors.aliyun.com", tokens)
        self.assertIn("mirrors.cloud.tencent.com", tokens)
        self.assertNotIn("github.com", tokens)
        self.assertEqual(out["NO_PROXY"], out["no_proxy"])
        self.assertEqual(out["HTTPS_PROXY"], "http://127.0.0.1:33128")

    def test_inherit_proxy_policy_does_not_change_no_proxy(self):
        env = {"PIP_INDEX_URL": "https://mirror.example/simple/", "NO_PROXY": "localhost"}
        out = pp._apply_mirror_proxy_policy(env, {"mirror_proxy_policy": "INHERIT"})
        self.assertEqual(out["NO_PROXY"], "localhost")

    def test_host_mirror_policy_accepts_shanxi_shape(self):
        with tempfile.TemporaryDirectory() as td:
            req = Path(td) / "requirements.txt"
            req.write_text("torch==2.8.0+cu126\nnumpy>=2.2\n")
            cfg = {
                "index_policy": "HOST_MIRROR_REQUIRED",
                "mirror_proxy_policy": "DIRECT_FOR_HOST_MIRRORS",
                "allow_embedded_index_directives": False,
                "accelerator_wheel_route_required": True,
            }
            env = pp._bridge_installer_routes({
                "PIP_INDEX_URL": "https://mirrors.volces.com/pypi/simple/",
                "PIP_EXTRA_INDEX_URL": "https://mirrors.aliyun.com/pytorch-wheels/cu126/",
                "UV_INDEX_URL": "https://mirrors.volces.com/pypi/simple/",
                "UV_EXTRA_INDEX_URL": "https://mirrors.cloud.tencent.com/pypi/simple/ https://mirrors.aliyun.com/pypi/simple/",
                "UV_FIND_LINKS": "https://mirrors.aliyun.com/pytorch-wheels/cu126/",
                "HTTPS_PROXY": "http://127.0.0.1:33128",
            })
            env = pp._apply_mirror_proxy_policy(env, cfg)
            summary = pp._validate_index_policy(cfg, [req], env)
            self.assertEqual(summary["index_policy"], "HOST_MIRROR_REQUIRED")
            self.assertEqual(summary["mirror_proxy_policy"], "DIRECT_FOR_HOST_MIRRORS")
            self.assertTrue(summary["pip_primary_index_present"])
            self.assertTrue(summary["uv_primary_index_present"])
            self.assertTrue(summary["uv_find_links_present"])
            self.assertTrue(summary["all_package_mirror_hosts_bypass_proxy"])
            self.assertEqual(summary["configured_package_mirror_host_count"], 3)
            self.assertEqual(summary["package_mirror_hosts_in_no_proxy_count"], 3)
            self.assertEqual(summary["embedded_index_directive_count"], 0)

    def test_host_mirror_policy_rejects_missing_accelerator_route(self):
        with tempfile.TemporaryDirectory() as td:
            req = Path(td) / "requirements.txt"
            req.write_text("torch==2.8.0+cu126\n")
            cfg = {
                "index_policy": "HOST_MIRROR_REQUIRED",
                "mirror_proxy_policy": "INHERIT",
                "allow_embedded_index_directives": False,
                "accelerator_wheel_route_required": True,
            }
            with self.assertRaisesRegex(RuntimeError, "ACCELERATOR_ROUTE_MISSING"):
                pp._validate_index_policy(cfg, [req], {"PIP_INDEX_URL": "https://mirror.example/simple/"})

    def test_host_mirror_policy_rejects_incomplete_direct_bypass(self):
        with tempfile.TemporaryDirectory() as td:
            req = Path(td) / "requirements.txt"
            req.write_text("torch==2.8.0+cu126\n")
            cfg = {
                "index_policy": "HOST_MIRROR_REQUIRED",
                "mirror_proxy_policy": "DIRECT_FOR_HOST_MIRRORS",
                "allow_embedded_index_directives": False,
                "accelerator_wheel_route_required": True,
            }
            env = {
                "PIP_INDEX_URL": "https://mirror.example/simple/",
                "PIP_EXTRA_INDEX_URL": "https://wheels.example/cu126/",
                "NO_PROXY": "mirror.example",
            }
            with self.assertRaisesRegex(RuntimeError, "NO_PROXY_BINDING_INCOMPLETE"):
                pp._validate_index_policy(cfg, [req], env)

    def test_host_mirror_policy_rejects_embedded_public_index(self):
        with tempfile.TemporaryDirectory() as td:
            req = Path(td) / "requirements.txt"
            req.write_text("--extra-index-url https://download.pytorch.org/whl/cu126\ntorch==2.8.0+cu126\n")
            cfg = {
                "index_policy": "HOST_MIRROR_REQUIRED",
                "mirror_proxy_policy": "INHERIT",
                "allow_embedded_index_directives": False,
                "accelerator_wheel_route_required": True,
            }
            env = {
                "PIP_INDEX_URL": "https://mirror.example/simple/",
                "PIP_EXTRA_INDEX_URL": "https://mirror.example/cu126/",
            }
            with self.assertRaisesRegex(RuntimeError, "EMBEDDED_INDEX_FORBIDDEN"):
                pp._validate_index_policy(cfg, [req], env)

    def test_r104_manifest_requires_host_mirror_and_repo_requirement_has_no_index_directive(self):
        manifest = json.loads((ROOT / "provision" / "environments" / "r104.json").read_text())
        py = manifest["python"]
        self.assertEqual(py["index_policy"], "HOST_MIRROR_REQUIRED")
        self.assertEqual(py["mirror_proxy_policy"], "DIRECT_FOR_HOST_MIRRORS")
        self.assertFalse(py["allow_embedded_index_directives"])
        self.assertTrue(py["accelerator_wheel_route_required"])
        self.assertFalse(py["allow_public_index_fallback"])
        req = (ROOT / "requirements-shanxi-pascal.txt").read_text()
        self.assertNotIn("download.pytorch.org", req)
        self.assertNotIn("--extra-index-url", req)
        self.assertIn("torch==2.8.0+cu126", req)


if __name__ == "__main__":
    unittest.main()
