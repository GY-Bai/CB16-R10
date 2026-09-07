#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "provision" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import provision_python as pp


def _safe_endpoint(url: str) -> dict:
    u = urllib.parse.urlsplit(url)
    return {
        "scheme": u.scheme,
        "host": u.hostname,
        "port": u.port,
        "path": u.path,
        "userinfo_redacted": True,
    }


def _probe(url: str, *, direct: bool) -> dict:
    cmd = ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--connect-timeout", "10", "--max-time", "30"]
    if direct:
        cmd += ["--noproxy", "*"]
    cmd.append(url)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    code = p.stdout.strip()
    return {
        "mode": "DIRECT_NO_PROXY" if direct else "INHERITED_RUNNER_PROXY_POLICY",
        "endpoint": _safe_endpoint(url),
        "curl_rc": p.returncode,
        "http_code": code,
        "reachable": p.returncode == 0 and code.isdigit() and 200 <= int(code) < 500,
        "stderr_class": "NONE" if not p.stderr.strip() else "PRESENT_REDACTED",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    manifest = json.loads((ROOT / "provision/environments/r104.json").read_text())
    reqs = [ROOT / p for p in manifest["requirements"]]
    prov_env = pp.load_provision_env()
    install_env = pp._install_env(prov_env)
    route_summary = pp._validate_index_policy(manifest["python"], reqs, install_env)

    route_keys = (
        "PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX",
        "PIP_EXTRA_INDEX_URL", "UV_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS", "UV_FIND_LINKS",
    )
    urls = []
    for k in route_keys:
        for raw in (install_env.get(k) or "").split():
            if raw.startswith(("http://", "https://")) and raw not in urls:
                urls.append(raw)

    probes=[]
    for url in urls:
        inherited=_probe(url,direct=False)
        direct=_probe(url,direct=True)
        probes.append({
            "endpoint": _safe_endpoint(url),
            "inherited": inherited,
            "direct": direct,
            "direct_recovers_proxy_failure": (not inherited["reachable"]) and direct["reachable"],
        })

    current_route_pass = bool(probes) and all(x["inherited"]["reachable"] for x in probes)
    direct_route_pass = bool(probes) and all(x["direct"]["reachable"] for x in probes)
    result = {
        "schema": "CB16_PROVISION_HOST_MIRROR_QUALIFICATION_V2",
        "status": "PASS" if current_route_pass else "FAIL_CURRENT_ROUTE",
        "current_route_pass": current_route_pass,
        "direct_route_pass": direct_route_pass,
        "direct_bypass_candidate": (not current_route_pass) and direct_route_pass,
        "route_summary": route_summary,
        "route_probes": probes,
        "requirements_embedded_indexes": pp._embedded_index_directives(reqs),
        "public_pytorch_index_present": any("download.pytorch.org" in p.read_text() for p in reqs),
        "network_install_attempted": False,
        "scientific_semantics_changed": False,
        "final_holdout_2025_09_accessed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["public_pytorch_index_present"] or result["requirements_embedded_indexes"]:
        return 2
    return 0 if current_route_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
