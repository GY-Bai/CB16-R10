#!/usr/bin/env python3
"""Sanitized Shanxi egress diagnostics for CB16 provisioning.

This script never prints proxy/index URLs, credentials, resolved IP addresses, or
host-local paths. It reports only presence booleans and coarse connectivity
outcomes so results are safe to paste into a CI/debug transcript.
"""
from __future__ import annotations

import json
import os
import socket
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "provision" / "scripts"))
from provision_python import load_provision_env  # noqa: E402

TARGETS = {
    "pypi": ("pypi.org", "https://pypi.org/simple/"),
    "pytorch_cu126": ("download.pytorch.org", "https://download.pytorch.org/whl/cu126/"),
    "huggingface": ("huggingface.co", "https://huggingface.co/"),
}
PROXY_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)
INDEX_KEYS = (
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX",
)


def _presence(env: dict[str, str], keys: tuple[str, ...]) -> bool:
    return any(bool(env.get(k)) for k in keys)


def _dns(host: str) -> dict[str, object]:
    try:
        # Intentionally discard resolved addresses.
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        return {"ok": True, "error_type": None}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__}


def _proxies(env: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    http = env.get("HTTP_PROXY") or env.get("http_proxy")
    https = env.get("HTTPS_PROXY") or env.get("https_proxy")
    all_proxy = env.get("ALL_PROXY") or env.get("all_proxy")
    if http:
        out["http"] = http
    elif all_proxy:
        out["http"] = all_proxy
    if https:
        out["https"] = https
    elif all_proxy:
        out["https"] = all_proxy
    return out


def _https(url: str, *, mode: str, provision_env: dict[str, str]) -> dict[str, object]:
    if mode == "direct":
        handler = urllib.request.ProxyHandler({})
    elif mode == "provision_env":
        handler = urllib.request.ProxyHandler(_proxies(provision_env))
    elif mode == "process_env":
        handler = urllib.request.ProxyHandler()
    else:
        raise ValueError(mode)

    opener = urllib.request.build_opener(handler, urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CB16-Network-Diagnostic/1"},
        method="GET",
    )
    try:
        with opener.open(request, timeout=12) as response:
            # Read only one byte. This is reachability, not an asset download.
            response.read(1)
            return {"ok": True, "http_status": int(response.status), "error_type": None}
    except urllib.error.HTTPError as exc:
        # 2xx/3xx is preferred, but an HTTP response still proves transport worked.
        return {
            "ok": 200 <= int(exc.code) < 400,
            "http_status": int(exc.code),
            "error_type": "HTTPError",
        }
    except Exception as exc:
        return {"ok": False, "http_status": None, "error_type": type(exc).__name__}


def main() -> int:
    provision_env = load_provision_env()
    process_env = dict(os.environ)
    result: dict[str, object] = {
        "schema": "CB16_SHANXI_NETWORK_DIAGNOSTIC_V1",
        "sensitive_values_emitted": False,
        "configuration_presence": {
            "process_proxy_present": _presence(process_env, PROXY_KEYS),
            "provision_proxy_present": _presence(provision_env, PROXY_KEYS),
            "process_python_index_present": _presence(process_env, INDEX_KEYS),
            "provision_python_index_present": _presence(provision_env, INDEX_KEYS),
            "provision_hf_endpoint_present": bool(provision_env.get("HF_ENDPOINT")),
        },
        "targets": {},
    }

    targets: dict[str, object] = {}
    for name, (host, url) in TARGETS.items():
        targets[name] = {
            "dns": _dns(host),
            "https": {
                "process_env": _https(url, mode="process_env", provision_env=provision_env),
                "provision_env": _https(url, mode="provision_env", provision_env=provision_env),
                "direct": _https(url, mode="direct", provision_env=provision_env),
            },
        }
    result["targets"] = targets
    print(json.dumps(result, sort_keys=True, indent=2))

    # A useful egress path exists if PyPI is reachable by at least one mode.
    pypi_https = targets["pypi"]["https"]  # type: ignore[index]
    return 0 if any(bool(v.get("ok")) for v in pypi_https.values()) else 2  # type: ignore[union-attr]


if __name__ == "__main__":
    raise SystemExit(main())
