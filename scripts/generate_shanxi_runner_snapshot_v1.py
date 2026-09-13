#!/usr/bin/env python3
"""Generate a sanitized Shanxi runner snapshot from a docker-inspect document.

The generator is deliberately read-only: it consumes JSON produced by an
authorized ``docker inspect`` read and never contacts Docker, the host or the
runner itself. Sensitive-looking environment values are dropped from the
sanitized output; their key names are recorded for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "CB16_SHANXI_RUNNER_SANITIZED_CONTAINER_SNAPSHOT_V1"
SENSITIVE_NAME_PATTERN = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|API_KEY)",
    re.IGNORECASE,
)
SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"BEGIN (OPENSSH|RSA|EC|DSA) PRIVATE KEY"),
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_sensitive_name(name: str) -> bool:
    if name in {"GPG_KEY", "PYTHON_SHA256"}:
        return False
    return bool(SENSITIVE_NAME_PATTERN.search(name))


def _is_sensitive_value(value: str) -> bool:
    return any(pattern.search(value) for pattern in SENSITIVE_VALUE_PATTERNS)


def _inspect_identity(inspect_obj: dict[str, Any]) -> dict[str, Any]:
    cfg = inspect_obj.get("Config", {})
    hc = inspect_obj.get("HostConfig", {})
    return {
        "name": (inspect_obj.get("Name") or "").lstrip("/") or None,
        "id": inspect_obj.get("Id"),
        "image": cfg.get("Image"),
        "image_id": inspect_obj.get("Image"),
        "image_size_bytes": inspect_obj.get("Size"),
        "image_created": inspect_obj.get("Created"),
        "cmd": cfg.get("Cmd"),
        "entrypoint": cfg.get("Entrypoint"),
        "working_dir": cfg.get("WorkingDir"),
        "restart_policy": hc.get("RestartPolicy"),
        "network_mode": hc.get("NetworkMode"),
        "runtime": hc.get("Runtime"),
        "privileged": hc.get("Privileged"),
        "shm_size_bytes": hc.get("ShmSize"),
        "gpu_device_requests": hc.get("DeviceRequests"),
        "memory_limit_bytes": hc.get("Memory"),
        "nano_cpus": hc.get("NanoCpus"),
        "started_at": (inspect_obj.get("State") or {}).get("StartedAt"),
        "running": (inspect_obj.get("State") or {}).get("Running"),
        "mount_count": len(inspect_obj.get("Mounts", [])),
        "managed_by": {
            "labels": cfg.get("Labels") or {},
            "compose_project": (cfg.get("Labels") or {}).get("com.docker.compose.project"),
            "compose_config_files": (cfg.get("Labels") or {}).get("com.docker.compose.project.config_files"),
            "systemd_unit": None,
        },
    }


def sanitize_environment(inspect_obj: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, str] = {}
    filtered_keys: list[str] = []
    for item in inspect_obj.get("Config", {}).get("Env", []) or []:
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        if _is_sensitive_name(name) or _is_sensitive_value(value):
            filtered_keys.append(name)
            continue
        values[name] = value
    return {
        "values": values,
        "filtered_keys": sorted(filtered_keys),
        "filtered_reason": "sensitive-looking environment names or values were omitted from the sanitized snapshot",
    }


def sanitize_mounts(inspect_obj: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for mount in inspect_obj.get("Mounts", []) or []:
        entries.append(
            {
                "volume": mount.get("Name"),
                "source": mount.get("Source"),
                "container_path": mount.get("Destination"),
                "type": mount.get("Type"),
                "rw": bool(mount.get("RW")),
                "propagation": mount.get("Propagation"),
            }
        )
    return entries


def sanitize_inspect_v1(
    inspect_obj: dict[str, Any],
    collected_at_utc: str | None = None,
    host_facts: dict[str, Any] | None = None,
    provision_env_keys: list[str] | None = None,
) -> dict[str, Any]:
    source_bytes = json.dumps(inspect_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": SCHEMA,
        "collected_at_utc": collected_at_utc or _now_utc(),
        "collection": {
            "method": "read-only docker inspect sanitization",
            "read_only": True,
            "secrets_read": False,
            "host_access_required": False,
            "source_inspect_sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "host": host_facts if host_facts is not None else "NOT_COLLECTED_BY_DOCKER_INSPECT",
        "container": _inspect_identity(inspect_obj),
        "environment": sanitize_environment(inspect_obj),
        "docker_mount_contract": sanitize_mounts(inspect_obj),
        "provision_env": {
            "keys": sorted(provision_env_keys or []),
            "values_redacted": True,
        },
        "scientific_boundary": {
            "runtime_behavior_changed": False,
            "model_accessed": False,
            "latent_accessed": False,
            "final_holdout_accessed": False,
            "note": "read-only infrastructure snapshot; no science payload opened",
        },
    }


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-json", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--collected-at-utc")
    parser.add_argument("--host-facts-json", type=Path)
    parser.add_argument("--provision-env-keys-json", type=Path)
    args = parser.parse_args(argv)

    inspect_obj = _load_json(args.inspect_json)
    if isinstance(inspect_obj, list):
        inspect_obj = inspect_obj[0]
    host_facts = _load_json(args.host_facts_json) if args.host_facts_json else None
    provision_env_keys = _load_json(args.provision_env_keys_json) if args.provision_env_keys_json else None
    snapshot = sanitize_inspect_v1(
        inspect_obj,
        collected_at_utc=args.collected_at_utc,
        host_facts=host_facts,
        provision_env_keys=provision_env_keys,
    )
    text = json.dumps(snapshot, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
