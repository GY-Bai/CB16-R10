from __future__ import annotations

import os
from pathlib import Path

_ALLOWED = {
    "CB16_R10_PACKAGE_ROOT",
    "CB16_R10_DATA_ROOT",
    "CB16_R10_R102_ROOT",
    "CB16_R10_R103_ROOT",
    "CB16_R10_R104_ROOT",
    "CB16_R10_PARENT_R101_ROOT",
    "CB16_R10_PARENT_G0",
}


def _host_env_files() -> tuple[Path, ...]:
    worker_root = Path(os.environ.get("CB16_CI_WORKER_ROOT", "/data/cb16_ci"))
    return (Path("/etc/cb16-ci/provision.env"), worker_root / "provision.env")


def host_binding(name: str, default: str) -> str:
    """Resolve an allow-listed host-local R10 path without exposing it in public evidence."""
    if name not in _ALLOWED:
        raise ValueError(f"R10_HOST_BINDING_NOT_ALLOWED:{name}")
    direct = os.environ.get(name)
    if direct:
        return direct
    for path in _host_env_files():
        try:
            lines = path.read_text().splitlines()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name and value:
                return value
    return default
