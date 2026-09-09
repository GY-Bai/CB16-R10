#!/usr/bin/env python3
"""Static fail-closed path/runner contract for the CB16 R11 Docker migration."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_RUNTIME_PATHS = (
    "/home/bgy",
    "/data/cb16_hdd",
    "/data/cb16_ci",
    "/cb16_r11_runtime",
)

DOCKER_WORKFLOWS = (
    ".github/workflows/cb16-r11-docker-preflight.yml",
    ".github/workflows/cb16-r11-science-g0-r104-authority.yml",
    ".github/workflows/cb16-r11-science-g0-s4f-preflight.yml",
    ".github/workflows/cb16-r11-science-g0-materialize-cache.yml",
    ".github/workflows/cb16-r11-science-g0-stage4-roots.yml",
    ".github/workflows/cb16-r11-science-g0-authority-adoption.yml",
)

RUNTIME_CODE = (
    "ci/resolve_verified_r104_python.py",
    "provision/scripts/provision_common.py",
    "provision/scripts/provision_python.py",
    "scripts/r11_science_g0_initialize_stage4_roots.py",
    "scripts/r11_science_g0_seal.py",
    "infra/docker/runner/Dockerfile",
    "infra/docker/runner/entrypoint.sh",
    "infra/docker/runner/compose.r11.yml",
)


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        raise RuntimeError(f"MISSING:{rel}")
    return path.read_text(encoding="utf-8")


def main() -> int:
    errors: list[str] = []

    for rel in DOCKER_WORKFLOWS:
        text = read(rel)
        if "shanxi-docker-r11" not in text:
            errors.append(f"{rel}:DOCKER_RUNNER_LABEL_MISSING")
        if "persist-credentials: false" not in text:
            errors.append(f"{rel}:CHECKOUT_CREDENTIAL_PERSISTENCE_GUARD_MISSING")
        for prefix in FORBIDDEN_RUNTIME_PATHS:
            if prefix in text:
                errors.append(f"{rel}:FORBIDDEN_RUNTIME_PATH:{prefix}")

    for rel in RUNTIME_CODE:
        text = read(rel)
        for prefix in FORBIDDEN_RUNTIME_PATHS:
            if prefix in text:
                errors.append(f"{rel}:FORBIDDEN_RUNTIME_PATH:{prefix}")

    # This script intentionally keeps legacy path strings only as canonical identity
    # fields so the already-qualified lineage hash remains byte-identical. They must
    # never become the physical I/O defaults in Docker.
    materialize = read("scripts/r11_science_g0_materialize_cache.py")
    required_identity = (
        'RAW_IDENTITY_ROOT = Path("/data/cb16_hdd/binance_usdm_1m_funding_2020_2026")',
        'R11_IDENTITY_ROOT = Path("/home/bgy/cb16_ssd/runtime/R11/G0")',
        'RAW_ROOT = Path(os.environ.get("CB16_RAW_ROOT", str(RAW_IDENTITY_ROOT)))',
        'R11_ROOT = Path(os.environ.get("CB16_R11_G0_ROOT", "/cb16/g0"))',
    )
    for needle in required_identity:
        if needle not in materialize:
            errors.append("scripts/r11_science_g0_materialize_cache.py:IDENTITY_ACCESS_SPLIT_DRIFT")

    resolver = read("ci/resolve_verified_r104_python.py")
    if 'CB16_VERIFIED_VENV", "/cb16/venv"' not in resolver:
        errors.append("ci/resolve_verified_r104_python.py:DOCKER_VENV_DEFAULT_MISSING")

    provision = read("provision/scripts/provision_python.py")
    if 'CB16_PROVISION_ENV", "/run/secrets/cb16-provision.env"' not in provision:
        errors.append("provision/scripts/provision_python.py:DOCKER_PROVISION_SECRET_MISSING")

    common = read("provision/scripts/provision_common.py")
    if 'CB16_CI_WORKER_ROOT", "/cb16/worker"' not in common:
        errors.append("provision/scripts/provision_common.py:DOCKER_WORKER_DEFAULT_MISSING")

    if errors:
        print("CB16_R11_DOCKER_PATH_POLICY=FAIL")
        for error in errors:
            print(error)
        return 1

    print("CB16_R11_DOCKER_PATH_POLICY=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
