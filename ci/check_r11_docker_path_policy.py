#!/usr/bin/env python3
"""Static fail-closed path/runner contract for the CB16 R11 Docker cutover."""
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

ACTIVE_R11_WORKFLOWS = (
    ".github/workflows/cb16-r11-docker-preflight.yml",
    ".github/workflows/cb16-r11-main-smoke.yml",
    ".github/workflows/cb16-r11-science-g0-inventory.yml",
    ".github/workflows/cb16-r11-science-g0-prereq.yml",
    ".github/workflows/cb16-r11-science-g0-storage-probe.yml",
    ".github/workflows/cb16-r11-science-g0-r104-authority.yml",
    ".github/workflows/cb16-r11-science-g0-s4f-preflight.yml",
    ".github/workflows/cb16-r11-science-g0-materialize-cache.yml",
    ".github/workflows/cb16-r11-science-g0-stage4-roots.yml",
    ".github/workflows/cb16-r11-science-g0-authority-adoption.yml",
)

# Active runtime code may not depend on host business paths. docker_preflight.py is
# excluded because it intentionally names those paths as negative visibility markers.
RUNTIME_CODE = (
    "ci/resolve_verified_r104_python.py",
    "provision/scripts/provision_common.py",
    "provision/scripts/provision_python.py",
    "scripts/r11_science_g0_inventory.py",
    "scripts/r11_science_g0_prereq_discovery.py",
    "scripts/r11_science_g0_initialize_stage4_roots.py",
    "scripts/r11_science_g0_seal.py",
)


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        raise RuntimeError(f"MISSING:{rel}")
    return path.read_text(encoding="utf-8")


def check_forbidden_paths(rel: str, text: str, errors: list[str]) -> None:
    for prefix in FORBIDDEN_RUNTIME_PATHS:
        if prefix in text:
            errors.append(f"{rel}:FORBIDDEN_RUNTIME_PATH:{prefix}")
    if "git-annex" in text.lower():
        errors.append(f"{rel}:GIT_ANNEX_FORBIDDEN_IN_ACTIVE_R11_DOCKER_PATH")


def main() -> int:
    errors: list[str] = []

    for rel in ACTIVE_R11_WORKFLOWS:
        text = read(rel)
        if "runs-on: [self-hosted, shanxi, shanxi-docker-r11]" not in text:
            errors.append(f"{rel}:SHANXI_DOCKER_R11_RUNNER_LABEL_MISSING")
        if "persist-credentials: false" not in text:
            errors.append(f"{rel}:CHECKOUT_CREDENTIAL_PERSISTENCE_GUARD_MISSING")
        if "python3 ci/docker_preflight.py" not in text:
            errors.append(f"{rel}:DOCKER_EXECUTION_PREFLIGHT_MISSING")
        check_forbidden_paths(rel, text, errors)

        # Transport is host/container infrastructure authority. R11 workflows must
        # not bypass provision_python.py with an unsourced ad-hoc UV install.
        lowered = text.lower()
        if "uv pip install" in lowered or "uv sync" in lowered:
            if "source /cb16/worker/provision.env" not in text and "source \"/cb16/worker/provision.env\"" not in text:
                errors.append(f"{rel}:DIRECT_UV_INSTALL_WITHOUT_PROVISION_ENV")

    for rel in RUNTIME_CODE:
        check_forbidden_paths(rel, read(rel), errors)

    docker_preflight = read("ci/docker_preflight.py")
    for marker in FORBIDDEN_RUNTIME_PATHS:
        if f'    "{marker}",' not in docker_preflight:
            errors.append(f"ci/docker_preflight.py:FORBIDDEN_HOST_MARKER_MISSING:{marker}")
    for needle in (
        "FORBIDDEN_HOST_PATHS = (",
        "check_no_host_paths",
        'Path("/proc/self/mountinfo")',
        'failures.append("HOST_BUSINESS_PATH_VISIBLE")',
        'Path("/run/secrets/provision.env")',
        'Path("/run/secrets/cb16-provision.env")',
    ):
        if needle not in docker_preflight:
            errors.append("ci/docker_preflight.py:DOCKER_PREFLIGHT_CONTRACT_DRIFT")
    if "git-annex" in docker_preflight.lower():
        errors.append("ci/docker_preflight.py:GIT_ANNEX_FORBIDDEN_IN_ACTIVE_R11_DOCKER_PATH")

    # Strong-lineage hashes already include historical path strings. Those strings
    # remain identity-only and must never become physical Docker I/O roots.
    materialize_rel = "scripts/r11_science_g0_materialize_cache.py"
    materialize = read(materialize_rel)
    required_identity = (
        'RAW_IDENTITY_ROOT = Path("/data/cb16_hdd/binance_usdm_1m_funding_2020_2026")',
        'R11_IDENTITY_ROOT = Path("/home/bgy/cb16_ssd/runtime/R11/G0")',
        'RAW_ROOT = Path(os.environ.get("CB16_RAW_ROOT", str(RAW_IDENTITY_ROOT)))',
        'R11_ROOT = Path(os.environ.get("CB16_R11_G0_ROOT", "/cb16/g0"))',
    )
    for needle in required_identity:
        if needle not in materialize:
            errors.append(f"{materialize_rel}:IDENTITY_ACCESS_SPLIT_DRIFT")
    scrubbed = materialize
    for needle in required_identity[:2]:
        scrubbed = scrubbed.replace(needle, "IDENTITY_ONLY_PATH_REDACTED")
    check_forbidden_paths(materialize_rel, scrubbed, errors)

    resolver = read("ci/resolve_verified_r104_python.py")
    if 'CB16_VERIFIED_VENV", "/cb16/venv"' not in resolver:
        errors.append("ci/resolve_verified_r104_python.py:DOCKER_VENV_DEFAULT_MISSING")
    if 'worker_root / "provision.env"' not in resolver:
        errors.append("ci/resolve_verified_r104_python.py:WORKER_PROVISION_ENV_FALLBACK_MISSING")

    provision = read("provision/scripts/provision_python.py")
    if 'Path(os.environ.get("CB16_CI_WORKER_ROOT", "/cb16/worker")) / "provision.env"' not in provision:
        errors.append("provision/scripts/provision_python.py:WORKER_PROVISION_ENV_FALLBACK_MISSING")

    common = read("provision/scripts/provision_common.py")
    if 'CB16_CI_WORKER_ROOT", "/cb16/worker"' not in common:
        errors.append("provision/scripts/provision_common.py:DOCKER_WORKER_DEFAULT_MISSING")

    inventory = read("scripts/r11_science_g0_inventory.py")
    if 'CB16_RAW_ROOT", str(identity_root)' not in inventory:
        errors.append("scripts/r11_science_g0_inventory.py:DOCKER_RAW_ACCESS_OVERRIDE_MISSING")

    prereq = read("scripts/r11_science_g0_prereq_discovery.py")
    for needle in (
        'CB16_RAW_ROOT", "/cb16/raw"',
        'CB16_G0_ROOT", os.environ.get("CB16_R11_G0_ROOT", "/cb16/g0")',
        'CB16_R104_ROOT", "/cb16/runtime/r104"',
    ):
        if needle not in prereq:
            errors.append("scripts/r11_science_g0_prereq_discovery.py:DOCKER_VOLUME_ROOT_DRIFT")

    if errors:
        print("CB16_R11_DOCKER_PATH_POLICY=FAIL")
        for error in errors:
            print(error)
        return 1

    print("CB16_R11_DOCKER_PATH_POLICY=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
