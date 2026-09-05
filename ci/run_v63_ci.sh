#!/usr/bin/env bash
# Fixed, auditable Infra V6.3 qualification entrypoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${CI_PYTHON:-python3}"

"$PYTHON_BIN" ci/v63_preflight.py
