#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest -v "$ROOT/tests/test_r102_core.py" "$ROOT/tests/test_r102_policy_trace.py"
python -m compileall -q "$ROOT/cb16_local_opt" "$ROOT/scripts"
python "$ROOT/scripts/verify_package_r102.py"
echo R10_2_TESTS_PASS
