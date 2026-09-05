#!/usr/bin/env bash
# CB16 Shanxi CI entrypoint.
# Called by cb16-ci-worker with repo checked out in current directory.
# Exit 0 = PASS, non-zero = FAIL/ERROR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CI_OUT="${CI_OUT:-$ROOT/ci_output}"
mkdir -p "$CI_OUT"

START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FAILED=0

{
  echo "# CB16 CI Report"
  echo
  echo "- commit: ${CI_COMMIT_SHA:-unknown}"
  echo "- started: $START_ISO"
  echo
} > "$CI_OUT/REPORT.md"

run_test() {
  name="$1"; shift || true
  echo "::group::$name"
  if "$@"; then
    echo "PASS $name" >> "$CI_OUT/REPORT.md"
    echo "PASS: $name"
  else
    echo "FAIL $name" >> "$CI_OUT/REPORT.md"
    echo "FAIL: $name"
    FAILED=1
  fi
  echo "::endgroup::"
}

run_test "repository_policy" python3 ci/check_repository_policy.py --root .

# A harmless smoke check proving the workspace is exact and Python runs.
run_test "python_import_smoke" python3 - <<'PY'
import json, pathlib, sys
p = pathlib.Path("PACKAGE_MANIFEST_R10_2.json")
assert p.exists(), "PACKAGE_MANIFEST_R10_2.json missing"
data = json.loads(p.read_text())
print("manifest schema:", data.get("schema", "unknown"))
print("python:", sys.version.split()[0])
PY

# Repository-local unit smoke (does not require datasets or model weights).
if [ -d tests ] && find tests -name 'test_*.py' | grep -q .; then
  run_test "repo_tests" python3 -m pytest tests -q --disable-warnings --maxfail=1 || true
fi

FINISH_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ "$FAILED" -eq 0 ]; then
  VERDICT="PASS"
else
  VERDICT="FAIL"
fi

TESTS_TOTAL=$(grep -cE '^(PASS|FAIL) ' "$CI_OUT/REPORT.md" || true)
TESTS_PASS=$(grep -c '^PASS ' "$CI_OUT/REPORT.md" || true)
TESTS_FAIL=$(grep -c '^FAIL ' "$CI_OUT/REPORT.md" || true)

cat > "$CI_OUT/result.json" <<JSON
{
  "schema": "CB16_CI_RESULT_V1",
  "job_id": "${CI_JOB_ID:-unknown}",
  "commit_sha": "${CI_COMMIT_SHA:-unknown}",
  "verdict": "$VERDICT",
  "started_at": "$START_ISO",
  "finished_at": "$FINISH_ISO",
  "tests_total": $TESTS_TOTAL,
  "tests_pass": $TESTS_PASS,
  "tests_fail": $TESTS_FAIL,
  "model_artifacts": []
}
JSON

cat >> "$CI_OUT/REPORT.md" <<EOF

## Result

- verdict: $VERDICT
- tests_total: $TESTS_TOTAL
- tests_pass: $TESTS_PASS
- tests_fail: $TESTS_FAIL
EOF

echo "VERDICT=$VERDICT"
exit "$FAILED"
