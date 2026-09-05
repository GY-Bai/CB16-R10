#!/usr/bin/env bash
# CB16 Shanxi CI entrypoint.
# Called by cb16-ci-worker with repo checked out in current directory.
# Exit 0 = PASS, non-zero = FAIL/ERROR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${CI_PYTHON:-python3}"
CI_OUT="${CI_OUT:-$ROOT/ci_output}"
CI_PROFILE="${CI_PROFILE:-smoke}"
mkdir -p "$CI_OUT"

case "$CI_PROFILE" in
  smoke|unit|r102|r103|r104|v63) ;;
  *)
    echo "UNKNOWN_CI_PROFILE=$CI_PROFILE" >&2
    exit 64
    ;;
esac

START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FAILED=0

{
  echo "# CB16 CI Report"
  echo
  echo "- commit: ${CI_COMMIT_SHA:-unknown}"
  echo "- profile: $CI_PROFILE"
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

run_test "repository_policy" "$PYTHON_BIN" ci/check_repository_policy.py --root .

run_test "python_import_smoke" "$PYTHON_BIN" - <<'PY'
import json, pathlib, sys
p = pathlib.Path("PACKAGE_MANIFEST_R10_2.json")
assert p.exists(), "PACKAGE_MANIFEST_R10_2.json missing"
data = json.loads(p.read_text())
print("manifest schema:", data.get("schema", "unknown"))
print("python:", sys.version.split()[0])
PY

run_repo_tests() {
  if [ -d tests ] && find tests -name 'test_*.py' -print -quit | grep -q .; then
    run_test "repo_tests" "$PYTHON_BIN" -m pytest tests -q --disable-warnings --maxfail=1
  else
    echo "SKIP repo_tests (no tests/test_*.py)" >> "$CI_OUT/REPORT.md"
  fi
}

case "$CI_PROFILE" in
  smoke)
    ;;
  unit)
    run_repo_tests
    ;;
  r102)
    run_repo_tests
    run_test "r102_5gen_qualification" "$PYTHON_BIN" scripts/run_r102_pipeline.py
    ;;
  r103)
    run_repo_tests
    run_test "r103_20gen_expansion" "$PYTHON_BIN" scripts/run_r103_expansion.py
    ;;
  r104)
    run_repo_tests
    run_test "r104_100gen_research" "$PYTHON_BIN" scripts/run_r104_long_research.py
    ;;
  v63)
    run_repo_tests
    if [ -f ci/run_v63_ci.sh ]; then
      run_test "infra_v63" bash ci/run_v63_ci.sh
    else
      echo "FAIL infra_v63 (ci/run_v63_ci.sh missing)" >> "$CI_OUT/REPORT.md"
      echo "FAIL: infra_v63: ci/run_v63_ci.sh missing"
      FAILED=1
    fi
    ;;
esac

FINISH_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ "$FAILED" -eq 0 ]; then
  VERDICT="PASS"
else
  VERDICT="FAIL"
fi

TESTS_TOTAL=$(grep -cE '^(PASS|FAIL) ' "$CI_OUT/REPORT.md" || true)
TESTS_PASS=$(grep -c '^PASS ' "$CI_OUT/REPORT.md" || true)
TESTS_FAIL=$(grep -c '^FAIL ' "$CI_OUT/REPORT.md" || true)

cat >> "$CI_OUT/REPORT.md" <<EOF

## Result

- verdict: $VERDICT
- profile: $CI_PROFILE
- finished: $FINISH_ISO
- tests_total: $TESTS_TOTAL
- tests_pass: $TESTS_PASS
- tests_fail: $TESTS_FAIL
EOF

cat > "$CI_OUT/result.json" <<JSON
{
  "schema": "CB16_CI_RESULT_V1",
  "job_id": "${CI_JOB_ID:-unknown}",
  "commit_sha": "${CI_COMMIT_SHA:-unknown}",
  "ci_profile": "$CI_PROFILE",
  "verdict": "$VERDICT",
  "started_at": "$START_ISO",
  "finished_at": "$FINISH_ISO",
  "tests_total": $TESTS_TOTAL,
  "tests_pass": $TESTS_PASS,
  "tests_fail": $TESTS_FAIL,
  "model_artifacts": []
}
JSON

# Evidence hashes are computed only after both canonical evidence files are final.
(
  cd "$CI_OUT"
  sha256sum result.json REPORT.md > SHA256SUMS
)

echo "VERDICT=$VERDICT"
exit "$FAILED"
