# CB16 CI Report

- commit: edc9f3781ad2c8306d0c3880fafabf0dcaae579e
- profile: r104
- started: 2026-09-06T08:24:46Z

PASS repository_policy
PASS python_import_smoke
PASS repo_tests

## R10 engineering blocker

- exception_type: "PermissionError"
- error_code: "HOST_PATH_PERMISSION"
- detail: "ON_POLICY_REAL_TRACE_RECEIPT.json.tmp"
- scientific_verdict_changed: false
- final_holdout_2025_09_accessed: false
FAIL r104_100gen_research

## Result

- verdict: FAIL
- profile: r104
- finished: 2026-09-06T09:10:14Z
- tests_total: 4
- tests_pass: 3
- tests_fail: 1
