# CB16 CI Report

- commit: beba12c24066f629d77bd3d69dafef7b8319e32b
- profile: r104
- started: 2026-09-06T05:09:15Z

PASS repository_policy
PASS python_import_smoke
PASS repo_tests

## R10 engineering blocker

- exception_type: "PermissionError"
- error_code: "HOST_PATH_PERMISSION"
- detail: "1713fb89a9cacf702177d4ef5821a7c5bff6fbce9c205b72d6c1c185ca3d184d.json.zlib"
- scientific_verdict_changed: false
- final_holdout_2025_09_accessed: false
FAIL r104_100gen_research

## Result

- verdict: FAIL
- profile: r104
- finished: 2026-09-06T07:00:57Z
- tests_total: 4
- tests_pass: 3
- tests_fail: 1
