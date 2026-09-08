# CB16 R11 Stage-4 INTH — Integration Receipt Compiler & Final Qualification Gates R0

## Scope

INTH is an evidence compiler and contract task only. It does not read sibling Integration branches, assemble the production runtime, merge Integration forks, run the final hostile matrix, or emit `R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED`.

The immutable Integration Seed is `86a4ac8a9080cd8382600cb998059e9585495a11`. Scientific status remains `DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`.

## Pre-consolidation compiler

`cb16_local_opt/stage4_integration_gate_compiler_r11.py` accepts eight explicit receipt candidates plus read-only repository evidence. It never discovers sibling branch refs. Final consolidation must supply immutable candidate metadata (`receipt_commit_sha` and `branch_tip_sha`) through the CLI manifest.

A positive result is only:

`STAGE4_INTEGRATION_FORK_RECEIPTS_READY_FOR_CONSOLIDATION`

It can never be the reserved final cutover verdict.

For every INTA–INTH receipt the compiler requires:

- exact task ID and assigned branch;
- exact Integration Seed SHA;
- `PASS` status and unchanged scientific status;
- exact Semantic Guard values;
- implementation head descended from the seed with no merge commit in the fork history;
- a distinct final receipt commit whose sole change is adding that task's receipt directly after `task_head_sha`;
- no post-creation receipt mutation;
- exact equality between actual seed→implementation changed paths and declared `touched_paths`;
- every touched path covered by `owned_paths`;
- no existing-file modification except INTB's exclusive `cb16_local_opt/integration_adapters_r11.py` allowance;
- no final-holdout/raw-data/Semantic-Freeze path mutation;
- exact Semantic Freeze blob at implementation and receipt commits;
- passed tests plus GitHub Actions run identity and success conclusion in `tests.evidence`;
- no sibling Integration dependency;
- no touched-path overlap across Integration tasks.

INTG additionally must explicitly preserve `integrated_runtime_qualification_claimed=false`; the reference/integration harness receipt is not allowed to masquerade as final-runtime hostile qualification.

## Explicit candidate manifest

`scripts/adjudicate_r11_stage4_integration_receipts.py` consumes a JSON object with a `receipts` array. Each row provides `task_id`, repository-relative receipt `path`, `receipt_commit_sha`, and `branch_tip_sha`. The receipt JSON itself stays schema-authoritative. The script is fail-closed and writes a machine-readable pre-consolidation report.

This indirection is intentional: INTH qualification uses only synthetic fixtures and never enumerates or opens another INTA–INTG branch.

## Synthetic qualification

`tests/stage4_fixtures/valid_integration_bundle_v1.json` models eight independent, linear Integration forks. Tests inject hostile evidence for missing/duplicate receipts, wrong branch/seed, non-descendant heads, merge history, broken two-commit protocol, receipt mutation, undeclared paths, existing-file ownership escalation, forbidden/frozen paths, scientific or semantic-guard drift, missing CI evidence, sibling dependency, INTG overclaim, and cross-task path overlap.

A synthetic PASS proves only that the compiler fails closed over its evidence contract. It is not evidence that the consolidated runtime exists or passes H01–H20.

## Final consolidated report contract

`authority/rearchitecture_r11/CB16_R11_STAGE4_FINAL_CUTOVER_REPORT_SCHEMA_V1.json` defines the later final-adjudicator report. The Python validator adds semantic constraints that require, before the reserved positive verdict can be accepted:

- all eight accepted Integration receipts and exact seed ancestry;
- controlled consolidation only;
- exact Semantic Freeze and unchanged scientific status;
- S4A writer registry rebuilt/re-audited at final head with `UNKNOWN_AUTHORITY = 0`;
- all Integration correctness tests passing;
- real S4H H01–H20 executed against the consolidated runtime, with integrated-runtime qualification explicitly true;
- exactly one live canonical authority writer;
- stale and legacy writers unable to mutate;
- Permission bypass impossible;
- replay unable to become new scientific Evidence;
- short startup/steady/drain/shutdown qualification on exactly `[self-hosted, shanxi, cb16-wss-qualification]`;
- verified R10.4 Python reuse in `READY / VERIFIED_CANONICAL_R104_VENV_REUSE` mode;
- FP32 canonical and `AMP=false`;
- final holdout beginning 2025-09 untouched;
- no fresh market data and no new scientific verdict.

Long endurance is not required in this Integration round.

## Qualification workflow

`.github/workflows/cb16-r11-stage4-inth-final-gates.yml` uses exactly the canonical Shanxi runner labels. It performs no network package installation, resolves `ci/resolve_verified_r104_python.py`, requires `READY` and `VERIFIED_CANONICAL_R104_VENV_REUSE`, verifies seed ancestry/Semantic Freeze, runs the synthetic fail-closed test suite, and runs the existing static semantic-authority guard.
