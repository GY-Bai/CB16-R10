# CB16 R11 Stage-4 S4I — Receipt Validator & Qualification Gate Compiler R0

## Scope

S4I is an infrastructure evidence compiler. It does **not** implement runtime authority, adoption, leasing/fencing, permission, storage ownership, legacy retirement, or hostile-cutover production behavior. It consumes the frozen Stage-4 Gatework manifest and task-receipt schema, then adjudicates Wave-1 receipt evidence fail closed.

Highest rule remains:

`SEMANTIC CONTRACTS_ARE_AUTHORITY__LEGACY_PYTHON_IMPLEMENTATION_IS_NOT_AUTHORITY`

The scientific status remains:

`DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`

The only successful S4I aggregate verdict is:

`STAGE4_WAVE1_RECEIPTS_READY_FOR_INTEGRATION`

S4I cannot emit `R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED`; that verdict belongs only to the final integrated qualification.

## Inputs

The compiler reads:

- `authority/rearchitecture_r11/CB16_R11_STAGE4_GATEWORK_V1.json`
- `authority/rearchitecture_r11/CB16_R11_STAGE4_TASK_RECEIPT_SCHEMA_V1.json`
- exactly one canonical receipt path for each `S4A` through `S4I`
- read-only Git evidence from the repository containing the receipts

S4I has no dependency on S4A-S4H implementations or branches. The included fixture set is synthetic and exists only to qualify the compiler logic before real sibling receipts exist.

## Fail-closed checks

For every Wave-1 task, the compiler requires:

1. exactly one receipt and the exact expected task ID;
2. exact expected branch and exact Gatework base `0f18e08ec7250b9b4e45c62803c25be966834390`;
3. receipt conformance to the frozen receipt schema;
4. `status == PASS`;
5. all frozen semantic guards present with exact values and no extra guard keys;
6. unchanged frozen scientific status;
7. `tests.passed == true`, at least one test command, and non-empty test evidence;
8. a real implementation commit descended from Gatework base and containing no merge commit between base and task head;
9. declared `touched_paths` exactly equal to `git diff --name-only <base>...<task_head>`;
10. no Semantic Freeze, final-holdout, 2025-09, or frozen raw-data manifest path change;
11. every changed implementation path declared in `owned_paths`;
12. no existing-file modification except the manifest-authorized S4E exception for `cb16_local_opt/integration_adapters_r11.py`;
13. all additive files Stage-4 namespaced;
14. one committed receipt-addition event, separate from the implementation head;
15. implementation head is an ancestor of the receipt commit and the receipt commit is an ancestor of the evidence HEAD;
16. receipt bytes at evidence HEAD match the receipt-creation commit, and the receipt path is clean in the working tree;
17. no exact or directory-prefix ownership overlap between different tasks.

The Gatework manifest itself is also checked so no task other than S4E can acquire an existing-file exception.

## Negative qualification matrix

The test suite rejects at least:

- missing and duplicate task receipts;
- FAIL, BLOCKED, and NOT_RUN receipts;
- wrong Gatework base or wrong branch;
- malformed implementation SHA;
- changed scientific status;
- final-holdout path changes;
- fresh-market-data guard violations;
- sibling dependency use;
- duplicate/overlapping path ownership;
- undeclared changed paths;
- Semantic Freeze blob mismatch;
- fake PASS with `tests.passed=false`;
- PASS with missing test evidence;
- non-commit or non-descendant implementation heads;
- implementation head not preceding the receipt commit;
- ambiguous/missing receipt commit evidence;
- unauthorized existing-file modifications;
- unauthorized use of the S4E integration-adapter exception;
- receipt mutation after creation or dirty receipt worktree state.

## CLI

Run from repository root:

```bash
python scripts/adjudicate_r11_stage4_wave1_receipts.py --repo-root .
```

The command scans only the canonical Stage-4 receipt directory. Any missing receipt, malformed receipt, invalid Git evidence, or violated ownership/semantic invariant returns a FAIL JSON object and non-zero exit code.

## Integration notes

Final integration should first bring accepted Wave-1 implementation and receipt commits into one evidence history while preserving each task's original implementation-head ancestry. Once all nine canonical receipt files exist in that history, run the S4I compiler. A successful S4I result means only that the Wave-1 receipts are structurally and evidentially ready for integration. The integrated runtime still must undergo the final hostile/cutover qualification before any final Stage-4 infrastructure verdict can be issued.
