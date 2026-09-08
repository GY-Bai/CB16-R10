# CB16 R11 Stage-4 — Canonical Authority Cutover & Legacy Retirement R0

## 1. Purpose

Stage-4 is an infrastructure/authority cutover stage. It does not create a new scientific verdict. Its purpose is to make the R11 runtime the only legitimate operational authority and to retire legacy R10/R10.2/R10.4 paths from authoritative-write status.

Highest rule:

`SEMANTIC CONTRACTS ARE AUTHORITY. LEGACY PYTHON IMPLEMENTATION IS NOT AUTHORITY.`

## 2. Frozen upstream

All Wave-1 task branches MUST fork from the final `ai/r11-stage4-gatework-r0` head created on top of:

- Stage-3 integration branch: `ai/r11-stage3-integration-r0`
- Stage-3 integration head: `35d6dccd85fe053f4d14d1d43b3110496947cdc1`
- Stage-3 integrated smoke run: `34184028833` — PASS
- Stage-2 qualified base: `92f1ca011aec27bbe9e89ac5ba9afe74817ee2de`
- Semantic Freeze blob: `3c401a0a350984381912f7860181e3e96eb8d7cf`

No task may rebase itself onto a sibling Stage-4 branch.

## 3. Frozen scientific boundary

Every task must preserve:

- `DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED / TRUE_WORSE_THAN_SHUFFLE`
- final holdout beginning `2025-09-01T00:00:00Z` unopened
- no fresh market data
- no mutation of frozen historical market bytes
- Frozen Physics and Supervisor semantics
- Teacher meaning
- Evidence meaning
- gradient ownership
- Champion/Challenger semantics
- requested-risk semantics (`requested_risk != confidence`)
- replay is engineering replay, never new scientific evidence
- `TRUTH != BELIEF != DECISION != PERMISSION`

Stage-4 may change ownership, lifecycle, recovery, storage/runtime layout, entrypoints, compatibility gates and fail-closed enforcement only.

## 4. Explicitly out of scope

The following are not Stage-4 work:

- 30m/2h/6h endurance escalation unless separately authorized by the user
- distributed/cross-machine authority or split-site runtime
- new scientific model architecture
- new Teacher/Physics/Evidence semantics
- performance tuning, queue-depth tuning, worker-count tuning
- final holdout opening
- fresh data download
- main-branch merge

## 5. Parallel Wave-1 rule

Wave-1 consists of S4A..S4I. They are intentionally decoupled.

A Wave-1 agent MUST NOT:

1. import a module created by a sibling Stage-4 task;
2. cherry-pick a sibling Stage-4 commit;
3. use a sibling result as an implementation prerequisite;
4. redefine another task's contract;
5. merge to the Stage-4 integration branch;
6. claim final Stage-4 qualification.

If a task contains internal dependencies, complete them in order inside that same task.

Cross-task wiring belongs only to the final integration/adjudication wave.

## 6. Wave-1 task partition

### S4A — Authority Surface Inventory and Writer Registry

Find and classify every path capable of changing authoritative generation/evidence/journal/snapshot/challenger/tournament/champion/checkpoint/release state. Produce a machine-readable writer registry and fail-closed static audit. Do not alter writer behavior.

### S4B — Canonical Runtime Entrypoint and Lifecycle

Create a production-shaped R11 runtime entrypoint/lifecycle controller independent of qualification harnesses. It must expose explicit boot/verify/recover/acquire/start/generation/drain/seal/stop phases and protocol interfaces for later integration. Do not depend on S4C/S4D/S4F/S4G implementations.

### S4C — One-Time Authority Adoption and Lineage Preservation

Define and implement a one-time R10/R11-source-state adoption receipt and verifier. Adoption preserves history; it never manufactures evidence or rewrites lineage. Repeating the identical adoption must be an idempotent no-op or fail closed; conflicting adoption must fail closed.

### S4D — Runtime Singleton Lease and Fencing

Implement single-owner authority lease/fencing semantics. Two runtimes must never simultaneously believe they own the authoritative writer role. Stale owners/tokens must be rejected after handoff/recovery.

### S4E — Permission Boundary and Legacy Bypass Closure

Audit and close direct production paths that could bypass ActionIntent -> Frozen Supervisor -> Frozen Physics/Permission. This task has exclusive Wave-1 ownership of `cb16_local_opt/integration_adapters_r11.py` if an existing-file change is required.

### S4F — Canonical Persistent State Roots and Storage Ownership

Define canonical SSD/HDD roots, ownership classes, mutable/append-only/immutable rules, startup verification, incomplete-seal/orphan handling and safe recovery contracts. Do not touch frozen raw 1m bytes.

### S4G — Legacy Runtime Retirement and Authoritative-Write Rejection

Implement a legacy-retirement policy/guard that classifies legacy code as oracle/reference/replay/diagnostic/compatibility only and rejects attempts to acquire authoritative write capability. Do not delete historical code merely to make tests pass.

### S4H — Hostile Cutover and Split-Brain Test Matrix

Build an independent hostile scenario harness/model covering duplicate runtime, stale lease, conflicting adoption, legacy writer attempt, crash around cutover, duplicate release/commit, permission bypass attempt and replay laundering. It may use existing R11 public contracts but no sibling Stage-4 implementation.

### S4I — Stage-4 Receipt Validator and Qualification Gate Compiler

Implement task-receipt validation and final gate aggregation logic against fixtures. It must be independently testable before sibling receipts exist. It must never turn missing/invalid evidence into PASS.

## 7. Task receipt

Every Wave-1 task must write exactly one primary receipt under:

`authority/rearchitecture_r11/stage4_receipts/<TASK_ID>_RECEIPT_V1.json`

It must conform to:

`authority/rearchitecture_r11/CB16_R11_STAGE4_TASK_RECEIPT_SCHEMA_V1.json`

A task may report `PASS`, `FAIL`, `BLOCKED`, or `NOT_RUN`. Only `PASS` is eligible for integration.

The receipt is not a scientific verdict.

## 8. Real-machine policy

Only tasks that genuinely require hardware/OS ownership behavior may use the Shanxi qualification runner:

`runs-on: [self-hosted, shanxi, cb16-wss-qualification]`

Do not use `cb16-r10-canonical`, broad `[self-hosted, shanxi]`, or OCI for qualification.

Short Stage-4 qualification is allowed. Long endurance is not implicitly authorized.

## 9. Final integration wave

The final integration branch is reserved:

`ai/r11-stage4-integration-r0`

It must be created only after Wave-1 receipts have been reviewed. The integrator may wire the accepted task APIs together and resolve integration-only conflicts, but must not silently rewrite a task's semantics to force PASS.

The integrated hostile qualification must prove at minimum:

1. exactly one authoritative runtime owner;
2. identical adoption is idempotent and conflicting adoption is rejected;
3. legacy authoritative-write attempts are rejected;
4. a second R11 runtime is rejected or fenced;
5. restart preserves one authoritative lineage;
6. stale writer/token cannot mutate state;
7. Permission/Physics cannot be bypassed;
8. replay creates zero new scientific evidence;
9. final holdout remains unopened;
10. Semantic Freeze remains byte-identical;
11. scientific status remains unchanged.

Only the integration adjudicator may emit:

`R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED`

That verdict is an infrastructure verdict only.
