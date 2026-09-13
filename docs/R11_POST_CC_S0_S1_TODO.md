# CB16 R11 Post-CC — Current S0-v2 / S1 Routing

**Status:** HISTORICAL S0 PASS / S0-v2 ACTIVE / S1 SCIENTIFIC QUALIFICATION PAUSED  
**Current scientific/code baseline:** `main@392063881a6ef0dd1776ac579f1a290a134fc49e`  
**CC status:** CLOSED  
**FINAL:** SEALED  
**Fresh market data:** FORBIDDEN  

This document is the current routing authority. It supersedes the earlier execution routing that sent implementers directly into the old S1 task package. It does **not** rewrite historical receipts or scientific verdicts.

## 1. Historical S0 remains frozen

The original post-CC S0 contract/economic-semantic migration is complete and remains PASS.

- frozen implementation base: `fc7102442e91a1c27cf705487c6d06bd64b8ea09`
- qualified implementation identity: `5760061d6c274e9f8796e6608e86bb173018148f`
- receipt-bearing head: `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f`
- evidence: `POST_CC_CONTRACT_MIGRATION_QUALIFIED`

Historical task package:

`docs/post_cc/S0_CONTRACT_MIGRATION_TODO.md`

Historical receipt:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json`

Do not reopen, re-run under the same identity, edit, or reinterpret that receipt.

## 2. Why a new S0-v2 exists

The 2026-09-13 executability review and live code audit show that the next real engineering dependency is not another status check. The canonical system still needs a durable, restart-safe, joint-action learning foundation:

- Brain-ready observation payload persistence;
- immutable observation storage;
- persistent replay materialization without collector-memory truth;
- canonical nominal direction + conditional-risk batch;
- target joint likelihood against persisted true `log_mu`;
- separate Critic / V-trace / Actor update path;
- exactly-once durable update/checkpoint recovery;
- generation switch with same logical account continuation.

These were previously embedded inside a broad S1 qualification package. For one-agent execution and clean review ownership they are now a new versioned engineering qualification stage:

`CB16_R11_S0V2_DURABLE_LEARNABILITY_FOUNDATION_R0`

Current detailed task authority:

**`docs/post_cc/S0_V2_DURABLE_LEARNABILITY_FOUNDATION_TODO.md`**

Implementation branch:

**`ai/r11-s0v2-durable-learnability-foundation-r0`**

Maximum evidence:

`POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`

This does not imply end-to-end learnability, ECONOMIC or TRANSFER evidence.

## 3. Single-agent execution model

The owner will assign one implementation Agent to the designated S0-v2 branch.

The Agent must implement the complete bounded package, add tests, run the dedicated GitHub Actions -> Shanxi Docker qualification, and leave the branch in `READY_FOR_SOL_REVIEW` or a concrete classified blocker state.

The Agent must not:

- create sibling implementation branches;
- split S0-v2 among multiple agents;
- merge its own changes to `main`;
- self-issue Sol's reviewer acceptance;
- stop merely because later pieces are not implemented yet.

Sol reviews the actual diff and exact-SHA CI evidence, with particular attention to formulas, comparisons, direct branch predicates, masks, empty/non-finite cases, boundary semantics, nominal-vs-executed action routing, true `log_mu`, exactly-once recovery and account continuity. Only after that review may Sol merge.

See `docs/ROLES_AND_REVIEW_PROTOCOL.md` and `docs/S_SERIES_TODO_AUTHORING_PRINCIPLES.md`.

## 4. Non-negotiable semantics

S0-v2 and later S1 preserve:

- `Truth != Belief != Decision != Permission != Execution`;
- nominal sampled action != executed action;
- requested risk != confidence;
- true behavior `log_mu` persisted at decision time, never reconstructed;
- Actor = categorical direction + conditional continuous risk;
- FLAT risk exactly 0;
- same logical account continuity;
- signed economics and failure retention;
- no handcrafted regime/cycle/resonance activation subsystem;
- arithmetic expected return as current economic orientation;
- FINAL/fresh-data firewall.

B&H and FLAT remain parallel benchmark components with no master precedence. Model ordering, benchmark comparison and promotion remain distinct.

## 5. Current execution sequence

```text
main@392063881a6ef0dd1776ac579f1a290a134fc49e
    -> S0-v2 task-authoring/navigation commits
    -> ai/r11-s0v2-durable-learnability-foundation-r0
    -> one Agent implements complete S0-v2 candidate
    -> GitHub Actions / Shanxi Docker on exact candidate SHA
    -> READY_FOR_SOL_REVIEW
    -> Sol diff + semantic + CI review
    -> changes requested OR accepted/merged
    -> only then route successor S1 scientific qualification
```

The former `docs/post_cc/S1_END_TO_END_LEARNABILITY_TODO.md` is retained as design/provenance and later-science input, but it is **not the current implementation entrypoint** while S0-v2 is active.

## 6. Successor S1 boundary

After S0-v2 is accepted, a successor S1 task package may qualify actual learnability using the scientific classes already identified:

1. `ACCOUNT_DEPENDENT_ACTION`
2. `DELAYED_CONSEQUENCE_CREDIT`
3. `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION`
4. `OFF_POLICY_VTRACE_CORRECTION`
5. `A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION`

with preregistered multi-seed thresholds and negative controls.

S0-v2 does not automatically start S1, historical training, S2/S3, capacity scaling, FINAL work or economic qualification.

## 7. Failure taxonomy

Use only:

- `READY_FOR_SOL_REVIEW` for a complete implementer candidate;
- `CONTRACT_MISMATCH` for a frozen-semantic conflict;
- `EXECUTION_BLOCKED` for a real tooling/infrastructure blocker;
- `HARDWARE_LIMIT` for a valid workload that cannot run within the declared resource envelope.

Ordinary unimplemented S0-v2 tasks are not blockers. Continue implementing them.