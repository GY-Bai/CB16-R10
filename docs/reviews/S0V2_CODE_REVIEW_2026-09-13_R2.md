# S0-v2 Sol R2 Code Review — PASS

Date: 2026-09-13

Stage: `CB16_R11_S0V2_DURABLE_LEARNABILITY_FOUNDATION_R0`

Reviewed implementation branch: `ai/r11-s0v2-durable-learnability-foundation-r0`

Reviewed record-binding SHA: `787431a15563f7a1f205b258d1bf49a27695002f`

Reviewed runtime/test/workflow SHA: `8ac88f834cd4b467f2daa61a136a01d0b68a5785`

R1 review commit: `f363344a4b53d63721aa6f7dd60fbed28da7e840`

## Verdict

`PASS`

All three R1 merge blockers are closed in the actual runtime path, with counterexample coverage and exact-SHA Shanxi Docker evidence. No new merge blocker was found in R2.

The strongest justified evidence level is exactly:

`POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`

This review does **not** claim S1 end-to-end learnability, economic evidence, transfer evidence, or any FINAL result.

## R1 blocker closure

### B1 — actual Critic/V-trace mechanical-terminal provenance

Closed.

`joint_actor_critic_losses_v1()` now reads the final durable sample and `batch.mechanical_terminals[end - 1]`, checks tensor/sample agreement, passes the actual mechanical flag into `bootstrap_decision_v1()`, and preserves semantic distinction in `BoundaryBootstrapDecisionV1.boundary_semantics` and diagnostics.

The actual loss path now distinguishes at least `MECHANICAL_TERMINAL` and `ECONOMIC_TERMINAL`; both remain zero-bootstrap terminals without collapsing their provenance.

Counterexamples cover both mechanical and non-mechanical economic terminal behavior and reject a batch/sample mechanical-terminal mismatch.

### B2 — committed child authority for generation switch

Closed.

`commit_child_generation_v1()` no longer accepts an optional caller-supplied record as sufficient authority. It requires a `DurableUpdateStoreV1` and `update_id`, reloads the durable update record, requires `STATUS_COMMITTED`, re-loads and verifies the child checkpoint bytes, and binds runtime policy SHA through `child_policy_identity_v1(child_checkpoint_sha256, policy_generation, policy_id)`.

Missing authority, PREPARED, STAGED, corrupted child checkpoint, and unrelated runtime policy SHA all fail closed.

### B3 — same-instance retry after post-mutation failure

Closed.

Before entering the region in which optimizer/model mutation may occur, `PostCCDurableReplayLearnerV1` marks the live instance restart-required. Any fault from that region leaves the instance in `RESTART_REQUIRED_FROM_DURABLE_PARENT`, and a same-instance retry is rejected before a new update ID or second gradient can be produced.

Recovery reconstructs from the durable parent checkpoint. Hostile tests cover `after_gradient_before_stage`, `after_stage_before_commit`, and `after_commit_before_ack`; each finishes with a single logical committed update and optimizer step `before + 1`.

## Exact-SHA Shanxi qualification

Runtime candidate qualification:

- run `34741816806`
- checkout `8ac88f834cd4b467f2daa61a136a01d0b68a5785`
- tree `669a81cac5fab4646cdab9cc293950e50bde4334`
- business job `103682679229` on `shanxi-docker-r11`
- 119 tests PASS
- 16/16 qualification gates PASS
- artifact `10313545447`
- artifact SHA256 `71f6d9264e9327451250e33745a975cee5c9cc041d3c5cfad2eb380d44099289`

Record-binding-head qualification:

- run `34742010342`
- checkout `787431a15563f7a1f205b258d1bf49a27695002f`
- tree `8ac4c0d334285b66aef637d3d8e00741bb63f3c2`
- business job `103683188989` on `shanxi-docker-r11`
- 119 tests PASS
- 16/16 qualification gates PASS
- artifact `10313111511`
- artifact SHA256 `f468197099fc9cae412714d50d7b5d86b13cb1e225a48ee99efb71f4dd15c38b`

The record-binding head differs from the runtime candidate only in `authority/rearchitecture_r11/CB16_R11_S0V2_REVIEW_CANDIDATE_V1.json`.

## Frozen-history and firewall checks

Historical CC/S0 receipts remain unchanged. FINAL remains closed. No fresh market data was used. No legacy performance fallback was admitted. No ECONOMIC/TRANSFER/end-to-end-learnability claim is authorized by this review.

## Acceptance

Sol accepts S0-v2 for merge to `main`.

Final reviewer receipt:

`authority/rearchitecture_r11/CB16_R11_S0V2_RECEIPT_V1.json`

Successor S1 must freeze its base from the exact accepted merged `main` state and must not use a moving branch base.
