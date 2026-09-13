# CB16 R11 S0-v2 — Durable Learnability Foundation TODO

**Stage identity:** `CB16_R11_S0V2_DURABLE_LEARNABILITY_FOUNDATION_R0`  
**Status:** AUTHORIZED FOR ONE IMPLEMENTER AGENT  
**Scientific/code baseline:** `main@392063881a6ef0dd1776ac579f1a290a134fc49e`  
**Historical S0 status:** PASS / FROZEN / DO NOT REWRITE  
**Historical S0 qualified implementation:** `5760061d6c274e9f8796e6608e86bb173018148f`  
**Historical S0 receipt-bearing head:** `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f`  
**Implementation branch:** `ai/r11-s0v2-durable-learnability-foundation-r0`  
**Executor model:** one implementation Agent only  
**Reviewer / merge authority:** Sol  
**FINAL:** SEALED  
**Fresh market data:** FORBIDDEN  

This is a **new versioned S0 stage**. It does not reopen, replace, invalidate, or edit the historical post-CC S0 contract-migration program or its receipts. The old S0 remains `POST_CC_CONTRACT_MIGRATION_QUALIFIED`.

The owner has deliberately simplified execution: one Agent implements the whole bounded S0-v2 package on the branch above; Sol reviews the actual diff, Shanxi CI evidence, formulas, comparisons, branches, masks, boundaries and verdict logic; only Sol decides whether the candidate may merge to `main`.

The implementer must not create sibling implementation branches, split this into parallel agents, merge its own PR, or issue the final reviewer acceptance on Sol's behalf.

---

# 0. Purpose and exact boundary

CC already proved a synthetic joined loop can execute one valid update. Historical S0 already closed economic-ordering / B&H / FLAT / promotion migration. The remaining engineering gap before a meaningful learnability experiment is a durable, restart-safe, joint-action learning foundation.

S0-v2 must implement and qualify this bounded chain:

```text
canonical Brain observation
  -> immutable content-addressed observation fact
  -> authoritative transition / immutable experience linkage
  -> persistent replay selection
  -> restart-safe replay materialization
  -> canonical nominal direction + target-risk batch
  -> persisted true behavior log_mu
  -> target joint log_pi on the nominal behavior action
  -> separate Critic + boundary-aware bootstrap + V-trace
  -> joint Actor/Critic update
  -> durable exactly-once update provenance
  -> child checkpoint
  -> explicit generation switch
  -> same logical account continuation
```

S0-v2 is a **foundation qualification**, not the final learnability-science verdict. It does not need to prove the five later known-answer tasks converge across the frozen multi-seed rule. Those tasks belong to the successor S1 scientific qualification after S0-v2 is reviewed and merged.

Maximum S0-v2 evidence if every gate passes:

`POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`

S0-v2 may **not** claim:

- `INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER`;
- `ECONOMIC`;
- `TRANSFER`;
- historical profitability;
- production readiness.

---

# 1. Authority and files to read before editing

The implementer must start from the designated branch and read, in this order:

1. `docs/CURRENT_STATE.md`
2. `docs/ROLES_AND_REVIEW_PROTOCOL.md`
3. `docs/S_SERIES_TODO_AUTHORING_PRINCIPLES.md`
4. this S0-v2 TODO
5. `docs/reviews/S0_S1_EXECUTABILITY_REVIEW_2026-09-13.md`
6. `docs/PRINCIPLE_ALIGNMENT.md`
7. `docs/COMPONENT_REQUIREMENTS.md`
8. `docs/TRAINING_ALGORITHM_R0.md`
9. `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
10. `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`
11. `authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json`
12. frozen successor contracts already on main, especially:
    - `cb16_local_opt/post_cc_observation_contract_v1.py`
    - `cb16_local_opt/post_cc_joint_replay_contract_v1.py`
13. reusable qualified CC implementation surfaces, including the canonical Brain, policy distribution/RNG, runtime/account/execution, immutable experience/replay, Critic/V-trace, learner transaction/checkpoint/recovery and generation-switch code.

Precedence is current owner principle/document authority -> frozen CC semantics -> historical S0 successor contracts -> this S0-v2 execution contract -> implementation details.

If a requested implementation requires changing a frozen scientific meaning rather than filling the authorized missing implementation, stop with `CONTRACT_MISMATCH` and identify the exact conflicting authority. Do not silently reinterpret it.

---

# 2. Non-negotiable semantics

All implementation and tests must preserve:

- `Truth != Belief != Decision != Permission != Execution`;
- nominal sampled action is not executed action;
- requested target risk is not confidence;
- canonical Actor action is categorical direction + conditional continuous target risk;
- `FLAT` has exact target risk `0` and a point-mass risk contribution;
- non-FLAT target risk follows the frozen continuous support semantics;
- true behavior `log_mu` is computed by the behavior policy at decision time and persisted; it is never reconstructed later from execution, quantity, reward or target policy;
- same logical account continuity across chunk, restart, checkpoint and allowed generation switch;
- negative equity/liability and failure facts are not erased;
- computational truncation, chunk boundary, task horizon and mechanical/economic terminal are distinct;
- frozen market organ gradient ownership remains frozen; authorized Brain/Critic paths remain trainable;
- no handcrafted regime/cycle/resonance activation subsystem;
- no hidden strategic SL/TP/max-hold/cooldown;
- arithmetic expected return remains the economic orientation for later science;
- FINAL remains sealed and fresh market data is not accessed.

These are fail-closed invariants. Do not add permissive fallbacks around them.

---

# 3. Execution rule: missing functionality is the task, not a blocker

The implementer must not stop after reporting that durable replay, joint learner, restart reconstruction, recovery or generation continuity are not yet implemented. Those are exactly the S0-v2 deliverables.

A normal execution must leave meaningful code commits on the designated branch.

Legitimate early stop categories are only:

- `CONTRACT_MISMATCH` — frozen authorities conflict with the requested implementation;
- `EXECUTION_BLOCKED` — repo/tool/infrastructure failure prevents valid implementation or test execution;
- `HARDWARE_LIMIT` — a valid required workload cannot execute inside the declared resource envelope;
- a genuinely new scientific owner choice that is not resolved anywhere in current authority.

Ordinary missing code, failing unit tests during development, or the absence of a final receipt are not early-stop reasons. Fix and continue.

---

# 4. Single-Agent implementation tasks

Use these IDs exactly. Do not introduce a competing task-numbering system.

## S0V2-001 — Freeze S0-v2 baseline manifest

Create:

`authority/rearchitecture_r11/CB16_R11_S0V2_BASELINE_V1.json`

It must record at minimum:

- stage identity;
- scientific/code baseline `392063881a6ef0dd1776ac579f1a290a134fc49e`;
- actual branch base / task-document handoff SHA;
- historical S0 receipt path and immutable blob identity;
- CC integration spec/receipt identities;
- observation/joint-replay frozen contract identities;
- canonical Brain/distribution identities;
- FINAL/fresh-data firewall state;
- declared evidence ceiling.

The baseline manifest must distinguish the scientific/code baseline from later documentation-only task-authoring commits. Do not repeat the prior qualified-head/receipt-head ambiguity.

## S0V2-002 — Implement `PostCCObservationFactV1`

Create or complete:

`cb16_local_opt/post_cc_observation_fact_v1.py`

Reuse `post_cc_observation_contract_v1.py`; do not invent an incompatible second observation contract.

The durable fact must deterministically encode the canonical separated Brain inputs and identities required by the frozen contract, including market/account/execution payloads, observation hash, normalizer/source identity, account lineage, decision index/environment time and content hash.

Required tests:

- deterministic serialization;
- round-trip reconstruction;
- same semantic observation -> same hash;
- payload/hash corruption -> fail closed;
- W-01 observation identity agreement;
- no future-information fields admitted by the codec.

## S0V2-003 — Add immutable content-addressed observation store

Create or complete:

`cb16_local_opt/post_cc_observation_store_v1.py`

Required behavior:

- idempotent write of identical content;
- immutable read by content/observation identity;
- reject conflicting content for an existing identity;
- safe close/reopen and process restart;
- deterministic on-disk representation;
- fail closed on checksum/hash mismatch;
- no mutation after commit.

Do not use a process-local dictionary as qualification truth.

## S0V2-004 — Bind observation persistence to canonical collection

At every policy decision that can enter S0-v2 replay:

1. build the canonical Brain observation;
2. persist the observation fact before the experience link is considered valid;
3. bind durable reference/hash to the policy decision/transition/experience path without changing W-01..W-05 meaning;
4. assert the policy decision observation hash matches the persisted fact;
5. never create action/log_mu observations for mechanical no-decision advances.

Add transaction-order and corruption tests.

## S0V2-005 — Implement `PostCCJointReplaySampleV1`

Create or complete:

`cb16_local_opt/post_cc_joint_replay_v1.py`

Reuse `post_cc_joint_replay_contract_v1.py` and carry at least the canonical equivalents of:

- sequence / transition / account lineage identities;
- decision index and environment time;
- reconstructed market/account/execution observation;
- nominal direction;
- nominal target risk;
- risk measure kind/mask;
- persisted behavior `log_mu`;
- behavior policy/generation/hash identity;
- reward and discount;
- explicit boundary/bootstrap semantics;
- replay sampling probability/weight;
- source fact hashes/provenance.

Executed direction/quantity may appear as consequence context only. It may not replace the nominal action used for policy likelihood.

## S0V2-006 — Build persistent replay materializer

Create or complete:

`cb16_local_opt/post_cc_replay_materializer_v1.py`

Inputs must be persistent sequence/transition references plus durable stores/contracts. Output is ordered `PostCCJointReplaySampleV1` materialization.

Hard qualification rule:

> The S0-v2 learner path may not receive collector-private rollout tensors or live `records` as training truth.

Mandatory restart sentinel:

1. collect and persist valid experience;
2. record expected materialized sample identities/hashes;
3. destroy all collector/live Python objects;
4. start a fresh materialization context/process path;
5. reconstruct solely from durable state + manifest;
6. require identical ordered sample identities/hashes, with only explicitly frozen floating serialization tolerances where unavoidable.

If the restart path secretly reads cached collector objects, the gate fails.

## S0V2-007 — Version joint-action sequence batch

Create or complete:

`cb16_local_opt/post_cc_joint_batch_v1.py`

The batch must expose canonical market/account/execution tensors plus nominal direction, nominal target risk, risk kind/mask, persisted behavior `log_mu`, rewards, discounts, bootstrap/boundary data, replay weights/probabilities and provenance IDs.

Validation must fail closed for at least:

- FLAT with non-zero risk;
- non-FLAT target risk outside frozen support;
- non-finite `log_mu`;
- missing behavior identity;
- observation hash mismatch;
- time-order violation within one account lineage;
- missing durable source reference.

## S0V2-008 — Implement canonical target joint `log_pi`

Create or complete:

`cb16_local_opt/post_cc_joint_policy_loss_v1.py`

Use `CCCentralBrain` and the existing canonical policy distribution. For every replayed action, evaluate target `log_pi` on the **nominal behavior direction + nominal target risk**.

Mandatory known-answer component tests:

- FLAT point mass;
- LONG/SHORT continuous risk density;
- same-policy `log_pi == log_mu` within frozen numeric tolerance;
- controlled off-policy likelihood ratio;
- execution rejection/clamping/quantity change does not alter the nominal likelihood input;
- reconstructed/fabricated `log_mu` is rejected by the S0-v2 path.

This module is high-risk for formula translation. Sol will independently inspect density, log-Jacobian, indexing, masking, numerical support and ratio logic before merge.

## S0V2-009 — Integrate separate Critic, bootstrap and V-trace

Reuse the qualified Critic/value and V-trace semantics instead of creating a second algorithm.

The new path must explicitly distinguish:

- compute truncation;
- replay/chunk boundary;
- task/dataset horizon;
- mechanical/economic terminal/account death.

Do not derive all bootstrap behavior from a single generic `done` branch.

Add analytic/known-answer tests for boundary masks and same-policy/off-policy V-trace behavior.

## S0V2-010 — Implement reusable durable-replay learner runtime

Create or complete:

`cb16_local_opt/post_cc_learner_v1.py`

It must:

- accept only validated S0-v2 joint batches produced from durable replay materialization;
- compute canonical target joint likelihood;
- compute Critic/V-trace/Actor objectives under existing semantics;
- preserve gradient ownership;
- keep behavior checkpoint immutable during its collection unit;
- update only authorized parameters;
- create a separate child checkpoint;
- expose deterministic provenance for the update;
- reject an in-memory collector-record shortcut.

Do not delete or rewrite historical categorical learner paths merely because S0-v2 no longer qualifies through them.

## S0V2-011 — Durable learning-update provenance and exactly-once recovery

Create a successor durable update record compatible with W-04 meaning. It must link:

- sampled sequence IDs;
- materialized sample hashes;
- sampling probability/weight;
- behavior/target/parent policy identities;
- Actor/Critic losses and V-trace diagnostics;
- gradient ownership summary;
- optimizer step before/after;
- parent/child checkpoint identities;
- transaction/commit status.

Qualify exactly-once behavior with hostile interruption points including, where architecture permits:

- before update commit;
- after model/checkpoint staging but before authority commit;
- after authority commit before caller acknowledgement;
- process restart/retry.

The same logical update must never be applied twice.

## S0V2-012 — Generation switch and same-account continuation

Bind a committed child checkpoint through the existing generation-switch authority.

Required proof:

- parent behavior checkpoint is not mutated in place;
- child identity is distinct and attributable;
- switch occurs only at the authorized boundary;
- same logical account lineage, holdings, liabilities and causal state continue across the switch;
- child actually performs a later policy decision;
- creating a new flat account to demonstrate child action fails the test.

## S0V2-013 — Foundation qualification runner

Create:

`scripts/run_r11_s0v2_foundation_qualification.py`

The runner must verify exact baseline/authority identities and execute the bounded foundation gate set without FINAL/fresh-data access. It must produce machine-readable raw results rather than only console text.

At minimum compile gates for:

- baseline identity;
- historical receipt immutability;
- observation fact integrity;
- durable observation store restart;
- observation/runtime linkage;
- durable replay reconstruction after live-object destruction;
- joint batch invariants;
- nominal-action likelihood integrity;
- true persisted `log_mu` integrity;
- Critic/bootstrap/V-trace component correctness;
- gradient ownership;
- exactly-once update recovery;
- child checkpoint immutability;
- same-account generation continuity;
- FINAL/fresh-data firewall;
- legacy performance fallback/import firewall where touched by the new path.

## S0V2-014 — Pre-merge Shanxi Docker CI

Add a dedicated GitHub Actions workflow, recommended:

`.github/workflows/cb16-r11-s0v2-foundation-qualification.yml`

Requirements from current review protocol:

- run through GitHub Actions, not an improvised ChatGPT sandbox business test;
- use shared preflight and `[self-hosted, shanxi-docker-r11]`;
- verify canonical Python/runtime before tests;
- print/record actual checkout SHA;
- execute focused S0-v2 tests plus relevant existing regression tests;
- run the S0-v2 qualification runner;
- archive machine-readable qualification artifacts;
- fail closed if expected tests were not collected/executed;
- do not claim PASS from repo-guard alone.

The implementer may trigger and debug this CI on the candidate branch. Green CI is necessary but does not replace Sol review.

## S0V2-015 — Candidate machine-readable review record

Create:

`authority/rearchitecture_r11/CB16_R11_S0V2_REVIEW_CANDIDATE_V1.json`

This is **not** a self-issued final receipt. It must report:

- exact branch/head SHA;
- baseline manifest identity;
- changed implementation surfaces;
- test inventory and counts;
- Shanxi workflow/run/job/artifact identities;
- all compiled gate results;
- FINAL/fresh-data state;
- strongest evidence requested;
- unresolved issues;
- terminal state `READY_FOR_SOL_REVIEW` when all implementer-side gates are green.

If a real blocker exists, use the classified blocker instead of fabricating `READY_FOR_SOL_REVIEW`.

## S0V2-016 — Handoff to Sol; do not self-merge

The implementer finishes by pushing the complete candidate to:

`ai/r11-s0v2-durable-learnability-foundation-r0`

and reporting:

- exact head SHA;
- concise diff inventory;
- focused/regression test results;
- Shanxi workflow/run/job/artifact IDs;
- candidate record path;
- any high-risk formula/control-flow locations that deserve reviewer attention.

Then stop. Do not merge to `main` and do not start successor S1.

---

# 5. Required test emphasis for high-density logic

Because formula translation and naked branch logic are known implementation risk areas, tests must include explicit counterexamples rather than only happy paths.

Sol will inspect at least:

1. direction index mapping SHORT/FLAT/LONG;
2. FLAT risk point-mass versus non-FLAT density;
3. transformed-risk density/log-Jacobian and endpoint support;
4. `log_pi - log_mu` sign/order and V-trace ratio construction;
5. boundary/bootstrap masks under each terminal/truncation class;
6. nominal versus permission/execution field routing;
7. account lineage continuity after child generation switch;
8. exactly-once transaction comparisons and retry branches;
9. observation/content hash comparisons and collision branches;
10. empty/missing/non-finite/mask cases;
11. conditions using `<`, `<=`, `>`, `>=` around thresholds/support boundaries;
12. any code path that could accidentally turn a failure into a silent fallback.

Tests must make incorrect inversions fail visibly.

---

# 6. Files the implementer must not rewrite

Do not edit historical machine-readable receipts merely to make new tests pass, including:

- `CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`;
- `CB16_R11_CC_THREAD_C_RECEIPT_V1.json`;
- `CB16_R11_POST_CC_S0_RECEIPT_V1.json`.

Do not rewrite historical S0 TODO to pretend S0-v2 was part of the original qualification.

Do not reopen CC A/B/C/D implementation branches or restore legacy performance fallbacks.

If a frozen historical file is inconsistent with a new successor need, create a versioned successor artifact and document the relationship.

---

# 7. S0-v2 gate and verdict semantics

Implementer-side terminal states:

- `READY_FOR_SOL_REVIEW`
- `CONTRACT_MISMATCH`
- `EXECUTION_BLOCKED`
- `HARDWARE_LIMIT`

The implementation Agent does not issue the final merge acceptance.

After receiving the branch, Sol independently reviews:

1. branch lineage and diff scope;
2. frozen authority/receipt immutability;
3. formulas and high-density numerical logic;
4. comparisons, direct branch predicates, masks, missing/empty handling and boundaries;
5. durable-only replay truth and restart proof;
6. nominal-action / true-log_mu integrity;
7. checkpoint/recovery/account continuity;
8. GitHub Actions -> Shanxi Docker evidence for the exact reviewed SHA;
9. machine-readable candidate record consistency.

Sol may request changes, reject the candidate, or accept and merge.

Only after Sol acceptance may the merged S0-v2 authority state claim at most:

`POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`

A separate final receipt may be created/accepted during review if the current role protocol requires reviewer-side evidence binding. Do not manufacture that receipt before the reviewed SHA and CI evidence exist.

---

# 8. What happens after S0-v2

S0-v2 ends at the durable learning foundation. It does not automatically authorize historical market training.

The successor S1 scientific program, after explicit routing from the accepted S0-v2 evidence, should exercise the already frozen scientific classes:

1. `ACCOUNT_DEPENDENT_ACTION`
2. `DELAYED_CONSEQUENCE_CREDIT`
3. `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION`
4. `OFF_POLICY_VTRACE_CORRECTION`
5. `A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION`

with preregistered multi-seed thresholds and negative controls.

That later S1 determines whether the durable foundation actually learns the known answers. S0-v2 only proves the foundation is semantically correct, durable, restart-safe, recoverable and capable of performing the canonical joint update without forbidden side channels.

STOP after S0-v2 handoff to Sol.