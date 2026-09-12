# CB16 R11 Post-CC — S0/S1 Execution TODO

**Status:** OPEN / canonical post-CC execution plan  
**Frozen planning baseline:** `main@fc7102442e91a1c27cf705487c6d06bd64b8ea09`  
**Planning date:** 2026-09-12  
**Execution order:** `S0 -> S1`  
**CC status:** CLOSED / do not reopen A/B/C/D  
**FINAL:** SEALED  
**Fresh data:** FORBIDDEN unless separately authorized  

This document converts the post-CC design documents into executable work packages. It is the current task-routing authority for the first two post-CC stages.

Read together with:

- `docs/ECONOMIC_ORDERING_AND_PROMOTION.md`
- `docs/POST_CC_SCIENTIFIC_PROGRAM_R0.md`
- `docs/CURRENT_STATE.md`
- `docs/CC_INTEGRATION_HANDOFF.md`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

Detailed packages:

- S0: `docs/post_cc/S0_CONTRACT_MIGRATION_TODO.md`
- S1: `docs/post_cc/S1_END_TO_END_LEARNABILITY_TODO.md`

---

## 1. Why S0/S1 exist

CC proved that the joined architecture can execute a synthetic closed loop with correct provenance, recovery, account continuity, V-trace update, generation switch and a semantically equivalent high-throughput path. Its strongest evidence remains:

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

CC did **not** prove economic edge, transfer, or robust learnability under repeated optimization.

Two post-CC gaps now have to be closed in order.

### Gap family A — economic ordering / baseline / promotion semantics

The current design is now explicit:

1. model-vs-model ordering is one question;
2. B&H comparison is an independent benchmark component;
3. FLAT comparison is an independent benchmark component;
4. promotion is a separate versioned decision rule;
5. **B&H and FLAT have no master precedence and no master winner.**

Historical CC code and receipts were qualified before this clarification. In particular, the old promotion path can still emit an unresolved-owner state when B&H and FLAT disagree. That historical fact must remain auditable, but it is no longer the canonical post-CC behavior.

S0 performs an explicit successor migration. Old receipts are never edited.

### Gap family B — integrated execution is not yet end-to-end learnability evidence

The CC integration canary proves wiring and one committed update, but it still contains qualification shortcuts that are unacceptable as the S1 learnability proof:

- training tensors can be assembled from in-memory rollout `records` instead of being reconstructed exclusively from durable replay facts;
- persistent experience carries observation identity/hash, but the canonical durable path does not yet guarantee materialization of the actual Brain-ready market/account/execution observation payload;
- generic `SequenceBatch` / `CCLearner.update_categorical()` are categorical-action oriented, while the canonical Actor uses a joint distribution: categorical direction + conditional continuous target risk;
- existing learning toys establish component/analytic properties, but do not by themselves prove repeated optimization through the full durable canonical loop produces the preregistered behavioral solution;
- checkpoint change, non-zero gradients or loss reduction are not sufficient S1 evidence.

S1 removes those shortcuts and proves learnability through the canonical post-CC path.

---

## 2. Frozen authority hierarchy

For S0/S1, precedence is:

1. current owner-confirmed principles in `VISION`, `DECISIONS`, `ECONOMIC_ORDERING_AND_PROMOTION` and `POST_CC_SCIENTIFIC_PROGRAM_R0`;
2. the frozen CC science identity and W-01..W-05 semantics;
3. the exact CC integration spec/receipt for what was actually qualified;
4. successor S0 authority once S0 PASSes;
5. S1 run spec frozen before S1 qualification execution.

Historical receipts describe historical qualification states. A newer decision may supersede future routing without rewriting those receipts.

If a proposed S0/S1 implementation would change CC science meaning rather than extend/migrate it, stop with `CONTRACT_MISMATCH`; do not silently patch the old receipt.

---

## 3. Non-negotiable scientific semantics

S0/S1 must preserve:

- `Truth != Belief != Decision != Permission != Execution`;
- requested target risk is not confidence;
- one logical account remains continuous across chunk/pause/recovery/generation boundaries;
- signed account economics and liabilities are preserved;
- nominal sampled action remains separate from permitted/executed action;
- true behavior likelihood `log_mu` is retained from the behavior policy and never reconstructed from execution;
- stochastic RNG / policy / generation provenance remains attributable;
- failures and terminal facts remain in raw experience;
- historical experience is not expired merely because it is old;
- no handcrafted cycle/regime/resonance activation subsystem is introduced;
- no hidden strategic SL/TP/max-hold/cooldown is reintroduced into the policy-neutral environment;
- arithmetic expected return remains the primary economic orientation unless a later owner decision explicitly changes it;
- high-bankruptcy/higher-arithmetic-expectation behavior must be allowed to win the relevant known-answer task;
- FINAL stays sealed and no fresh market data is consumed in S0/S1.

---

## 4. S0 — Contract closure and semantic migration

S0 is a prerequisite for S1. It does **not** reopen CC and does not claim new economic evidence.

Primary responsibilities:

1. freeze the exact post-CC successor contract surface;
2. migrate old B&H/FLAT disagreement behavior to the no-precedence semantics;
3. separate model ordering, benchmark components and promotion in code/API vocabulary;
4. preserve old receipts and old code provenance;
5. freeze S1 task identities, model identities, run manifest, evidence rules and data firewall;
6. emit a machine-readable S0 spec/receipt.

Suggested branch:

`ai/r11-post-cc-s0-contract-migration-r0`

Frozen implementation base:

`fc7102442e91a1c27cf705487c6d06bd64b8ea09`

Required detailed tasks are in `docs/post_cc/S0_CONTRACT_MIGRATION_TODO.md`.

S0 is DONE only when a successor path can represent all four of these without ambiguity:

```text
model_ordering_result
buy_and_hold_component_result
flat_component_result
promotion_result
```

No field, enum, helper or gate may require choosing a master baseline winner.

---

## 5. S1 — End-to-end learnability qualification

S1 may begin implementation only from a qualified S0 authority/head. It is not allowed to assume S0 PASS from planning text.

Primary responsibilities:

1. make observations durably replayable without changing the frozen meaning of W-01..W-05;
2. materialize canonical joint-action learning samples from persistent experience;
3. train through the canonical `direction + conditional risk` likelihood rather than a categorical-only surrogate;
4. prohibit rollout-memory side channels in the qualification path;
5. perform repeated learner updates, committed checkpoints and account-preserving generation switches;
6. demonstrate preregistered behavioral/return improvement on synthetic known-answer tasks across multiple seeds;
7. include negative controls that prevent a false learnability PASS;
8. emit a machine-readable S1 learnability receipt.

Suggested branch:

`ai/r11-post-cc-s1-learnability-r0`

S1 base:

`S0 qualified head`, recorded in the S0 receipt.

Required detailed tasks are in `docs/post_cc/S1_END_TO_END_LEARNABILITY_TODO.md`.

---

## 6. Gap registry at `fc710244...`

| ID | Gap | Stage | Required disposition |
|---|---|---|---|
| P0-G01 | old promotion code can treat B&H/FLAT disagreement as unresolved owner precedence | S0 | successor semantic migration; old path historical only |
| P0-G02 | old Thread-C/integration receipts preserve the old unresolved state | S0 | retain bytes; add successor migration authority, never rewrite |
| P0-G03 | model ordering, benchmark components and promotion can still be conflated by old API names/statuses | S0 | explicit separate schemas/results |
| P0-G04 | no machine-readable post-CC successor economic/promotion identity | S0 | S0 spec + receipt |
| P0-G05 | S1 task/run/model/evidence identities are design prose rather than frozen executable contracts | S0 | preregister S1 registry/run manifest |
| P1-G01 | integration canary learner input can be sourced from live in-memory rollout records | S1 | persistent-replay-only qualification path |
| P1-G02 | durable facts identify observations but do not yet guarantee Brain-ready observation materialization | S1 | content-addressed observation fact/sidecar contract |
| P1-G03 | generic learner batch is categorical-only | S1 | versioned canonical joint-action learning batch |
| P1-G04 | generic learner API does not natively optimize the joint direction+risk likelihood | S1 | joint Actor loss path using canonical distribution |
| P1-G05 | current toys do not constitute full repeated closed-loop convergence evidence | S1 | integrated known-answer training harness |
| P1-G06 | one update + child acts is wiring evidence, not learnability | S1 | preregistered behavior/return success thresholds |
| P1-G07 | insufficient negative controls for false learning PASS | S1 | shuffled/no-signal/impossible controls |
| P1-G08 | exact durable replay/recovery equivalence for repeated learning not yet the central gate | S1 | restart/recovery qualification |
| P1-G09 | S1 evidence name and ceiling are not yet machine-frozen | S0/S1 | preregister then emit receipt |

---

## 7. Cross-stage execution rules

### S0 may modify

- new post-CC successor modules/tests/authority;
- explicit routing away from obsolete post-CC use of old promotion code;
- current navigation/state docs where needed.

### S0 must not

- edit historical CC receipts;
- claim the old receipt was wrong at the time it was issued;
- run market economic qualification;
- open FINAL or consume fresh market data;
- silently redefine W-01..W-05.

### S1 may modify

- new post-CC observation/replay/learning materialization modules;
- successor learner/batch adapters required for canonical joint actions;
- S1 synthetic task harnesses/tests/workflows/receipts.

### S1 must not

- read training tensors from qualification-harness private rollout memory if those tensors are supposed to represent durable replay;
- fabricate `log_mu` or behavior probabilities;
- train against executed action as if it were the nominal sampled action;
- pass solely because loss decreases or parameters change;
- change the economic objective to make a toy easier;
- use FINAL/fresh market data;
- claim ECONOMIC or TRANSFER evidence.

---

## 8. S0 -> S1 handoff contract

S0 receipt must publish at least:

```text
s0_qualified_head_sha
successor_science_contract_id
successor_economic_ordering_contract_id
legacy_promotion_migration_status
historical_receipts_unchanged
s1_task_registry_id
s1_run_spec_id
s1_model_identity_rule
s1_data_scope
s1_evidence_ceiling
FINAL_opened=false
fresh_data_used=false
```

S1 refuses to start canonical qualification if any required S0 field is missing or unresolved.

---

## 9. Evidence ladder

S0 may claim at most:

`POST_CC_CONTRACT_MIGRATION_QUALIFIED`

S1 target evidence, if all gates pass:

`INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER`

This means the canonical post-CC loop learned preregistered synthetic tasks end-to-end. It does **not** mean profitable historical trading, market edge, transfer, or production readiness.

`ECONOMIC` and `TRANSFER` remain later-stage claims.

---

## 10. Failure classification

Every S0/S1 run or task must use one of:

- `PASS`
- `SCIENTIFIC_FAIL` — valid experiment, hypothesis/threshold not met;
- `CONTRACT_MISMATCH` — attempted implementation conflicts with frozen semantics;
- `EXECUTION_BLOCKED` — infrastructure/software failure prevents valid result;
- `HARDWARE_LIMIT` — valid workload cannot execute within declared hardware resource bounds;
- `EVIDENCE_INSUFFICIENT` — execution completed but preregistered evidence requirements were not met.

Do not rescue a `SCIENTIFIC_FAIL` by changing thresholds, task distribution, objective or seeds in place. Any changed scientific question receives a new version/run identity.

---

## 11. Integration / merge discipline

S0 is one bounded successor-contract program. S1 is one bounded learnability program. They are not four new independent CC threads.

Recommended sequence:

```text
main@fc710244...
      |
      +-- S0 contract migration -> S0 qualification receipt
                                  |
                                  +-- S1 learnability -> S1 qualification receipt
```

After S0 PASS, merge/update `main` before freezing the S1 implementation base. Do not develop S1 against an imagined S0 API and later reinterpret mismatches.

---

## 12. Immediate Sol handoff

Sol should execute in this order:

1. read this master TODO;
2. execute every mandatory S0 item in `S0_CONTRACT_MIGRATION_TODO.md`;
3. qualify S0 and publish its receipt;
4. freeze S1 base at the qualified S0 head;
5. execute every mandatory S1 item in `S1_END_TO_END_LEARNABILITY_TODO.md`;
6. stop after the S1 verdict and receipt;
7. do **not** automatically begin historical economic qualification or capacity scaling after S1.

The next post-S1 program must be authorized from the actual S1 evidence, not presumed by this plan.
