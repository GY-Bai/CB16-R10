# CB16 R11 Post-CC S1 — End-to-End Learnability Execution & Qualification TODO

**Stage:** S1  
**Status:** ACTIVE / AUTHORIZED FOR IMPLEMENTATION  
**S0 prerequisite:** PASS  
**S0 frozen implementation base:** `fc7102442e91a1c27cf705487c6d06bd64b8ea09`  
**S0 qualified implementation identity:** `5760061d6c274e9f8796e6608e86bb173018148f`  
**S0 receipt-bearing handoff head / S1 working base:** `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f`  
**S1 working branch:** `ai/r11-post-cc-s1-end-to-end-learnability-r0`  
**Parent plan:** `docs/R11_POST_CC_S0_S1_TODO.md`  
**Purpose:** implement and then qualify the canonical post-CC durable end-to-end learning loop on preregistered synthetic known-answer tasks  
**FINAL:** SEALED  
**Fresh data:** FORBIDDEN  

Target evidence if every mandatory gate passes:

`INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER`

This evidence does **not** imply ECONOMIC or TRANSFER capability.

---

# 0. Critical execution rule — S1 is not blocked

S0 is already qualified and receipt-backed. S1 is authorized to implement.

The following are **not** reasons to stop:

- durable observation storage is not yet implemented;
- persistent replay materialization is not yet implemented;
- the joint direction+risk learner is not yet implemented;
- restart reconstruction is not yet implemented;
- exactly-once learner recovery is not yet implemented;
- generation/account continuity tests are not yet implemented;
- known-answer tasks or negative controls are not yet implemented;
- the final S1 receipt cannot yet be issued.

Those are the work items of S1.

An agent must not perform S1-001, observe that later gates are unfinished, and return a status-only report. Missing S1 functionality means **continue implementing the next mandatory task**.

The only legitimate early-stop conditions before qualification are:

1. `CONTRACT_MISMATCH` — implementation would require changing a frozen scientific semantic rather than implementing the authorized successor;
2. `EXECUTION_BLOCKED` — repository/tool/infrastructure failure prevents implementation or valid execution;
3. `HARDWARE_LIMIT` — the scientifically valid frozen workload cannot execute inside the declared resource envelope;
4. a genuinely new owner scientific decision is required and current authority does not resolve it.

Otherwise continue through the execution waves below.

---

# 1. S0 identity versus S1 working base

Do not conflate the certified implementation identity with the receipt-bearing handoff commit.

```text
S0 frozen implementation base
fc7102442e91a1c27cf705487c6d06bd64b8ea09
        |
        v
S0 qualified implementation identity
5760061d6c274e9f8796e6608e86bb173018148f
        |
        v
S0 receipt/report-bearing handoff head
 ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f
        |
        v
S1 working branch starts here
ai/r11-post-cc-s1-end-to-end-learnability-r0
```

The S0 receipt at `ec30185...` certifies `qualified_head_sha == 5760061d...`.

Therefore:

- `5760061d...` remains the inherited qualified S0 implementation identity;
- `ec30185...` is the correct S1 working/handoff base because it contains the receipt and frozen S1 preregistration artifacts;
- do **not** reset S1 back to `5760061d...` merely because that SHA appears in the receipt;
- all S1 authority must record both identities explicitly.

---

# 2. Frozen S1 preregistration

Read and verify:

- `authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json`
- `authority/rearchitecture_r11/CB16_R11_POST_CC_S1_TASK_REGISTRY_V1.json`
- `authority/rearchitecture_r11/CB16_R11_POST_CC_S1_RUN_SPEC_V1.json`

Current frozen essentials include:

- seeds: `1701, 1702, 1703, 1704, 1705`;
- positive-task qualification threshold: at least `4/5` seeds unless a task-specific frozen rule says otherwise;
- persistent-replay-only training truth;
- canonical joint direction + conditional risk likelihood;
- true behavior `log_mu` from decision time;
- optimizer family / LR as frozen in the run spec (`SGD`, `0.01` at the current S0 preregistration);
- maximum policy-decision budget per task/seed: `100000`;
- no rescue after qualification begins;
- FINAL/fresh-data firewall CLOSED;
- evidence ceiling: `INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER`.

Do not silently modify frozen values after observing qualification outcomes.

---

# 3. What S1 must prove

S1 must demonstrate the following through one auditable canonical path:

```text
synthetic causal environment
  -> canonical Brain observes market/account/execution state
  -> stochastic joint action is sampled
  -> nominal direction + nominal target risk are retained
  -> true behavior joint log_mu is persisted
  -> A-authoritative account/runtime consequences occur
  -> durable immutable experience is written
  -> durable content-addressed observation fact is written
  -> live collector objects may be destroyed
  -> replay reconstructs the training sample only from durable state
  -> joint direction+risk batch is materialized
  -> target joint log_pi is evaluated on the nominal behavior action
  -> Critic/V-trace/Actor update occurs
  -> child checkpoint commits exactly once
  -> generation switch preserves the same logical account where required
  -> child policy actually acts
  -> repeated optimization solves preregistered known-answer tasks
  -> negative controls remain negative
```

Every arrow matters.

A component smoke test that bypasses durable replay may remain for diagnostics, but it cannot qualify S1.

---

# 4. Existing qualified code to reuse, not replace

Reuse qualified CC surfaces wherever semantics already match:

- `cc_policy_brain_r0.py` — canonical market/account/execution Brain composition;
- `cc_policy_distribution_r0.py` — joint categorical direction + conditional continuous risk distribution;
- `cc_policy_rng_r0.py` — policy RNG provenance;
- Thread-A runtime/account/execution authority;
- Thread-C immutable raw facts and replay-selection surfaces;
- `cc_policy_reward_r0.py` — arithmetic-equity reward contract where task-compatible;
- `cc_critic_value_r0.py` and bootstrap helpers;
- `cc_vtrace_r0.py` — V-trace math;
- learner transaction/checkpoint/recovery machinery;
- generation-switch semantics;
- canonical fast path only when it preserves S1 observability and semantics.

Do not create a second policy distribution, alternate account physics, alternate economic objective, or handcrafted regime subsystem merely because a synthetic task would be easier that way.

---

# 5. S1 known gaps

## S1-G01 — rollout-memory learner side channel

The CC integration canary may assemble training tensors from in-memory rollout `records`. That remains wiring evidence only.

S1 qualification must train exclusively from durable replay/materialization.

## S1-G02 — replayable observation payload

Observation identity/hash exists, but S1 needs deterministic reconstruction of the separated Brain-ready:

```text
market
account
execution
```

payload from durable content-addressed facts.

## S1-G03 — categorical-only generic batch

The generic categorical batch does not fully encode canonical joint action semantics. S1 needs nominal direction, nominal target risk, risk kind/mask and true joint `log_mu`.

## S1-G04 — one update is not learnability

Checkpoint change, non-zero gradient, optimizer advancement or one changed child action are diagnostics, not learnability evidence.

## S1-G05 — component toys are not full-loop qualification

Positive tasks must converge through the complete persistent path, with negative controls.

---

# 6. Execution waves

S1 is executed as five sequential waves. These waves are an execution discipline, not independent scientific programs.

```text
S1-A Durable Data Plane
      -> S1-B Joint Learning Plane
      -> S1-C Scientific Known-Answer Tasks
      -> S1-D Qualification Runtime
      -> S1-E Gate Compiler + Receipt
```

An agent may continue across waves in one session. If work is split across agents, each wave must leave durable commits and tests so the next agent continues from code rather than re-performing status review.

**Required progress rule:** after implementation begins, a normal execution session must leave actual S1 commits beyond `ec30185...`, unless it terminates under one of the explicit early-stop classifications in Section 0.

---

# 7. Wave S1-A — Durable Data Plane (`S1-002` through `S1-007`)

## S1-001 — Verify S0 authority and emit S1 baseline authority

The prerequisite is already satisfied at the repository-program level, but the S1 branch must make the handoff machine-readable.

Before changing learning code:

- read S0 receipt and assert `status == PASS`;
- assert receipt certifies `qualified_head_sha == 5760061d6c274e9f8796e6608e86bb173018148f`;
- assert working base contains the receipt at `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f` lineage;
- assert historical CC receipts remain unchanged;
- load frozen task registry/run spec;
- assert synthetic-only and FINAL/fresh firewall closed.

Create if absent:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S1_BASELINE_V1.json`

The baseline artifact must record both:

- inherited S0 qualified implementation identity;
- S0 receipt-bearing S1 working base.

If the receipt is absent or not PASS, classify `EXECUTION_BLOCKED`. Do not use this clause after a valid PASS receipt has already been verified.

## S1-002 — Implement `PostCCObservationFactV1`

Recommended surface:

`cb16_local_opt/post_cc_observation_fact_v1.py`

Required payload includes at minimum:

```text
science_semantic_version
observation_schema
observation_hash
market_payload
account_payload
execution_payload
market_source/version identity
account lineage
decision index
environment time
normalizer identity
content hash
```

Requirements:

- no future information;
- deterministic codec/hash;
- corruption/hash mismatch fails closed;
- payload round-trip reconstructs exact or explicitly tolerance-approved Brain inputs;
- observation hash agrees with W-01 reference.

## S1-003 — Durable immutable observation store

Recommended surface:

`cb16_local_opt/post_cc_observation_store_v1.py`

Must support:

- idempotent content-addressed write;
- exact read by observation/content hash;
- collision/conflict rejection;
- restart/reopen;
- deterministic serialization;
- no mutation after commit.

## S1-004 — Bind observation facts to canonical collection

For every policy-decision transition used by S1 replay:

1. construct canonical Brain observation;
2. persist `PostCCObservationFactV1`;
3. link transition/experience through durable reference/hash;
4. verify nominal decision observation hash matches stored fact;
5. commit experience only when the durable observation reference is valid.

Mechanical no-decision advances remain raw facts and must not receive fabricated action observations or `log_mu`.

## S1-005 — Implement `PostCCJointReplaySampleV1`

Recommended surface:

`cb16_local_opt/post_cc_joint_replay_v1.py`

Carry at minimum:

```text
sequence_id
transition_ref
account_lineage_id
decision_index
environment_time
market observation
account observation
execution observation
nominal_direction
nominal_target_risk
risk_measure_kind
behavior_log_mu
behavior_policy_id/generation/hash
reward
discount
boundary/bootstrap fields
sampling_probability/weight
source fact hashes
```

Executed action/quantity may appear only as consequence context and must never overwrite the nominal action used for policy likelihood.

## S1-006 — Persistent replay materializer

Recommended surface:

`cb16_local_opt/post_cc_replay_materializer_v1.py`

Input:

- persistent sequence IDs / transition refs;
- durable observation store;
- immutable experience/raw-fact stores;
- task/run contract.

Output:

- ordered validated `PostCCJointReplaySampleV1` objects.

Hard rule:

> The S1 qualification learner may not receive collector-side live tensors or in-memory `records` as training truth.

Mandatory restart sentinel:

1. collect and persist experience;
2. record durable sample identities/checksums;
3. destroy rollout/live Python objects;
4. reopen/restart materializer;
5. reconstruct samples exclusively from durable state;
6. require identical discrete/content-addressed identities and frozen numeric tolerance where necessary.

## S1-007 — Joint-action sequence batch

Recommended surface:

`cb16_local_opt/post_cc_joint_batch_v1.py`

Required tensors/metadata include:

- market/account/execution observations;
- nominal direction indices;
- nominal target risks;
- risk kind/masks;
- persisted behavior `log_mu`;
- rewards/discounts;
- bootstrap/boundary fields;
- sampling weights/probabilities;
- sequence/provenance IDs.

Validation must reject:

- FLAT with nonzero nominal target risk;
- non-FLAT risk outside canonical support;
- non-finite or missing `log_mu`;
- missing behavior identity;
- time-order corruption;
- observation hash mismatch.

### S1-A exit condition

Do not exit S1-A merely because S1-B is unfinished.

S1-A exits only when:

- S1-002..007 implementation exists;
- focused tests pass;
- restart/durable replay sentinel passes;
- a durable S1-A implementation commit is pushed.

Then continue to S1-B.

---

# 8. Wave S1-B — Joint Learning Plane (`S1-008` through `S1-012`)

## S1-008 — Canonical joint target log-prob and Actor loss

Recommended surface:

`cb16_local_opt/post_cc_joint_policy_loss_v1.py`

For every replay sample:

1. run `CCCentralBrain` on materialized market/account/execution inputs;
2. use the canonical CC policy distribution;
3. evaluate target joint `log_pi` on the **nominal behavior action**;
4. compare against persisted true `log_mu`;
5. compute V-trace ratios and Actor objective under the frozen semantics.

Mandatory tests:

- FLAT point-mass likelihood with exact risk=0;
- LONG/SHORT conditional continuous-risk density;
- same-policy `log_pi == log_mu` known answer;
- controlled off-policy ratios;
- rejected/clamped execution does not alter nominal likelihood.

## S1-009 — Separate Critic + boundary-aware bootstrap

Use the qualified Critic/value path. Distinguish explicitly:

- computational truncation;
- replay/sequence chunk boundary;
- synthetic task horizon;
- mechanical terminal/account death.

Do not infer bootstrap only from a generic `done` boolean.

## S1-010 — Reusable S1 learner runtime

Recommended surface:

`cb16_local_opt/post_cc_learner_v1.py`

Responsibilities:

- consume only validated durable joint replay batches;
- compute target joint likelihood, V-trace, Critic loss and Actor loss;
- enforce gradient ownership;
- update only authorized trainable parameters;
- preserve frozen market organs;
- persist Actor/Critic gradient and optimizer diagnostics;
- create a child checkpoint without mutating behavior checkpoint in place;
- use exactly-once transaction semantics.

Do not delete or rewrite historical categorical learner paths solely to make S1 work.

## S1-011 — Durable learning-update provenance

Each update must record canonical equivalents of:

- sampled sequence IDs;
- materialized sample hashes;
- sampling probabilities/weights;
- parent checkpoint;
- behavior and target policy identities;
- Actor/Critic losses;
- V-trace diagnostics;
- gradient ownership summary;
- optimizer step before/after;
- child checkpoint;
- commit status.

## S1-012 — Generation switch + same-account continuation

Where a task spans a generation boundary:

- behavior checkpoint remains fixed during its collection unit;
- learner commits child separately;
- switch occurs only at an explicit A-authoritative boundary;
- logical account lineage, holdings, liabilities and causal state continue;
- child policy makes actual subsequent decisions.

Creating a new flat account to make the child act is not continuity and cannot satisfy this gate.

### S1-B exit condition

S1-B exits only when:

- S1-008..012 implementation exists;
- joint-likelihood/Critic/V-trace/learner tests pass;
- durable update provenance works;
- generation/account-continuity test passes;
- a durable S1-B implementation commit is pushed.

Then continue to S1-C.

---

# 9. Wave S1-C — Mandatory Scientific Known-Answer Tasks (`S1-013` through `S1-020`)

All task identities, seeds, budgets and thresholds come from the S0-frozen task registry/run spec. Implement generators/harnesses without tuning pass criteria after observing results.

## S1-013 — `ACCOUNT_DEPENDENT_ACTION`

Same market observation, different reachable account states, different known-optimal actions.

Require after learning:

- probability moves toward the account-conditional known answer;
- frozen-evaluation arithmetic task return improves;
- market-only/no-account ablation fails to achieve the same qualification result where the frozen task requires account information.

## S1-014 — `DELAYED_CONSEQUENCE_CREDIT`

The correct early action is only identifiable from delayed downstream account consequence.

Require:

- no immediate label leaks the answer;
- reward/boundary semantics are frozen;
- early action probability and task return improve;
- shuffled-credit control materially degrades learning under the frozen relation.

## S1-015 — `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION`

Construct the preregistered case where one choice has:

- higher failure/bankruptcy frequency;
- higher complete-sample arithmetic expected return.

The correct answer is the higher arithmetic-EV choice.

Forbidden substitutions:

- survivor filtering;
- log wealth/log-growth objective;
- Sharpe objective;
- hidden drawdown penalty;
- hidden bankruptcy/survival preference;
- clipping that reverses the frozen arithmetic-EV ordering.

All failures remain in the denominator.

## S1-016 — `OFF_POLICY_VTRACE_CORRECTION`

Use deliberate behavior/target mismatch.

Require:

- true behavior `log_mu` persisted at collection time;
- target joint `log_pi` recomputed from materialized nominal action;
- known V-trace ratios/targets verified on controlled cases;
- preregistered off-policy learning task improves;
- fabricated/reconstructed behavior probability fails closed.

## S1-017 — `A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION`

Synthetic recurrence sequence A -> B -> A.

Require:

- no explicit A/B regime detector that activates a handcrafted rule;
- old A experience is not expired merely because B is recent;
- after B adaptation, return to A satisfies frozen retention/relearning criterion;
- replay/source attribution stays explicit.

## S1-018 — No-signal / zero-reward negative control

Matched control with no learnable reward relation or zero reward under frozen definition.

Must not falsely satisfy the positive known-answer qualification merely from optimizer noise.

## S1-019 — Shuffled-credit negative control

Break action-to-consequence alignment while preserving marginal structure where feasible.

The positive delayed-credit task must outperform this control according to frozen criteria.

## S1-020 — Random/impossible negative control

Where applicable, verify the system cannot satisfy the positive known-answer threshold by memorizing run artifacts or exploiting leakage.

### S1-C exit condition

S1-C exits only when:

- all five positive task generators/harnesses exist;
- mandatory negative controls exist;
- component known answers/oracles are independently checked;
- task definitions exactly match frozen registry identities;
- task/control implementation tests pass;
- implementation commits are pushed.

Do not issue the final S1 verdict at S1-C.

---

# 10. Wave S1-D — Qualification Runtime (`S1-021` through `S1-031`)

## S1-021 — Unified S1 runner

Recommended surface:

`scripts/run_r11_post_cc_s1_learnability.py`

It must:

- consume one frozen run manifest;
- verify exact S0/task/code/model identities;
- run positive tasks and controls non-interactively;
- isolate seeds deterministically;
- persist raw run artifacts;
- never access FINAL/fresh data;
- emit per-task/per-seed results and aggregate gate summary.

## S1-022 — Multi-seed qualification

Use exactly the frozen seed policy and aggregation rules. Do not cherry-pick seeds or hide failures.

## S1-023 — Pre/post behavioral evaluation

Evaluate the same frozen statistics before and after learning, such as:

- probability assigned to known-correct direction/risk region;
- complete-sample arithmetic expected task return;
- Critic value error where analytic values exist.

Training loss is diagnostic only.

## S1-024 — Durable restart reconstruction gate

Qualification must explicitly destroy collector/live state and reconstruct training samples solely from persistent stores + manifest. Sample identities/checksums must satisfy the frozen equality/tolerance contract.

## S1-025 — Exactly-once hostile learner transaction matrix

Exercise failures at multiple transaction points, including at minimum:

- before update commit;
- after partial durable write where applicable;
- around child checkpoint commit;
- recovery/reopen;
- duplicate-retry prevention.

No learner update may be applied twice.

## S1-026 — Provenance audit

Verify every qualified update/action can be traced through policy generation, RNG, observation facts, experience facts, replay sample IDs, update record and child checkpoint.

## S1-027 — Gradient ownership audit

Verify authorized trainable Actor/Critic parameters may move and frozen organs may not.

## S1-028 — Reward/objective firewall

Verify arithmetic-equity reward semantics and high-bankruptcy/higher-EV orientation are not silently replaced by log wealth, Sharpe, survivor filtering or hidden risk penalties.

## S1-029 — B&H/FLAT semantic firewall

S1 must retain S0 successor semantics:

- B&H and FLAT are sibling benchmark components;
- no master precedence/winner;
- promotion is separate/versioned.

## S1-030 — FINAL/fresh-data firewall

Require machine-verifiable:

```text
FINAL_opened = false
fresh_data_used = false
historical_market_corpus_accessed = false
```

S1 is synthetic qualification only.

## S1-031 — Performance is a guard, not the scientific winner

Use canonical qualified fast surfaces when semantically equivalent. Do not reopen CC performance competition. Optimize only if profiling reveals a concrete execution blocker; do not simplify scientific semantics for throughput.

### S1-D exit condition

Run the real qualification workload under the frozen manifest and preserve all results, including failed seeds/controls.

A failed preregistered valid experiment is evidence; do not rescue it in place.

---

# 11. Wave S1-E — Gate Compiler, Artifacts & Final Receipt (`S1-032` through `S1-035`)

## S1-032 — Qualification gate compiler

Compile all mandatory component, durability, scientific, control, recovery, provenance and firewall gates into one deterministic machine-readable result.

The compiler must distinguish:

- implementation/invariant failure;
- valid scientific failure;
- execution block;
- hardware limit;
- insufficient evidence.

## S1-033 — Durable per-task result artifacts

Persist, at minimum:

- task identity/hash;
- seed;
- code/model/checkpoint identities;
- replay/sample identities;
- pre/post behavior statistics;
- complete-sample arithmetic task return;
- oracle/gap metric;
- control relation;
- decision budget consumed;
- pass/fail classification;
- raw artifact checksums.

## S1-034 — Final S1 receipt

Expected authority path:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S1_LEARNABILITY_RECEIPT_V1.json`

The receipt must record at minimum:

- S1 working base / handoff SHA;
- inherited S0 qualified implementation identity;
- exact qualified S1 head SHA;
- task registry and run-spec identities;
- model/optimizer identities;
- seed list and complete outcomes;
- durable replay/restart status;
- joint learner status;
- recovery/exactly-once status;
- generation/account-continuity status;
- positive-task and negative-control verdicts;
- strongest justified evidence level;
- FINAL/fresh-data firewall state;
- unresolved items.

## S1-035 — Navigation update and STOP

After the final verdict:

- update canonical navigation/state docs to the receipt-backed S1 status;
- preserve all failed scientific evidence;
- stop.

Do **not** automatically start S2, historical execution, capacity scaling, economic qualification or FINAL work.

---

# 12. Hard fail-closed invariants

These are qualification invariants. They do not forbid implementation work; they forbid an invalid path from receiving PASS.

## Action semantics

- executed action cannot replace nominal sampled action;
- executed quantity cannot reconstruct nominal policy likelihood;
- FLAT must have exact nominal target risk `0`;
- non-FLAT risk semantics must use canonical policy distribution/support.

## Behavior likelihood

- true `log_mu` comes from behavior policy at decision time;
- `log_mu` cannot be fabricated/reconstructed from execution or stored outcomes.

## Observation integrity

- observation hash mismatch fails closed;
- learner must consume the same causally valid market/account/execution observation represented by the Actor decision identity.

## Durable replay

- qualification learner consumes only durable replay/materialized samples;
- collector-side live tensors/records are not training truth;
- restart reconstruction must reproduce frozen sample identity.

## Generation/account continuity

- child generation cannot fake continuation by creating a new flat account when task semantics require continuity.

## Economic objective

- `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION` uses complete-sample arithmetic expectation;
- no survivor filtering, log wealth, Sharpe or hidden risk penalty may reverse the intended known answer.

## Recurrence

- A -> B -> A may not be solved with a handcrafted regime detector/rule activation subsystem.

---

# 13. What is NOT sufficient for S1 PASS

None of the following alone qualifies S1:

- lower training loss;
- non-zero gradients;
- changed model parameters;
- optimizer step advanced;
- checkpoint file written;
- child policy acts once;
- one action differs from parent;
- in-memory toy convergence;
- collector-side tensor training;
- one lucky seed;
- high throughput.

These are diagnostics only.

---

# 14. No-rescue qualification discipline

Before the first qualifying multi-seed run, freeze all remaining implementation-dependent run identities required by the existing S0 preregistration without changing its scientific question.

Once qualification begins:

- do not raise decision budget because a seed failed;
- do not delete bad seeds;
- do not change success thresholds after observing results;
- do not change reward/objective to produce the desired answer;
- do not redesign the task after seeing qualification outcomes;
- do not tune on qualification results and rerun under the same identity.

A changed scientific question requires a new version/run identity.

---

# 15. Verdict taxonomy

Use exactly one final classification appropriate to the evidence:

- `PASS` — every mandatory qualification gate passes;
- `SCIENTIFIC_FAIL` — implementation is valid but preregistered scientific criterion fails;
- `CONTRACT_MISMATCH` — implementation requires violating frozen semantics;
- `EXECUTION_BLOCKED` — software/infrastructure prevents a valid result;
- `HARDWARE_LIMIT` — valid workload cannot run in declared hardware envelope;
- `EVIDENCE_INSUFFICIENT` — execution completed but frozen evidence requirements are not met.

Do not use `EXECUTION_BLOCKED` merely because later S1 tasks have not yet been implemented. Implement them.

---

# 16. Agent handoff discipline

Every Agent taking over S1 must first compare the S1 branch against `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f` and identify the first unfinished task.

Then:

1. continue from existing S1 commits;
2. do not re-run status-only prerequisite analysis already settled by receipt unless verifying integrity;
3. implement the first unfinished wave;
4. add focused tests;
5. run them;
6. commit/push working code;
7. continue to the next wave if execution capacity remains;
8. report the exact first unfinished task only after durable progress has been made, unless a legitimate early-stop condition occurs.

A handoff that only says “S1 cannot yet receive a receipt because later work is unfinished” is **not task completion**.

---

# 17. Final completion condition

S1 is complete only when one of the following is durably established:

### PASS

All mandatory implementation, durability, known-answer, negative-control, recovery, provenance and firewall gates pass and the S1 receipt claims no more than:

`INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER`

### Valid non-PASS scientific verdict

The full preregistered qualification ran validly and produced `SCIENTIFIC_FAIL` or `EVIDENCE_INSUFFICIENT`; preserve complete evidence and stop.

### Genuine execution/contract verdict

A concrete `CONTRACT_MISMATCH`, `EXECUTION_BLOCKED`, or `HARDWARE_LIMIT` prevents valid completion; preserve exact evidence and owning layer.

Until one of those conditions is reached, unfinished S1 work means **continue implementation**, not “S1 is blocked.”
