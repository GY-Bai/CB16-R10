# CB16 R11 Post-CC S1 — End-to-End Learnability Qualification TODO

**Stage:** S1  
**Status:** BLOCKED UNTIL S0 PASS  
**Base:** exact S0 qualified head recorded in `CB16_R11_POST_CC_S0_RECEIPT_V1.json`  
**Parent plan:** `docs/R11_POST_CC_S0_S1_TODO.md`  
**Prerequisite:** `docs/post_cc/S0_CONTRACT_MIGRATION_TODO.md` PASS  
**Purpose:** prove that the canonical post-CC loop can actually learn preregistered synthetic known-answer tasks end-to-end  
**FINAL:** SEALED  
**Fresh data:** FORBIDDEN  

Suggested branch after S0 PASS:

`ai/r11-post-cc-s1-learnability-r0`

S1 is **not** a historical profitability experiment. It is the bridge between CC wiring evidence and later historical science.

Target evidence if every mandatory gate passes:

`INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER`

This evidence does not imply ECONOMIC or TRANSFER capability.

---

## 0. What S1 must prove

S1 must demonstrate all of the following in one auditable program:

```text
synthetic causal environment
  -> canonical Brain observes market/account/execution state
  -> stochastic joint action is sampled
  -> true behavior log_mu is persisted
  -> A-authoritative account/runtime consequences occur
  -> durable immutable experience is written
  -> observation payload is reconstructed from durable content-addressed facts
  -> replay selects persistent sequences
  -> joint direction+risk training batch is materialized
  -> Critic/V-trace/Actor update occurs
  -> child checkpoint commits exactly once
  -> same logical account can continue across generation switch where the task requires it
  -> child policy actually acts
  -> repeated training measurably solves the preregistered known-answer task
  -> negative controls do not produce a false PASS
```

Every arrow matters.

A test that bypasses durable replay and directly reuses live rollout tensors may remain as a component smoke test, but it **cannot** qualify S1.

---

## 1. Existing code to reuse, not replace

S1 should reuse qualified CC components wherever their semantics already match:

- `cc_policy_brain_r0.py` — canonical market/account/execution Brain composition;
- `cc_policy_distribution_r0.py` — joint categorical direction + conditional continuous risk distribution;
- `cc_policy_rng_r0.py` — policy RNG provenance;
- Thread-A runtime/account/execution surfaces;
- Thread-C immutable raw fact and replay-selection surfaces;
- `cc_policy_reward_r0.py` — arithmetic-equity reward contract where task-compatible;
- `cc_critic_value_r0.py` / bootstrap helpers;
- `cc_vtrace_r0.py` — V-trace math;
- learner transaction/checkpoint/recovery machinery;
- generation-switch semantics;
- canonical fast path only where it does not obstruct S1 observability.

Do not create a second policy distribution or a new account physics just because a toy is easier that way.

---

## 2. S1 gaps that must be closed

### S1-G01 — rollout-memory learner side channel

The integration canary can assemble training tensors from in-memory `records`. This is valid wiring evidence but not durable-replay learnability evidence.

S1 must prohibit this shortcut in the qualification path.

### S1-G02 — observation identity exists, replayable payload is incomplete

CC experience retains observation identity/hash, but S1 needs deterministic reconstruction of the actual separated Brain inputs:

```text
market
account
execution
```

S1 must add a content-addressed observation fact/sidecar under the S0-frozen successor contract.

### S1-G03 — generic batch is categorical-action oriented

The existing generic `SequenceBatch.actions` / categorical learner path does not fully represent the canonical joint action.

S1 needs a versioned joint replay batch that carries nominal direction, nominal target risk, risk measure kind and true joint `log_mu`.

### S1-G04 — one update is not learnability

CC proves an update can commit and a child can act. S1 must prove repeated optimization reaches a preregistered known answer.

### S1-G05 — component toys are not full-loop convergence evidence

Existing toys remain useful unit references. S1 must execute the task through the complete persistent path.

---

# 3. Mandatory implementation tasks

## S1-001 — Verify S0 authority and freeze exact S1 head base

Before changing code:

- read S0 receipt;
- assert S0 status PASS;
- assert historical CC receipts remained unchanged;
- record exact S0 qualified head as S1 base;
- load frozen S1 task registry/run-spec identities;
- assert FINAL/fresh scope is synthetic-only.

Create:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S1_BASELINE_V1.json`

If S0 receipt is absent or not PASS, stop `EXECUTION_BLOCKED`.

---

## S1-002 — Implement `PostCCObservationFactV1`

Implement the S0-frozen observation-fact contract.

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
account lineage / decision index / environment time
normalizer identity
content hash
```

Requirements:

- no future information;
- same logical observation -> deterministic hash under the frozen codec;
- corruption/hash mismatch -> fail closed;
- payload round-trip reconstructs exact/tolerance-approved Brain inputs;
- observation hash agrees with W-01 reference.

---

## S1-003 — Add durable observation store

Recommended surface:

`cb16_local_opt/post_cc_observation_store_v1.py`

Use content-addressed immutable storage semantics compatible with existing fact-store discipline.

Must support:

- idempotent write;
- exact read by observation/content hash;
- collision/conflict rejection;
- restart/reopen;
- deterministic serialization;
- no mutation after commit.

A learner must be able to restart in a new process and reconstruct inputs without access to the collector's Python objects.

---

## S1-004 — Bind observation facts to runtime collection

At every policy-decision transition used for S1 replay:

1. construct the canonical Brain observation;
2. persist `PostCCObservationFactV1`;
3. put only the durable reference/hash into the transition/experience linkage required by the frozen contracts;
4. verify the nominal decision's `observation_hash` matches the stored fact;
5. commit experience only when the observation reference is valid according to the successor transaction rule.

Mechanical no-decision advances remain raw facts and must not receive fabricated action observations/log_mu.

---

## S1-005 — Implement `PostCCJointReplaySampleV1`

Recommended surface:

`cb16_local_opt/post_cc_joint_replay_v1.py`

The materialized sample must carry:

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

Executed action/quantity may be supplied as consequence context but must never overwrite the nominal action used for policy likelihood.

---

## S1-006 — Build persistent replay materializer

Recommended surface:

`cb16_local_opt/post_cc_replay_materializer_v1.py`

Input:

- persistent sequence ID / transition refs selected by canonical replay logic;
- observation store;
- immutable experience/raw fact store;
- task/run contract.

Output:

- ordered `PostCCJointReplaySampleV1` objects.

Hard rule:

> The S1 qualification learner may not receive original rollout tensors or the collector's in-memory `records` as training truth.

Add a sentinel test that deletes all live rollout objects, restarts the materializer, and still produces the identical training sample hashes.

---

## S1-007 — Version a joint-action sequence batch

Recommended surface:

`cb16_local_opt/post_cc_joint_batch_v1.py`

Required batched tensors/metadata:

- market observations `[T,B,...]`;
- account observations;
- execution observations;
- nominal direction indices;
- nominal target risks;
- risk measure kinds/masks;
- behavior `log_mu`;
- rewards;
- discounts;
- bootstrap states/taus where applicable;
- sampling weights/probabilities;
- sequence/provenance IDs.

Validation must reject:

- FLAT with nonzero nominal target risk;
- non-FLAT endpoint values outside the canonical support contract;
- non-finite log_mu;
- missing behavior identity;
- shuffled time order inside the same account sequence;
- mismatched observation hashes.

---

## S1-008 — Implement canonical joint target log-prob path

Recommended surface:

`cb16_local_opt/post_cc_joint_policy_loss_v1.py`

For each replay sample:

1. run `CCCentralBrain` on materialized market/account/execution inputs;
2. construct/use the canonical CC policy distribution;
3. evaluate target joint `log_pi` of the **nominal behavior action**;
4. compute V-trace ratio against persisted true `log_mu`;
5. compute Actor loss from the versioned policy objective.

No executed action substitution.

Qualification tests must include:

- FLAT point-mass likelihood;
- LONG/SHORT continuous risk density;
- same-policy `log_pi == log_mu` known answer;
- off-policy known ratios;
- rejected/clamped execution does not alter nominal likelihood.

---

## S1-009 — Integrate separate Critic and boundary-aware bootstrap

Use the qualified Critic/value path with the S0-frozen task boundary semantics.

Must distinguish:

- computational truncation;
- sequence chunk boundary;
- dataset/task horizon;
- mechanical terminal/account death.

Do not infer bootstrap solely from a generic `done` boolean.

Critic target and realized reward remain distinct.

---

## S1-010 — Implement reusable S1 learner runtime

Recommended surface:

`cb16_local_opt/post_cc_learner_v1.py`

Responsibilities:

- consume only validated joint replay batches;
- compute target joint likelihood/V-trace/Critic loss/Actor loss;
- respect gradient ownership;
- update only authorized trainable parameters;
- preserve frozen market organs;
- report actor/critic gradients and optimizer state;
- produce a child checkpoint without mutating the behavior checkpoint in place;
- use exactly-once transaction semantics.

Do not delete the old categorical learner; it can remain a component/history path.

---

## S1-011 — Make update provenance durable

Each update must emit a successor learning-update record linked to W-04 semantics and include:

- sampled sequence IDs;
- materialized sample hashes;
- sampling probabilities/weights;
- parent checkpoint;
- target/behavior policy identities;
- Actor/Critic losses;
- V-trace diagnostics;
- gradient ownership summary;
- optimizer step before/after;
- child checkpoint;
- commit status.

Exactly-once recovery must be verified after process failure at multiple transaction points.

---

## S1-012 — Generation switch + same-account continuation gate

Where the known-answer task spans a generation boundary:

- behavior checkpoint remains fixed during its collection unit;
- learner commits child separately;
- switch happens at an explicit A-authoritative boundary;
- account lineage, holdings, liabilities and causal state continue;
- child policy actually makes subsequent decisions.

A test that resets to a fresh flat account at every generation cannot satisfy this gate.

---

# 4. Mandatory known-answer tasks

All task identities, budgets, seeds and thresholds must come from the S0-frozen task registry. S1 may implement the generators/harnesses but may not tune pass criteria after seeing results.

## S1-013 — `ACCOUNT_DEPENDENT_ACTION`

Construct states where the same market observation has different known-optimal actions because account state differs.

S1 must show, after training:

- behavior probability moves toward the account-conditional known answer;
- task return improves under the frozen evaluation policy;
- a market-only/no-account ablation cannot obtain the same qualification result if the task is designed to require account information.

Purpose: prove actual account information is learnable and affects behavior, not merely present in a tensor.

---

## S1-014 — `DELAYED_CONSEQUENCE_CREDIT`

Construct a task where the correct early action is identifiable only through delayed downstream account consequence.

Requirements:

- no immediate label tells the Actor the answer;
- causal reward/boundary semantics are preregistered;
- training must improve the early action probability/expected return;
- a shuffled-credit control must materially degrade learning.

Purpose: test sequence credit assignment rather than static classification.

---

## S1-015 — `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION`

Construct a known-answer distribution where one strategy/action has:

- higher failure/bankruptcy frequency;
- higher complete-sample arithmetic expected return.

The correct solution under the frozen owner objective is the higher arithmetic-expectation choice.

Requirements:

- all failures stay in the denominator;
- no log-wealth/Sharpe/survival override;
- no survivor-only filtering;
- learned policy must move toward the high-expectation action within the preregistered threshold.

Purpose: prove the learner/evaluator did not silently reintroduce risk aversion.

---

## S1-016 — `OFF_POLICY_VTRACE_CORRECTION`

Use a behavior policy deliberately different from the target policy.

Requirements:

- persist true behavior `log_mu` at collection time;
- recompute target joint `log_pi` from the materialized nominal action;
- verify known V-trace ratios/targets on a controlled case;
- then demonstrate learning under the preregistered off-policy task;
- fabricated or reconstructed behavior probability must fail qualification.

Purpose: qualify the real off-policy replay path.

---

## S1-017 — `A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION`

Train/evaluate a synthetic recurring-pattern task with A then B then A conditions.

Requirements:

- no hand-coded condition detector activates an A-specific rule;
- old A experience is not expired solely because B was recent;
- after B adaptation, returning to A must show the preregistered retention/relearning criterion;
- replay/source attribution remains explicit.

Purpose: test historical retention/recurrence philosophy through weights/replay rather than manual cycle logic.

---

# 5. Mandatory negative controls

## S1-018 — No-signal / zero-reward control

Run a matched control where observations/actions contain no learnable reward relation or reward is zero according to the frozen control definition.

Expected behavior must be preregistered. The system must not report known-answer learnability merely from optimizer noise.

---

## S1-019 — Shuffled-credit control

Break the action-to-consequence alignment while preserving marginal distributions where feasible.

The positive delayed-credit task must outperform this control according to the S0-frozen relation.

If shuffled credit performs indistinguishably under a task designed to require credit assignment, return `SCIENTIFIC_FAIL` or `EVIDENCE_INSUFFICIENT`, not PASS.

---

## S1-020 — Random/impossible-label control where applicable

For any task that can be mirrored with an impossible/random target, verify the training system does not satisfy the positive known-answer threshold merely by memorizing run artifacts or exploiting a leak.

---

# 6. Qualification harness

## S1-021 — Build a unified S1 task runner

Recommended surface:

`scripts/run_r11_post_cc_s1_learnability.py`

It must:

- consume one frozen run manifest;
- verify exact code/S0/task-registry identities;
- execute positive tasks and required controls without interactive rescue;
- isolate seeds deterministically;
- persist raw run artifacts;
- never access FINAL/fresh data;
- produce per-task result JSON and one aggregate gate summary.

No task-specific manual intervention after the run starts.

---

## S1-022 — Multi-seed / repeat qualification

Use the S0-preregistered seed policy.

The exact aggregation rule must already be frozen; examples may include:

- minimum number/fraction of seeds meeting the known-answer threshold;
- paired improvement over initialization/control;
- confidence interval or bootstrap diagnostic where appropriate.

Do not select the best seed after the fact.

Report every seed, including failures.

---

## S1-023 — Pre/post behavioral evaluation

For each task, evaluate the same preregistered policy statistic before and after learning.

Examples:

- probability assigned to known-correct direction/risk region;
- arithmetic expected task return under deterministic/stochastic frozen evaluation rule;
- known value error for Critic where the task has analytic values.

Training loss is diagnostic only.

---

## S1-024 — Exact durable replay/restart qualification

At a preregistered interruption point:

1. stop after persistent experience exists;
2. destroy collector/in-memory rollout state;
3. restart a new learner process/runtime;
4. reconstruct observation/replay batches only from durable stores + run manifest;
5. verify sample/source hashes and ordering;
6. continue training;
7. compare to the declared exact/tolerance recovery rule.

This is a mandatory S1 gate.

---

## S1-025 — Exactly-once learner transaction hostile matrix

Inject failures at least around:

- before update compute;
- after compute before commit;
- after child checkpoint write before transaction commit;
- after transaction commit before caller acknowledgement.

A logical update ID must commit at most once.

---

## S1-026 — Provenance audit

Automated audit must verify, for every qualified replay action:

```text
behavior policy identity exists
behavior generation exists
RNG provenance exists
true log_mu finite
nominal action exists
observation fact hash resolves
experience transition links correctly
sequence ordering valid
target policy identity recorded
child checkpoint lineage valid
```

Any provenance break fails qualification.

---

## S1-027 — Gradient ownership audit

Automated before/after checks must prove:

- authorized Actor/Critic parameters can change;
- frozen market organs do not change;
- no unexpected parameter receives gradient;
- checkpoint hashes/parameter counts match declared model identity.

---

## S1-028 — Reward telescoping and objective firewall

For every task using arithmetic equity-delta reward, retain a known-answer check that the declared finite-horizon cumulative reward corresponds to the intended arithmetic equity change under the frozen discount convention.

Add sentinels rejecting accidental substitution with:

- log equity;
- sign reward;
- clipped reward;
- drawdown penalty as hidden primary objective;
- survivor-only return.

---

## S1-029 — B&H/FLAT semantic firewall

S1 is not an economic benchmark stage, but any baseline-related helper used in synthetic scoring must follow the S0 successor semantics:

- B&H and FLAT remain independent components;
- no master precedence;
- baseline disagreement never becomes an owner question;
- promotion rule, if exercised only as a contract test, is explicit/versioned.

This prevents S1 from reintroducing the old ambiguity indirectly.

---

## S1-030 — FINAL/fresh-data firewall

Qualification workflow must prove:

```text
FINAL_opened=false
fresh_data_used=false
scientific_scope=SYNTHETIC_ONLY
```

The presence of the user's historical corpus on the host does not authorize S1 to read it.

---

# 7. Performance rules for S1

## S1-031 — Use performance as a guard, not the scientific winner

S1 may use the canonical hard-cutover fast spine where semantics and observability remain qualified.

Record:

- transitions/s;
- learner updates/s;
- wall clock per task;
- peak memory;
- queue/backpressure health.

But S1 is not a new performance tournament. Do not change task semantics, batch population or failure retention for speed.

If performance blocks a valid S1 run within declared limits, classify `HARDWARE_LIMIT` or `EXECUTION_BLOCKED`; do not silently simplify the task.

---

# 8. Qualification compiler and receipt

## S1-032 — Build gate compiler

Recommended surface:

`cb16_local_opt/post_cc_s1_qualification_v1.py`

Mandatory gates:

```text
S0_AUTHORITY_PASS
DURABLE_OBSERVATION_FACT_PASS
PERSISTENT_REPLAY_ONLY_PASS
JOINT_ACTION_BATCH_PASS
TRUE_LOG_MU_PASS
JOINT_TARGET_LOG_PI_PASS
VTRACE_KNOWN_ANSWER_PASS
BOUNDARY_BOOTSTRAP_PASS
EXACTLY_ONCE_UPDATE_PASS
GENERATION_ACCOUNT_CONTINUITY_PASS
ACCOUNT_DEPENDENT_ACTION_PASS
DELAYED_CONSEQUENCE_PASS
HIGH_BANKRUPTCY_HIGHER_EXPECTATION_PASS
OFF_POLICY_LEARNING_PASS
A_B_A_RETENTION_PASS
NO_SIGNAL_CONTROL_PASS
SHUFFLED_CREDIT_CONTROL_PASS
DURABLE_RESTART_PASS
PROVENANCE_AUDIT_PASS
GRADIENT_OWNERSHIP_PASS
OBJECTIVE_FIREWALL_PASS
NO_MASTER_BASELINE_PRECEDENCE_PASS
FINAL_FRESH_FIREWALL_PASS
MULTI_SEED_AGGREGATION_PASS
```

All mandatory gates must be true for S1 PASS.

---

## S1-033 — Emit per-task durable result artifacts

Recommended directory:

`artifacts/post_cc_s1/`

Each task result should include:

- run/task identity;
- seed;
- code/model/data/contract identities;
- pre-training metrics;
- training budget actually consumed;
- post-training metrics;
- control metrics;
- account/failure counts where relevant;
- checkpoint lineage;
- replay/sample hashes;
- verdict/classification;
- no-rescue statement.

---

## S1-034 — Emit final S1 machine-readable receipt

Required path:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S1_LEARNABILITY_RECEIPT_V1.json`

Minimum contents:

```text
status
classification
s0_receipt path/hash
s1_base_sha
qualified_head_sha
science contract identity
task registry path/hash
run spec path/hash
model identities / parameter counts
positive task results
negative control results
seed aggregation
persistent replay-only proof
joint action likelihood proof
exactly-once/recovery proof
generation/account continuity proof
provenance/gradient audits
FINAL_opened=false
fresh_data_used=false
strongest_justified_evidence_level
ECONOMIC_evidence_claimed=false
TRANSFER_evidence_claimed=false
```

If PASS, strongest justified evidence:

`INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER`

---

## S1-035 — Update navigation only after the verdict

After the receipt exists, update:

- `docs/CURRENT_STATE.md`
- `docs/README.md`
- `AGENTS.md`
- `docs/POST_CC_SCIENTIFIC_PROGRAM_R0.md`

The next stage must be chosen from the actual S1 verdict.

Do not automatically start historical training/economic qualification or capacity scaling merely because S1 PASSes.

---

# 9. Tests that explicitly do NOT qualify S1 by themselves

The following are useful but individually insufficient:

- policy distribution unit tests;
- V-trace formula unit tests;
- reward telescoping unit tests;
- one learner optimizer step;
- checkpoint checksum changed;
- child policy emitted one action;
- in-memory toy learner converged;
- a task trained only from collector-side tensors;
- a single successful seed;
- lower actor/critic loss;
- high throughput.

S1 is specifically the conjunction of **durable replay + canonical joint policy learning + repeated optimization + known-answer behavioral success + negative controls + recovery/provenance**.

---

# 10. Suggested new path ownership

Prefer successor prefixes:

```text
cb16_local_opt/post_cc_observation_*.py
cb16_local_opt/post_cc_replay_*.py
cb16_local_opt/post_cc_joint_*.py
cb16_local_opt/post_cc_learner_*.py
cb16_local_opt/post_cc_s1_qualification_*.py
scripts/run_r11_post_cc_s1_*.py
tests/test_post_cc_s1_*.py
authority/rearchitecture_r11/CB16_R11_POST_CC_S1_*.json
```

Do not mutate historical CC modules solely to avoid having successor names.

Where a bug in a shared canonical CC component is discovered, classify it explicitly. If fixing it changes frozen CC semantics, open a new versioned contract rather than silently editing history.

---

# 11. Definition of Done

S1 is DONE only when all of the following are true:

1. S0 is qualified and frozen;
2. Brain-ready observations are durably reconstructable;
3. learner training truth comes from persistent replay, not rollout-memory shortcuts;
4. canonical joint direction+risk likelihood is used with true persisted behavior `log_mu`;
5. Critic/V-trace/boundary semantics are integrated;
6. updates/checkpoints are exactly-once and recoverable;
7. child generations act while required account continuity is preserved;
8. every mandatory known-answer task meets its preregistered multi-seed criterion;
9. negative controls behave as preregistered;
10. objective/provenance/gradient/firewall audits pass;
11. the machine-readable S1 receipt exists;
12. no ECONOMIC/TRANSFER claim is made;
13. FINAL remains sealed and no fresh market data is used.

If any scientific task validly fails, report `SCIENTIFIC_FAIL`. Do not rescue it inside S1. A new hypothesis requires a new versioned program.
