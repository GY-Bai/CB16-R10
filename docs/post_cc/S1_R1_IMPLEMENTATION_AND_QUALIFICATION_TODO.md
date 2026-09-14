# CB16 R11 S1 R1 — End-to-End Durable Learnability Implementation & Qualification TODO

**Role owner:** Sol TODO / code-review layer  
**Code executor:** DS Flash  
**Upstream alignment:** user-approved principles → Astra documents → this Sol TODO  
**Repository:** `GY-Bai/CB16-R10`  
**Canonical S1 branch:** `ai/r11-post-cc-s1-end-to-end-learnability-r1`  
**Accepted S0-v2 merged parent:** `fc80b472236e7a4df8094563f8adac826fc42231`  
**S1 base authority:** `authority/rearchitecture_r11/CB16_R11_POST_CC_S1_BASELINE_V1.json`  
**Parent evidence:** `POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`  
**Target S1 evidence ceiling:** `INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER`  
**FINAL:** SEALED  
**Fresh / historical market corpus:** FORBIDDEN  
**Status at TODO publication:** IMPLEMENTATION AUTHORIZED / S1 SCIENTIFIC QUALIFICATION NOT YET STARTED

This document is the **post-S0-v2 execution delta** for S1. It does not rewrite historical S1 V1 authority, receipts, or the qualified S0-v2 foundation. Where the legacy `S1_END_TO_END_LEARNABILITY_TODO.md` still describes observation/replay/joint-learner/exactly-once/generation-continuity components as missing, the post-S0-v2 baseline and handoff supersede that implementation-gap text only.

Read together with:

- `docs/post_cc/S1_POST_S0V2_BASE_HANDOFF.md`
- `docs/post_cc/S1_END_TO_END_LEARNABILITY_TODO.md`
- `docs/S_SERIES_TODO_AUTHORING_PRINCIPLES.md`
- `docs/ROLES_AND_REVIEW_PROTOCOL.md`
- `authority/rearchitecture_r11/CB16_R11_S0V2_RECEIPT_V1.json`
- `authority/rearchitecture_r11/CB16_R11_POST_CC_S1_BASELINE_V1.json`
- `authority/rearchitecture_r11/CB16_R11_POST_CC_S1_TASK_REGISTRY_V1.json`
- `authority/rearchitecture_r11/CB16_R11_POST_CC_S1_RUN_SPEC_V1.json`

---

# 0. Role boundary and communication protocol

The responsibility chain is fixed:

```text
user principles
  -> Astra document alignment
  -> Sol TODO + scientific concretization + code review
  -> DS Flash implementation + tests + Shanxi CI evidence
  -> Sol review / CHANGES_REQUIRED or acceptance
  -> only then merge authority
```

DS Flash is the **code executor**, not the scientific authority and not the final reviewer.

DS must:

1. implement the TODO on the canonical S1 branch;
2. add tests and GitHub Actions coverage;
3. run runtime tests only through GitHub Actions → `shanxi-docker-r11`;
4. open a PR to `main` when the implementation candidate is ready;
5. put the exact implementation state, run evidence, failures, and unresolved items in the PR body / candidate record;
6. respond to Sol review on the **same PR and same branch** unless Sol explicitly versions the program;
7. never merge the PR itself unless separately instructed by the owner / accepted reviewer flow.

Sol will independently inspect the actual diff and exact SHA. DS self-report, green CI, or unit tests do not substitute for Sol review.

## 0.1 Two-phase PR rule

S1 uses one PR as the multi-round communication surface, but formal qualification is separated from implementation review.

### Phase A — implementation candidate / no official S1 scientific verdict

DS implements all code, task generators, controls, deterministic gate compiler, workflow, artifact writer, and bounded smoke. DS then opens the PR with terminal state:

`READY_FOR_SOL_CODE_REVIEW`

The Phase-A smoke is engineering evidence only. It must not be presented as S1 PASS/FAIL.

### Phase B — formal qualification only after Sol acceptance of the implementation candidate

Formal 5-seed S1 qualification may begin only after Sol records:

`READY_FOR_S1_QUALIFICATION`

The qualification must use the exact reviewed implementation SHA and exact frozen execution manifest. Once formal qualification begins, scientific task parameters, seeds, success thresholds, reward, model, optimizer, evaluation population, and qualification budget cannot be changed to rescue results.

If a later Sol review finds an **implementation bug** in formula translation, branch logic, persistence, or gate aggregation, the affected qualification run is classified implementation-invalid and is not a scientific FAIL. DS may fix the bug on the same PR, but the scientific manifest must remain unchanged; the changed code requires renewed Sol review and renewed exact-SHA Shanxi qualification.

---

# 1. Exact start rule — do not reopen S0-v2

Before code changes, DS must verify and report:

- current branch is `ai/r11-post-cc-s1-end-to-end-learnability-r1`;
- lineage descends from `fc80b472236e7a4df8094563f8adac826fc42231`;
- `CB16_R11_S0V2_RECEIPT_V1.json` is `QUALIFIED` with Sol R2 `PASS`;
- `CB16_R11_POST_CC_S1_BASELINE_V1.json` is present;
- frozen task registry and run spec are present;
- FINAL/fresh/historical-market firewalls remain closed;
- old branch `ai/r11-post-cc-s1-end-to-end-learnability-r0@9d991143...` is **not** imported wholesale, merged, rebased, or used as authority.

The following S0-v2 surfaces are inherited and must be **reused, not reimplemented merely because the old S1 TODO lists them as missing**:

- `post_cc_observation_fact_v1.py`
- `post_cc_observation_store_v1.py`
- `post_cc_durable_collection_v1.py`
- `post_cc_joint_replay_v1.py`
- `post_cc_replay_materializer_v1.py`
- `post_cc_joint_batch_v1.py`
- `post_cc_joint_policy_loss_v1.py`
- `post_cc_critic_vtrace_v1.py`
- `post_cc_update_transaction_v1.py`
- `post_cc_learner_v1.py`
- `post_cc_generation_continuity_v1.py`

Changes to these qualified files are allowed only when a concrete S1 requirement cannot be met through extension/adaptation. Any such change must be called out explicitly in the PR as **qualified-foundation modification** and receives heightened Sol review.

---

# 2. Frozen scientific values that DS must not choose or change

The following remain frozen from the S1 task registry/run spec:

- task IDs:
  - `ACCOUNT_DEPENDENT_ACTION`
  - `DELAYED_CONSEQUENCE_CREDIT`
  - `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION`
  - `OFF_POLICY_VTRACE_CORRECTION`
  - `A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION`
- seeds: `1701, 1702, 1703, 1704, 1705`;
- minimum positive seeds passing: `4/5`;
- Actor: `CCCentralBrain`, market/account/execution dimensions `2/3/2`, hidden `8`;
- frozen market organ remains frozen;
- separate `SeparateCritic`, input dimension `7`, hidden `8`;
- Actor SGD LR `0.01`;
- Critic SGD LR `0.01`;
- persistent replay only;
- true decision-time behavior `log_mu` required;
- target joint `log_pi` recomputed on nominal behavior action;
- maximum policy decisions per task/seed `100000`;
- complete-sample arithmetic return orientation;
- failures/bankruptcies remain in denominator;
- no survivor-only, log-growth, Sharpe, hidden drawdown, or hidden survival objective;
- no handcrafted regime detector / rule activation for A→B→A;
- no result-driven rescue.

No task ID, phase flag, oracle action, future outcome, or correct label may be inserted into Actor inputs.

---

# 3. S1 execution-manifest concretization — must be frozen before formal results

The historical V1 registry names the scientific problems but does not fully numericize every generator/evaluation detail. Before formal qualification, DS must implement and commit:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S1_EXECUTION_MANIFEST_V1.json`

This manifest is an **execution concretization under the frozen scientific contracts**, not permission for DS to redesign the science.

It must encode the constants in this section exactly. If implementation canaries prove a listed constant makes the task internally impossible or violates an inherited contract, DS must stop that task and report the minimal counterexample in the PR. DS must not silently change the constant. Sol will revise/version the TODO/manifest before formal qualification if a true pre-result contract defect exists.

## 3.1 Common evaluation and aggregation definitions

All score-oriented positive tasks use a higher-is-better scalar `score` declared by the task spec.

For a task/context with a declared oracle comparator:

```text
initial_gap = oracle_score - initial_score
final_gap   = oracle_score - final_score
```

Qualification precondition:

```text
initial_gap > 1e-9
```

If `initial_gap <= 1e-9`, the task instance is not allowed to auto-PASS. Classify the seed/task as `EVIDENCE_INSUFFICIENT` unless a task-specific frozen rule gives another interpretation.

Oracle-gap reduction:

```text
gap_reduction = (initial_gap - final_gap) / initial_gap
```

Positive threshold:

```text
gap_reduction >= 0.50
```

Do not use absolute values, swap subtraction order, clamp a negative final gap to zero, or change denominator after seeing results. A model that exceeds the finite oracle comparator may have `gap_reduction > 1`; this is allowed.

All floating comparisons that are intended to represent strict scientific ordering use tolerance `1e-9` unless the frozen task says otherwise. The analytic V-trace tolerance remains `1e-6`.

## 3.2 Oracle comparator action class

For task environments that need a finite action comparator, enumerate exactly:

```text
FLAT, risk = 0
LONG, risk ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
SHORT, risk ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
```

The oracle comparator is the highest complete-sample arithmetic-return member of this declared finite class under the task environment. It is a comparator class, not a claim of the global optimum over every continuous risk value.

Oracle evaluation must use the same mechanical execution/account rules as the policy evaluation. Test code may independently enumerate the finite class; it must not call the production gate/oracle function to manufacture its expected answer.

## 3.3 Common bounded training schedule

Formal qualification schedule per ordinary positive task/seed:

- maximum qualification policy decisions actually consumed: `16384`;
- collection unit: `128` policy decisions under a behavior checkpoint fixed within the unit;
- after each complete collection unit: `1` durable replay learner update;
- replay batch target size: `128` decision samples when at least that many validated samples exist, otherwise all available validated samples;
- sampling: uniform from all eligible durable replay for that task/run, using a dedicated replay RNG derived deterministically from the frozen seed;
- no age-based expiry;
- each successful update commits a child checkpoint exactly once;
- child becomes behavior policy only through the qualified committed-child generation switch;
- evaluation uses a dedicated RNG stream not reused by training/action/replay RNG.

A task may terminate earlier only for mechanical terminal/account death at the episode level; the **qualification training program does not early-stop a seed merely because the success threshold was reached**. This avoids unequal positive/control budgets.

Controls matched to a positive task use the same decision/update budget unless their contract is a pure integrity-rejection check that does not train.

A bounded engineering smoke may use a smaller explicit `SMOKE_ONLY` budget, but its results cannot enter the S1 scientific gate compiler.

## 3.4 Policy evaluation population

At each frozen pre/post evaluation point:

- evaluate every declared task context;
- draw `2048` nominal actions per context from the policy with a dedicated deterministic evaluation RNG;
- environment stochastic branches are enumerated exactly where the task defines a finite branch distribution rather than re-sampled;
- aggregate all sampled actions and all outcome branches with declared probabilities;
- report mean complete-sample arithmetic return and task-specific behavior statistics;
- evaluation decisions do not enter training replay.

All evaluation failures/bankruptcies remain in the denominator.

---

# 4. Implementation task map

Existing S1 task numbering remains authoritative. This document narrows what remains after S0-v2.

| Legacy task IDs | Post-S0-v2 treatment |
|---|---|
| `S1-001` | verify new S0-v2 baseline / execution manifest |
| `S1-002..012` | inherited qualified foundation; regression/adoption only |
| `S1-013..020` | **implement now**: task generators, oracles, controls |
| `S1-021..031` | **implement now**: repeated-learning qualification runtime and audits |
| `S1-032..033` | **implement now**: deterministic gate compiler + artifacts |
| `S1-034` | DS emits review candidate only; final authoritative receipt is reviewer-owned |
| `S1-035` | navigation/merge only after final reviewer verdict |

---

# 5. S1-001 — baseline adoption and machine-readable execution manifest

DS must create the execution manifest from Section 3 and a loader/validator used by both tests and the qualification runner.

Mandatory fail-closed checks:

- accepted parent SHA is `fc80b472236e7a4df8094563f8adac826fc42231`;
- S0-v2 receipt status/review verdict match authority;
- task IDs/seeds/model/LRs/max budget match frozen V1 files;
- all scientific constants in Sections 3 and 6 are represented in one manifest hash;
- FINAL/fresh/historical market access are false;
- no runtime CLI flag may override seeds, thresholds, reward, optimizer, model size, or qualification budget in `qualification` mode.

Allowed CLI differences may select only non-scientific behavior such as `smoke` versus `qualification`, output root, or logging verbosity.

---

# 6. S1-013..017 — five positive task environments

Task modules may be split into several files; recommended root is:

`cb16_local_opt/post_cc_s1_tasks_v1.py`

Every task spec must provide at least:

- task ID and spec hash;
- causal generator inputs;
- initial account construction and reachability proof;
- Actor-visible observation definition;
- action/execution mapping;
- environment transitions/outcome probabilities;
- reward and boundary semantics;
- oracle comparator enumeration;
- pre/post evaluation function;
- positive predicate;
- required negative controls;
- durable source/provenance fields.

## 6.1 S1-013 — `ACCOUNT_DEPENDENT_ACTION`

Purpose: prove the Brain uses Account state, not Market-only information.

Frozen construction:

- scored market observation is identical across two contexts: mark price `100.0`, same causal market features and same execution features before the scored decision;
- genesis account is initial capital `1000.0`;
- two scored account states must be generated from genesis through the canonical execution path, not manually forged:
  - context `HELD_LONG`: previously established LONG target near risk `0.5`;
  - context `HELD_SHORT`: previously established SHORT target near risk `0.5`;
- setup transitions are provenance facts but are excluded from the scored policy-learning population;
- scored horizon is one policy decision followed by a flat-price terminal mark `100.0`;
- scored execution cost must be positive: fee rate `0.001`, slippage `0`, max gross leverage `2.0`, initial margin rate `0.5`, maintenance margin rate `0.25`;
- no funding, external flow, or hidden penalty;
- oracle is independently enumerated over the common finite action class.

Pre-training task validity test must prove:

1. `HELD_LONG` and `HELD_SHORT` Actor market/execution vectors are identical;
2. account vectors differ;
3. both states are reachable from canonical genesis via valid execution;
4. oracle direction differs between the two contexts;
5. Market-only/account-zero ablation removes the information required to distinguish them.

Primary score: mean complete-sample arithmetic terminal return over the two contexts.

Seed success requires:

- oracle-gap reduction `>= 0.50`;
- post score > initial score;
- the full model assigns higher probability to each context's oracle direction than at initialization;
- the matched account-input ablation **does not** meet the same positive threshold for that seed.

Task PASS requires at least `4/5` full-model seeds passing and at least `4/5` matched ablation seeds failing the positive predicate.

## 6.2 S1-014 — `DELAYED_CONSEQUENCE_CREDIT`

Purpose: prove credit reaches an earlier decision through durable downstream account consequences and a compute/restart boundary.

Frozen construction:

- initial capital `1000.0`;
- two equally weighted causal contexts:
  - `UP_SIGNAL`: decision-time mark `101.0`;
  - `DOWN_SIGNAL`: decision-time mark `99.0`;
- the first post-decision environment advance keeps the mark unchanged and contributes no directional market PnL;
- a durable compute/restart boundary occurs after the first decision transition;
- before the next policy decision is allowed, a later **no-decision mechanical/environment advance** moves:
  - `UP_SIGNAL`: `101.0 -> 111.0`;
  - `DOWN_SIGNAL`: `99.0 -> 89.0`;
- terminal liquidation/settlement produces the final account consequence;
- fee/slippage are zero in this task so the first transition contains no action-sign reward clue;
- the Actor receives no task ID, outcome label, or future price.

Because the consequence occurs across a no-decision advance, DS must implement a durable decision-interval credit/provenance adapter if the existing S0-v2 transition record cannot represent this directly. The adapter must:

- keep the no-decision advance as raw/environment truth with **no fabricated nominal action or `log_mu`**;
- bind the downstream account-equity consequence to the preceding decision through explicit source hashes and time ordering;
- survive process/object destruction and rebuild from durable state;
- never rewrite the original raw transition facts;
- expose one validated decision-level reward/boundary view to the learner.

This is a high-risk review surface. Do not fake the delayed task by assigning the future answer directly as an immediate label.

Oracle comparator is enumerated over the common finite action class using complete terminal arithmetic return. Expected oracle directions must be LONG for `UP_SIGNAL` and SHORT for `DOWN_SIGNAL`; task preflight fails if independent enumeration does not prove this.

Primary score: mean complete terminal arithmetic return over both contexts.

Seed success requires:

- oracle-gap reduction `>= 0.50`;
- post score > initial score;
- probability of the correct **early** direction rises in both contexts;
- matched shuffled-credit control has lower gap reduction than the positive run by `> 1e-9` for that seed.

At least `4/5` seeds must satisfy the full predicate.

## 6.3 S1-015 — `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION`

Purpose: prove complete-sample arithmetic expectation wins even when its preferred action has higher failure/bankruptcy frequency.

Frozen construction:

- initial capital `1000.0`;
- one scored decision at mark `100.0`;
- action/environment execution uses zero fee and zero slippage for analytic clarity;
- max gross leverage `2.0`, initial margin rate `0.5`;
- terminal environment has two exact branches:
  - `WIN`, probability `0.80`, terminal mark `120.0`;
  - `LOSS`, probability `0.20`, terminal mark `40.0`;
- after terminal marking, exposure is mechanically liquidated;
- a failure/bankruptcy fact is `terminal_equity <= 0.0`; if the runtime requires explicit account closure to encode mechanical terminal, implement a policy-neutral insolvency closure adapter and preserve its provenance;
- all branch outcomes, including negative/zero equity, remain in the arithmetic denominator.

The task preflight must independently enumerate the common finite action comparator and prove that at least one non-FLAT comparator:

1. has **strictly higher complete-sample arithmetic expected return** than FLAT; and
2. has **strictly higher failure/bankruptcy frequency** than FLAT.

The positive known-answer family is the highest-arithmetic-EV comparator, regardless of its bankruptcy frequency.

Primary score: exact branch-probability-weighted arithmetic terminal return.

Additional reported statistic: exact branch-probability-weighted bankruptcy frequency.

Seed success requires:

- oracle-gap reduction `>= 0.50`;
- post score > initial score;
- policy probability mass moves toward the higher-arithmetic-EV action family;
- no survivor filtering or alternative risk-adjusted objective is used.

The gate must **not** reject the preferred action because its bankruptcy frequency is higher.

## 6.4 S1-016 — `OFF_POLICY_VTRACE_CORRECTION`

Purpose: prove persisted behavior likelihood and target/behavior correction are mathematically correct and usable for learning.

This task has two mandatory layers.

### Layer A — analytic scalar V-trace fixture

Use a hand-written reference implementation in tests, not the production `vtrace()` function, with:

```text
rewards   = [0.0, 0.0, 1.0]
discounts = [1.0, 1.0, 0.0]
values    = [0.20, 0.10, 0.00]
bootstrap = 0.0
log_rhos  = [ln(2.0), ln(0.5), ln(1.5)]
rho_bar   = 1.0
c_bar     = 1.0
pg_rho_bar= 1.0
```

Reference equations, evaluated backward in time:

```text
rho_t      = exp(log_rho_t)
clipped_rho_t = min(rho_bar, rho_t)
c_t        = min(c_bar, rho_t)
delta_t    = clipped_rho_t * (r_t + gamma_t * V_{t+1} - V_t)
v_s        = V_s + delta_s + gamma_s * c_s * (v_{s+1} - V_{s+1})
pg_adv_t   = min(pg_rho_bar, rho_t) * (r_t + gamma_t * v_{t+1} - V_t)
```

Analytic tolerance: `1e-6` for `rhos`, clipped rhos, `c`, `vs`, and policy-gradient advantages.

### Layer B — durable off-policy learning

- behavior checkpoint is deliberately distinct from the target checkpoint;
- behavior action and true joint `log_mu` are persisted at decision time;
- target `log_pi` is recomputed from the durable **nominal** action, not executed quantity;
- at least one controlled population must contain ratios both above and below `1.0`;
- fabricated/reconstructed `log_mu` must fail closed;
- same-policy sentinel must reduce to `log_pi - log_mu ≈ 0` within existing component tolerance;
- the end-to-end off-policy task must meet the common oracle-gap reduction threshold in at least `4/5` seeds.

Any sign reversal (`log_mu - log_pi`), comparison reversal, or execution-action substitution is a merge blocker even if tests are green.

## 6.5 S1-017 — `A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION`

Purpose: prove old A experience remains usable through a B phase and a returning-A phase without explicit regime rule activation.

Frozen construction:

- no phase/task ID enters Actor observation;
- A and B are naturally distinguishable only through causal market observation:
  - A context: mark `101.0`, one-step terminal mark `111.0`, correct direction family LONG;
  - B context: mark `99.0`, one-step terminal mark `89.0`, correct direction family SHORT;
- fee/slippage zero for this task;
- each episode begins from the same declared flat `1000.0` account distribution so the condition is market-driven, not leaked through account reset labels;
- old durable A replay may not be deleted, downweighted to zero, or expired merely because B is newer;
- no `if phase == A`, regime detector, cycle index, or rule-switch path may select policy behavior.

Training phases per seed:

```text
A1: 4096 policy decisions + normal durable updates
B : 4096 new B policy decisions + normal durable updates using replay pool that still contains A1
A2: 4096 new A policy decisions + normal durable updates using the full retained replay pool
```

The combined `12288` decisions remain below the frozen `100000` maximum.

Evaluation checkpoints:

1. `BASELINE_A` before A1;
2. `POST_A1_A` after A1;
3. `POST_B_B` after B;
4. `RETURN_A_PRE_A2` immediately after B and **before any A2 learning**;
5. `POST_A2_A` after A2.

Seed success requires all of:

- A1 oracle-gap reduction `>= 0.50`;
- B oracle-gap reduction `>= 0.50`;
- `RETURN_A_PRE_A2` A score > `BASELINE_A` A score by `> 1e-9`;
- A2 A oracle-gap reduction relative to `BASELINE_A` `>= 0.50`;
- durable A1 samples still exist after B and are eligible for generic replay sampling;
- no handcrafted regime activation exists.

At least `4/5` seeds must satisfy the full predicate.

This task explicitly distinguishes **immediate retention evidence** (`RETURN_A_PRE_A2`) from **bounded relearning evidence** (`POST_A2_A`). Report both; do not merge them into one vague score.

---

# 7. S1-018..020 — negative controls and integrity attacks

Controls are not optional decorations. Their generator, evaluation population, seed mapping, and relation to the matched positive task must be machine-readable.

## 7.1 S1-018 — no-signal / zero-reward control

For every positive task family to which this control is declared applicable:

- preserve observation/account/action marginal shapes;
- remove the learnable action→reward relation or set the declared task reward to zero;
- use the same training decision/update budget;
- do not change model/optimizer.

Control PASS rule:

- no more than `1/5` seeds may accidentally satisfy the matched positive task's complete positive predicate.

A control is not required to show literally zero parameter movement.

## 7.2 S1-019 — shuffled-credit control

For delayed-credit and off-policy matched populations:

- deterministically permute decision→consequence association using a dedicated control RNG derived from the seed;
- preserve the multiset of observations, actions, rewards/outcomes, and sequence lengths as far as the task permits;
- prove in test that the permutation actually breaks the original action/consequence pairing;
- do not shuffle fields inside one sample in a way that creates impossible hash/provenance combinations.

Matched relation:

- positive seed gap reduction must exceed shuffled-control gap reduction by `> 1e-9` for the seed to satisfy the control relation.

## 7.3 S1-020 — random/impossible and integrity controls

Random/impossible target control:

- target assignment must be generated independently of Actor-visible causal inputs;
- training and frozen evaluation must use separate deterministic target realizations so the control is not a finite-table memorization test;
- no more than `1/5` seeds may satisfy the matched positive predicate.

Mandatory fail-closed integrity attacks are deterministic and require 100% rejection:

- observation content/hash mismatch;
- missing/corrupted behavior identity;
- fabricated `log_mu`;
- nominal action replaced by executed action/quantity;
- cross-account replay splice;
- time-order corruption;
- non-FLAT risk outside support;
- FLAT with nonzero risk;
- PREPARED/STAGED child generation used as runtime authority;
- committed child checkpoint tamper;
- same-instance retry after post-gradient failure;
- terminal supplied with bootstrap;
- truncation missing durable bootstrap.

---

# 8. S1-021..031 — repeated-learning runtime and qualification audits

Recommended new surfaces:

- `cb16_local_opt/post_cc_s1_training_loop_v1.py`
- `cb16_local_opt/post_cc_s1_qualification_v1.py`
- `scripts/run_r11_post_cc_s1_learnability.py`

Names may differ, but there must be one canonical auditable path.

## 8.1 Required loop — every positive learning result must come through this path

```text
synthetic causal environment
 -> canonical Brain observation
 -> stochastic nominal direction + conditional risk
 -> true decision-time log_mu
 -> A-authoritative execution/account consequence
 -> durable immutable observation/transition/raw facts
 -> collector/live objects destroyed at declared restart sentinel
 -> replay reopened/materialized only from durable state
 -> validated joint batch
 -> target log_pi on nominal action
 -> separate Critic + explicit boundary/bootstrap + V-trace
 -> Actor/Critic gradient update
 -> exactly-once child checkpoint commit
 -> committed-child policy identity
 -> explicit authorized generation switch
 -> same logical account continues where task requires it
 -> child policy actually produces later decisions
 -> repeated collection/learning
 -> frozen evaluation
```

A shortcut around any arrow may remain a diagnostic but cannot feed S1 qualification.

## 8.2 S1-021 — unified runner

The runner must support exactly two scientific modes:

- `smoke`: bounded engineering evidence, not scientific verdict;
- `qualification`: exact frozen manifest, no scientific override flags.

Qualification mode must fail closed if the execution manifest, task registry, run spec, code identity, model identity, or firewall does not match the expected frozen identity.

## 8.3 S1-022/023 — all-seed pre/post evaluation

For every positive task and control:

- retain every seed result;
- retain every failed episode/failure/bankruptcy;
- record initial, intermediate where required, and final scores;
- record policy behavior statistics and arithmetic return;
- never select best seed only.

## 8.4 S1-024 — durable restart sentinel

At least once per collection unit used for qualification:

1. close/flush durable stores;
2. destroy collector-side Python objects that contain rollout truth;
3. reconstruct materializer from durable roots;
4. verify sample/content identities;
5. build learner batch only from reopened durable state.

An in-memory object may be used after it has been reconstructed/validated from durable truth; the collector's private rollout objects may not be the only source of training truth.

## 8.5 S1-025 — exactly-once hostile update matrix

Reuse S0-v2 transaction machinery and keep the R1 blocker closure intact.

At minimum test faults:

- after gradient / before stage;
- after stage / before commit;
- after commit / before acknowledgement.

For post-mutation failures:

- same learner instance retry must be rejected;
- state on rejected retry must remain unchanged;
- recovery must rebuild from durable parent/checkpoint authority;
- exactly one committed child exists;
- final optimizer step is `before + 1`, never `+2`.

## 8.6 S1-026 — provenance audit

Every qualified update must be traceable through:

- task/spec/manifest hash;
- seed and RNG stream identities;
- account lineage;
- observation content hash;
- policy generation/id/hash;
- nominal direction/risk;
- persisted true `log_mu`;
- transition/raw consequence hashes;
- replay sample/materialization hash;
- batch hash;
- update ID;
- parent checkpoint;
- child checkpoint;
- generation-switch receipt.

Missing links fail qualification.

## 8.7 S1-027 — gradient ownership

Verify:

- frozen market organ unchanged byte/semantic hash across qualification;
- authorized Actor paths receive finite gradients when task signal requires them;
- separate Critic receives finite gradients;
- Actor and Critic parameters do not alias.

Do not interpret “nonzero gradient” alone as learnability PASS.

## 8.8 S1-028/029 — objective and benchmark firewalls

Machine-check:

- high-bankruptcy task uses complete arithmetic expectation;
- all failures remain in denominator;
- no log wealth, Sharpe, drawdown or survival objective enters training/evaluation;
- B&H and FLAT remain sibling benchmark components with no master precedence/winner;
- S1 synthetic task success does not perform production promotion.

## 8.9 S1-030 — data firewall

Qualification result must explicitly contain:

```text
FINAL_opened = false
fresh_data_used = false
historical_market_corpus_accessed = false
```

Any true value invalidates S1 qualification under this program.

## 8.10 S1-031 — performance is only an execution guard

Do not reopen CC performance competition. Reuse the canonical fast path only when semantics/provenance are identical. Throughput cannot decide the scientific winner and cannot justify dropping durable facts, controls, failed seeds, or replay validation.

---

# 9. S1-032 — deterministic gate compiler must exist before formal qualification

Recommended surface:

`cb16_local_opt/post_cc_s1_gate_compiler_v1.py`

The gate compiler must be complete and tested **before** `READY_FOR_S1_QUALIFICATION`.

It must consume result records, not re-run training logic to decide expected answers.

Mandatory synthetic compiler fixtures:

1. all-pass record → `PASS`;
2. exactly 3/5 positive seeds pass → positive task FAIL;
3. exactly 4/5 positive seeds pass → positive task PASS;
4. 2/5 no-signal control seeds falsely meet positive predicate → control FAIL;
5. 1/5 falsely meet → control PASS;
6. positive delayed score equals shuffled score → relation FAIL;
7. positive delayed score exceeds shuffled by `>1e-9` → relation PASS;
8. missing seed/result → `EVIDENCE_INSUFFICIENT`, never PASS;
9. valid implementation + failed frozen learning criterion → `SCIENTIFIC_FAIL`;
10. corrupted provenance/invariant → implementation/contract failure, not scientific FAIL;
11. FINAL/fresh firewall violation → fail closed;
12. NaN/Inf/zero denominator invalidity → never PASS.

Final taxonomy is restricted to:

- `PASS`
- `SCIENTIFIC_FAIL`
- `CONTRACT_MISMATCH`
- `EXECUTION_BLOCKED`
- `HARDWARE_LIMIT`
- `EVIDENCE_INSUFFICIENT`

Do not add a vague `PARTIAL_PASS` that hides a mandatory failure.

---

# 10. S1-033 — durable artifacts

Formal qualification must emit at minimum:

```text
artifacts/post_cc_s1/
  S1_RESULT.json
  S1_REPORT.md
  execution_manifest.json
  task_results/<task>/<seed>.json
  control_results/<control>/<seed>.json
  provenance/
  checkpoints/
  replay_manifests/
```

`S1_RESULT.json` must contain machine-readable gate inputs and final classification.

`S1_REPORT.md` is explanatory only; it cannot override machine result.

Persist hashes for the execution manifest, task specs, result files, replay/sample manifests, parent/child checkpoints and the final artifact bundle where the workflow supports it.

Failed seeds and failed controls are retained; no deletion before aggregation.

---

# 11. GitHub Actions → Shanxi Docker is mandatory

DS must add or bind a dedicated S1 workflow. Recommended path:

`.github/workflows/cb16-r11-post-cc-s1-learnability.yml`

Requirements:

- use the existing shared Shanxi preflight;
- business jobs run on `[self-hosted, shanxi-docker-r11]`;
- use the verified canonical Python in the existing runner contract;
- print and record actual checkout SHA/tree;
- print Python / torch / GPU identity;
- no nested ad-hoc container redesign;
- no FINAL/fresh/historical-market access;
- timeout must be explicit;
- qualification mode must upload required result artifacts;
- workflow inputs must not allow seed/threshold/reward/model/LR/budget overrides in qualification mode.

## 11.1 Required CI stages

### CI-A — implementation/unit/invariant suite

Run focused S0-v2 regression plus all new S1 task/oracle/control/gate tests on Shanxi Docker.

### CI-B — bounded smoke

- one frozen seed (`1701`);
- explicit `SMOKE_ONLY` reduced budget;
- execute the complete durable loop for every positive task and representative controls;
- proves wiring/executability only;
- must not enter scientific gate compiler as qualification evidence.

### CI-C — formal qualification

Only after Sol says `READY_FOR_S1_QUALIFICATION`:

- exact reviewed SHA;
- all five frozen seeds;
- exact qualification manifest;
- all mandatory positive tasks and controls;
- artifact upload;
- no rescue rerun under modified scientific values.

### CI-D — record-binding head verification

After DS commits the review-candidate record, rerun exact-head verification. If the only net change from the qualified runtime candidate is the review-candidate/report metadata, the workflow may reuse deterministic scientific artifacts only if it verifies the runtime code/tree identity and artifact hashes exactly. Otherwise rerun full qualification.

Repo Guard alone is not Shanxi runtime evidence.

The temporary ChatGPT sandbox may be used for reading/editing/static preparation only. Do not run business tests there and report them as runtime evidence.

---

# 12. DS Flash high-density logic review hotspots

DS should write code for readability and auditability: named intermediate values, small functions, explicit decision tables. Dense one-line boolean logic is discouraged in the following surfaces.

Sol will independently review these before merge.

## 12.1 Formula hotspots

- joint risk log-density and Jacobian;
- `log_ratio = log_pi - log_mu` direction;
- `rho = exp(log_ratio)` and clipping location;
- V-trace backward recursion and temporal indices;
- terminal zero-bootstrap vs truncation durable bootstrap;
- Actor loss sign and advantage detach behavior;
- Critic target detach;
- arithmetic return denominator;
- oracle gap and gap-reduction subtraction order;
- exact branch probability weighting in high-bankruptcy task;
- replay sampling weight interpretation.

## 12.2 Direct branch / comparison hotspots

Tests must cover below/equal/above for:

- `>= 0.50` oracle-gap threshold;
- `4/5` versus `3/5` seed aggregation;
- no-signal `<=1/5` accidental-positive rule;
- strict positive-vs-shuffled `>1e-9` relation;
- `terminal_equity <= 0.0` bankruptcy definition;
- terminal vs truncation boundary precedence;
- missing/PREPARED/STAGED/COMMITTED update authority;
- same-instance retry after mutation;
- FLAT risk exactly zero;
- non-FLAT risk strict `(0,1)` support;
- NaN/Inf/missing result behavior;
- empty replay/control result sets.

## 12.3 Independent expected answers

Do not generate test expected values by calling the production function being tested.

Use:

- hand arithmetic for finite action/oracle tables;
- the scalar V-trace equations in Section 6.4;
- explicit truth tables for gate aggregation;
- manually constructed terminal/truncation examples;
- known fixed hashes/provenance corruption cases.

---

# 13. PR contract — the PR is the multi-round review surface

Target branch: `main`  
Source branch: `ai/r11-post-cc-s1-end-to-end-learnability-r1`

Recommended title:

`R11 S1: durable end-to-end learnability candidate`

Do not squash/rebase away reviewed history while review is active unless Sol explicitly requests it.

## 13.1 Required PR body

DS must keep the PR body updated with this structure:

```markdown
## S1 candidate state

- branch:
- exact base / parent:
- implementation candidate SHA:
- tree SHA:
- execution manifest SHA256:
- terminal state: READY_FOR_SOL_CODE_REVIEW | READY_FOR_SOL_REVIEW

## Task coverage

| Task | Status | Main files/tests | Notes |
|---|---|---|---|
| S1-001 | ... | ... | ... |
| S1-013 | ... | ... | ... |
...
| S1-033 | ... | ... | ... |

## Qualified S0-v2 foundation touched?

- files changed:
- reason:
- regression tests:

## Scientific concretization

- task spec hashes:
- seeds:
- model/optimizer:
- training budget:
- evaluation population:
- thresholds:
- no-rescue confirmation:

## Shanxi CI evidence

### implementation/unit
- workflow/run/attempt:
- checkout SHA/tree:
- preflight job:
- business job:
- runner:
- Python/torch/GPU:
- tests passed/failed/skipped:

### bounded smoke
- workflow/run/attempt:
- checkout SHA/tree:
- result/artifact:
- explicit statement: NOT SCIENTIFIC QUALIFICATION

### formal qualification (only after Sol approval)
- workflow/run/attempt:
- checkout SHA/tree:
- business job:
- artifact ID/hash:
- S1_RESULT semantic hash:

## Scientific results

| Task/control | seed 1701 | 1702 | 1703 | 1704 | 1705 | aggregate |
|---|---:|---:|---:|---:|---:|---:|
| ACCOUNT_DEPENDENT_ACTION | | | | | | |
| DELAYED_CONSEQUENCE_CREDIT | | | | | | |
| HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION | | | | | | |
| OFF_POLICY_VTRACE_CORRECTION | | | | | | |
| A_B_A_RETENTION... | | | | | | |
| controls | | | | | | |

## Firewall

- FINAL_opened:
- fresh_data_used:
- historical_market_corpus_accessed:
- economic evidence claimed:
- transfer evidence claimed:

## Known issues / unresolved items

- ...

## First unfinished item

- ...

## Review request

- requested Sol state: READY_FOR_SOL_CODE_REVIEW / READY_FOR_SOL_REVIEW
```

Do not write `PASS` merely because implementation is complete. The PR requests review; reviewer authority determines acceptance.

---

# 14. Candidate authority record — DS writes candidate, not final receipt

Before requesting final Sol review, DS must create/update:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S1_REVIEW_CANDIDATE_V1.json`

Minimum fields:

- schema/stage/status;
- exact accepted parent `fc80b472...`;
- Sol TODO path;
- implementation candidate SHA/tree;
- record-binding head SHA/tree when available;
- execution manifest path/hash;
- task spec hashes;
- model/optimizer identities;
- seed list;
- CI run/job/artifact IDs;
- per-task/per-seed result references;
- positive/control aggregate verdicts;
- durability/restart status;
- V-trace/critic status;
- exactly-once status;
- generation/account-continuity status;
- firewall status;
- strongest evidence **claimed by candidate**, capped at S1 target;
- unresolved items;
- review history array;
- `terminal_state = READY_FOR_SOL_REVIEW`.

DS must **not** create or modify the final authoritative:

`CB16_R11_POST_CC_S1_LEARNABILITY_RECEIPT_V1.json`

unless Sol explicitly delegates that mechanical write after acceptance. Final review verdict/receipt remains reviewer-owned.

Historical receipts are immutable.

---

# 15. Multi-round Sol review loop

## Round R1

Sol reviews the actual PR SHA, especially Sections 12.1–12.3 hotspots, task validity, no leakage, durable-only learning, CI coverage, and candidate-record consistency.

Possible outcomes:

- `CHANGES_REQUIRED`
- `READY_FOR_S1_QUALIFICATION` (Phase-A implementation accepted; formal run authorized)
- after formal run, `PASS` / valid non-PASS verdict at final review

## If `CHANGES_REQUIRED`

DS must on the same branch/PR:

1. fix only the identified implementation/contract issue;
2. add a counterexample/regression test that would fail before the fix;
3. update the candidate record review-history section;
4. rerun affected Shanxi CI at the new exact SHA;
5. update the PR body with new head/tree/run IDs;
6. request the next Sol review round.

Do not change scientific thresholds/tasks/seeds/reward/model/budget to make results greener.

## After `READY_FOR_S1_QUALIFICATION`

DS runs CI-C formal qualification on the exact reviewed runtime SHA. Any subsequent runtime code/test/workflow change invalidates the reviewed runtime identity and requires renewed review before a replacement formal qualification is authoritative.

---

# 16. Stop / verdict rules

DS should continue implementation until one of these is true:

1. implementation candidate is complete and ready for Sol code review;
2. a concrete `CONTRACT_MISMATCH` blocks valid implementation;
3. repository/CI infrastructure causes reproducible `EXECUTION_BLOCKED`;
4. the frozen valid workload exceeds declared hardware envelope → `HARDWARE_LIMIT`.

Missing later S1 code is **not** `EXECUTION_BLOCKED`; implement it.

After formal qualification, preserve one of the allowed final classifications:

- `PASS`
- `SCIENTIFIC_FAIL`
- `CONTRACT_MISMATCH`
- `EXECUTION_BLOCKED`
- `HARDWARE_LIMIT`
- `EVIDENCE_INSUFFICIENT`

A valid scientific FAIL is not to be rescued in place.

S1 completion never authorizes automatic S2, historical-market execution, capacity scaling, economic qualification, transfer claims, or FINAL access.

---

# 17. Definition of done for DS implementation candidate

DS may request `READY_FOR_SOL_CODE_REVIEW` only when all are true:

- exact post-S0-v2 base verified;
- execution manifest exists and validates;
- five positive task generators/oracles implemented;
- mandatory negative controls implemented;
- task-validity/oracle tests are independent of production gate logic;
- complete persistent repeated-learning loop implemented;
- delayed consequence has durable causal credit across the declared restart/downstream consequence path;
- multi-generation durable replay learning implemented;
- exact-once/generation-continuity foundation reused and regression-tested;
- deterministic gate compiler exists and all PASS/FAIL/boundary fixtures pass;
- S1 result/artifact writer exists;
- dedicated Shanxi workflow exists;
- CI-A passes on the candidate SHA;
- CI-B bounded smoke executes the full path on Shanxi Docker;
- FINAL/fresh/historical-market firewall remains closed;
- PR body contains the required evidence and unresolved items;
- no final receipt or merge has been self-issued.

This is the handoff point for Sol R1 code review, not the S1 scientific PASS point.
