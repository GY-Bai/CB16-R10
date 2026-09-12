# CB16 R11 Actor–Critic Code Alignment TODO

**Status:** OPEN / implementation not authorized by this document alone  
**Target branch:** `main`  
**Review baseline:** `875ab16f92c6504a2bdd5fcc12d5ea6172464990`  
**Purpose:** turn the latest autonomous-Trader / continuous-account / Actor–Critic / V-trace design into an explicit file-level implementation plan without rewriting the frozen historical R10/R11 scientific runtime.

---

## 0. Executive verdict

The current repository is **not yet code-aligned** with the new learning design, but the infrastructure is largely reusable.

The main mismatch is scientific, not infrastructural:

```text
current historical lane
Market + Account
    -> deterministic Brain
    -> one nominal action
    -> legacy Supervisor / frozen Physics
    -> H72 continuation with FLAT/0 after the first step
    -> terminal H72 log-equity utility
    -> Teacher-target CE + SmoothL1 Student training

required new lane
Market_t + Account_t
    -> stochastic Actor_t
    -> Permission / execution authority
    -> Physics
    -> Account_t+1
    -> reward_t
    -> Actor_t+1 ...
    -> immutable continuous trajectory
    -> Critic + V-trace replay learning
    -> economic evaluation on a common finite objective horizon
```

The implementation must therefore add a **new versioned scientific lane** rather than silently changing historical files whose existing tests encode legacy semantics.

---

# 1. Non-negotiable migration rules

## 1.1 Historical runtime must remain reproducible

The following files are historical/frozen compatibility surfaces and must **not** be converted in-place into the new Actor–Critic runtime:

- `[KEEP/FROZEN] cb16_local_opt/trace_runtime_r11.py`
- `[KEEP/FROZEN] cb16_local_opt/training_runtime_r11.py`
- `[KEEP/FROZEN] cb16_local_opt/typed_central_brain_r10.py`
- `[KEEP/FROZEN] cb16_local_opt/r102_physics.py`
- `[KEEP/FROZEN] tests/test_r11_trace_runtime.py`

Reason: the existing H72 tests intentionally require the optimized R11 trace runtime to reproduce the legacy H72 branch, snapshot sequence, supervisor decisions, utility, termination and finalize behavior exactly. The new scientific semantics must not invalidate that oracle.

## 1.2 Reuse infrastructure; do not reimplement it unless a concrete blocker appears

The following should remain infrastructure authorities and be reused through adapters:

- `[REUSE] cb16_local_opt/market_runtime_cache_r11.py`
- `[REUSE] cb16_local_opt/sharded_experience_lake.py`
- `[REUSE] cb16_local_opt/checkpoint_store_r11.py`

New RL semantics should live above these files. If a generic capability is missing, add only the smallest backward-compatible extension; do not put Actor–Critic semantics directly into generic storage code.

## 1.3 No fake behavior-policy likelihoods

Historical H72 / Teacher demonstrations that do not contain a valid behavior-policy probability must not be assigned fabricated `log_mu` values and then treated as ordinary V-trace replay.

## 1.4 Failure is experience

Liquidation, account death, very negative return and other failed trajectories must remain first-class training/evaluation observations. They must not disappear through censoring simply because terminal equity is non-positive.

## 1.5 Account continuity is authoritative

The post-step account state is the next decision's account state unless a true terminal/reset contract explicitly says otherwise. Chunk boundaries, learner updates, generation switches and process restarts must not silently reset the account.

## 1.6 Final holdout and fresh-data restrictions remain unchanged

This TODO does not authorize final-holdout access, new-data download, or any weakening of existing fail-closed market-data authority.

---

# 2. Existing files: disposition matrix

| File | Disposition | Required action |
|---|---|---|
| `cb16_local_opt/trace_runtime_r11.py` | KEEP/FROZEN | Preserve legacy H72 behavior. Do not retrofit continuous Actor calls into `_simulate_h72_branch_r11`. |
| `cb16_local_opt/training_runtime_r11.py` | KEEP/FROZEN | Preserve Teacher soft-target CE + risk SmoothL1 historical training. New Actor–Critic learner goes in a new file. |
| `cb16_local_opt/typed_central_brain_r10.py` | KEEP/FROZEN | Preserve deterministic `compose_action()` behavior for historical experiments. Do not change it into stochastic sampling. |
| `cb16_local_opt/r102_physics.py` | KEEP/FROZEN | Preserve frozen R10.2/H72 authority. New target-position semantics must enter through a versioned adapter/new execution contract. |
| `cb16_local_opt/market_runtime_cache_r11.py` | REUSE | Use as the market-array source for the new continuous collector. |
| `cb16_local_opt/sharded_experience_lake.py` | REUSE | Keep generic CAS/immutable-object semantics. Add RL payloads through `trajectory_lake_r0.py`. |
| `cb16_local_opt/checkpoint_store_r11.py` | REUSE | Keep semantic tensor-object CAS. Composite Actor/Critic/optimizer/RNG state goes through a new adapter. |
| `tests/test_r11_trace_runtime.py` | KEEP/FROZEN | Historical regression guard; it must remain green after the new lane is added. |

---

# 3. P0 — semantic and experience-production blockers

## T0 — Create a versioned Actor–Critic scientific contract

**Priority:** P0 / must be first

### Files

- `[NEW] cb16_local_opt/actor_critic_contract_r0.py`
- `[NEW] tests/test_actor_critic_contract_r0.py`

### Implementation TODO

- [ ] Define immutable version identifiers for:
  - observation schema;
  - action schema;
  - permission/execution schema;
  - reward schema;
  - trajectory schema;
  - policy distribution schema;
  - critic/value schema;
  - replay compatibility schema.
- [ ] Define canonical hashing/serialization helpers for science-level identities where the existing generic helpers are insufficient.
- [ ] Encode invariants:
  - nominal action != permission != executed action;
  - requested target risk != confidence;
  - outcome != correct-action label;
  - account state at `t+1` must descend from executed state at `t`;
  - failure trajectories remain admissible experience;
  - no final-holdout access;
  - no fresh-data authority.
- [ ] Provide fail-closed validators used by all new Actor–Critic modules.

### Acceptance tests

- [ ] Same semantic object => same hash independent of dict insertion order.
- [ ] Schema/version mismatch fails closed.
- [ ] Missing required lineage/version fields fails closed.
- [ ] Historical R11 modules import and run without depending on this new contract.

---

## T1 — Introduce a target-position action contract

**Priority:** P0

### Files

- `[NEW] cb16_local_opt/action_contract_r0.py`
- `[NEW] tests/test_action_contract_r0.py`
- `[KEEP/FROZEN] cb16_local_opt/typed_central_brain_r10.py`

### Implementation TODO

- [ ] Define `TargetPositionActionR0` (exact class name may change once frozen) containing at minimum:
  - direction target: `SHORT / FLAT / LONG`;
  - requested target risk/exposure in `[0,1]`;
  - policy/version identity;
  - optional sampling metadata needed to reconstruct behavior likelihood.
- [ ] Make action semantics **target state**, not "one-time entry command".
- [ ] Support the required transitions:
  - FLAT -> LONG;
  - FLAT -> SHORT;
  - LONG -> smaller LONG;
  - LONG -> FLAT;
  - LONG -> SHORT through the authorized execution transition;
  - SHORT -> smaller SHORT;
  - SHORT -> FLAT;
  - SHORT -> LONG through the authorized execution transition.
- [ ] Add canonical serialization and validation.
- [ ] Explicitly separate `requested_target_risk` from realized executable quantity.
- [ ] Do not modify `typed_central_brain_r10.py`; old deterministic action composition remains historical.

### Acceptance tests

- [ ] All legal target-state transitions serialize deterministically.
- [ ] Invalid direction/risk values fail closed.
- [ ] LONG->FLAT and LONG->SHORT are representable before any Physics call.

---

## T2 — Add permission and Physics adaptation for target-state actions

**Priority:** P0

### Files

- `[NEW] cb16_local_opt/actor_critic_supervisor_r0.py`
- `[NEW] cb16_local_opt/actor_critic_physics_adapter_r0.py`
- `[NEW] tests/test_actor_critic_supervisor_r0.py`
- `[NEW] tests/test_actor_critic_physics_adapter_r0.py`
- `[KEEP/FROZEN] cb16_local_opt/r102_physics.py`

### Implementation TODO

- [ ] `actor_critic_supervisor_r0.py` must consume nominal target action + current AccountState and produce a permission result:
  - ACCEPT;
  - CLAMP;
  - REJECT;
  - explicit reason code;
  - permitted target exposure.
- [ ] Permission must remain external authority; the Actor cannot bypass legality, margin or risk limits.
- [ ] `actor_critic_physics_adapter_r0.py` must translate the permitted target into the smallest legal executable delta against the current position.
- [ ] Freeze the mapping from `requested_target_risk in [0,1]` to target exposure/quantity.
- [ ] Include exchange/physics constraints in the executable conversion:
  - available equity/margin;
  - maximum authorized exposure;
  - quantity precision;
  - minimum quantity/notional if applicable;
  - fees/funding through existing Physics authority;
  - liquidation/termination state.
- [ ] Reverse-position behavior must be deterministic and explicitly specified. Do not allow an ambiguous implicit flip.
- [ ] Preserve `r102_physics.py` byte/semantic behavior for historical callers.

### Acceptance tests

- [ ] Known account + same action always produces the same executable quantity.
- [ ] Reduce / close / reverse tests exist for both LONG and SHORT.
- [ ] Clamp/reject reason is recorded and reproducible.
- [ ] Model cannot create quantity beyond permission authority.
- [ ] Legacy `tests/test_r11_trace_runtime.py` remains green.

---

## T3 — Add a stochastic Actor policy contract

**Priority:** P0

### Files

- `[NEW] cb16_local_opt/actor_policy_r0.py`
- `[NEW] cb16_local_opt/actor_critic_brain_r0.py`
- `[NEW] tests/test_actor_policy_r0.py`
- `[NEW] tests/test_actor_critic_brain_r0.py`
- `[KEEP/FROZEN] cb16_local_opt/typed_central_brain_r10.py`

### Implementation TODO

- [ ] Implement a policy distribution with three distinct interfaces:
  - `sample(...)` for training/experience generation;
  - `log_prob(...)` for exact behavior/target likelihood calculation;
  - `deterministic_action(...)` for evaluation/deployment diagnostics.
- [ ] Direction should be a categorical distribution over SHORT/FLAT/LONG.
- [ ] Risk/exposure must use an explicitly bounded distribution compatible with `[0,1]`; the chosen parameterization must permit exact `log_prob` reconstruction.
- [ ] Store all parameters needed to recompute `log_mu` from the frozen behavior policy.
- [ ] Separate sampling RNG state from model weights.
- [ ] `actor_critic_brain_r0.py` may reuse existing sensory representations, but must expose the new Actor interface rather than calling the historical `compose_action()`.
- [ ] Do not reinterpret the historical sigmoid-risk head as automatically equivalent to the new stochastic risk distribution.

### Acceptance tests

- [ ] Fixed model + fixed RNG => identical sampled action sequence.
- [ ] Stored behavior distribution + sampled action => exact reproducible `log_mu`.
- [ ] `deterministic_action()` never consumes RNG.
- [ ] Probability normalization and finite-log-prob tests cover extreme logits/risk parameters.

---

## T4 — Build a true continuous policy rollout runtime

**Priority:** P0 / central code gap

### Files

- `[NEW] cb16_local_opt/continuous_rollout_r0.py`
- `[NEW] tests/test_continuous_rollout_r0.py`
- `[REUSE] cb16_local_opt/market_runtime_cache_r11.py`
- `[KEEP/FROZEN] cb16_local_opt/trace_runtime_r11.py`

### Implementation TODO

- [ ] At **every authorized policy decision clock**:
  1. obtain current market observation;
  2. obtain current authoritative account snapshot;
  3. construct Actor observation;
  4. sample nominal action;
  5. record behavior `log_mu`;
  6. pass through permission;
  7. execute through Physics adapter;
  8. obtain new account snapshot;
  9. compute reward;
  10. append transition;
  11. feed `AccountState_{t+1}` into the next policy decision.
- [ ] No `j > 0 => FLAT/0` shortcut is allowed in this new runtime.
- [ ] Keep one account strictly chronological even when different accounts are parallelized.
- [ ] Define deterministic collector seeding and stable trace IDs.
- [ ] Collector must be restartable from a sealed account/policy/RNG checkpoint.
- [ ] Reuse market cache arrays without mutating them.

### Acceptance tests

- [ ] A multi-step fixture proves the Brain is called at every decision point.
- [ ] Step `t+1` sees the exact post-execution account state from step `t`.
- [ ] Same seed and same starting snapshot reproduce identical nominal actions, permission outcomes, executions and transition hashes.
- [ ] Different accounts may execute concurrently without violating within-account order.
- [ ] Legacy H72 runtime output is unchanged.

---

## T5 — Replace learning reward with arithmetic-equity delta

**Priority:** P0

### Files

- `[NEW] cb16_local_opt/reward_r0.py`
- `[NEW] tests/test_reward_r0.py`

### Implementation TODO

- [ ] Implement the primary transition reward:

```text
r_t = (E_{t+1} - E_t) / E_ref
```

- [ ] Freeze how `E_ref` is chosen for one objective episode/cohort.
- [ ] Ensure fees, funding, realized/unrealized PnL and liquidation effects are reflected through authoritative equity rather than patched into the reward twice.
- [ ] Never drop a transition because `E_{t+1} <= 0`.
- [ ] Make reward finite/fail-closed for corrupt or non-finite accounting data.
- [ ] Keep legacy H72 `log(wt/w0)` utility untouched in historical runtime.

### Acceptance tests

- [ ] For a complete finite objective episode: `sum(r_t) == (E_T - E_0)/E_ref` within the frozen numerical tolerance.
- [ ] Positive, zero, negative and liquidation paths are tested.
- [ ] Transaction cost and funding effects are counted exactly once.

---

## T6 — Separate true terminal from objective horizon and chunk truncation

**Priority:** P0

### Files

- `[NEW] cb16_local_opt/episode_boundary_r0.py`
- `[NEW] tests/test_episode_boundary_r0.py`

### Implementation TODO

- [ ] Define explicit boundary kinds, at minimum:
  - `TRUE_TERMINAL`;
  - `OBJECTIVE_T`;
  - `CHUNK_TRUNCATION`;
  - `DATA_END`;
  - `PAUSE` / checkpoint interruption if needed.
- [ ] Define bootstrap mask/value semantics for each boundary.
- [ ] `OBJECTIVE_T` ends the scoring/learning objective segment but must not automatically assert that the physical account died.
- [ ] `CHUNK_TRUNCATION` must preserve account and bootstrap continuity.
- [ ] True liquidation/account-death semantics must remain explicit.

### Acceptance tests

- [ ] Split one rollout into multiple chunks and show value/return semantics equal an unsplit rollout.
- [ ] No bootstrap across a true terminal.
- [ ] Account snapshot survives non-terminal chunk/objective boundaries according to the frozen contract.

---

## T7 — Define V-trace-ready transition and sequence schemas

**Priority:** P0

### Files

- `[NEW] cb16_local_opt/trajectory_schema_r0.py`
- `[NEW] cb16_local_opt/trajectory_lake_r0.py`
- `[NEW] tests/test_trajectory_schema_r0.py`
- `[NEW] tests/test_trajectory_lake_r0.py`
- `[REUSE] cb16_local_opt/sharded_experience_lake.py`

### Required transition fields

Each primary transition must carry enough information to audit behavior, execution and learning without reconstructing hidden state from mutable runtime context. At minimum:

- `CausalTraceID` / transition ID;
- account ID;
- generation/update ID;
- behavior policy hash/version;
- observation schema/version;
- `x_t` identity or immutable observation payload/reference;
- nominal action;
- behavior `log_mu`;
- permission decision and reason;
- executed action/delta;
- `E_t`;
- `E_{t+1}`;
- `E_ref`;
- reward;
- `x_{t+1}` identity/reference;
- boundary kind;
- bootstrap admissibility;
- market/source lineage;
- account pre/post snapshot hashes;
- RNG provenance sufficient for deterministic replay diagnostics;
- action/execution/reward semantic version IDs.

### Implementation TODO

- [ ] `trajectory_schema_r0.py` owns dataclasses + validation + canonical identity.
- [ ] `trajectory_lake_r0.py` maps transition/sequence objects onto `ShardedExperienceLake` without teaching the generic Lake about RL semantics.
- [ ] Add sequence sealing: ordered transition IDs + policy/version identities + start/end account snapshot hashes.
- [ ] Fail closed on continuity breaks.
- [ ] Keep immutable exactly-once semantics.

### Acceptance tests

- [ ] Death/liquidation transition can be stored and loaded normally.
- [ ] ACCEPT/CLAMP/REJECT cases preserve nominal vs executed action.
- [ ] Sequence whose pre/post account hashes do not chain is rejected.
- [ ] Same logical trajectory serializes to the same identity.

---

# 4. P1 — learner and qualification

## T8 — Add a Critic

**Priority:** P1

### Files

- `[NEW] cb16_local_opt/critic_r0.py`
- `[NEW] tests/test_critic_r0.py`
- `[NEW or COMPOSE] cb16_local_opt/actor_critic_brain_r0.py`

### Implementation TODO

- [ ] Critic must consume sufficient state for account-conditioned value prediction, including:
  - market representation;
  - account state;
  - equity normalized consistently with objective reward;
  - legality/termination information where required;
  - remaining objective horizon `tau = T - t` or an equivalent frozen representation.
- [ ] Keep policy and value heads distinguishable for optimizer/diagnostic ownership.
- [ ] Do not use realized winning action as a supervised value label.

### Acceptance tests

- [ ] Critic learns exact/near-exact values in a deterministic known-answer environment.
- [ ] Value changes correctly when only AccountState or remaining horizon changes.

---

## T9 — Implement pure V-trace math and Actor–Critic training runtime

**Priority:** P1

### Files

- `[NEW] cb16_local_opt/vtrace_r0.py`
- `[NEW] cb16_local_opt/actor_critic_training_runtime_r0.py`
- `[NEW] tests/test_vtrace_r0.py`
- `[NEW] tests/test_actor_critic_training_runtime_r0.py`
- `[KEEP/FROZEN] cb16_local_opt/training_runtime_r11.py`

### Implementation TODO

`vtrace_r0.py`:

- [ ] Implement pure tensor V-trace recurrence separately from optimizer/runtime concerns.
- [ ] Inputs must include target-policy `log_pi`, stored behavior `log_mu`, rewards, discounts/bootstrap masks and values.
- [ ] Freeze `rho_bar`, `c_bar` and policy-gradient clipping semantics in config/authority.
- [ ] Numerically stable importance ratios from log-prob differences.

`actor_critic_training_runtime_r0.py`:

- [ ] Batch **ordered sequences**, not independent shuffled rows.
- [ ] Compute current `log_pi(a_t|x_t)` and compare to immutable behavior `log_mu`.
- [ ] Separate Actor and Critic losses.
- [ ] Use separate optimizer state or explicitly frozen parameter ownership.
- [ ] Keep training FP32 unless a later qualified change explicitly authorizes another precision.
- [ ] Gradient clipping, entropy term and optimizer hyperparameters must be frozen in the training recipe.
- [ ] Record update identity, input sequence identities and output checkpoint identity.

### Acceptance tests

- [ ] Hand-computed short sequence matches V-trace outputs element-by-element.
- [ ] When `pi == mu`, importance ratios are 1 and the result reduces to the expected on-policy recurrence.
- [ ] Extreme probability ratios remain finite after clipping.
- [ ] Learner never shuffles transition order inside a sequence.
- [ ] Historical `training_runtime_r11.py` remains untouched and independently runnable.

---

## T10 — Add fail-closed replay compatibility checks

**Priority:** P1

### Files

- `[NEW] cb16_local_opt/replay_compatibility_r0.py`
- `[NEW] tests/test_replay_compatibility_r0.py`

### Implementation TODO

- [ ] Before a sequence enters Actor–Critic replay, verify compatibility for:
  - action schema/version;
  - observation schema/normalizer identity;
  - permission/execution semantics;
  - reward semantics;
  - valid behavior `log_mu`;
  - policy-distribution version;
  - account-state schema;
  - boundary/bootstrap semantics.
- [ ] Reject incompatible historical demonstrations from primary V-trace replay.
- [ ] Permit explicit diagnostic/demo routes without silently upgrading them to RL trajectories.

### Acceptance tests

- [ ] Missing `log_mu` => fail closed for V-trace replay.
- [ ] Changed action semantics => fail closed.
- [ ] Same compatible schema across older policy generations => accepted if behavior likelihood is valid.

---

## T11 — Extend checkpointing through a composite Actor–Critic adapter

**Priority:** P1

### Files

- `[NEW] cb16_local_opt/actor_critic_checkpoint_r0.py`
- `[NEW] tests/test_actor_critic_checkpoint_r0.py`
- `[REUSE] cb16_local_opt/checkpoint_store_r11.py`

### Composite checkpoint must bind

- Actor tensor-object hash;
- Critic tensor-object hash;
- Actor optimizer state identity;
- Critic optimizer state identity;
- collector RNG state;
- learner RNG state;
- generation number;
- learner update counter;
- training recipe hash;
- replay/trajectory snapshot identity;
- authoritative account snapshot hash;
- market/data authority identity;
- schema/semantic versions.

### Implementation TODO

- [ ] Reuse `CheckpointStoreR11` tensor CAS for model tensor objects where possible.
- [ ] Do not break the existing generation checkpoint schema used by historical runtime.
- [ ] Add a new composite manifest/schema in the adapter.
- [ ] Save/restore RNG states explicitly.
- [ ] Crash recovery must restore the exact next collection/update state, not merely weights.

### Acceptance tests

- [ ] Save -> restore -> next sampled action is identical.
- [ ] Save -> restore -> next learner update produces identical tensor hashes under deterministic test conditions.
- [ ] Missing optimizer/RNG/account component fails closed.

---

## T12 — Build a training qualification gate before market-scale training

**Priority:** P1

### Files

- `[NEW] cb16_local_opt/training_qualification_r0.py`
- `[NEW] tests/test_training_qualification_r0.py`

### Qualification gate must aggregate

- [ ] action-contract tests;
- [ ] permission/Physics adapter tests;
- [ ] reward conservation tests;
- [ ] terminal/truncation tests;
- [ ] behavior log-prob tests;
- [ ] trajectory continuity tests;
- [ ] V-trace hand-calculation tests;
- [ ] replay compatibility tests;
- [ ] checkpoint/recovery tests;
- [ ] historical regression tests.

### Gate rule

No large historical Actor–Critic market training is authorized until this qualification suite passes as one receipt-producing gate.

---

## T13 — Add known-answer learning environments

**Priority:** P1

### Files

- `[NEW] cb16_local_opt/known_answer_envs_r0.py`
- `[NEW] tests/test_actor_critic_known_answer_r0.py`

### Required toy environments

- [ ] **Account-conditioned action:** same market observation, different account state => different optimal action.
- [ ] **Delayed loss:** locally attractive action creates a later penalty; validates temporal credit.
- [ ] **Risky higher arithmetic mean:** low-probability large gain / frequent loss case that distinguishes arithmetic objective behavior from survival-only preference.
- [ ] **Horizon reversal:** optimal action changes as `tau=T-t` changes.
- [ ] **Off-policy replay:** behavior and target policies deliberately differ, with analytically checkable importance correction.

### Acceptance criteria

- [ ] Pre-register multi-seed success thresholds.
- [ ] Actor must learn the known optimum above the frozen threshold.
- [ ] Critic error must meet the frozen threshold.
- [ ] V-trace replay must outperform/bound the deliberately wrong uncorrected replay baseline in the designated test.
- [ ] Failure of these gates stops expansion to expensive market training.

---

# 5. P2 — economic evaluation, generation lineage and Stage-4 integration

## T14 — Add economic evaluation based on common finite horizons

**Priority:** P2

### Files

- `[NEW] cb16_local_opt/economic_evaluator_r0.py`
- `[NEW] tests/test_economic_evaluator_r0.py`

### Implementation TODO

- [ ] Evaluate every preregistered account/cohort on a common objective horizon `T`.
- [ ] Primary return basis is arithmetic account return/equity, consistent with the new reward objective.
- [ ] Include at least:
  - Trader return distribution;
  - Buy-and-Hold baseline under the same market interval/account assumptions;
  - always-FLAT/no-position baseline;
  - mean/median;
  - lower-tail statistics;
  - account-death/liquidation probability;
  - drawdown/terminal-equity distribution;
  - permission rejection/clamp diagnostics.
- [ ] Do not remove blown-up accounts before computing means.
- [ ] Keep statistical/economic evaluation separate from the learner loss.

### Acceptance tests

- [ ] A deliberately high-variance strategy with rare large gains retains failed accounts in the arithmetic mean.
- [ ] B&H and FLAT use exactly the same preregistered cohort/window.
- [ ] Changing evaluation membership after seeing outcomes fails closed where the frozen protocol requires preregistration.

---

## T15 — Explicitly separate legacy demonstration data from primary replay

**Priority:** P2

### Files

- `[NEW] cb16_local_opt/legacy_experience_adapter_r0.py`
- `[NEW] tests/test_legacy_experience_adapter_r0.py`
- `[KEEP/FROZEN] cb16_local_opt/trace_runtime_r11.py`

### Implementation TODO

- [ ] Classify old H72 / counterfactual / Teacher evidence as `DEMONSTRATION`, `DIAGNOSTIC`, or another explicit non-primary-replay source type.
- [ ] Do not synthesize behavior probabilities for deterministic/legacy traces that never recorded a compatible stochastic policy.
- [ ] Allow historical experience to be used only by an explicitly authorized auxiliary objective/diagnostic route.
- [ ] If old market contexts are to become primary V-trace experience, recollect them under the new continuous Actor policy and new trajectory schema.

---

## T16 — Add Actor–Critic generation handoff and account continuity

**Priority:** P2

### Files

- `[NEW] cb16_local_opt/actor_critic_generation_r0.py`
- `[NEW] tests/test_actor_critic_generation_r0.py`
- `[REFERENCE ONLY until separately integrated] cb16_local_opt/continuous_generation_binding_r0.py` from branch `ai/r11-continuous-generation-qualification-r0`

### Implementation TODO

- [ ] Bind parent Actor/Critic checkpoint -> child Actor/Critic checkpoint.
- [ ] Bind optimizer and RNG state.
- [ ] Bind exact training recipe and experience/replay snapshot.
- [ ] Bind account snapshot at generation transition.
- [ ] Enforce monotonic generation/update lineage.
- [ ] Switching policy generation must not reset or rewrite the live account unless a true reset contract is explicitly invoked.
- [ ] Port useful concepts from the unmerged continuous-generation qualification work only after reconciling it with current `main`:
  - account-continuity assertion;
  - future-invariance assertion;
  - optimizer snapshot identity;
  - training/evaluation evidence quarantine;
  - generation lineage.
- [ ] Do not treat that branch as already implementing continuous RL; its binding module deliberately does not change Physics, Teacher, Student loss or sensory semantics.

### Acceptance tests

- [ ] `G(n+1).parent == G(n).child` for all bound policy/value components.
- [ ] Account post-state before generation switch equals account pre-state after switch.
- [ ] Stale/rejected parent cannot silently become the next lineage parent.

---

## T17 — Integrate the new scientific lane into Stage-4 through one adapter

**Priority:** P2

### Files

- `[NEW] cb16_local_opt/stage4_actor_critic_adapter_r0.py`
- `[NEW] tests/test_stage4_actor_critic_adapter_r0.py`
- `[REUSE] cb16_local_opt/market_runtime_cache_r11.py`
- `[REUSE] cb16_local_opt/sharded_experience_lake.py`
- `[REUSE] cb16_local_opt/checkpoint_store_r11.py`

### Implementation TODO

- [ ] Make `stage4_actor_critic_adapter_r0.py` the narrow integration boundary between the new science lane and existing lifecycle/orchestration infrastructure.
- [ ] Reuse existing singleton/fencing/state-root/recovery/orchestration mechanisms rather than recreating them inside the learner.
- [ ] Keep the legacy H72 and Actor–Critic runtime independently selectable and independently receipted.
- [ ] Do not let new Actor–Critic receipts overwrite or reinterpret historical R11 receipts.
- [ ] Before modifying any existing Stage-4 dispatcher/entrypoint, identify the active `main` call site and make the smallest registration/import change; record that exact existing filename in the implementation PR/receipt.
- [ ] New adapter must fail closed if Stage-4 authority selects an incompatible legacy/new runtime combination.

### Acceptance tests

- [ ] Historical runtime boots and runs exactly as before.
- [ ] Actor–Critic runtime boots through the same lifecycle authority but uses its own semantic/version receipt.
- [ ] Crash/restart restores correct account/model/RNG lineage.

---

# 6. New Python file map

The target file layout after implementation should be approximately:

```text
cb16_local_opt/
    # historical / reusable
    typed_central_brain_r10.py                 # KEEP/FROZEN
    r102_physics.py                            # KEEP/FROZEN
    trace_runtime_r11.py                       # KEEP/FROZEN
    training_runtime_r11.py                    # KEEP/FROZEN
    market_runtime_cache_r11.py                # REUSE
    sharded_experience_lake.py                 # REUSE
    checkpoint_store_r11.py                    # REUSE

    # new Actor-Critic science lane
    actor_critic_contract_r0.py
    action_contract_r0.py
    actor_critic_supervisor_r0.py
    actor_critic_physics_adapter_r0.py
    actor_policy_r0.py
    actor_critic_brain_r0.py
    reward_r0.py
    episode_boundary_r0.py
    trajectory_schema_r0.py
    trajectory_lake_r0.py
    continuous_rollout_r0.py
    critic_r0.py
    vtrace_r0.py
    replay_compatibility_r0.py
    actor_critic_training_runtime_r0.py
    actor_critic_checkpoint_r0.py
    training_qualification_r0.py
    known_answer_envs_r0.py
    economic_evaluator_r0.py
    legacy_experience_adapter_r0.py
    actor_critic_generation_r0.py
    stage4_actor_critic_adapter_r0.py
```

Corresponding tests:

```text
tests/
    test_r11_trace_runtime.py                   # KEEP/FROZEN regression guard

    test_actor_critic_contract_r0.py
    test_action_contract_r0.py
    test_actor_critic_supervisor_r0.py
    test_actor_critic_physics_adapter_r0.py
    test_actor_policy_r0.py
    test_actor_critic_brain_r0.py
    test_reward_r0.py
    test_episode_boundary_r0.py
    test_trajectory_schema_r0.py
    test_trajectory_lake_r0.py
    test_continuous_rollout_r0.py
    test_critic_r0.py
    test_vtrace_r0.py
    test_replay_compatibility_r0.py
    test_actor_critic_training_runtime_r0.py
    test_actor_critic_checkpoint_r0.py
    test_training_qualification_r0.py
    test_actor_critic_known_answer_r0.py
    test_economic_evaluator_r0.py
    test_legacy_experience_adapter_r0.py
    test_actor_critic_generation_r0.py
    test_stage4_actor_critic_adapter_r0.py
```

Names ending in `_r0.py` intentionally identify this as a new, independently qualifiable scientific protocol rather than a silent mutation of R10/R11 historical authority. Exact revision number may be changed by a later authority freeze, but the version separation itself is mandatory.

---

# 7. Recommended implementation dependency order

## Phase A — freeze semantics before generating new experience

```text
T0 actor_critic_contract_r0.py
 -> T1 action_contract_r0.py
 -> T2 actor_critic_supervisor_r0.py
      actor_critic_physics_adapter_r0.py
```

**Gate A:** active position can be reduced, closed and reversed through deterministic permission/execution semantics without changing legacy R10.2/H72 behavior.

## Phase B — produce scientifically valid continuous experience

```text
T3 actor_policy_r0.py + actor_critic_brain_r0.py
 -> T5 reward_r0.py
 -> T6 episode_boundary_r0.py
 -> T7 trajectory_schema_r0.py + trajectory_lake_r0.py
 -> T4 continuous_rollout_r0.py
```

**Gate B:** one account can execute a multi-decision continuous trajectory with valid `log_mu`, exact account-state chaining, arithmetic reward and immutable transition identity.

## Phase C — learner

```text
T8 critic_r0.py
 -> T9 vtrace_r0.py
 -> T10 replay_compatibility_r0.py
 -> T9 actor_critic_training_runtime_r0.py
 -> T11 actor_critic_checkpoint_r0.py
```

**Gate C:** hand-calculated V-trace, deterministic recovery and compatible replay all pass.

## Phase D — qualification before market-scale learning

```text
T12 training_qualification_r0.py
 -> T13 known_answer_envs_r0.py
```

**Gate D:** all preregistered toy-learning gates pass across the frozen seed set. Failure here blocks expensive historical market training.

## Phase E — economic and lifecycle closure

```text
T14 economic_evaluator_r0.py
 -> T15 legacy_experience_adapter_r0.py
 -> T16 actor_critic_generation_r0.py
 -> T17 stage4_actor_critic_adapter_r0.py
```

**Gate E:** continuous generation, crash recovery, account continuity and economic comparison can run without altering historical receipts or final-holdout authority.

---

# 8. Suggested PR / task decomposition

To keep scientific authority reviewable, do not land the entire migration as one large patch.

### PR A — Action & execution semantics

Files:

- `actor_critic_contract_r0.py`
- `action_contract_r0.py`
- `actor_critic_supervisor_r0.py`
- `actor_critic_physics_adapter_r0.py`
- corresponding tests

No learner yet.

### PR B — Continuous experience

Files:

- `actor_policy_r0.py`
- `actor_critic_brain_r0.py`
- `reward_r0.py`
- `episode_boundary_r0.py`
- `trajectory_schema_r0.py`
- `trajectory_lake_r0.py`
- `continuous_rollout_r0.py`
- corresponding tests

No market-scale training yet.

### PR C — Actor–Critic / V-trace learner

Files:

- `critic_r0.py`
- `vtrace_r0.py`
- `replay_compatibility_r0.py`
- `actor_critic_training_runtime_r0.py`
- `actor_critic_checkpoint_r0.py`
- corresponding tests

### PR D — Scientific qualification

Files:

- `training_qualification_r0.py`
- `known_answer_envs_r0.py`
- corresponding tests

Only after PR D passes may the project consider bounded real-market training.

### PR E — Economic / generation / Stage-4 integration

Files:

- `economic_evaluator_r0.py`
- `legacy_experience_adapter_r0.py`
- `actor_critic_generation_r0.py`
- `stage4_actor_critic_adapter_r0.py`
- corresponding tests

---

# 9. Explicit anti-TODOs

The following shortcuts are prohibited because they would create apparent code alignment without scientific alignment:

- [ ] **Do not** rewrite `trace_runtime_r11.py` so its historical H72 test now means something different.
- [ ] **Do not** call the existing first-action-plus-72h continuation a continuous on-policy rollout.
- [ ] **Do not** convert `training_runtime_r11.py` in place from Teacher/Student loss to V-trace.
- [ ] **Do not** overwrite `typed_central_brain_r10.py` deterministic action semantics.
- [ ] **Do not** fabricate `log_mu` for old demonstrations.
- [ ] **Do not** delete liquidation/failed trajectories to make the dataset numerically convenient.
- [ ] **Do not** treat `OBJECTIVE_T` or a batching chunk as automatic account death.
- [ ] **Do not** let the Actor directly write executable quantity around the permission/Physics authority.
- [ ] **Do not** introduce a hand-engineered regime/cycle/resonance rule subsystem as part of this migration.
- [ ] **Do not** open final holdout or download fresh market data for qualification.
- [ ] **Do not** begin large historical training merely because the new Python modules import successfully; the known-answer qualification gate is mandatory first.

---

# 10. Definition of done

This code-alignment TODO is complete only when all of the following are true:

1. Historical R10/R11 regression tests still pass unchanged.
2. Active-position reduce/close/reverse actions are explicitly representable and permission-controlled.
3. The Actor is a real stochastic policy with reconstructible behavior `log_mu`.
4. One continuous account trajectory calls the policy repeatedly and preserves exact AccountState chronology.
5. Reward telescopes to the intended finite-horizon arithmetic account return.
6. Failure/liquidation trajectories remain present in experience and evaluation.
7. Transition/sequence schemas are immutable, hashable and V-trace-ready.
8. Critic and V-trace math pass known-answer tests.
9. Replay rejects semantically incompatible experience rather than guessing compatibility.
10. Checkpoint/recovery binds Actor, Critic, optimizers, RNG, account and replay identity.
11. All toy learning gates pass across preregistered seeds.
12. Economic evaluation includes every preregistered account and compares Trader vs B&H vs FLAT on the same horizon.
13. Generation switches preserve account continuity.
14. Stage-4 lifecycle can run the new lane without changing old receipts/authority.
15. Final holdout remains unopened and no fresh market data is introduced.

Until these conditions are satisfied, the repository should be described as:

> **Actor–Critic design documented; infrastructure reusable; autonomous continuous learning runtime not yet fully code-aligned.**
