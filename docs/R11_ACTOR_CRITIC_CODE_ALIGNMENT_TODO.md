# CB16 R11 Actor–Critic Code Alignment TODO

**Status:** OPEN / planning authority only; this document does not itself authorize scientific implementation or final-holdout access  
**Target branch:** `main`  
**Review baseline:** `4d47062585578e66b7fc47eb244d7aa0c0ccb7cf`  
**Expanded decomposition:** `AC-001` through `AC-058`  
**Purpose:** convert the autonomous-Trader / continuous-account / stochastic Actor–Critic / V-trace design into independently implementable, independently testable code tasks while preserving the historical R10/R11 scientific lane.

---

# 0. Executive verdict

The current repository is **not yet scientifically aligned** with the new learning design, but the existing infrastructure remains largely reusable.

The current historical lane is approximately:

```text
Market + Account
    -> deterministic Brain
    -> one nominal action
    -> legacy Supervisor / frozen Physics
    -> H72 continuation with FLAT/0 after the first step
    -> terminal H72 log-equity utility
    -> Teacher-target CE + SmoothL1 Student training
```

The required new lane is:

```text
Market_t + Account_t
    -> stochastic Actor_t
    -> nominal target-position action + log_mu
    -> Permission / execution authority
    -> Physics
    -> Account_t+1
    -> arithmetic-equity reward_t
    -> Actor_t+1 ...
    -> immutable continuous trajectory
    -> Critic + V-trace replay learning
    -> common-horizon economic evaluation
```

This is therefore a **new versioned scientific lane**, not an in-place rewrite of the legacy H72 / Teacher–Student runtime.

The key ordering rule is:

> **Do not implement V-trace first.** Action semantics, execution semantics, continuous rollout semantics, reward semantics, and trajectory semantics must be frozen and qualified before off-policy learning is allowed to consume the generated experience.

---

# 1. Non-negotiable migration rules

## 1.1 Preserve the historical lane

The following surfaces remain historical compatibility authorities and must not be converted in place:

- `[KEEP/FROZEN] cb16_local_opt/trace_runtime_r11.py`
- `[KEEP/FROZEN] cb16_local_opt/training_runtime_r11.py`
- `[KEEP/FROZEN] cb16_local_opt/typed_central_brain_r10.py`
- `[KEEP/FROZEN] cb16_local_opt/r102_physics.py`
- `[KEEP/FROZEN] tests/test_r11_trace_runtime.py`
- `[KEEP/FROZEN] authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0/**`
- `[KEEP/FROZEN] authority/control_plane_r1/risk_supervisor_r1.py`

Existing regression/oracle behavior must remain reproducible.

## 1.2 Reuse infrastructure instead of rebuilding it

Prefer adapters over replacement for:

- `[REUSE] cb16_local_opt/market_runtime_cache_r11.py`
- `[REUSE] cb16_local_opt/sharded_experience_lake.py`
- `[REUSE] cb16_local_opt/checkpoint_store_r11.py`
- `[REUSE] cb16_local_opt/minute_physics_binding_r2.py` where its generic snapshot/transition utilities remain valid
- `[REUSE] existing Stage-4 state roots, fencing, recovery, async orchestration and authority-adoption infrastructure where scientifically compatible`

Generic infrastructure must not silently acquire Actor–Critic-specific meaning.

## 1.3 No fabricated behavior likelihood

Historical demonstrations without valid behavior-policy probability information must not be assigned fake `log_mu` and inserted into ordinary V-trace replay.

## 1.4 Failure remains first-class experience

Liquidation, account death, large negative return and other failed trajectories remain valid experience. They must not disappear through positive-equity censoring or survivorship filtering.

## 1.5 Account continuity is authoritative

`AccountState_{t+1}` is the next decision's account state unless a true terminal/reset contract explicitly says otherwise. Chunk boundaries, learner updates, generation switches, checkpoint/restart and process scheduling must not silently reset the logical account.

## 1.6 Objective remains expected arithmetic return

The primary objective must not silently become log-growth, Sharpe, drawdown minimization, sign reward, clipped reward, bankruptcy avoidance, pessimistic value, or another risk-adjusted surrogate.

Risk/tail statistics are diagnostics unless separately authorized as decision criteria.

## 1.7 Final holdout and fresh-data restrictions remain unchanged

This TODO authorizes neither final-holdout access nor fresh-data download. Existing fail-closed data authority remains in force.

---

# 2. Hard gates

The new lane has four explicit blocking gates.

## Gate A — Semantic execution gate

Requires `AC-001` through `AC-014` PASS.

No trajectory generated before this gate may be presumed replay-compatible with the final execution semantics.

## Gate B — Experience-production gate

Requires `AC-015` through `AC-035` PASS.

Until then, no collected trajectory is admitted as canonical Actor–Critic training evidence.

## Gate C — Learner qualification gate

Requires `AC-036` through `AC-050` PASS.

> **AC-050 must PASS before any real historical market scale-up.**

Failure of a known-answer toy is a scientific blocker, not a throughput problem.

## Gate D — Economic evaluation gate

Requires `AC-051` through `AC-054` PASS before Champion/Challenger promotion can rely on the new economic evaluator.

`AC-055` through `AC-058` then complete migration/infrastructure/CI/documentation closure.

---

# 3. T0–T17 to AC task mapping

The earlier T0–T17 workstreams remain the high-level authority. The AC tasks below are their implementation decomposition.

| Existing workstream | Meaning | Decomposed tasks |
|---|---|---|
| T0 | versioned scientific protocol/runtime lane | AC-001–AC-004 |
| T1 | target-position Action semantics | AC-005–AC-009 |
| T2 | requested risk -> permission -> executable exposure | AC-010–AC-014 |
| T3 | stochastic Actor | AC-015–AC-022 |
| T4 | true continuous policy collector | AC-023–AC-027 |
| T5 | arithmetic-equity reward | AC-028–AC-030 |
| T6 | boundary/terminal/bootstrap semantics | AC-031–AC-033 |
| T7 | Transition/Sequence experience schema | AC-034–AC-035 |
| T8 | separate Critic | AC-036–AC-039 |
| T9 | V-trace Actor–Critic | AC-040–AC-043 |
| T10 | replay compatibility | AC-044–AC-045 |
| T11 | full generation checkpoint/recovery | AC-046 |
| T12 | training qualification math/contracts | AC-047 |
| T13 | known-answer toy environments | AC-048–AC-050 |
| T14 | economic evaluator | AC-051–AC-054 |
| T15 | old demo/new replay boundary | AC-055 |
| T16 | generation switch/account continuity | AC-056 |
| T17 | infrastructure adoption/CI/docs closure | AC-057–AC-058 |

---

# 4. Branch and receipt discipline

Each AC task should be independently reviewable.

Recommended branch form:

```text
ai/r11-ac-<NNN>-<short-name>-r0
```

Rules:

1. One task owns only its declared files unless an authority blocker requires a documented expansion.
2. No undeclared dependency on an unmerged sibling branch.
3. A task may depend only on earlier AC tasks explicitly listed below.
4. Every task must leave a machine-readable or test-visible PASS/FAIL condition.
5. `EXECUTION_BLOCKED` / `HARDWARE_LIMIT` must not be reported as `SCIENTIFIC_FAIL`.
6. A scientific failure must be preserved rather than rescued by changing objective, seed, horizon or semantics inside the same task.
7. Historical regression tests remain mandatory throughout the migration.

---

# 5. Detailed implementation tasks

## Phase A — Authority, Action and execution semantics

### AC-001 — Actor–Critic science version registry

**Maps to:** T0  
**Priority:** P0  
**Branch:** `ai/r11-ac-001-science-contract-r0`

**Files**

- `[NEW] cb16_local_opt/actor_critic_contract_r0.py`
- `[NEW] tests/test_actor_critic_contract_r0.py`

**Implement**

- Freeze explicit version IDs for observation, action, permission/execution, reward, trajectory, Actor distribution, Critic/value, replay compatibility and checkpoint bundle.
- Provide fail-closed validation helpers.
- Distinguish scientific semantic version from code/git revision.

**Dependencies:** none.  
**Gate:** unknown/missing/mixed versions fail closed; same declared contract validates deterministically.

---

### AC-002 — Canonical science serialization and semantic hashing

**Maps to:** T0  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/actor_critic_contract_r0.py`
- `[MODIFY] tests/test_actor_critic_contract_r0.py`

**Implement**

- Canonical serialization for science identities.
- Stable hashes independent of dict insertion order or incidental process state.
- Required lineage fields for account, policy, execution and data source.

**Dependencies:** AC-001.  
**Gate:** semantically identical objects hash identically; changed semantic field changes hash; absent lineage fails closed.

---

### AC-003 — Legacy freeze regression sentinel

**Maps to:** T0  
**Priority:** P0

**Files**

- `[NEW] tests/test_actor_critic_legacy_freeze_r0.py`
- `[KEEP/FROZEN] tests/test_r11_trace_runtime.py`

**Implement**

- Explicit regression test that imports/runs legacy deterministic Brain, H72 path and historical training surfaces unchanged.
- Record expected separation between historical and Actor–Critic lanes.

**Dependencies:** AC-001.  
**Gate:** any accidental semantic mutation of frozen legacy behavior blocks the new lane.

---

### AC-004 — Runtime lane router and fail-closed protocol selection

**Maps to:** T0  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/actor_critic_runtime_router_r0.py`
- `[NEW] tests/test_actor_critic_runtime_router_r0.py`

**Implement**

- Explicitly select `LEGACY_R11` versus `ACTOR_CRITIC_R0`.
- Forbid implicit fallback from the new lane to legacy semantics or vice versa.
- Emit selected science-contract hash into receipts.

**Dependencies:** AC-001–AC-003.  
**Gate:** unknown lane or contract mismatch fails closed.

---

### AC-005 — `TargetPositionActionR0` datatype

**Maps to:** T1  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/action_contract_r0.py`
- `[NEW] tests/test_action_contract_r0.py`

**Implement**

- Direction target: `SHORT / FLAT / LONG`.
- `requested_target_risk` in `[0,1]`.
- Action/schema/policy identity fields.
- Target-state semantics, not one-time-entry semantics.

**Dependencies:** AC-001–AC-002.  
**Gate:** all valid actions round-trip canonically; invalid direction/risk fails closed.

---

### AC-006 — Action canonicalization and invariant validation

**Maps to:** T1  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/action_contract_r0.py`
- `[MODIFY] tests/test_action_contract_r0.py`

**Implement**

- Canonical encoding of direction/risk.
- Freeze `FLAT` target-risk semantics.
- Forbid `requested_target_risk` from being interpreted as confidence.

**Dependencies:** AC-005.  
**Gate:** equivalent action payloads serialize identically; contradictory fields fail closed.

---

### AC-007 — Full target-state transition matrix

**Maps to:** T1  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/action_contract_r0.py`
- `[MODIFY] tests/test_action_contract_r0.py`

**Implement/test**

- FLAT -> LONG / SHORT.
- LONG -> larger/smaller LONG / FLAT / SHORT.
- SHORT -> larger/smaller SHORT / FLAT / LONG.
- Same-target no-op representation.

**Dependencies:** AC-005–AC-006.  
**Gate:** every intended transition is representable before Physics; no ambiguous reversal encoding.

---

### AC-008 — Nominal / permitted / executed action record separation

**Maps to:** T1  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/execution_record_r0.py`
- `[NEW] tests/test_execution_record_r0.py`

**Implement**

- Immutable records for nominal Actor output, Supervisor permission result and actual executed delta.
- Prevent downstream learner/evaluator from conflating them.

**Dependencies:** AC-005–AC-007.  
**Gate:** one record can show Actor request != permission != execution without losing any field.

---

### AC-009 — Action behavior metadata contract

**Maps to:** T1  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/action_contract_r0.py`
- `[MODIFY] cb16_local_opt/execution_record_r0.py`
- `[MODIFY] tests/test_action_contract_r0.py`

**Implement**

- Fields required to link sampled action to behavior-policy version/hash and later `log_mu` verification.
- No probability fabrication at this layer.

**Dependencies:** AC-005–AC-008.  
**Gate:** sampled-action provenance can be bound to one immutable behavior policy identity.

---

### AC-010 — Supervisor permission result contract

**Maps to:** T2  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/actor_critic_supervisor_r0.py`
- `[NEW] tests/test_actor_critic_supervisor_r0.py`

**Implement**

- Permission outcomes: `ACCEPT / CLAMP / REJECT` plus reason code and permitted target exposure.
- Preserve external legality/risk authority.
- Actor cannot bypass margin/legality/termination constraints.

**Dependencies:** AC-005–AC-009.  
**Gate:** permission output is deterministic for identical state/action/authority.

---

### AC-011 — Held-position reduce/close permission

**Maps to:** T2  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/actor_critic_supervisor_r0.py`
- `[MODIFY] tests/test_actor_critic_supervisor_r0.py`

**Implement**

- Remove the new lane's dependence on legacy `POSITION_ALREADY_OPEN -> FORCED_NOOP` behavior.
- Authorize mechanically legal resize and close operations.

**Dependencies:** AC-010.  
**Gate:** LONG/SHORT held accounts can reduce and close when legal; permission reason is auditable.

---

### AC-012 — Explicit reversal semantics

**Maps to:** T2  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/actor_critic_supervisor_r0.py`
- `[MODIFY] tests/test_actor_critic_supervisor_r0.py`

**Implement**

- Freeze how LONG->SHORT and SHORT->LONG are represented and authorized.
- No ambiguous implicit flip.
- Record any required close-then-open sequence as an explicit execution contract.

**Dependencies:** AC-011.  
**Gate:** reversal has one deterministic legal interpretation.

---

### AC-013 — Requested risk -> target exposure/quantity mapping

**Maps to:** T2  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/target_exposure_r0.py`
- `[NEW] tests/test_target_exposure_r0.py`

**Implement**

- Freeze mathematical mapping from `[0,1]` requested target risk to target exposure/quantity authority.
- Account for equity/margin and declared maximum legal exposure without converting this into a trading heuristic.
- Precision/minimum-size handling must remain mechanical.

**Dependencies:** AC-010–AC-012.  
**Gate:** same state/action/contract -> identical target exposure; no hidden strategy rule inside sizing.

---

### AC-014 — Target-position Physics adapter

**Maps to:** T2  
**Priority:** P0 / **Gate A completion**

**Files**

- `[NEW] cb16_local_opt/actor_critic_physics_adapter_r0.py`
- `[NEW] tests/test_actor_critic_physics_adapter_r0.py`
- `[KEEP/FROZEN] cb16_local_opt/r102_physics.py`
- `[KEEP/FROZEN] authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0/**`

**Implement**

- Convert permitted target exposure to the smallest legal executable delta against current position.
- Preserve authoritative fees/funding/margin/liquidation mechanics.
- Cover open, resize, close, reverse and no-op.

**Dependencies:** AC-005–AC-013.  
**Gate:** deterministic known-state execution tests PASS; legacy H72 regression remains green. **Gate A closes only here.**

---

## Phase B — Stochastic Actor and continuous experience production

### AC-015 — Actor policy interface

**Maps to:** T3  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/actor_policy_r0.py`
- `[NEW] tests/test_actor_policy_r0.py`

**Implement**

- `sample(...)`.
- `log_prob(...)`.
- `deterministic_action(...)`.
- Explicit policy distribution/version object.

**Dependencies:** Gate A.  
**Gate:** interfaces are distinct; evaluation call consumes no RNG.

---

### AC-016 — Categorical direction distribution

**Maps to:** T3  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/actor_policy_r0.py`
- `[MODIFY] tests/test_actor_policy_r0.py`

**Implement**

- Stable categorical distribution over SHORT/FLAT/LONG.
- Numerically stable probabilities/log-probabilities.

**Dependencies:** AC-015.  
**Gate:** probabilities normalize and remain finite under extreme logits.

---

### AC-017 — Bounded conditional risk distribution

**Maps to:** T3  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/actor_policy_r0.py`
- `[MODIFY] tests/test_actor_policy_r0.py`

**Implement**

- Explicit bounded distribution for LONG/SHORT risk in `[0,1]`.
- Parameterization must support exact sample likelihood reconstruction.
- FLAT risk remains contract-defined rather than sampled from a meaningless distribution.

**Dependencies:** AC-015–AC-016.  
**Gate:** samples are bounded; `log_prob` finite for legal samples.

---

### AC-018 — Endpoint-mass semantics

**Maps to:** T3  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/actor_policy_r0.py`
- `[MODIFY] tests/test_actor_policy_r0.py`

**Implement**

- If endpoint masses at risk 0/1 are retained, encode them as an explicit mixed distribution rather than pretending a continuous density supplies point probability.
- If endpoints are disallowed, freeze that instead and validate strictly.

**Dependencies:** AC-017.  
**Gate:** exact and auditable endpoint likelihood behavior.

---

### AC-019 — Joint action `log_prob`

**Maps to:** T3  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/actor_policy_r0.py`
- `[MODIFY] tests/test_actor_policy_r0.py`

**Implement**

- Freeze joint likelihood for direction + conditional risk.
- `log_mu` saved at collection must equal later recomputation from the frozen behavior policy.

**Dependencies:** AC-016–AC-018.  
**Gate:** sampled action -> stored `log_mu` -> recomputed `log_mu` matches within frozen tolerance.

---

### AC-020 — Actor RNG and sampling provenance

**Maps to:** T3  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/policy_rng_r0.py`
- `[NEW] tests/test_policy_rng_r0.py`

**Implement**

- Separate Actor sampling RNG from model weights and unrelated process RNG.
- Serializable/restorable RNG state.
- Stable provenance fields for trajectory receipts.

**Dependencies:** AC-015–AC-019.  
**Gate:** fixed model + restored RNG produces identical next sampled action sequence.

---

### AC-021 — Deterministic evaluation action

**Maps to:** T3  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/actor_policy_r0.py`
- `[MODIFY] tests/test_actor_policy_r0.py`

**Implement**

- Deterministic policy readout for evaluation/diagnostics.
- Must not mutate RNG state.
- Must be explicitly distinct from sampled behavior used for training collection.

**Dependencies:** AC-015–AC-020.  
**Gate:** evaluation before/after does not change the subsequent stochastic sample sequence.

---

### AC-022 — Actor–Critic Brain integration surface

**Maps to:** T3  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/actor_critic_brain_r0.py`
- `[NEW] tests/test_actor_critic_brain_r0.py`
- `[KEEP/FROZEN] cb16_local_opt/typed_central_brain_r10.py`

**Implement**

- Reuse compatible sensory representations without reusing historical deterministic action semantics.
- Expose stochastic Actor distribution/action API.
- Bind policy hash/version and RNG provenance.

**Dependencies:** AC-015–AC-021.  
**Gate:** new Brain can sample, score and deterministically evaluate the same observation while legacy Brain remains untouched.

---

### AC-023 — Actor–Critic observation schema

**Maps to:** T4  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/actor_critic_observation_r0.py`
- `[NEW] tests/test_actor_critic_observation_r0.py`

**Implement**

- Explicit causal observation containing authorized market sensory state + Account state + legal/execution state required by policy.
- Include normalized `equity/E_ref` if used.
- Include remaining objective time `tau=T-t` only if the policy/value problem is horizon-conditioned.

**Dependencies:** AC-001–AC-022.  
**Gate:** no future information; observation identity is deterministic.

---

### AC-024 — Observation builder and account-lineage validation

**Maps to:** T4  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/actor_critic_observation_r0.py`
- `[MODIFY] tests/test_actor_critic_observation_r0.py`

**Implement**

- Build `x_t` from the exact current authoritative account snapshot.
- Validate account lineage across successive observations.
- Reject stale/mismatched account snapshots.

**Dependencies:** AC-023.  
**Gate:** `x_{t+1}` demonstrably descends from execution at `t`.

---

### AC-025 — Policy decision-clock contract

**Maps to:** T4  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/decision_clock_r0.py`
- `[NEW] tests/test_decision_clock_r0.py`

**Implement**

- Freeze when the Actor is allowed/required to make a new decision.
- Separate market-bar progression from learner-update cadence and storage chunking.
- No hidden H72-first-step-only behavior.

**Dependencies:** AC-023–AC-024.  
**Gate:** fixture enumerates exact decision timestamps and rejects duplicate/skipped unauthorized decisions.

---

### AC-026 — Continuous rollout state machine

**Maps to:** T4  
**Priority:** P0 / central collector task

**Files**

- `[NEW] cb16_local_opt/continuous_rollout_r0.py`
- `[NEW] tests/test_continuous_rollout_r0.py`
- `[REUSE] cb16_local_opt/market_runtime_cache_r11.py`
- `[KEEP/FROZEN] cb16_local_opt/trace_runtime_r11.py`

**Implement**

At each authorized decision point:

1. current market state;
2. current Account state;
3. observation build;
4. Actor sample;
5. behavior `log_mu`;
6. permission;
7. Physics execution;
8. post-step Account state;
9. reward hook;
10. transition append;
11. next decision from `AccountState_{t+1}`.

**Dependencies:** AC-014, AC-022–AC-025.  
**Gate:** multi-step fixture proves Actor is called at every decision clock and no `j>0 => FLAT/0` shortcut exists.

---

### AC-027 — Rollout execution integration and within-account serialization

**Maps to:** T4  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/continuous_rollout_r0.py`
- `[MODIFY] tests/test_continuous_rollout_r0.py`
- `[REUSE] existing async/orchestration primitives where compatible`

**Implement**

- Wire nominal -> permission -> executable -> Physics -> account transition.
- Allow different accounts to run asynchronously while preserving strict chronological order within one account.
- Stable trace/account IDs.

**Dependencies:** AC-026.  
**Gate:** concurrent-account test shows no cross-account contamination and no within-account reordering.

---

### AC-028 — Arithmetic-equity reward primitive

**Maps to:** T5  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/reward_r0.py`
- `[NEW] tests/test_reward_r0.py`

**Implement**

```text
r_t = (E_{t+1} - E_t) / E_ref
```

- Freeze `E_ref` semantics for an objective cohort/episode.
- Keep legacy H72 log utility untouched.

**Dependencies:** AC-026–AC-027.  
**Gate:** positive/zero/negative rewards match authoritative equity deltas exactly.

---

### AC-029 — Reward accounting invariants

**Maps to:** T5  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/reward_r0.py`
- `[MODIFY] tests/test_reward_r0.py`

**Implement**

- Fees, funding, realized and unrealized PnL enter reward only through authoritative equity changes unless separately proven necessary.
- Prevent double counting.
- Fail closed on non-finite/corrupt accounting state.

**Dependencies:** AC-028.  
**Gate:** cost/funding fixtures are counted exactly once.

---

### AC-030 — Reward telescoping and failure-path qualification

**Maps to:** T5  
**Priority:** P0

**Files**

- `[MODIFY] tests/test_reward_r0.py`

**Implement/test**

For common finite horizon and `gamma=1`:

```text
sum_t r_t == (E_T - E_0) / E_ref
```

Include liquidation/non-positive terminal equity.

**Dependencies:** AC-028–AC-029.  
**Gate:** identity passes within frozen numerical tolerance and failed accounts are not censored.

---

### AC-031 — Boundary taxonomy

**Maps to:** T6  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/episode_boundary_r0.py`
- `[NEW] tests/test_episode_boundary_r0.py`

**Implement**

At minimum:

- `TRUE_TERMINAL`;
- `OBJECTIVE_T`;
- `CHUNK_TRUNCATION`;
- `DATA_END`;
- `PAUSE`.

**Dependencies:** AC-026–AC-030.  
**Gate:** every rollout stop has one explicit boundary reason.

---

### AC-032 — Bootstrap semantics by boundary type

**Maps to:** T6  
**Priority:** P0

**Files**

- `[MODIFY] cb16_local_opt/episode_boundary_r0.py`
- `[MODIFY] tests/test_episode_boundary_r0.py`

**Implement**

- True terminal -> no future bootstrap.
- Non-terminal chunking/pause -> bootstrap from the next real state/value.
- Objective horizon ends the objective segment without falsely claiming physical account death.
- Data exhaustion must not masquerade as realized terminal outcome.

**Dependencies:** AC-031.  
**Gate:** split versus unsplit rollout has equivalent return/value semantics where mathematically expected.

---

### AC-033 — Pause/resume and non-terminal account continuity

**Maps to:** T6  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/rollout_resume_r0.py`
- `[NEW] tests/test_rollout_resume_r0.py`

**Implement**

- Seal and restore account/policy/RNG/decision-clock state at non-terminal interruption.
- Resume without resetting logical account or skipping/repeating a decision.

**Dependencies:** AC-020, AC-026–AC-032.  
**Gate:** interrupted+resumed rollout reproduces uninterrupted rollout transition-for-transition.

---

### AC-034 — `TransitionV2` schema

**Maps to:** T7  
**Priority:** P0

**Files**

- `[NEW] cb16_local_opt/trajectory_schema_r0.py`
- `[NEW] tests/test_trajectory_schema_r0.py`

**Required fields**

At minimum:

- transition/causal trace ID;
- account ID;
- generation/update identity;
- behavior policy hash/version;
- observation schema/version and immutable `x_t` reference/payload;
- nominal sampled action;
- `log_mu`;
- permission decision/reason;
- executed action/delta;
- `E_t`, `E_{t+1}`, `E_ref`;
- reward;
- immutable `x_{t+1}` reference/payload;
- boundary type/bootstrap mask;
- market/data lineage;
- action/execution/reward/normalizer versions;
- RNG provenance;
- source classification.

**Dependencies:** AC-008–AC-033.  
**Gate:** transition is self-auditable without mutable external runtime state.

---

### AC-035 — `SequenceV2`, Experience Lake adapter and failure persistence

**Maps to:** T7  
**Priority:** P0 / **Gate B completion**

**Files**

- `[MODIFY] cb16_local_opt/trajectory_schema_r0.py`
- `[NEW] cb16_local_opt/trajectory_lake_r0.py`
- `[NEW] tests/test_trajectory_lake_r0.py`
- `[REUSE] cb16_local_opt/sharded_experience_lake.py`

**Implement**

- Immutable ordered sequences of compatible transitions.
- Content-addressed persistence through existing Experience Lake.
- Preserve bankrupt/liquidated/negative-return trajectories.
- Detect same-ID/different-content conflict.
- Record generation/policy/account lineage.

**Dependencies:** AC-034.  
**Gate:** write/read round-trip is bit/semantic stable; failure trajectories persist; incompatible sequence composition fails closed. **Gate B closes only here.**

---

## Phase C — Critic, V-trace learner, replay and recovery

### AC-036 — Separate Critic model API

**Maps to:** T8  
**Priority:** P1

**Files**

- `[NEW] cb16_local_opt/critic_value_r0.py`
- `[NEW] tests/test_critic_value_r0.py`

**Implement**

- Independent value estimator; not the old Teacher.
- Output scalar expected future arithmetic-return value under the frozen objective semantics.

**Dependencies:** Gate B.  
**Gate:** shape/device/dtype and deterministic forward behavior are tested.

---

### AC-037 — Critic causal input contract

**Maps to:** T8  
**Priority:** P1

**Files**

- `[MODIFY] cb16_local_opt/critic_value_r0.py`
- `[MODIFY] tests/test_critic_value_r0.py`

**Implement**

- Critic sees only causal information available at decision time.
- Account state and legal/execution state must be represented where necessary.
- Horizon conditioning must match Actor/objective semantics.

**Dependencies:** AC-023–AC-024, AC-036.  
**Gate:** future-poison perturbation does not alter current Critic input/value.

---

### AC-038 — Critic regression loss

**Maps to:** T8  
**Priority:** P1

**Files**

- `[MODIFY] cb16_local_opt/critic_value_r0.py`
- `[MODIFY] tests/test_critic_value_r0.py`

**Implement**

- Mean-value regression objective compatible with arithmetic expected return.
- No silent quantile/pessimistic/twin-Q replacement of the owner objective.

**Dependencies:** AC-036–AC-037.  
**Gate:** exact-value toy target can be fitted within preregistered tolerance.

---

### AC-039 — Bootstrap value calculator

**Maps to:** T8  
**Priority:** P1

**Files**

- `[NEW] cb16_local_opt/value_bootstrap_r0.py`
- `[NEW] tests/test_value_bootstrap_r0.py`

**Implement**

- Apply boundary-specific bootstrap rules from AC-032.
- Explicit value at sequence end.

**Dependencies:** AC-032, AC-036–AC-038.  
**Gate:** true terminal/non-terminal truncation known-answer cases return correct bootstrap values.

---

### AC-040 — V-trace recurrence kernel

**Maps to:** T9  
**Priority:** P1

**Files**

- `[NEW] cb16_local_opt/vtrace_r0.py`
- `[NEW] tests/test_vtrace_r0.py`

**Implement**

- Frozen equations for importance ratio, `rho_bar`, `c_bar`, temporal deltas and backward recurrence.
- Use behavior `log_mu` and current target-policy `log_pi` from compatible action semantics.

**Dependencies:** AC-019, AC-032, AC-034–AC-039.  
**Gate:** outputs finite; shapes/masks correct; no accidental probability-space underflow path.

---

### AC-041 — V-trace mathematical known-answer tests

**Maps to:** T9  
**Priority:** P1

**Files**

- `[MODIFY] tests/test_vtrace_r0.py`

**Implement/test**

- Hand-computed short sequence.
- `pi == mu` reduction to the on-policy recurrence.
- Clipping boundary cases.
- True terminal versus truncation.

**Dependencies:** AC-040.  
**Gate:** exact known-answer values within frozen tolerance.

---

### AC-042 — Actor policy-gradient loss

**Maps to:** T9  
**Priority:** P1

**Files**

- `[NEW] cb16_local_opt/actor_critic_loss_r0.py`
- `[NEW] tests/test_actor_critic_loss_r0.py`

**Implement**

- Policy loss from the frozen V-trace advantage/target formulation.
- If entropy regularization exists, expose it as a transparent coefficient and metric.
- Do not silently stack PPO clipping/GAE/SAO or another correction into the baseline V-trace task.

**Dependencies:** AC-040–AC-041.  
**Gate:** sign/direction of gradient matches simple hand-built policy examples.

---

### AC-043 — Actor–Critic training step/runtime

**Maps to:** T9  
**Priority:** P1

**Files**

- `[NEW] cb16_local_opt/actor_critic_training_r0.py`
- `[NEW] tests/test_actor_critic_training_r0.py`
- `[KEEP/FROZEN] cb16_local_opt/training_runtime_r11.py`

**Implement**

- Separate Actor/Critic optimizers.
- Sequence minibatch update.
- Explicit gradient ownership.
- Metrics for policy loss, critic loss, entropy if enabled, importance ratios and clipping rates.
- No mutation of old Teacher/Student training runtime.

**Dependencies:** AC-036–AC-042.  
**Gate:** one deterministic fixed-seed update produces expected parameter deltas; frozen historical training still passes.

---

### AC-044 — Replay compatibility gate

**Maps to:** T10  
**Priority:** P1

**Files**

- `[NEW] cb16_local_opt/replay_compat_r0.py`
- `[NEW] tests/test_replay_compat_r0.py`

**Implement**

Before replay require compatible:

- action schema;
- behavior probability semantics;
- observation/normalizer version;
- permission/execution semantics;
- reward schema;
- boundary semantics;
- required lineage.

**Dependencies:** AC-001–AC-043.  
**Gate:** every deliberate mismatch fixture fails closed with explicit reason.

---

### AC-045 — Sequence replay sampler and weighting

**Maps to:** T10  
**Priority:** P1

**Files**

- `[NEW] cb16_local_opt/sequence_replay_r0.py`
- `[NEW] tests/test_sequence_replay_r0.py`

**Implement**

- Initial sequence length/batch semantics from the training design, while keeping these configuration values explicit.
- Sample only compatible sequences.
- If account/trajectory resampling weights are used, make estimator weighting explicit and testable.
- No survivor-only filtering.

**Dependencies:** AC-035, AC-044.  
**Gate:** deterministic fixed-seed sample order; empirical weighting matches declared probabilities.

---

### AC-046 — Full Actor–Critic generation checkpoint and exact recovery

**Maps to:** T11  
**Priority:** P1

**Files**

- `[NEW] cb16_local_opt/actor_critic_checkpoint_r0.py`
- `[NEW] tests/test_actor_critic_checkpoint_r0.py`
- `[REUSE] cb16_local_opt/checkpoint_store_r11.py`

**Checkpoint bundle must bind**

- Actor weights;
- Critic weights;
- both optimizer states;
- Actor sampling RNG;
- learner/process RNG needed for deterministic continuation;
- update counter/generation;
- replay/data snapshot identity;
- normalizer/science contract identity;
- logical account/collector continuation identities where applicable.

**Dependencies:** AC-020, AC-033, AC-035–AC-045.  
**Gate:** crash/restart reproduces the same next sampled action **and** same next gradient update from the same sealed state.

---

## Phase D — Mathematical and known-answer qualification

### AC-047 — Core math/contract qualification suite

**Maps to:** T12  
**Priority:** P1

**Files**

- `[NEW] tests/test_actor_critic_math_qualification_r0.py`

**Must include**

- reward telescoping;
- liquidation retention;
- nominal/permitted/executed split;
- behavior/action probability consistency;
- V-trace hand recurrence;
- `pi==mu` reduction;
- bootstrap/terminal boundaries;
- recovery equivalence;
- causal-input/future-poison invariance;
- replay weighting sanity.

**Dependencies:** AC-001–AC-046.  
**Gate:** all tests preregistered and green; failures cannot be bypassed by deleting the failing case.

---

### AC-048 — Known-answer toy suite A: account dependence and high-risk expected return

**Maps to:** T13  
**Priority:** P1

**Files**

- `[NEW] cb16_local_opt/toy_envs_actor_critic_r0.py`
- `[NEW] tests/test_actor_critic_toy_envs_r0.py`

**Toy cases**

1. Same market state, different Account state -> different optimal action.
2. A strategy with higher bankruptcy frequency but higher true expected arithmetic return must win when the expected return is genuinely larger.
3. Failures remain in both learning and evaluation samples.

**Dependencies:** AC-047.  
**Gate:** learned/ranked policy converges to preregistered known answer within fixed seed/budget tolerance.

---

### AC-049 — Known-answer toy suite B: delayed consequence, horizon reversal and off-policy correction

**Maps to:** T13  
**Priority:** P1

**Files**

- `[MODIFY] cb16_local_opt/toy_envs_actor_critic_r0.py`
- `[MODIFY] tests/test_actor_critic_toy_envs_r0.py`

**Toy cases**

4. Short-term gain followed by delayed loss, proving long-horizon credit can reverse a myopic choice.
5. Environment where extending objective horizon genuinely reverses policy ranking.
6. Old-policy replay with known `mu` and new `pi`: compare correct V-trace, pure on-policy and intentionally uncorrected replay.

**Dependencies:** AC-048.  
**Gate:** each method matches its mathematical known answer; V-trace correction must not be accepted merely because loss decreases.

---

### AC-050 — Qualification compiler and hard PASS receipt

**Maps to:** T12/T13  
**Priority:** P1 / **Gate C completion**

**Files**

- `[NEW] cb16_local_opt/actor_critic_qualification_r0.py`
- `[NEW] tests/test_actor_critic_qualification_r0.py`
- `[NEW] authority/rearchitecture_r11/CB16_R11_ACTOR_CRITIC_QUALIFICATION_R0_SPEC_V1.json`
- `[NEW on PASS only] authority/rearchitecture_r11/CB16_R11_ACTOR_CRITIC_QUALIFICATION_R0_RECEIPT_V1.json`

**Implement**

- Compile AC-047–AC-049 outcomes plus required legacy regressions into one fail-closed qualification result.
- Receipt records code SHA, contract hashes, seeds, budgets, tolerances and exact test inventory.
- No PASS if any required test is skipped/xfail/unknown.

**Dependencies:** AC-001–AC-049.  
**Gate:** one machine-checkable `PASS` receipt. **No real historical market scale-up is authorized before this task passes.**

---

## Phase E — Economic evaluation

### AC-051 — Common-horizon cohort evaluator

**Maps to:** T14  
**Priority:** P2

**Files**

- `[NEW] cb16_local_opt/economic_evaluator_r0.py`
- `[NEW] tests/test_economic_evaluator_r0.py`

**Implement**

- Evaluate all preregistered accounts on common objective horizon `T` and cohort semantics.
- Arithmetic return is primary.
- Include all failed accounts.

**Dependencies:** Gate C.  
**Gate:** synthetic cohort mean matches hand calculation including bankrupt paths.

---

### AC-052 — Buy-and-Hold and FLAT baseline engine

**Maps to:** T14  
**Priority:** P2

**Files**

- `[NEW] cb16_local_opt/economic_baselines_r0.py`
- `[NEW] tests/test_economic_baselines_r0.py`

**Implement**

- B&H baseline and all-FLAT/no-position baseline on exactly the same cohort/horizon/data mechanics.
- No favorable data-window mismatch.

**Dependencies:** AC-051.  
**Gate:** identical cohort IDs/data lineage across Trader and baselines.

---

### AC-053 — Failure/tail/outcome-distribution reporting

**Maps to:** T14  
**Priority:** P2

**Files**

- `[MODIFY] cb16_local_opt/economic_evaluator_r0.py`
- `[NEW] tests/test_economic_failure_reporting_r0.py`

**Report, without changing the primary objective**

- liquidation/account-death frequency;
- return distribution/tails;
- median/quantiles;
- maximum loss diagnostics where defined;
- success/failure counts and reasons;
- survival-sensitive diagnostics explicitly labeled as diagnostics.

**Dependencies:** AC-051–AC-052.  
**Gate:** removing a failed account changes a protected fixture result and is detected.

---

### AC-054 — Champion/Challenger economic comparison contract

**Maps to:** T14  
**Priority:** P2 / **Gate D completion**

**Files**

- `[NEW] cb16_local_opt/economic_promotion_r0.py`
- `[NEW] tests/test_economic_promotion_r0.py`

**Implement**

- Compare Champion/Challenger under the same frozen cohort/horizon/evaluator.
- Primary reported decision quantity remains arithmetic expected return and declared baseline deltas.
- Do not silently substitute Sharpe/log-growth/drawdown ranking.
- If the exact master ranking rule is still owner-open, report `DECISION_RULE_UNRESOLVED` instead of inventing one.

**Dependencies:** AC-051–AC-053.  
**Gate:** evaluator can distinguish measurement from promotion authority; unresolved ranking cannot silently promote.

---

## Phase F — Migration, generation continuity, CI and documentation closure

### AC-055 — Historical demo / new replay boundary router

**Maps to:** T15  
**Priority:** P2

**Files**

- `[NEW] cb16_local_opt/experience_source_router_r0.py`
- `[NEW] tests/test_experience_source_router_r0.py`

**Implement**

- Classify legacy H72/Teacher evidence as demonstration/diagnostic unless regenerated under compatible Actor–Critic semantics.
- Reject old nine-action/H72 records from V-trace replay when valid behavior likelihood is absent.
- Never fabricate `log_mu`.

**Dependencies:** AC-044–AC-054.  
**Gate:** known incompatible legacy sample is rejected with explicit reason while remaining readable as historical evidence.

---

### AC-056 — Generation switch, policy lineage and account continuity integration

**Maps to:** T16  
**Priority:** P2

**Files**

- `[NEW or ADAPT] cb16_local_opt/actor_critic_generation_binding_r0.py`
- `[NEW] tests/test_actor_critic_generation_binding_r0.py`
- `[REUSE/ADAPT] concepts from continuous-generation qualification/binding work`

**Implement**

- Fixed behavior checkpoint for each registered collection unit.
- Generation/policy switch is explicitly recorded.
- Switching generation must not reset the logical account.
- Mixed-generation trajectories retain per-transition behavior-policy identity; do not attribute the entire trajectory to the final policy.

**Dependencies:** AC-035, AC-046, AC-055.  
**Gate:** account lineage remains continuous across a controlled generation switch and replay sees correct per-transition policy identity.

---

### AC-057 — Stage-4 infrastructure adoption and CI qualification workflow

**Maps to:** T17  
**Priority:** P2

**Files**

- `[NEW/ADAPT] .github/workflows/cb16-r11-actor-critic-qualification-r0.yml`
- `[ADAPT ONLY AS NEEDED] Stage-4 canonical runtime/state-root/fencing/recovery/orchestration integration surfaces`
- `[NEW] tests/test_actor_critic_infrastructure_adoption_r0.py`

**Implement**

- Reuse existing singleton/fencing/state-root/recovery/async infrastructure.
- Add only scientifically required adapter surfaces.
- CI must run legacy regressions + new qualification gates.
- Do not open final holdout or download fresh data.

**Dependencies:** AC-050, AC-055–AC-056.  
**Gate:** clean CI run produces complete artifacts/receipts and historical lane remains green.

---

### AC-058 — Canonical docs/current-state/decision-log closure

**Maps to:** T17  
**Priority:** P2 / final planning closure

**Files**

- `[MODIFY] docs/R11_ACTOR_CRITIC_CODE_ALIGNMENT_TODO.md`
- `[MODIFY] docs/TRAINING_ALGORITHM_R0.md`
- `[MODIFY] docs/TRAINING_QUALIFICATION_R0.md`
- `[MODIFY] docs/CURRENT_STATE.md`
- `[MODIFY] docs/ARCHITECTURE_MAP.md`
- `[MODIFY] docs/DECISIONS.md`
- `[MODIFY] docs/OPEN_QUESTIONS.md`
- `[MODIFY] docs/README.md` if navigation changes

**Implement**

- Record exact PASS/FAIL/BLOCKED state of AC-001–AC-057.
- Update architecture map to show historical lane versus Actor–Critic lane.
- Close decisions only when authority exists; unresolved choices stay explicitly unresolved.
- Record exact code/receipt SHAs.

**Dependencies:** AC-001–AC-057.  
**Gate:** docs agree with live code/receipts; no claim of qualification without corresponding machine-readable evidence.

---

# 6. Dependency DAG

The intended critical path is:

```text
AC-001..004
    -> AC-005..014                # semantic Action / Permission / Physics gate
    -> AC-015..022                # stochastic Actor
    -> AC-023..035                # continuous experience production
    -> AC-036..046                # Critic / V-trace / replay / recovery
    -> AC-047..050                # mathematical + known-answer qualification
    -> AC-051..054                # economic evaluation
    -> AC-055..058                # migration + generation + infra + docs closure
```

Parallelism is allowed only where the declared dependency set is already satisfied. In particular:

- AC-015 Actor internals may be developed after Gate A, but they must not generate canonical training experience before AC-035.
- Critic architecture experiments may be coded after the observation contract stabilizes, but no canonical learner qualification occurs before Gate B.
- Economic evaluator scaffolding may be written early, but no real-market scientific promotion authority exists before AC-050.
- Infrastructure work must not outrun an unresolved scientific contract.

---

# 7. Definition of done for one AC task

A task is not DONE merely because code imports.

Each task must satisfy all applicable items:

- [ ] declared files only, or documented authority-approved scope expansion;
- [ ] deterministic unit tests where determinism is part of the contract;
- [ ] negative/fail-closed test;
- [ ] lineage/version fields recorded where relevant;
- [ ] no final-holdout access;
- [ ] no fresh-data download;
- [ ] no silent mutation of historical runtime;
- [ ] no hidden objective change;
- [ ] no fabricated behavior likelihood;
- [ ] failure cases retained;
- [ ] exact dependency SHAs/contract hashes recorded in receipt when required;
- [ ] result classified as PASS, SCIENTIFIC_FAIL, EXECUTION_BLOCKED or HARDWARE_LIMIT as appropriate.

---

# 8. Scientific no-rescue rules

Once an AC qualification experiment begins, do not rescue it inside the same task by:

- deleting bankrupt accounts;
- replacing arithmetic return with log return;
- changing the horizon because the result is inconvenient;
- changing seed/budget after seeing the answer;
- fabricating `log_mu` for historical data;
- changing execution semantics after trajectories have already been collected;
- changing `E_ref` definition after looking at results;
- turning a known-answer toy into a different problem;
- introducing a new handcrafted cycle/regime/rule engine to force recurrence behavior;
- silently converting an execution blocker into a scientific PASS.

A legitimate redesign starts a new revision/task with a new preregistered contract.

---

# 9. Owner-open decisions that must remain visibly open until frozen

The TODO may recommend implementations, but the following should not be silently invented if still unresolved at implementation time:

1. Exact target-position/exposure semantics beyond the approved high-level `LONG/FLAT/SHORT + requested target risk` intent, including any leverage/exposure cap that is strategic rather than mechanical.
2. Exact stochastic bounded-risk distribution if multiple mathematically valid candidates remain.
3. Exact finite objective horizon `T` used for canonical economic ranking.
4. Exact Champion/Challenger master ranking rule when arithmetic-return deltas to B&H and FLAT do not point to the same decision.
5. Any entropy coefficient or other regularization strength that could materially alter the objective.
6. Exact replay sequence length/batch size if not yet frozen; these are implementation hyperparameters, not semantic permission to change the learning objective.

When unresolved, code/receipts must say `UNRESOLVED` and fail closed where the missing decision affects scientific validity.

---

# 10. Explicitly prohibited shortcuts

Do **not**:

- retrofit stochastic sampling into `typed_central_brain_r10.py`;
- convert `training_runtime_r11.py` into Actor–Critic training;
- edit `_simulate_h72_branch_r11` into the new continuous collector;
- reinterpret old H72 log utility as the new reward;
- give historical demonstrations fake behavior probabilities;
- drop non-positive-equity trajectories;
- reset Account state at learner/checkpoint/generation boundaries without a true terminal contract;
- use final holdout for debugging or task qualification;
- rebuild Experience Lake/Checkpoint Store merely because new payloads are needed;
- add hand-engineered recurrence/cycle activation logic to substitute for model learning.

---

# 11. Final readiness checklist

Real bounded historical Actor–Critic training is not authorized until all of the following are true:

- [ ] Gate A PASS: AC-001–AC-014.
- [ ] Gate B PASS: AC-015–AC-035.
- [ ] Gate C PASS with machine-readable receipt: AC-036–AC-050.
- [ ] All known-answer toys pass without post-hoc rescue.
- [ ] Legacy H72/Teacher–Student regression remains green.
- [ ] Behavior likelihood is real and reproducible.
- [ ] Failed accounts remain present.
- [ ] Pause/restart/generation switch preserves account continuity.
- [ ] Actor/Critic/optimizer/RNG/replay state can recover exactly.
- [ ] Final holdout remains sealed.

Economic Champion/Challenger promotion under the new lane additionally requires:

- [ ] Gate D PASS: AC-051–AC-054.
- [ ] Same preregistered cohort/horizon for Trader, B&H and FLAT.
- [ ] Arithmetic expected return remains the primary objective.
- [ ] Tail/liquidation metrics are reported rather than used as silent replacement objectives.
- [ ] AC-055–AC-058 integration/CI/documentation closure is complete before declaring the lane canonical.

---

# 12. Immediate next implementation boundary

The next engineering boundary is **Phase A only**:

```text
AC-001 -> AC-002 -> AC-003 -> AC-004
       -> AC-005 -> ... -> AC-014
```

Do not jump directly to `vtrace_r0.py`.

The first irreversible scientific asset is the meaning of an action executed against a live account state. Once trajectories are generated under one action/execution contract, later semantic changes can invalidate their behavior likelihoods and replay meaning. Therefore Action/Permission/Physics semantics must qualify before experience production and learning.

**Current overall state:** `PLANNED / NOT YET IMPLEMENTED / HISTORICAL LANE PRESERVED`.
