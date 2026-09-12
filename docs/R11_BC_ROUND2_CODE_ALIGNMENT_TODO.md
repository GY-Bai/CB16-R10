# CB16 R11 BC Round 2 — Code Alignment / Closure TODO

**Status:** OPEN / Round-2 execution plan  
**Audit baseline:** `main@c373d230e218e2fb83064d3a7fed529e174482b6`  
**Audit date:** 2026-09-12  
**Task namespace:** `BC-001` through `BC-105`  
**Target:** a self-contained implementation plan that a new Agent can execute without reconstructing the detailed AC-001–AC-058 history.  
**Scope:** audit and repair already-landed Actor–Critic R0 surfaces where necessary, then complete the continuous-account stochastic Actor–Critic learning loop, generation continuity, retention qualification, economic evaluation, CI and documentation closure.

---

# 0. How BC Round 2 relates to AC

BC is a new task namespace and the current Round-2 execution plan. It is **not** merely `AC-059+` and does not require the executing Agent to know each AC task.

The existing AC TODO remains historical/design provenance. Code already merged from AC work is treated as **input implementation to audit**, not as automatically correct authority for Round 2.

Rules:

1. Start every BC task from the live `main` SHA, not from an old AC branch.
2. Existing AC R0 code may be reused when it satisfies the current owner vision and BC acceptance tests.
3. If BC discovers a semantic defect in AC code, BC may repair or supersede it.
4. A semantic change must not be hidden behind the same science identity. Prefer a new R1/Round-2 contract or an explicit compatibility adapter; preserve R0 regression evidence.
5. Do not rewrite historical R10/R11 frozen authority or prior scientific verdicts merely because Round 2 needs different semantics.
6. Unmerged AC branches are **not authority**. At audit time `ai/r11-ac-015-actor-policy-interface-r0` exists one commit ahead of main with `actor_policy_r0.py` and its tests. BC may inspect it as a candidate, but must not depend on it or assume it will merge.
7. FINAL remains sealed and fresh market-data download remains forbidden unless separately authorized.

The implementation Agent should read, in this order:

- `AGENTS.md`
- `docs/VISION.md`
- `docs/PRINCIPLE_ALIGNMENT.md`
- `docs/COMPONENT_REQUIREMENTS.md`
- this file
- then only the code/authority/docs needed for the current BC task.

The old AC TODO is optional historical context for a BC implementer.

---

# 1. Live-baseline audit verdict

## 1.1 What is present on main

At `c373d23` main contains the implemented R0 surfaces corresponding to the old AC-001–AC-014 progression:

- `cb16_local_opt/actor_critic_contract_r0.py`
- `cb16_local_opt/actor_critic_runtime_router_r0.py`
- `cb16_local_opt/action_contract_r0.py`
- `cb16_local_opt/execution_record_r0.py`
- `cb16_local_opt/actor_critic_supervisor_r0.py`
- `cb16_local_opt/target_exposure_r0.py`
- `cb16_local_opt/actor_critic_physics_adapter_r0.py`
- their dedicated tests and legacy-freeze sentinel.

Reusable infrastructure already on main includes, among other surfaces:

- `cb16_local_opt/continuous_generation_binding_r0.py`
- `cb16_local_opt/demonstration_shard_store_r11.py`
- `cb16_local_opt/stage4_state_roots_mount_r0.py`
- `cb16_local_opt/sharded_experience_lake.py`
- `cb16_local_opt/checkpoint_store_r11.py`
- `cb16_local_opt/market_runtime_cache_r11.py`
- Stage-4 fencing/state-root/recovery/orchestration infrastructure.

`c373d23` Repo Guard and R11 Main Smoke both completed successfully. This proves repository/runtime smoke integrity for that commit; it does **not** prove Round-2 scientific closure.

## 1.2 Concrete semantic gaps found in the current code

These are BC inputs, not speculative future concerns.

### Gap G-01 — nominal same-risk is classified as NOOP too early

`action_contract_r0.py` classifies same direction + same `requested_target_risk` as `SAME_TARGET_NOOP` before current equity, price, legal notional envelope and quantity are recomputed.

But `target_exposure_r0.py` defines risk as a fraction of a **current** legal notional envelope. Therefore the same `LONG, 0.5` request can require a different target quantity after equity/price/margin/limits move.

Round 2 must make **execution NOOP a post-sizing fact**, not a nominal-action fact.

### Gap G-02 — Supervisor infers exposure increase from risk scalar

`actor_critic_supervisor_r0.py` decides whether new margin is needed via `permitted_risk > current_risk`. That can disagree with the actual target quantity/notional after the legal envelope changes.

Round 2 must separate:

- policy request legality / hard risk cap;
- target sizing using current account/exchange authority;
- actual execution feasibility of the resulting delta.

### Gap G-03 — current adapter inherits strategy preferences from the frozen kernel

`actor_critic_physics_adapter_r0.py` explicitly delegates funding, intrabar SL/TP, liquidation and max-hold to frozen `step_account`.

The frozen V5.5 kernel also contains stop-loss, take-profit, time-stop and cooldown behavior, and entry sizing/capital validation references ATR-derived protective prices.

Liquidation, fees, funding, legal quantities and genuine exchange/account constraints may be environment mechanics. Fixed SL/TP/max-hold/cooldown are strategy preferences unless separately justified. They must not silently make trading choices for the autonomous Round-2 Trader.

### Gap G-04 — economic loss may be erased by cash clamping

Current `_partial_reduce` in the Actor–Critic adapter performs `st.cash = max(0.0, st.cash)`.

The frozen kernel `_force_close_at` and insolvency path also clamp cash to zero. This can be valid for a specifically defined limited-liability settlement model, but it cannot be assumed when the Round-2 objective requires complete economic consequence accounting, including possible negative equity/liability until responsibility truly ends.

Round 2 must freeze the legal/economic responsibility model and prove ledger conservation.

### Gap G-05 — target adapter is not an environment clock

`execute_target_position_r0(...)` explicitly does not advance a market bar. Continuous trading therefore still needs an independent environment-advance contract so that market PnL, funding and liquidation continue between policy decisions even when no order is sent or an action is rejected.

### Gap G-06 — Account observation is not the authoritative ledger

Current philosophy explicitly distinguishes complete account truth/recovery state from Account6 or another policy observation. Round 2 must define both and prove their relationship.

### Gap G-07 — state sufficiency is not established

No current qualification proves that the policy-visible market/account representation is sufficient whenever the known-answer optimal action depends on account/history. If two causally different states alias to the same observation while requiring different actions, that is a representation gap, not a reason to increase training epochs.

### Gap G-08 — no complete stochastic-policy / collector / learner chain exists on main

At the baseline, main does not yet contain the complete Round-2 stochastic Actor, causal observation builders, continuous collector, trajectory schema/lake adapter, arithmetic reward runtime, Critic, V-trace learner, replay-support health gate, exact learner recovery or full closed-loop qualification.

### Gap G-09 — continual knowledge retention has no scientific gate

The owner requirement that new experience may calibrate current behavior without simply erasing reusable historical knowledge is documented, while hand-built regime/cycle activation is forbidden. No current task proves A→B→A retention/adaptation behavior.

### Gap G-10 — component PASS and closed-loop/economic PASS are not the same evidence

Round 2 needs explicit evidence levels. A unit test or workflow success cannot be upgraded into “learned the task” or “economic ability improved.”

---

# 2. Non-negotiable Round-2 principles

Every BC task must preserve these owner-aligned constraints.

- **P-01 Single-asset account Trader:** policy uses market + account + necessary causal history; humans choose asset/capital/macro scope.
- **P-02 Experience comes from interaction:** nominal action -> permission -> actual execution -> account consequence -> next decision.
- **P-03 Repeated practice is allowed:** existing experience may be replayed; changed policy should also produce new legal account trajectories.
- **P-04 Account consequences persist:** chunk, pause, process restart, learner update and generation switch do not silently reset the logical account.
- **P-05 Risk choice belongs to the model:** exchange/account mechanics constrain legality; fixed strategy preferences do not silently choose stop/exit/holding policy.
- **P-06 Primary economic preference is expected arithmetic return:** complete failures count; no silent Sharpe/log-growth/drawdown/bankruptcy objective substitution.
- **P-07 Facts, replay, demonstrations and evaluation are different views:** winners may be separately selected, but failure facts remain and evaluation is not survivor-filtered.
- **P-08 Frozen organs vs trainable Brain are distinct:** no gradients into frozen market organs; declared account/fusion/Actor/Critic paths must actually receive gradients.
- **P-09 Fixed running checkpoint, generation learning:** behavior checkpoint stays fixed during a registered collection unit; replacement does not reset the account; history is not deleted by age.
- **P-10 Evidence claims are scoped:** contract, component, closed-loop, known-answer, economic and transfer evidence are distinct.

Additional Round-2 invariants:

- Truth != Belief != Decision != Permission != Execution.
- `requested_target_risk` != confidence.
- nominal action probability is never reconstructed from a clamped/executed action.
- market action does not alter the historical price path under the current price-taker sandbox assumption.
- no fabricated `log_mu`.
- no deletion of bankrupt/negative-return trajectories to rescue results.
- no free recapitalization of a dead account under the same account identity.
- no final holdout debugging and no fresh data download.
- no handcrafted cycle/regime activation engine used as a substitute for learned recurrence.

---

# 3. Evidence levels

Every BC completion report must declare the strongest level actually proven:

1. `CONTRACT` — versioned fields/semantics exist and fail closed.
2. `COMPONENT` — controlled inputs produce specified outputs, including negative cases.
3. `CLOSED_LOOP` — real component chain passes state/action/account consequences end-to-end.
4. `KNOWN_ANSWER` — a pre-solved task is learned or ranked correctly under frozen budget/seeds.
5. `ECONOMIC` — a frozen policy comparison under common cohort/horizon and complete failures improves the declared arithmetic objective.
6. `TRANSFER` — evidence uses independence appropriate to the transfer claim.

No task may report a stronger level than its evidence supports.

---

# 4. Branch, dependency and receipt discipline

Recommended branch:

```text
ai/r11-bc-<NNN>-<short-name>-r0
```

Rules:

1. Re-read live `main` immediately before branch creation. If main moved, reconcile before coding.
2. One BC task owns only its declared files unless a blocker forces a documented scope expansion.
3. Do not depend on an unmerged sibling branch. Depend only on explicit prior BC tasks already on main.
4. Existing AC branches may be inspected but are not dependencies or authority.
5. Semantic changes require a version bump/new semantic identity and migration tests.
6. Historical frozen authority files stay byte-stable unless a separate authority migration explicitly permits otherwise.
7. Every task has at least one positive and one fail-closed/negative test where meaningful.
8. Keep exact SHAs, semantic hashes, authority versions and lineage in machine-readable receipts at Gate tasks.
9. Classify outcomes as `PASS`, `SCIENTIFIC_FAIL`, `EXECUTION_BLOCKED`, `HARDWARE_LIMIT`, or `UNRESOLVED_OWNER_DECISION`.
10. A scientific result is never rescued inside the same task by changing objective, seed, T, sampling weight or semantics after observing the result.

---

# 5. Round-2 gates

## Gate BC-A — Corrected execution/account semantics

Requires `BC-001` through `BC-020` PASS.

No newly collected trajectory may be called canonical Round-2 experience before BC-A.

## Gate BC-B — State and stochastic policy qualification

Requires `BC-021` through `BC-043` PASS.

This freezes the policy-visible state, state-sufficiency handling, stochastic action distribution, likelihood and gradient ownership.

## Gate BC-C — Continuous experience production

Requires `BC-044` through `BC-066` PASS.

This proves four-clock behavior, continuous account evolution, exact resume, immutable trajectory facts and replay admission boundaries.

## Gate BC-D — Learner and exact recovery

Requires `BC-067` through `BC-078` PASS.

No real historical scale-up before BC-D plus the known-answer Gate below.

## Gate BC-E — Generation and historical-retention qualification

Requires `BC-079` through `BC-085` PASS.

## Gate BC-F — Known-answer and closed-loop science qualification

Requires `BC-086` through `BC-092` PASS and a machine-readable qualification receipt.

**No real historical Actor–Critic scale-up before BC-F.**

## Gate BC-G — Economic, infrastructure and documentation closure

Requires `BC-093` through `BC-105` PASS before the Round-2 lane may be called canonical.

---

# 6. Detailed tasks

## Phase A — Baseline authority and Round-2 semantic migration

### BC-001 — Freeze live Round-2 baseline inventory

**Principles:** P-10  
**Files:**
- `[NEW] authority/rearchitecture_r11/CB16_R11_BC_ROUND2_BASELINE_V1.json`
- `[NEW] tests/test_bc_round2_baseline_r0.py`

**Implement:** bind exact `main` base SHA, critical R0 file blob SHAs, frozen authority hashes, current docs blobs and the fact that FINAL/fresh-data boundaries remain closed. Record unmerged branch names only as observations, never dependencies.

**Dependencies:** none.  
**Gate:** baseline verifier fails if any bound semantic surface differs from the preregistered inventory.

### BC-002 — Principle-to-component traceability manifest

**Principles:** P-01–P-10  
**Files:**
- `[NEW] authority/rearchitecture_r11/CB16_R11_BC_ROUND2_PRINCIPLE_TRACE_V1.json`
- `[NEW] tests/test_bc_round2_principle_trace_r0.py`

**Implement:** map each P requirement to owning BC tasks, observable evidence and forbidden shortcuts. Require all ten P IDs to have at least one implementation owner and one qualification owner.

**Dependencies:** BC-001.  
**Gate:** missing principle owner/evidence mapping fails closed.

### BC-003 — Current semantic-gap registry

**Files:**
- `[NEW] authority/rearchitecture_r11/CB16_R11_BC_ROUND2_SEMANTIC_GAPS_V1.json`
- `[NEW] tests/test_bc_round2_semantic_gaps_r0.py`

**Implement:** encode G-01 through G-10 from this document plus exact source file/function references and resolution task IDs. Do not label a gap resolved until the owning BC gate passes.

**Dependencies:** BC-001–002.  
**Gate:** every baseline gap has a resolution owner; stale “resolved” status without evidence fails.

### BC-004 — Round-2 science semantic contract

**Files:**
- `[NEW] cb16_local_opt/actor_critic_contract_r1.py`
- `[NEW] tests/test_actor_critic_contract_r1.py`
- `[KEEP] cb16_local_opt/actor_critic_contract_r0.py`

**Implement:** version the corrected execution/account/observation/reward/trajectory/policy/value/replay/checkpoint semantics. Bind account-economics version and environment-profile version explicitly. Keep code revision separate from science identity.

**Dependencies:** BC-003.  
**Gate:** mixed R0/R1 semantic payloads fail closed unless an explicit compatibility adapter exists.

### BC-005 — Round-2 runtime lane router

**Files:**
- `[NEW] cb16_local_opt/actor_critic_runtime_router_r1.py`
- `[NEW] tests/test_actor_critic_runtime_router_r1.py`
- `[KEEP] cb16_local_opt/actor_critic_runtime_router_r0.py`

**Implement:** explicit `LEGACY_R11`, `ACTOR_CRITIC_R0`, `BC_ROUND2_R1`; no implicit fallback. Emit exact Round-2 science hash into runtime receipts.

**Dependencies:** BC-004.  
**Gate:** unknown/mixed lane fails closed.

### BC-006 — R0 and historical regression fence

**Files:**
- `[NEW] tests/test_bc_round2_legacy_and_r0_freeze.py`

**Implement:** prove frozen R10/R11 historical lane and already-landed AC R0 regression fixtures remain reproducible. Round-2 changes may supersede R0 semantics only through new versions, not by silently changing old expected outputs.

**Dependencies:** BC-004–005.  
**Gate:** unintended historical or R0 behavior drift blocks merge.

### BC-007 — Evidence-level contract

**Files:**
- `[NEW] cb16_local_opt/evidence_level_r0.py`
- `[NEW] tests/test_evidence_level_r0.py`

**Implement:** typed evidence levels `CONTRACT/COMPONENT/CLOSED_LOOP/KNOWN_ANSWER/ECONOMIC/TRANSFER`; receipts state maximum justified level and source artifacts.

**Dependencies:** BC-002.  
**Gate:** a component-only fixture cannot be serialized as ECONOMIC/TRANSFER evidence.

### BC-008 — Round-2 receipt compiler scaffold

**Files:**
- `[NEW] cb16_local_opt/bc_round2_receipt_r0.py`
- `[NEW] tests/test_bc_round2_receipt_r0.py`

**Implement:** common receipt schema for code SHA, science hash, dependencies, gates, evidence level, final/fresh firewalls and result classification.

**Dependencies:** BC-004, BC-007.  
**Gate:** unknown/missing dependency or skipped required check prevents PASS receipt.

---

## Phase B — Execution semantics and complete account economics

### BC-009 — Execution-rule source taxonomy

**Principles:** P-05  
**Files:**
- `[NEW] authority/rearchitecture_r11/CB16_R11_BC_EXECUTION_RULE_TAXONOMY_V1.json`
- `[NEW] tests/test_bc_execution_rule_taxonomy_r0.py`

**Implement:** inventory fee, slippage, funding, margin, liquidation, min qty/notional, leverage cap, SL, TP, max-hold, cooldown, risk limits and finalize behavior. Classify each as `MARKET_EXCHANGE_MECHANIC`, `USER_RESOURCE_BOUNDARY`, `STRATEGY_PREFERENCE`, or `HISTORICAL_ONLY` with source/version.

**Dependencies:** BC-001–003.  
**Gate:** no forced behavior in Round-2 environment may have an unclassified source.

### BC-010 — Policy-neutral environment profile

**Files:**
- `[NEW] cb16_local_opt/actor_critic_environment_profile_r1.py`
- `[NEW] tests/test_actor_critic_environment_profile_r1.py`

**Implement:** define the Round-2 execution profile that retains real mechanics but disables/default-excludes strategy-owned SL/TP/max-hold/cooldown unless separately authorized. Do not mutate frozen V5.5 authority.

**Dependencies:** BC-009.  
**Gate:** profile cannot silently enable a STRATEGY_PREFERENCE rule.

### BC-011 — Target-action semantic repair: no premature NOOP

**Files:**
- `[NEW] cb16_local_opt/action_contract_r1.py`
- `[NEW] tests/test_action_contract_r1.py`
- `[KEEP] cb16_local_opt/action_contract_r0.py`

**Implement:** same direction/risk means “same nominal request,” not execution NOOP. Remove/replace any transition classification that claims no trade before target quantity is recomputed.

**Dependencies:** BC-004, BC-010.  
**Gate:** same `LONG,0.5` with changed equity/price/envelope is allowed to produce a different target quantity later.

### BC-012 — Dynamic target-exposure authority R1

**Files:**
- `[NEW] cb16_local_opt/target_exposure_r1.py`
- `[NEW] tests/test_target_exposure_r1.py`

**Implement:** recompute permitted target notional/quantity from current authoritative account, price and legal envelope every decision. Risk remains an exposure request, not confidence or a cached quantity.

**Dependencies:** BC-011.  
**Gate:** changed equity/price/legal cap changes quantity exactly when the frozen mapping predicts.

### BC-013 — Supervisor legality vs execution-feasibility separation

**Files:**
- `[NEW] cb16_local_opt/actor_critic_supervisor_r1.py`
- `[NEW] tests/test_actor_critic_supervisor_r1.py`

**Implement:** Supervisor applies termination, legal directions and hard risk caps without inferring quantity increase/decrease solely from the scalar risk. Preserve nominal request and reason codes.

**Dependencies:** BC-011–012.  
**Gate:** equal risk with increased required quantity is not falsely labeled “same exposure”; reduced quantity is not blocked for lack of new margin.

### BC-014 — Post-sizing execution feasibility contract

**Files:**
- `[NEW] cb16_local_opt/execution_feasibility_r0.py`
- `[NEW] tests/test_execution_feasibility_r0.py`

**Implement:** using current position and freshly computed target quantity, determine actual delta, added margin requirement, quantity/notional legality and maintenance/liquidation feasibility. No profitability/ATR/stop heuristic.

**Dependencies:** BC-012–013.  
**Gate:** feasibility is a deterministic function of current account + target + mechanical authority.

### BC-015 — Reduce/close mechanical reachability

**Files:**
- `[MODIFY] tests/test_execution_feasibility_r0.py`
- `[MODIFY] tests/test_actor_critic_supervisor_r1.py`

**Implement/test:** mechanically legal reduction/close remains reachable even when no new margin is available. Closing must never require entry-only risk capacity.

**Dependencies:** BC-014.  
**Gate:** fixtures prove reduce/close succeeds under zero new-exposure capacity where mechanically legal.

### BC-016 — Two-phase reversal with re-evaluated second leg

**Files:**
- `[NEW] cb16_local_opt/reversal_execution_r1.py`
- `[NEW] tests/test_reversal_execution_r1.py`

**Implement:** close current side -> authoritative intermediate flat account -> recompute target sizing and feasibility -> attempt opposite open. Record both legs. If open fails, actual final state is flat and must not be reported as reversed.

**Dependencies:** BC-014–015.  
**Gate:** second-leg rejection leaves an auditable flat state with close costs preserved.

### BC-017 — Complete account-economics schema

**Principles:** P-04/P-06/P-07  
**Files:**
- `[NEW] cb16_local_opt/account_economics_r0.py`
- `[NEW] tests/test_account_economics_r0.py`

**Implement:** define cash, position, cost basis, realized/unrealized PnL, fees, funding, margin/collateral, liabilities/debt, equity, terminal responsibility and external capital flows. Define authoritative equity calculation.

**Dependencies:** BC-009.  
**Gate:** every economic field has one source and conservation meaning; Account6 is explicitly not this ledger.

### BC-018 — Negative-equity and liability preservation

**Files:**
- `[MODIFY] cb16_local_opt/account_economics_r0.py`
- `[MODIFY] tests/test_account_economics_r0.py`

**Implement:** forbid silent `max(0, cash/equity)` unless a versioned legal settlement explicitly converts liability into another field. Preserve economic loss through terminal settlement.

**Dependencies:** BC-017.  
**Gate:** constructed negative-equity fixture remains economically negative until declared responsibility termination; no loss disappears numerically.

### BC-019 — True terminal / responsibility-end contract

**Files:**
- `[NEW] cb16_local_opt/account_terminal_r0.py`
- `[NEW] tests/test_account_terminal_r0.py`

**Implement:** distinguish liquidation event, account trading disabled, remaining debt/fees/settlement, and final economic responsibility termination. Define absorbing state only when future economic increments are genuinely zero.

**Dependencies:** BC-017–018.  
**Gate:** liquidation cannot prematurely zero bootstrap/reward while unresolved economic responsibility exists.

### BC-020 — Corrected target-position execution adapter and BC-A receipt

**Files:**
- `[NEW] cb16_local_opt/actor_critic_physics_adapter_r1.py`
- `[NEW] tests/test_actor_critic_physics_adapter_r1.py`
- `[NEW] authority/rearchitecture_r11/CB16_R11_BC_GATE_A_RECEIPT_V1.json` on PASS
- `[KEEP] cb16_local_opt/actor_critic_physics_adapter_r0.py`

**Implement:** target delta execution using R1 action/sizing/supervisor/feasibility + account economics + policy-neutral profile. Open/resize/reduce/close/reverse/no-op; complete fees/margin/account consequences; no inherited strategy exit rule in the Round-2 path.

**Dependencies:** BC-001–019.  
**Gate:** conservation, negative-equity, dynamic same-risk rebalance, reversal failure, legacy regression and taxonomy tests all pass. **BC-A closes here.**

---

## Phase C — Account identity, observations and state sufficiency

### BC-021 — External capital-flow contract

**Files:**
- `[NEW] cb16_local_opt/capital_flow_r0.py`
- `[NEW] tests/test_capital_flow_r0.py`

**Implement:** explicit initial funding, deposit, withdrawal, transfer and new-account funding events. No implicit recapitalization after failure.

**Dependencies:** BC-017–019.  
**Gate:** new capital always changes a capital-flow ledger and cannot appear as ordinary trading PnL.

### BC-022 — Account identity and restart lineage

**Files:**
- `[NEW] cb16_local_opt/account_lineage_r0.py`
- `[NEW] tests/test_account_lineage_r0.py`

**Implement:** immutable logical account ID, parent/new-account relation, reset reason and capital source. A newly funded account is not the same failed account.

**Dependencies:** BC-021.  
**Gate:** restart cannot inherit old identity without an authorized continuity event.

### BC-023 — Ledger vs policy observation projection

**Files:**
- `[NEW] cb16_local_opt/account_observation_r0.py`
- `[NEW] tests/test_account_observation_r0.py`

**Implement:** build a causal policy-visible account projection from the full ledger while retaining a separate restore/economic state. Record projection version and omitted fields.

**Dependencies:** BC-017–022.  
**Gate:** future/hidden truth cannot leak through observation; restoring from policy projection alone is forbidden unless lossless by contract.

### BC-024 — Actor observation schema

**Files:**
- `[NEW] cb16_local_opt/actor_observation_r0.py`
- `[NEW] tests/test_actor_observation_r0.py`

**Implement:** causal market sensory representation + policy account projection + necessary legal/execution state + optional declared causal memory. The Actor does **not** receive objective countdown `tau=T-t` in the Round-2 baseline.

**Dependencies:** BC-023.  
**Gate:** future-poison and objective-T changes do not alter the Actor observation at a fixed causal state.

### BC-025 — Critic observation schema

**Files:**
- `[NEW] cb16_local_opt/critic_observation_r0.py`
- `[NEW] tests/test_critic_observation_r0.py`

**Implement:** causal state plus account information required for value estimation; may include normalized equity/E_ref and known objective remainder `tau=T-t`. No future market information.

**Dependencies:** BC-023.  
**Gate:** Critic can receive tau while Actor cannot; future market perturbation leaves current input unchanged.

### BC-026 — Actor/Critic information firewall

**Files:**
- `[NEW] tests/test_actor_critic_information_firewall_r0.py`

**Implement/test:** exact field whitelist, tau firewall, future-outcome poison tests, no Teacher/future tensors in Actor/Critic online inputs.

**Dependencies:** BC-024–025.  
**Gate:** any forbidden field fails qualification.

### BC-027 — Legal/execution-state observation projection

**Files:**
- `[NEW] cb16_local_opt/execution_observation_r0.py`
- `[NEW] tests/test_execution_observation_r0.py`

**Implement:** expose only contemporaneously known legality/resources needed for policy decisions, without leaking future execution or encoding a strategy recommendation.

**Dependencies:** BC-013–014, BC-024.  
**Gate:** policy can distinguish legal from illegal choices while profitability remains the model’s problem.

### BC-028 — Observation normalization and scale identity

**Files:**
- `[NEW] cb16_local_opt/observation_normalization_r0.py`
- `[NEW] tests/test_observation_normalization_r0.py`

**Implement:** versioned normalization; avoid asset-name/absolute-money identity shortcuts while preserving real effects of min quantity, fees, leverage and capacity. Bind normalizer identity to trajectories/checkpoints.

**Dependencies:** BC-024–027.  
**Gate:** scale-equivariant fixtures are equivalent only where mechanics are truly scale-equivariant; non-equivariant exchange constraints remain observable.

### BC-029 — State alias detector

**Files:**
- `[NEW] cb16_local_opt/state_sufficiency_r0.py`
- `[NEW] tests/test_state_sufficiency_r0.py`

**Implement:** detect known-answer pairs where policy-visible observation is identical but authoritative causal state implies different optimal actions. Classify as `REPRESENTATION_GAP`, not optimization failure.

**Dependencies:** BC-023–028.  
**Gate:** deliberately aliased fixture is detected.

### BC-030 — Account-dependent known-answer state test

**Files:**
- `[MODIFY] tests/test_state_sufficiency_r0.py`

**Implement/test:** same market observation, two analytically chosen account states with provably different optimal actions; confirm observation can represent the necessary distinction. Do not require arbitrary different accounts to always emit different actions.

**Dependencies:** BC-029.  
**Gate:** test either proves distinguishability or returns representation gap before learner scale-up.

### BC-031 — Optional policy-memory contract

**Files:**
- `[NEW] cb16_local_opt/policy_memory_r0.py`
- `[NEW] tests/test_policy_memory_r0.py`

**Implement:** a versioned causal memory interface that is disabled by default. Activate only if state-sufficiency evidence requires history beyond the frozen feed-forward observation. Memory must be derived from causal prefixes and recoverable across pause/restart.

**Dependencies:** BC-029–030.  
**Gate:** no RNN/hidden-state addition merely to hide an unresolved observation bug; disabled mode remains fully valid.

### BC-032 — Frozen sensory / trainable path ownership

**Principles:** P-08  
**Files:**
- `[NEW] cb16_local_opt/gradient_ownership_r0.py`
- `[NEW] tests/test_gradient_ownership_r0.py`

**Implement:** declare frozen market organs vs trainable account stem/fusion/Actor/Critic parameters. Verify zero gradient/mutation on frozen organs and nonzero eligible gradient paths on trainable components in known-answer fixtures.

**Dependencies:** BC-024–031.  
**Gate:** frozen and trainable ownership are both positively tested.

### BC-033 — Observation/state qualification compiler

**Files:**
- `[NEW] cb16_local_opt/state_qualification_r0.py`
- `[NEW] tests/test_state_qualification_r0.py`

**Implement:** compile BC-021–032, including unresolved representation gap status. A representation gap cannot be converted into PASS by increasing model size.

**Dependencies:** BC-021–032.  
**Gate:** machine-readable state qualification result.

---

## Phase D — Stochastic Actor / Central Brain

### BC-034 — Stochastic Actor policy API

**Files:**
- `[NEW] cb16_local_opt/actor_policy_r1.py`
- `[NEW] tests/test_actor_policy_r1.py`

**Implement:** `sample`, `log_prob`, `deterministic_action`, distribution identity and device/dtype contracts. Inspect but do not depend on the unmerged AC-015 candidate.

**Dependencies:** BC-A, BC-024–033.  
**Gate:** sampling/scoring/evaluation APIs are distinct and fail closed on incompatible observation versions.

### BC-035 — Categorical direction distribution

**Files:** `[MODIFY] actor_policy_r1.py`, `[MODIFY] tests/test_actor_policy_r1.py`

**Implement:** stable SHORT/FLAT/LONG categorical probabilities and finite log-probabilities for extreme logits.

**Dependencies:** BC-034.  
**Gate:** probabilities normalize and sampled frequencies match known probabilities within fixed statistical tolerance.

### BC-036 — Conditional bounded risk distribution

**Files:** same as BC-035.

**Implement:** LONG/SHORT risk on `[0,1]`, conditioned on sampled direction. FLAT uses the exact zero-risk point contract.

**Dependencies:** BC-035.  
**Gate:** legal samples bounded; exact likelihood reconstructable.

### BC-037 — Endpoint point-mass semantics

**Files:** same as BC-035.

**Implement:** explicit mixed distribution for any 0/1 masses, or explicitly disallow endpoint masses. Never treat a continuous density as point probability.

**Dependencies:** BC-036.  
**Gate:** endpoint likelihood known-answer tests.

### BC-038 — Joint nominal-action log probability

**Files:** same as BC-035.

**Implement:** `log p(direction) + log p(risk|direction)` under exact mixture semantics; FLAT does not add meaningless risk density.

**Dependencies:** BC-035–037.  
**Gate:** sampled `log_mu` exactly matches later recomputation under frozen behavior checkpoint.

### BC-039 — Actor RNG and provenance

**Files:**
- `[NEW] cb16_local_opt/policy_rng_r1.py`
- `[NEW] tests/test_policy_rng_r1.py`

**Implement:** isolated serializable Actor RNG; deterministic restore; bind RNG stream identity/position to trajectory provenance.

**Dependencies:** BC-034–038.  
**Gate:** restored checkpoint reproduces the exact next sampled action sequence.

### BC-040 — Deterministic evaluation policy identity

**Files:** `[MODIFY] actor_policy_r1.py`, `[MODIFY] tests/test_actor_policy_r1.py`

**Implement:** evaluation action does not consume sampling RNG and is explicitly a different execution policy identity when used for economic evaluation.

**Dependencies:** BC-039.  
**Gate:** deterministic evaluation before/after leaves stochastic sequence unchanged.

### BC-041 — Round-2 Central Brain integration

**Files:**
- `[NEW] cb16_local_opt/actor_critic_brain_r1.py`
- `[NEW] tests/test_actor_critic_brain_r1.py`
- `[KEEP] cb16_local_opt/typed_central_brain_r10.py`

**Implement:** frozen market organs + account/fusion path + stochastic Actor. Bind observation, normalizer, policy, distribution and science hashes.

**Dependencies:** BC-032–040.  
**Gate:** new Brain samples/scores/evaluates without touching legacy Brain semantics.

### BC-042 — Nominal-action likelihood / execution separation

**Files:**
- `[NEW] tests/test_nominal_execution_probability_firewall_r1.py`

**Implement/test:** clamp/reject/partial execution/reversal failure must retain original sampled action and `log_mu`; learner importance ratio never uses permitted/executed action probability fabricated after the fact.

**Dependencies:** BC-013–020, BC-038–041.  
**Gate:** deliberate clamp/reject fixtures preserve nominal likelihood exactly.

### BC-043 — Gate BC-B stochastic-policy qualification

**Files:**
- `[NEW] cb16_local_opt/actor_policy_qualification_r1.py`
- `[NEW] tests/test_actor_policy_qualification_r1.py`
- `[NEW on PASS] authority/rearchitecture_r11/CB16_R11_BC_GATE_B_RECEIPT_V1.json`

**Dependencies:** BC-021–042.  
**Gate:** state, causality, gradient ownership, probability and RNG tests all PASS. **BC-B closes here.**

---

## Phase E — Four clocks and continuous account rollout

### BC-044 — Four-clock contract

**Principles:** P-02/P-04  
**Files:**
- `[NEW] cb16_local_opt/clock_contract_r0.py`
- `[NEW] tests/test_clock_contract_r0.py`

**Implement:** separately represent environment clock, policy decision clock, learning/storage chunk clock and economic objective horizon. No overloaded `done` or timestamp semantics.

**Dependencies:** BC-B.  
**Gate:** changing one clock cannot silently mutate another.

### BC-045 — Policy decision schedule

**Files:**
- `[NEW] cb16_local_opt/decision_clock_r1.py`
- `[NEW] tests/test_decision_clock_r1.py`

**Implement:** exact decision instants and policy-call rules; no first-step-only H72 shortcut.

**Dependencies:** BC-044.  
**Gate:** fixture detects skipped/duplicated unauthorized decisions.

### BC-046 — Environment advancement primitive

**Files:**
- `[NEW] cb16_local_opt/environment_advance_r1.py`
- `[NEW] tests/test_environment_advance_r1.py`

**Implement:** advance market/account mechanics between policy decisions independent of whether a new target order exists. Apply market PnL, funding, mechanical liquidation and approved execution mechanics.

**Dependencies:** BC-010, BC-017–020, BC-044.  
**Gate:** held positions evolve economically even with zero new orders.

### BC-047 — Reject/NOOP must not freeze the world

**Files:** `[MODIFY] tests/test_environment_advance_r1.py`

**Implement/test:** Supervisor reject, same executable target, below-minimum target and policy abstention do not stop environment clock or existing-position consequences.

**Dependencies:** BC-046.  
**Gate:** account PnL/funding/liquidation changes under no-order fixtures exactly as mechanics require.

### BC-048 — Policy-neutral bar lifecycle

**Files:**
- `[NEW] cb16_local_opt/actor_critic_bar_lifecycle_r1.py`
- `[NEW] tests/test_actor_critic_bar_lifecycle_r1.py`

**Implement:** Round-2 bar lifecycle using the BC execution taxonomy/profile. No automatic SL/TP/max-hold/cooldown unless the profile explicitly authorizes them. Preserve genuine liquidation/funding.

**Dependencies:** BC-009–010, BC-046–047.  
**Gate:** strategy-preference poison fixture cannot change Round-2 account path.

### BC-049 — Continuous rollout state machine

**Files:**
- `[NEW] cb16_local_opt/continuous_rollout_r1.py`
- `[NEW] tests/test_continuous_rollout_r1.py`

**Implement:** market/account -> Actor -> nominal/log_mu -> Supervisor -> sizing -> feasibility -> execution -> environment progression -> next account -> next Actor decision, repeatedly.

**Dependencies:** BC-020, BC-041–048.  
**Gate:** multi-decision fixture proves each policy decision uses the actual previous account consequence.

### BC-050 — Strict within-account serialization

**Files:** `[MODIFY] continuous_rollout_r1.py`, `[MODIFY] tests/test_continuous_rollout_r1.py`

**Implement:** one logical account may not process state transitions out of chronological order or concurrently.

**Dependencies:** BC-049.  
**Gate:** deliberate race/reorder fails closed.

### BC-051 — Cross-account asynchronous execution

**Files:**
- `[NEW] cb16_local_opt/actor_critic_async_binding_r1.py`
- `[NEW] tests/test_actor_critic_async_binding_r1.py`

**Implement:** reuse compatible async/orchestrator infrastructure while preserving independent account serialization and deterministic identities.

**Dependencies:** BC-050.  
**Gate:** no cross-account state contamination under concurrent fixture.

### BC-052 — Boundary taxonomy R1

**Files:**
- `[NEW] cb16_local_opt/episode_boundary_r1.py`
- `[NEW] tests/test_episode_boundary_r1.py`

**Implement:** at minimum `ECONOMIC_TERMINAL`, `TRADING_DISABLED_PENDING_SETTLEMENT`, `OBJECTIVE_T`, `CHUNK_TRUNCATION`, `PAUSE`, `PROCESS_FAILURE`, `DATA_END`. Distinguish physical/economic/account/computation meanings.

**Dependencies:** BC-019, BC-044–051.  
**Gate:** every stop has one explicit typed reason; no generic done ambiguity.

### BC-053 — Exact pause/resume continuation

**Files:**
- `[NEW] cb16_local_opt/rollout_resume_r1.py`
- `[NEW] tests/test_rollout_resume_r1.py`

**Implement:** seal/restore full account truth, policy identity, policy memory if enabled, RNG, clocks and pending settlement. No repeated/skipped decision.

**Dependencies:** BC-039, BC-049–052.  
**Gate:** uninterrupted vs pause/resume trajectories are semantically identical.

### BC-054 — Objective horizon and data-end semantics

**Files:** `[MODIFY] episode_boundary_r1.py`, `[MODIFY] tests/test_episode_boundary_r1.py`

**Implement:** objective T ends a comparison target without forcing physical flat/account death; data exhaustion is unresolved/truncated realized evidence, not terminal success/failure.

**Dependencies:** BC-052.  
**Gate:** finite objective and physical account continuity coexist in fixtures.

### BC-055 — Generation switch during continuous account

**Files:**
- `[NEW] cb16_local_opt/collector_generation_switch_r1.py`
- `[NEW] tests/test_collector_generation_switch_r1.py`

**Implement:** registered policy switch preserves account/clock/environment truth and records exact transition ownership before/after switch.

**Dependencies:** BC-049–054.  
**Gate:** switching checkpoint does not reset capital, position or pending consequence.

### BC-056 — Gate BC-C collector qualification

**Files:**
- `[NEW] cb16_local_opt/collector_qualification_r1.py`
- `[NEW] tests/test_collector_qualification_r1.py`
- `[NEW on PASS] authority/rearchitecture_r11/CB16_R11_BC_GATE_C_RECEIPT_V1.json`

**Dependencies:** BC-044–055.  
**Gate:** four clocks, repeated policy calls, account continuity, environment advancement, pause/resume and generation switch all PASS.

---

## Phase F — Immutable experience, views and replay admission

### BC-057 — Round-2 Transition schema

**Files:**
- `[NEW] cb16_local_opt/trajectory_schema_r1.py`
- `[NEW] tests/test_trajectory_schema_r1.py`

**Required content:** account/generation/policy IDs; Actor observation; nominal action; log_mu; permission; target sizing; feasibility; execution legs; pre/post full-account economic hashes; E_t/E_t+1/E_ref; reward slot; next observation; all four clock identities; boundary; market lineage; science versions; RNG provenance; source classification.

**Dependencies:** BC-C.  
**Gate:** transition is self-auditable without mutable runtime memory.

### BC-058 — Ordered Sequence schema

**Files:** `[MODIFY] trajectory_schema_r1.py`, `[MODIFY] tests/test_trajectory_schema_r1.py`

**Implement:** immutable ordered transitions with continuity hashes and typed chunk boundaries; sequences never cross an unauthorized account reset.

**Dependencies:** BC-057.  
**Gate:** reorder/drop/duplicate/cross-account composition fails.

### BC-059 — Raw fact store adapter

**Files:**
- `[NEW] cb16_local_opt/trajectory_lake_r1.py`
- `[NEW] tests/test_trajectory_lake_r1.py`
- `[REUSE] cb16_local_opt/sharded_experience_lake.py`

**Implement:** content-addressed immutable storage of raw facts; exactly-once object identity; no algorithmic filtering at fact-ingest layer.

**Dependencies:** BC-057–058.  
**Gate:** semantic round-trip stable; same-ID/different-content conflict hard fails.

### BC-060 — Failure and terminal persistence

**Files:** `[MODIFY] tests/test_trajectory_lake_r1.py`

**Implement/test:** liquidation, negative equity/liability, reversal-open failure, rejected action and ordinary paths all persist. Terminal serialization must not require constructing an impossible “healthy” observation.

**Dependencies:** BC-059.  
**Gate:** removing or censoring any protected failure fixture is detected.

### BC-061 — Fork / mother-state lineage

**Files:**
- `[NEW] cb16_local_opt/experience_lineage_r1.py`
- `[NEW] tests/test_experience_lineage_r1.py`

**Implement:** parent state, branch action, continuation policy and market future lineage for counterfactual/demo branches. Do not count branches as independent market futures.

**Dependencies:** BC-059–060.  
**Gate:** fork identity and shared-market lineage are queryable.

### BC-062 — Exactly-once transition commit

**Files:**
- `[NEW] cb16_local_opt/trajectory_transaction_r1.py`
- `[NEW] tests/test_trajectory_transaction_r1.py`

**Implement:** crash-safe commit IDs/fencing so retry cannot append the same transition twice; distinguish planned replay from duplicate environment transaction.

**Dependencies:** BC-059.  
**Gate:** crash-before/after-commit fixtures produce one fact only.

### BC-063 — Four evidence views

**Principles:** P-07/P-10  
**Files:**
- `[NEW] cb16_local_opt/experience_views_r1.py`
- `[NEW] tests/test_experience_views_r1.py`

**Implement:** explicit views: `RAW_FACT`, `REPLAY_ADMISSIBLE`, `DEMONSTRATION_SELECTED`, `EVALUATION_COHORT`. A fact may belong to multiple views under explicit rules; one view never deletes the raw fact.

**Dependencies:** BC-059–062.  
**Gate:** demonstration winner filtering cannot alter evaluation denominator or raw storage.

### BC-064 — Legacy demonstration/source router

**Files:**
- `[NEW] cb16_local_opt/experience_source_router_r1.py`
- `[NEW] tests/test_experience_source_router_r1.py`

**Implement:** old H72/Teacher/demonstration material remains readable as historical evidence but is excluded from ordinary V-trace unless all required behavior/observation/execution semantics are genuinely compatible. Never synthesize `log_mu`.

**Dependencies:** BC-063.  
**Gate:** incompatible legacy sample is rejected with an explicit reason while remaining queryable as fact/demo evidence.

### BC-065 — Replay semantic compatibility gate

**Files:**
- `[NEW] cb16_local_opt/replay_compat_r1.py`
- `[NEW] tests/test_replay_compat_r1.py`

**Implement:** require compatible action distribution, observation/normalizer, account economics, execution profile, reward, boundaries, policy likelihood and lineage.

**Dependencies:** BC-057–064.  
**Gate:** every deliberate version mismatch fixture fails closed.

### BC-066 — Replay support / off-policy health diagnostics

**Files:**
- `[NEW] cb16_local_opt/replay_support_r0.py`
- `[NEW] tests/test_replay_support_r0.py`

**Implement:** importance-ratio distribution, clipping fraction, effective support/ESS-like diagnostics, generation contribution and policy-distance indicators. Admission/weighting may respond to mathematical support, **not sample age alone**.

**Dependencies:** BC-038, BC-065.  
**Gate:** extreme `mu`/`pi` mismatch is reported; old-but-well-supported experience is not rejected merely for age. **BC-C experience gate is now complete.**

---

## Phase G — Reward, Critic, V-trace and learner

### BC-067 — Arithmetic-equity reward R1

**Files:**
- `[NEW] cb16_local_opt/reward_r1.py`
- `[NEW] tests/test_reward_r1.py`

**Implement:** `r_t=(E_{t+1}-E_t)/E_ref`, fixed declared `E_ref` for an objective comparison, no log/sign/clipping/drawdown substitution.

**Dependencies:** BC-017–020, BC-057.  
**Gate:** positive/zero/negative/negative-equity transitions match authoritative ledger delta.

### BC-068 — Reward accounting and telescoping qualification

**Files:** `[MODIFY] tests/test_reward_r1.py`

**Implement/test:** fees/funding/realized/unrealized PnL counted once through equity; `sum r = (E_T-E_0)/E_ref` over complete finite path including failure/liability.

**Dependencies:** BC-067.  
**Gate:** hand-ledger fixtures telescope within frozen tolerance.

### BC-069 — Boundary-specific bootstrap calculator

**Files:**
- `[NEW] cb16_local_opt/value_bootstrap_r1.py`
- `[NEW] tests/test_value_bootstrap_r1.py`

**Implement:** zero only at true objective/economic terminal as contract says; bootstrap chunk/pause; unresolved data-end is not realized terminal; pending settlement is not zeroed early.

**Dependencies:** BC-019, BC-052–054, BC-067–068.  
**Gate:** split/unsplit known-answer paths agree.

### BC-070 — Separate Critic model

**Files:**
- `[NEW] cb16_local_opt/critic_value_r1.py`
- `[NEW] tests/test_critic_value_r1.py`

**Implement:** scalar expected future arithmetic-return value; no old Teacher semantics; separate trainable parameters from Actor by default.

**Dependencies:** BC-025–033, BC-069.  
**Gate:** shape/dtype/device and deterministic forward tests.

### BC-071 — Critic causal/tau contract

**Files:** `[MODIFY] critic_value_r1.py`, `[MODIFY] tests/test_critic_value_r1.py`

**Implement:** Critic sees only causal state plus allowed `tau`; future-poison invariance; Actor remains tau-free.

**Dependencies:** BC-070.  
**Gate:** future poison and Actor/critic field firewall pass.

### BC-072 — Mean-value regression objective

**Files:** `[MODIFY] critic_value_r1.py`, `[MODIFY] tests/test_critic_value_r1.py`

**Implement:** mean target consistent with arithmetic expectation; no silent quantile, pessimistic twin-Q minimum or robust-center substitution.

**Dependencies:** BC-070–071.  
**Gate:** exact-value toy converges under preregistered budget/tolerance.

### BC-073 — V-trace recurrence R1

**Files:**
- `[NEW] cb16_local_opt/vtrace_r1.py`
- `[NEW] tests/test_vtrace_r1.py`

**Implement:** exact log_mu/log_pi ratios, rho/c clipping, backward recurrence, masks and gamma semantics for bounded target. Record chosen clipping constants explicitly.

**Dependencies:** BC-038, BC-065–072.  
**Gate:** hand-computed sequence, pi==mu and clipping boundary known answers PASS.

### BC-074 — Actor policy-gradient loss

**Files:**
- `[NEW] cb16_local_opt/actor_critic_loss_r1.py`
- `[NEW] tests/test_actor_critic_loss_r1.py`

**Implement:** policy loss from frozen V-trace advantage/targets. Any entropy term must be explicit and preregistered; do not silently stack PPO/GAE/SAO corrections.

**Dependencies:** BC-073.  
**Gate:** gradient sign matches hand-built policy cases.

### BC-075 — Sequence replay sampler and weighting

**Files:**
- `[NEW] cb16_local_opt/sequence_replay_r1.py`
- `[NEW] tests/test_sequence_replay_r1.py`

**Implement:** compatible sequences only; deterministic RNG; explicit sampling probabilities/weights; no survivor-only filtering. Training sampling weights do not redefine economic cohort weights.

**Dependencies:** BC-063–066, BC-073.  
**Gate:** enumerated pool matches declared sampling distribution.

### BC-076 — Actor–Critic learner runtime

**Files:**
- `[NEW] cb16_local_opt/actor_critic_training_r1.py`
- `[NEW] tests/test_actor_critic_training_r1.py`
- `[KEEP] cb16_local_opt/training_runtime_r11.py`

**Implement:** separate Actor/Critic optimizers, sequence updates, explicit gradient ownership, policy/value/support metrics and no old Teacher/Student mutation.

**Dependencies:** BC-032, BC-070–075.  
**Gate:** fixed-seed update produces deterministic expected parameter deltas; frozen organs unchanged.

### BC-077 — Exactly-once learner transaction

**Files:**
- `[NEW] cb16_local_opt/learner_transaction_r0.py`
- `[NEW] tests/test_learner_transaction_r0.py`

**Implement:** unique update transaction binds replay sample IDs, parent checkpoint, optimizer state and update counter. Crash before commit retries safely; crash after committed gradient must not apply it twice.

**Dependencies:** BC-075–076.  
**Gate:** before-gradient / after-gradient-before-receipt / after-receipt crash fixtures result in exactly one committed update.

### BC-078 — Full learner checkpoint and exact recovery / Gate BC-D

**Files:**
- `[NEW] cb16_local_opt/actor_critic_checkpoint_r1.py`
- `[NEW] tests/test_actor_critic_checkpoint_r1.py`
- `[REUSE] cb16_local_opt/checkpoint_store_r11.py`
- `[NEW on PASS] authority/rearchitecture_r11/CB16_R11_BC_GATE_D_RECEIPT_V1.json`

**Bundle:** Actor/Critic, both optimizers, Actor/learner RNGs, update transaction state, replay snapshot, normalizer/science identity, generation identity, collector/account continuation identity where applicable.

**Gate:** restart yields same next sampled action and same next committed gradient transaction. **BC-D closes here.**

---

## Phase H — Generation, replay evolution and historical knowledge retention

### BC-079 — Fixed behavior-checkpoint collection unit

**Principles:** P-09  
**Files:**
- `[NEW] cb16_local_opt/generation_collection_r1.py`
- `[NEW] tests/test_generation_collection_r1.py`

**Implement:** freeze behavior policy checkpoint per registered collection unit; learner trains a separate copy and cannot mutate running weights in place.

**Dependencies:** BC-056, BC-078.  
**Gate:** running policy hash is invariant during a collection unit.

### BC-080 — Generation switch with account continuity

**Files:**
- `[NEW] cb16_local_opt/actor_critic_generation_binding_r1.py`
- `[NEW] tests/test_actor_critic_generation_binding_r1.py`
- `[REUSE] concepts from continuous_generation_binding_r0.py`

**Implement:** exact parent/child policy lineage; switch preserves logical account and clocks; transition at switch boundary records correct behavior identity.

**Dependencies:** BC-055, BC-079.  
**Gate:** account hashes/ledger continuity survive controlled generation switch.

### BC-081 — Mixed-generation trajectory attribution

**Files:** `[MODIFY] trajectory_schema_r1.py`, `[MODIFY] tests/test_trajectory_schema_r1.py`

**Implement:** every transition retains its own behavior policy; a long account path is not retroactively attributed to final checkpoint.

**Dependencies:** BC-080.  
**Gate:** multi-generation trajectory query returns exact transition ownership.

### BC-082 — Historical/new replay mixture ledger

**Files:**
- `[NEW] cb16_local_opt/replay_mixture_r0.py`
- `[NEW] tests/test_replay_mixture_r0.py`

**Implement:** explicitly record training mixture by source/generation/support; recent experience may calibrate current behavior without deleting old compatible facts. No age-expiration rule by default.

**Dependencies:** BC-066, BC-075, BC-081.  
**Gate:** changing mixture is a versioned training decision and visible in checkpoint lineage.

### BC-083 — A→B→A continual-retention known-answer environment

**Files:**
- `[NEW] cb16_local_opt/continual_retention_r0.py`
- `[NEW] tests/test_continual_retention_r0.py`

**Implement:** analytically solvable tasks/regimes A and B; train A, adapt on B, return to A. Measure adaptation and retention under fixed budgets for no-replay vs compatible historical replay.

**Dependencies:** BC-076, BC-082.  
**Gate:** reports both B adaptation and A retention; no post-hoc weighting rescue.

### BC-084 — No handcrafted recurrence/regime activation sentinel

**Files:**
- `[NEW] tests/test_no_handcrafted_regime_activation_r0.py`

**Implement/test:** Round-2 policy path must not contain rule-based calendar/cycle/regime switches introduced to make A→B→A pass. Historical recurrence should be expressed through learned weights/memory/replay, not manual activation tables.

**Dependencies:** BC-083.  
**Gate:** explicit prohibited activation hooks fail qualification.

### BC-085 — Gate BC-E generation/retention receipt

**Files:**
- `[NEW] cb16_local_opt/generation_retention_qualification_r0.py`
- `[NEW] tests/test_generation_retention_qualification_r0.py`
- `[NEW on PASS] authority/rearchitecture_r11/CB16_R11_BC_GATE_E_RECEIPT_V1.json`

**Dependencies:** BC-079–084.  
**Gate:** fixed-running-policy, account continuity, attribution, replay-mixture provenance and A→B→A known-answer evidence compiled. **BC-E closes here.**

---

## Phase I — Known-answer and full closed-loop qualification

### BC-086 — Core mathematical/contract qualification suite

**Files:**
- `[NEW] tests/test_bc_round2_math_qualification_r0.py`

**Must cover:** account conservation; negative liability; dynamic same-risk rebalance; nominal/executed probability firewall; four clocks; reward telescoping; boundary bootstrap; V-trace; recovery; replay weighting/support; future-poison; gradient ownership.

**Dependencies:** BC-001–085.  
**Gate:** all preregistered cases green; no skipped/xfail required case.

### BC-087 — Account-dependent policy toy

**Files:**
- `[NEW] cb16_local_opt/toy_envs_bc_r0.py`
- `[NEW] tests/test_toy_envs_bc_r0.py`

**Implement:** same market observation, analytically different account states with known different optimal actions. Distinguish representation failure from optimization failure.

**Dependencies:** BC-086.  
**Gate:** policy learns known account dependence under frozen seed/budget tolerance.

### BC-088 — Delayed consequence and horizon toy

**Files:** `[MODIFY] toy_envs_bc_r0.py`, `[MODIFY] tests/test_toy_envs_bc_r0.py`

**Implement:** short-term gain followed by delayed loss; separate case where increasing objective horizon genuinely reverses optimal ranking.

**Dependencies:** BC-087.  
**Gate:** learner follows analytical long-horizon answer, not immediate reward.

### BC-089 — High-bankruptcy / higher-arithmetic-expectation toy

**Files:** same toy files.

**Implement:** exact distribution where risky policy has higher bankruptcy frequency and higher arithmetic expectation. Complete failures remain in training/evaluation.

**Dependencies:** BC-087.  
**Gate:** ranking chooses higher expected arithmetic return; tail diagnostics do not override objective.

### BC-090 — Off-policy correction known-answer toy

**Files:** same toy files.

**Implement:** known behavior `mu` and target `pi`; compare correct V-trace, same-policy path and deliberately uncorrected replay.

**Dependencies:** BC-073, BC-087.  
**Gate:** recurrence matches analytical expectation and support diagnostics identify extreme mismatch.

### BC-091 — Execution-policy contamination toy

**Files:**
- `[NEW] tests/test_execution_policy_contamination_r0.py`

**Implement/test:** construct a path where fixed SL/TP/max-hold would change the outcome relative to policy-neutral execution. Prove Round-2 lane does not inherit those historical strategy choices; prove genuine liquidation still operates.

**Dependencies:** BC-010, BC-020, BC-048.  
**Gate:** contamination fixture detects accidental historical-rule leakage.

### BC-092 — Full closed-loop canary and Gate BC-F

**Files:**
- `[NEW] cb16_local_opt/actor_critic_closed_loop_qualification_r1.py`
- `[NEW] tests/test_actor_critic_closed_loop_qualification_r1.py`
- `[NEW] authority/rearchitecture_r11/CB16_R11_BC_ROUND2_QUALIFICATION_SPEC_V1.json`
- `[NEW on PASS] authority/rearchitecture_r11/CB16_R11_BC_GATE_F_RECEIPT_V1.json`

**Canary chain:** policy decision -> real execution -> environment advance -> next account observation -> raw trajectory -> replay -> learner update -> new checkpoint -> generation switch -> same logical account -> new policy actually acts. Include at least one terminal/failure path and one crash/recovery path.

**Dependencies:** BC-086–091.  
**Gate:** one machine-readable CLOSED_LOOP + KNOWN_ANSWER receipt. **No real historical scale-up before PASS.**

---

## Phase J — Economic evaluation, promotion and Round-2 closure

### BC-093 — Economic cohort / common horizon contract

**Principles:** P-06/P-10  
**Files:**
- `[NEW] cb16_local_opt/economic_cohort_r1.py`
- `[NEW] tests/test_economic_cohort_r1.py`

**Implement:** preregister account/capital/start-state distribution, common objective horizon T, weighting and realized-vs-estimated status. Objective T does not force physical close.

**Dependencies:** BC-F.  
**Gate:** cohort membership/weights/horizon cannot change after scoring starts.

### BC-094 — B&H and FLAT baseline engine

**Files:**
- `[NEW] cb16_local_opt/economic_baselines_r1.py`
- `[NEW] tests/test_economic_baselines_r1.py`

**Implement:** B&H-to-common-T and all-FLAT/no-position on identical cohort/data/capital mechanics. Baselines continue to common T even if Trader account terminates earlier where mathematically defined.

**Dependencies:** BC-093.  
**Gate:** identical cohort/data lineage across Trader/B&H/FLAT.

### BC-095 — Capital denominator and restart fairness

**Files:**
- `[NEW] cb16_local_opt/economic_capital_accounting_r0.py`
- `[NEW] tests/test_economic_capital_accounting_r0.py`

**Implement:** initial capital and every new-account capital injection are explicit. Multiple restarts cannot be collapsed into one lucky surviving account or treated as free replenishment.

**Dependencies:** BC-021–022, BC-093.  
**Gate:** injected capital changes denominator/identity exactly as contract declares.

### BC-096 — Failure/tail/outcome-distribution reporting

**Files:**
- `[NEW] cb16_local_opt/economic_evaluator_r1.py`
- `[NEW] tests/test_economic_evaluator_r1.py`

**Implement:** arithmetic return primary; all included accounts/failures; liquidation/debt/termination/unfinished classifications; mean, median, quantiles, tails and survival diagnostics clearly secondary.

**Dependencies:** BC-093–095.  
**Gate:** deleting a failed account changes protected expected-return fixture and fails.

### BC-097 — Mixed-generation evaluation identity

**Files:**
- `[MODIFY] economic_evaluator_r1.py`
- `[NEW] tests/test_mixed_generation_evaluation_r0.py`

**Implement:** distinguish a single frozen checkpoint evaluation from a deployment/generation-chain policy. An account run by G3→G4→G5 is not automatically credited entirely to G5.

**Dependencies:** BC-080–081, BC-096.  
**Gate:** report declares exact policy object being evaluated.

### BC-098 — Champion/Challenger promotion contract

**Files:**
- `[NEW] cb16_local_opt/economic_promotion_r1.py`
- `[NEW] tests/test_economic_promotion_r1.py`

**Implement:** same cohort/horizon/evaluator; arithmetic expected return and baseline deltas reported. If exact master rule for B&H-vs-FLAT conflict is still owner-open, return `UNRESOLVED_OWNER_DECISION`; never invent Sharpe/log/drawdown ranking.

**Dependencies:** BC-094–097.  
**Gate:** measurement can PASS while promotion remains unresolved; no silent auto-promotion.

### BC-099 — Economic Gate receipt

**Files:**
- `[NEW] cb16_local_opt/economic_qualification_r1.py`
- `[NEW] tests/test_economic_qualification_r1.py`
- `[NEW on PASS] authority/rearchitecture_r11/CB16_R11_BC_ECONOMIC_GATE_RECEIPT_V1.json`

**Dependencies:** BC-093–098.  
**Gate:** common cohort/T/capital/baselines/failures/policy identity are machine-bound.

### BC-100 — Stage-4 infrastructure adoption

**Files:**
- `[NEW/ADAPT] cb16_local_opt/bc_round2_infrastructure_binding_r0.py`
- `[NEW] tests/test_bc_round2_infrastructure_binding_r0.py`
- `[REUSE] Stage-4 state roots/fencing/recovery/orchestration`

**Implement:** adopt existing infrastructure only where compatible; preserve single-account serialization, durable state, lease/fencing and recovery semantics.

**Dependencies:** BC-F, BC-078, BC-085.  
**Gate:** infrastructure does not alter science semantics and fail-closed ownership tests pass.

### BC-101 — Round-2 CI qualification workflow

**Files:**
- `[NEW] .github/workflows/cb16-r11-bc-round2-qualification-r0.yml`

**Implement:** repo guard/preflight -> historical/R0 regressions -> BC gates -> no FINAL/fresh access. Use appropriate hosted/self-hosted runners without parallel mutation of the same canonical account/state root.

**Dependencies:** BC-099–100.  
**Gate:** clean workflow produces complete artifacts and gate receipts; no required test skipped.

### BC-102 — FINAL/fresh-data firewall proof

**Files:**
- `[NEW] tests/test_bc_round2_data_firewall_r0.py`
- `[ADAPT] CI workflow from BC-101`

**Implement/test:** qualification cannot read/open FINAL or download fresh data. Fail closed on path/config/network attempt where detectable.

**Dependencies:** BC-101.  
**Gate:** explicit firewall artifact in CI.

### BC-103 — Canonical current-state / architecture update

**Files:**
- `[MODIFY] docs/CURRENT_STATE.md`
- `[MODIFY] docs/ARCHITECTURE_MAP.md`
- `[MODIFY] docs/COMPONENT_REQUIREMENTS.md` only if behavior requirements changed by an owner decision

**Implement:** exact main SHA, deepest proven evidence level, active Round-2 lane, historical/R0 compatibility lane and remaining unresolved gaps. Do not write planned work as completed.

**Dependencies:** BC-001–102.  
**Gate:** docs match live code/receipts.

### BC-104 — Decisions / open questions / navigation closure

**Files:**
- `[MODIFY] docs/DECISIONS.md`
- `[MODIFY] docs/OPEN_QUESTIONS.md`
- `[MODIFY] docs/README.md`
- `[MODIFY] AGENTS.md`

**Implement:** close only owner-resolved questions; retain unresolved T/master ranking/deployment choices; make BC Round 2 the current implementation-plan entry and AC TODO historical/reference material.

**Dependencies:** BC-103.  
**Gate:** navigation no longer requires a new Agent to reconstruct AC task history.

### BC-105 — Final Round-2 compiler / canonical readiness receipt

**Files:**
- `[NEW] cb16_local_opt/bc_round2_final_gate_r0.py`
- `[NEW] tests/test_bc_round2_final_gate_r0.py`
- `[NEW on PASS] authority/rearchitecture_r11/CB16_R11_BC_ROUND2_FINAL_RECEIPT_V1.json`
- `[MODIFY] docs/R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md`

**Compile:** BC-A through BC-G receipts, exact main SHA, science identities, historical regressions, FINAL/fresh firewalls, known-answer receipt, closed-loop receipt, economic receipt, unresolved owner decisions and evidence-level ceiling.

**Dependencies:** BC-001–104.  
**Gate:** no canonical declaration unless all required BC gates PASS. An unresolved promotion master rule may limit evidence to ECONOMIC measurement without authorizing promotion. **BC-G / Round 2 closes here.**

---

# 7. Dependency DAG

```text
BC-001..008      baseline / semantic identity / receipts
      |
      v
BC-009..020      execution + full account economics repair        [Gate BC-A]
      |
      v
BC-021..033      account identity / observation / state sufficiency
      |
      v
BC-034..043      stochastic Actor / Brain / probability           [Gate BC-B]
      |
      v
BC-044..056      four clocks / environment / continuous rollout   [collector half of BC-C]
      |
      v
BC-057..066      immutable facts / views / replay admission        [Gate BC-C]
      |
      v
BC-067..078      reward / Critic / V-trace / learner / recovery    [Gate BC-D]
      |
      v
BC-079..085      generation / replay mixture / A-B-A retention     [Gate BC-E]
      |
      v
BC-086..092      known-answer + full closed-loop canary            [Gate BC-F]
      |
      v
BC-093..099      economic measurement / promotion boundary
      |
BC-100..105      infrastructure / CI / docs / final receipt        [Gate BC-G]
```

Parallel work is allowed only where all declared dependencies are already merged to main. No sibling branch may be treated as a hidden dependency.

---

# 8. Definition of Done for every BC task

A BC task is DONE only when all applicable conditions hold:

- declared file scope respected or expansion explicitly documented;
- task starts from verified live main and records base/head SHA;
- semantic version changes are explicit;
- positive known-answer test exists;
- negative/fail-closed case exists;
- exact authority/lineage hashes recorded where relevant;
- no FINAL access;
- no fresh-data download;
- historical frozen and R0 regressions remain green unless the task is an explicitly versioned successor;
- no fabricated behavior likelihood;
- no survivor-only fact deletion;
- no account reset at chunk/pause/checkpoint/generation boundary without an account-lineage event;
- no free capital injection;
- no strategy preference disguised as environment legality;
- no objective substitution;
- result classification and evidence level are explicit.

---

# 9. Scientific no-rescue rules

Once a BC qualification or scientific comparison begins, do not rescue it within the same task by:

- changing `T`, cohort, seed or budget after viewing the result;
- deleting failures or negative liabilities;
- replacing arithmetic return with log return, Sharpe or drawdown utility;
- changing execution profile after trajectories are collected;
- changing `E_ref` after looking at outcomes;
- fabricating log_mu for legacy experience;
- modifying replay support/clipping thresholds after seeing which result wins;
- resetting/re-funding a failed account and calling it the same account;
- adding handcrafted cycle/regime activation logic;
- converting runner/hardware failure into scientific FAIL or PASS;
- calling workflow success economic improvement.

A redesign requires a new version/task and preserved old result.

---

# 10. Owner-open decisions that remain open

BC implementation should proceed on everything not dependent on these choices, but must fail closed at the point a decision is scientifically required.

1. Exact canonical economic horizon `T` and start-state/cohort weighting for formal market ranking.
2. Exact master promotion rule if B&H and FLAT baseline deltas disagree.
3. Any strategic leverage/exposure cap that is a user preference rather than an exchange/resource limit.
4. Whether policy memory beyond the causal current observation is needed; this should first be driven by BC state-sufficiency evidence.
5. Exact update cadence / replay mixture / sequence length / batch size when these become experiment parameters; freeze before the relevant run.
6. Exact future simulated-account promotion/rollback operational authority.

Already closed principles must not be re-asked: complete failure retention, failure participation in long-horizon learning, account continuity, arithmetic expected-return preference, and no handcrafted recurrence engine.

---

# 11. Immediate execution order for the next Agent

Do **not** begin by implementing V-trace or merging the unreviewed AC-015 branch.

Start here:

```text
BC-001 -> BC-002 -> BC-003 -> BC-004 -> BC-005 -> BC-006 -> BC-007 -> BC-008
       -> BC-009 ... BC-020
```

The first Round-2 blocking objective is to prove that one target action against one real account has policy-neutral, conserved, economically complete consequences. Until that is true, new stochastic trajectories would be semantically unstable and should not become canonical replay data.

After Gate BC-A, proceed through state sufficiency and stochastic Actor, then continuous experience, then learner, then continual-retention and closed-loop qualification.

**Current Round-2 state at document creation:**

`AUDITED / BC PLAN FROZEN FOR EXECUTION / MAIN c373d23 / AC R0 INPUTS PRESERVED / BC-A NOT YET STARTED`
