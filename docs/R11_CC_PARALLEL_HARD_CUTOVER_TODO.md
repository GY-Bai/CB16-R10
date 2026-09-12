# CB16 R11 CC — Four-Thread Parallel Hard-Cutover TODO

**Status:** CLOSED / IMPLEMENTED + INTEGRATED / historical execution plan  
**Frozen implementation baseline:** `main@89d62bf966f476e598f0e2f5c5e8e03c15a8db51`  
**Planning-doc note:** CC handoff documents were committed after this SHA so agents could read them from main; implementation branches started from the frozen baseline.  
**Audit date:** 2026-09-12  
**Execution model:** four independent sub-agent task packages + one final integration pass  
**Performance policy:** **hard cutover / performance first / no legacy performance compatibility layer**  

> **Closure notice — 2026-09-12:** The four CC threads and final integration are complete. PR #99 was merged to `main` at `daa889d758ce80c7d1ce73ea37110937e1b146c0`. Current authority is `CB16_R11_CC_INTEGRATION_SPEC_V1.json` + `CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`; see `docs/CURRENT_STATE.md` and `docs/CC_INTEGRATION_HANDOFF.md`. The task/gap sections below are historical implementation provenance, not still-open work. The later baseline design clarification is in `docs/ECONOMIC_ORDERING_AND_PROMOTION.md`; the next program is `docs/POST_CC_SCIENTIFIC_PROGRAM_R0.md`. Preserve frozen receipt identity.

CC replaced the serial execution shape of BC with four independently executable task packages. Each sub-agent started from the same frozen implementation baseline and completed its package without importing, cherry-picking, waiting for, or treating any sibling CC branch as authority.

The four packages were:

1. **Thread A — Runtime / Account / Continuous Interaction**  
   `docs/cc/CC_THREAD_A_RUNTIME_ACCOUNT.md`
2. **Thread B — Policy / Critic / Learner / Retention**  
   `docs/cc/CC_THREAD_B_POLICY_LEARNING.md`
3. **Thread C — Experience / Replay / Economic Evaluation**  
   `docs/cc/CC_THREAD_C_EXPERIENCE_ECONOMICS.md`
4. **Thread D — Performance Hard Cutover / High-Throughput Spine**  
   `docs/cc/CC_THREAD_D_PERFORMANCE_HARD_CUTOVER.md`

The old AC and BC TODO files remain design/history sources. A CC executor did **not** execute AC/BC task numbering. CC tasks could repair, supersede, or retire BC surfaces where the frozen baseline proved them incomplete, stale, or structurally hostile to parallel/high-throughput execution.

> **Historical implementation-branch rule:** agents read CC planning docs from main but created the four implementation branches from `89d62bf...` without automatically absorbing later code commits.

---

## 1. Frozen-baseline audit verdict

At `89d62bf...` the repository was materially ahead of the BC document's creation snapshot.

Observed implementation already on the frozen baseline included the Round-2 R1 science lane, execution/account-economics repair, account lineage/observation, Actor/Critic observation firewalls, state-sufficiency machinery, optional policy memory, gradient ownership, and stochastic Actor work through BC-037. Gate BC-A had a machine-readable PASS receipt at COMPONENT evidence level. The current Actor implementation already contained categorical direction and conditional squashed-Normal risk semantics, with FLAT as an exact zero-risk point mass and non-FLAT endpoints disallowed.

The live code therefore invalidated any plan that still assumed BC-001 or AC-001 was the current implementation start.

### 1.1 Management-document drift found at the frozen baseline

The following drift was itself a CC input:

- `docs/CURRENT_STATE.md` still described an AC-013/014-era snapshot and said the new Physics adapter / Actor / learner were absent in contexts where this was no longer true.
- `docs/ARCHITECTURE_MAP.md` still described the R0 incremental Actor-Critic surface and did not map the landed R1 Round-2 modules through BC-037.
- `docs/R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md` still ended with a creation-state sentence saying `MAIN c373d23 / BC-A NOT YET STARTED`, even though BC-A was qualified and main had advanced through BC-037.
- prior `docs/README.md` and `AGENTS.md` routing directed implementation agents to BC; the CC planning change replaced that routing.
- `docs/PERFORMANCE_STRATEGY_3700X_1060.md` was design guidance, not an implementation series. CC Thread D converted that design into an explicit hard-cutover implementation package.

Shared management docs were intentionally **not edited by the four implementation sub-agents**. Their implementation-state updates were reserved for the later integration pass to avoid merge conflicts.

### 1.2 Current performance-code mismatch

The repository already contained useful historical performance ideas, but their runtime semantics were not the CC target:

- `gpu_inference_broker.py` used the old Market64 / Account6 request shape and deterministic direction/risk outputs.
- `multiprocess_trajectory_farm.py` replayed precomputed action schedules through `vectorized_physics`; it was not a live stochastic policy-each-decision collector.
- `vectorized_physics.py` carried historical semantics and was not a qualified substitute for the current R1 account/execution lane.
- `market_runtime_cache_r11.py` cached hourly R10.2 payloads in one process; it was useful as a design reference but did not define the CC cross-process/shared-market cache.

**CC did not build compatibility adapters for those performance APIs.** Thread D built a new CC performance spine and retired the old performance runtime from the active CC path.

---

## 2. Hard-cutover directive

The user's CC directive was:

> **Performance first. Hard switch. No compatibility mode for legacy performance runtime.**

For CC this meant:

- no dual-run production mode between old and new performance paths;
- no compatibility wrapper preserving old `InferenceRequest`, `TrajectoryReplayJob`, or `VectorizedPhysics` API contracts;
- no fallback from CC runtime to old broker/farm/vectorized execution if the new path fails;
- old performance modules may be read as implementation references and used as benchmark baselines, but are not runtime dependencies of the CC lane;
- after the CC fast path qualified, the active CC route points only to the new high-throughput spine;
- legacy historical/scientific lanes remain preserved as historical evidence; **hard cutover applies to the CC performance/runtime path, not to deletion or rewriting of historical scientific evidence.**

Performance priority did **not** authorize changing the scientific question. The following were correctness constraints of the new fast implementation, not compatibility obligations to old APIs:

- signed account economics and explicit liabilities are preserved;
- actual account consequences reach the next policy decision;
- nominal sampled action and true `log_mu` are preserved even if execution is clamped/rejected;
- failure/terminal facts are not dropped for throughput;
- one logical account remains temporally serialized;
- policy generation / RNG / science identity remain attributable;
- FINAL stays sealed and fresh-data download stays forbidden;
- strategy preferences are not reintroduced as hidden Physics rules;
- arithmetic-return/evaluation meaning is not replaced by log utility because an older vectorized path used it.

Within those constraints Thread D was free to change data layout, process topology, batching, cache design, language, serialization format, worker scheduling, storage tiering, and kernel implementation to maximize measured end-to-end throughput.

---

## 3. How four CC threads remained independent

All four implementation threads started from the exact frozen baseline `89d62bf...`.

### 3.1 Forbidden sibling dependencies

A CC sub-agent could not:

- import a Python module created only on another CC branch;
- cherry-pick another CC branch before completing its own task package;
- wait for another thread's PR to merge before writing or testing its package;
- mutate a file owned by another thread;
- edit shared navigation/state docs during thread execution;
- use a sibling branch's receipt as a prerequisite.

### 3.2 Allowed common inputs

Every thread could rely on:

- the code tree at `main@89d62bf...`;
- the design documents present at that SHA plus the later CC planning/handoff documents;
- this CC master file and its frozen cross-thread payload definitions;
- its own local fixtures/mocks that implement those payload definitions;
- repository-frozen non-FINAL data already allowed for component qualification;
- thread-local synthetic/known-answer data.

### 3.3 Disjoint ownership

Each thread owned a disjoint path family:

| Thread | Primary owned prefix / surfaces | Must not edit |
|---|---|---|
| A | `cb16_local_opt/cc_runtime_*`, `cc_clock_*`, `cc_environment_*`, `cc_account_*`, A-specific tests/receipts | B/C/D modules; shared docs |
| B | `cb16_local_opt/cc_policy_*`, `cc_critic_*`, `cc_vtrace_*`, `cc_learner_*`, B-specific tests/receipts | A/C/D modules; shared docs |
| C | `cb16_local_opt/cc_experience_*`, `cc_replay_*`, `cc_economic_*`, C-specific tests/receipts | A/B/D modules; shared docs |
| D | `cb16_local_opt/cc_fast_*`, optional `native/cc_*`, performance workflows/benchmarks, D-specific tests/receipts | A/B/C modules; shared docs |

Each task package gives a more exact file list.

---

## 4. Frozen cross-thread payload contracts

These contracts were frozen **in documentation** so sibling branches did not need a shared new Python module during parallel work. Every thread implemented thread-local fixtures/serializers around the same field meanings. Final integration centralized/bound them under the integrated science identity.

Unknown required fields fail closed. Extra fields must be namespaced extensions and cannot silently change a defined field's meaning.

### W-01 — `CCPolicyDecisionV1`

Required semantic fields:

```text
science_semantic_version
account_lineage_id
decision_index
environment_time
policy_generation
policy_id
policy_sha256
observation_schema
observation_hash
normalizer_id
nominal_direction        # SHORT | FLAT | LONG
nominal_target_risk      # [0,1], FLAT exactly 0
log_mu                   # log probability/density under behavior policy
risk_measure_kind        # point_mass | continuous_density as applicable
rng_stream_id
rng_position_or_counter
```

Rules:

- this is the **nominal sampled decision**, not the permitted or executed action;
- `log_mu` is never reconstructed from the later execution result;
- exact policy identity and RNG provenance are mandatory for stochastic collection;
- Actor does not receive economic objective countdown `tau` in the current Round-2 baseline.

### W-02 — `CCEnvironmentTransitionV1`

Required semantic fields:

```text
account_lineage_id
decision_index
environment_time_before
environment_time_after
pre_account_truth_hash
policy_decision_ref
permission_status
permission_reason
permitted_target_direction
permitted_target_risk
target_quantity
execution_legs[]
fees
funding
realized_pnl
unrealized_pnl_delta
liability_delta
post_account_truth_hash
post_equity
boundary_type
mechanical_terminal
external_capital_flow_ref_or_null
```

Rules:

- actual execution is distinct from policy decision;
- reject/NOOP does not stop environment/account consequences;
- negative equity/liability is not silently clamped away;
- reversal is represented by actual ordered legs;
- strategy SL/TP/max-hold/cooldown is not part of the CC policy-neutral environment unless separately versioned as an explicit restricted experiment.

### W-03 — `CCExperienceSequenceV1`

Required semantic fields:

```text
sequence_id
account_lineage_id
science_semantic_version
market_lineage_id
source_classification
transition_refs[]             # ordered
first_decision_index
last_decision_index
behavior_policy_identities[]
normalizer_identities[]
chunk_boundary_type
bootstrap_state_ref_or_null
raw_fact_content_sha256
```

Rules:

- sequence order is immutable and account-local;
- facts are retained independently of replay/demo/evaluation admission;
- failure/terminal sequences remain valid raw facts;
- old experience without valid behavior likelihood is not silently promoted into V-trace-compatible replay.

### W-04 — `CCLearningUpdateV1`

Required semantic fields:

```text
update_id
parent_checkpoint_sha256
science_semantic_version
sampled_sequence_ids[]
sampling_probabilities_or_weights
behavior_support_summary
actor_loss
critic_loss
vtrace_summary
gradient_ownership_summary
optimizer_step_before
optimizer_step_after
child_checkpoint_sha256
commit_status
```

Rules:

- one update ID commits at most once;
- the running behavior checkpoint is not mutated in place by the learner;
- frozen market organs receive no gradient/mutation;
- replay sampling weights do not redefine the economic evaluation cohort.

### W-05 — `CCEconomicResultV1`

Required semantic fields:

```text
evaluation_id
policy_object_type            # frozen_checkpoint | generation_chain | baseline
policy_identity
cohort_id
common_horizon_id
capital_denominator_id
account_results[]
mean_arithmetic_return
buy_hold_delta
flat_delta
failure_counts
tail_diagnostics
result_scope
```

Rules:

- all included failures remain in the denominator;
- injected restart capital is explicit;
- mixed-generation accounts are not automatically credited to the final checkpoint;
- arithmetic expected return is primary; tails are diagnostics unless the owner later changes the objective.

---

## 5. Historical CC gap registry at `89d62bf`

The following entries describe the gaps that motivated the four task packages. They are **not current open gaps** after integration; current status is in `CURRENT_STATE.md` and the integration receipt.

### CC-G01 — BC management docs were stale

Implementation had advanced through BC-037 while current-state/architecture/BC creation-state prose still described much earlier code.

### CC-G02 — Stochastic Actor was only partially complete

Direction and conditional risk were present, but joint nominal log probability, RNG serialization/provenance, deterministic evaluation identity, and full Brain integration were incomplete at the frozen baseline.

### CC-G03 — No qualified continuous R1 runtime spine

There was no finished four-clock policy-each-decision environment/account collector with exact pause/resume and generation switch semantics.

### CC-G04 — No qualified immutable CC experience path

Round-2 transition/sequence schemas, raw-fact storage, replay views, failure persistence, and support-health contracts were incomplete.

### CC-G05 — No complete reward/Critic/V-trace learner chain

Arithmetic reward, bootstrap rules, Critic, V-trace, policy loss, deterministic replay sampler, exactly-once learner update, and complete checkpoint recovery were open.

### CC-G06 — No generation/retention known-answer closure

Fixed behavior collection, account-preserving generation replacement, mixed-generation attribution, replay mixture provenance, and A→B→A retention qualification were open.

### CC-G07 — No economic evaluator for the new objective

Common cohort/T/capital, B&H/FLAT baselines, complete failures, mixed-generation policy identity and promotion boundary were not implemented as the Round-2 evaluator.

### CC-G08 — Existing high-throughput code was semantically obsolete for CC

Old broker/farm/vectorized execution used different request schemas, deterministic actions, historical reward/physics semantics or precomputed action schedules. CC chose retirement/hard cutover rather than compatibility.

### CC-G09 — No shared-market / many-account performance spine

Frozen market representations were not yet exposed as a new CC shared cache keyed by exact market/preprocessing/organ identity for many concurrent accounts.

### CC-G10 — No CC stochastic batch-inference broker

The old broker did not carry `log_mu`, RNG provenance, stochastic distribution identity, generation identity or current observation schema.

### CC-G11 — No CC live account worker farm

The existing trajectory farm replayed an action schedule instead of interleaving real Actor decisions with account evolution.

### CC-G12 — No bounded asynchronous fact writer/backpressure contract

High-throughput collection needed a bounded writer path that could not trade correctness for unlimited queue growth or dropped failure records.

### CC-G13 — Storage tiering was design-only

The SSD/RAM/HDD hot/warm/cold plan and ordered IO optimization had not been implemented or benchmarked on the Shanxi workflow.

### CC-G14 — No semantic-throughput benchmark harness

The project lacked a single benchmark that simultaneously reported compliant transitions/s, wall-clock known-answer completion, memory/IO/GPU metrics and semantic-equivalence checks.

### CC-G15 — Native-code escalation was not executable policy

The performance document described Numba/Rust/Go choices, but there was no profile-driven gate deciding when a hotspot warranted native migration.

### CC-G16 — Thread-safe integration identity was absent

Four independently built components needed a final integration gate verifying wire-schema identity rather than Python import lineage.

---

## 6. Four task packages — completed

### Thread A — Runtime / Account / Continuous Interaction

Goal: build a semantically correct single-account/multi-account reference runtime using current R1 execution/account truth, independent of learner, storage engine, and performance spine.

Historical task package: `docs/cc/CC_THREAD_A_RUNTIME_ACCOUNT.md`.

### Thread B — Policy / Critic / Learner / Retention

Goal: complete stochastic policy probability/RNG semantics, Brain/Critic/reward/V-trace/learner/checkpoint logic and known-answer learning using synthetic in-memory W-01/W-03/W-04 fixtures.

Historical task package: `docs/cc/CC_THREAD_B_POLICY_LEARNING.md`.

### Thread C — Experience / Replay / Economic Evaluation

Goal: define immutable experience and data views, replay compatibility/support health, raw-fact durability, failure retention, generation attribution and full economic evaluation using synthetic W-02/W-03/W-05 fixtures.

Historical task package: `docs/cc/CC_THREAD_C_EXPERIENCE_ECONOMICS.md`.

### Thread D — Performance Hard Cutover / High-Throughput Spine

Goal: implement the new single active CC high-throughput path for 3700X/GTX1060 with shared market reuse, stochastic batching, account-parallel CPU execution, bounded IO and benchmark-driven native optimization.

Historical task package: `docs/cc/CC_THREAD_D_PERFORMANCE_HARD_CUTOVER.md`.

---

## 7. Historical branch and delivery protocol

Each sub-agent created one primary thread branch from `89d62bf...`:

```text
ai/r11-cc-thread-a-runtime-r0
ai/r11-cc-thread-b-learning-r0
ai/r11-cc-thread-c-experience-r0
ai/r11-cc-thread-d-fast-cutover-r0
```

Each thread finished with:

```text
authority/rearchitecture_r11/CB16_R11_CC_THREAD_<A|B|C|D>_RECEIPT_V1.json
```

Final integration froze their exact heads and receipt hashes; moving branch names are no longer authority.

---

## 8. Final integration — completed

The integration agent performed the planned join:

1. bound W-01..W-05 central representations;
2. bound Thread B policy to Thread A runtime;
3. bound A transitions to Thread C raw-fact/replay/evaluation path;
4. replaced reference scheduling/transport with Thread D fast spine;
5. proved fast/reference semantic equivalence within preregistered rules;
6. ran closed-loop known-answer qualification;
7. ran Shanxi performance benchmark;
8. hard-routed CC to the fast path only;
9. enforced no runtime fallback to old `gpu_inference_broker.py`, `multiprocess_trajectory_farm.py`, or `vectorized_physics.py`;
10. emitted final integration authority and canonical handoff.

Result: `QUALIFIED_HARD_CUTOVER_PASS`.

---

## 9. Final performance decision

For active CC runtime, the fastest semantic-PASS implementation wins under the frozen rule:

`PERFORMANCE FIRST AMONG SEMANTICALLY QUALIFIED IMPLEMENTATIONS`

Receipt-selected topology:

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

Frozen benchmark summary:

- 16 accounts × 64 market steps = 1024 transitions/run;
- reference and fast each 7 alternating repetitions;
- reference median 7.535366 transitions/s;
- fast median 638.765768 transitions/s;
- median speedup 84.769×;
- semantic and final-account checksums matched.

No implementation wins solely because one microkernel is faster; science-semantic PASS remains prerequisite.

---

## 10. Final evidence level

Strongest justified evidence:

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

This is not ECONOMIC or TRANSFER evidence. High throughput cannot upgrade a scientific claim.

---

## 11. Post-integration design clarification

The former B&H/FLAT precedence question is closed at the design level: no default baseline priority; comparable complete arithmetic returns order policies; adequately supported improvement may promote a sandbox champion without requiring both baseline deltas to be positive. See [Economic ordering and promotion](ECONOMIC_ORDERING_AND_PROMOTION.md).

Historical C-20 code and receipts remain unchanged. Implement the successor evaluation semantics with new qualification under S0 of [Post-CC scientific program R0](POST_CC_SCIENTIFIC_PROGRAM_R0.md), then proceed to actual learnability and historical science. This does not reopen completed CC threads. Horizon/cohort/budget/replay/model-size choices are versioned experiment parameters.

---

## 12. Historical Definition of Done — satisfied

The original per-thread DoD required exact frozen base, no sibling dependency during independent implementation, owned file discipline, positive/negative tests, W-contract conformance, FINAL protection, machine-readable receipts and no Thread-D legacy fallback.

Final integration receipt records the frozen thread heads and qualifications; final joined qualification passed.

---

## 13. Canonical handoff after closure

Do **not** dispatch new agents from this TODO as though CC implementation were still open.

Use:

1. `docs/CURRENT_STATE.md`
2. `docs/CC_INTEGRATION_HANDOFF.md`
3. `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
4. `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

The four-thread plan remains only as implementation provenance.
