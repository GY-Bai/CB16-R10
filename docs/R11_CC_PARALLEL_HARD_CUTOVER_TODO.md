# CB16 R11 CC — Four-Thread Parallel Hard-Cutover TODO

**Status:** OPEN / parallel execution handoff  
**Frozen implementation baseline:** `main@89d62bf966f476e598f0e2f5c5e8e03c15a8db51`  
**Planning-doc note:** CC handoff documents are committed after this SHA so agents can read them from current main; implementation branches still start from the frozen baseline unless explicitly re-frozen.  
**Audit date:** 2026-09-12  
**Execution model:** four independent sub-agent task packages + one later integration pass  
**Performance policy:** **hard cutover / performance first / no legacy performance compatibility layer**  

CC replaces the serial execution shape of BC with four independently executable task packages. Each sub-agent starts from the same frozen implementation baseline and must be able to finish its package without importing, cherry-picking, waiting for, or treating any sibling CC branch as authority.

The four packages are:

1. **Thread A — Runtime / Account / Continuous Interaction**  
   `docs/cc/CC_THREAD_A_RUNTIME_ACCOUNT.md`
2. **Thread B — Policy / Critic / Learner / Retention**  
   `docs/cc/CC_THREAD_B_POLICY_LEARNING.md`
3. **Thread C — Experience / Replay / Economic Evaluation**  
   `docs/cc/CC_THREAD_C_EXPERIENCE_ECONOMICS.md`
4. **Thread D — Performance Hard Cutover / High-Throughput Spine**  
   `docs/cc/CC_THREAD_D_PERFORMANCE_HARD_CUTOVER.md`

The old AC and BC TODO files remain design/history sources. A CC executor does **not** execute AC/BC task numbering. CC tasks may repair, supersede, or retire BC surfaces where the frozen baseline proves they are incomplete, stale, or structurally hostile to parallel/high-throughput execution.

> **Implementation-branch rule:** read CC planning docs from current main, but create the four implementation branches from `89d62bf...`. Do not automatically absorb code commits after the frozen baseline.

---

## 1. Frozen-baseline audit verdict

At `89d62bf...` the repository is materially ahead of the BC document's creation snapshot.

Observed implementation already on the frozen baseline includes the Round-2 R1 science lane, execution/account-economics repair, account lineage/observation, Actor/Critic observation firewalls, state-sufficiency machinery, optional policy memory, gradient ownership, and stochastic Actor work through BC-037. Gate BC-A has a machine-readable PASS receipt at COMPONENT evidence level. The current Actor implementation already contains categorical direction and conditional squashed-Normal risk semantics, with FLAT as an exact zero-risk point mass and non-FLAT endpoints disallowed.

The live code therefore invalidates any plan that still assumes BC-001 or AC-001 is the current implementation start.

### 1.1 Management-document drift found at the frozen baseline

The following drift is itself a CC input:

- `docs/CURRENT_STATE.md` still describes an AC-013/014-era snapshot and says the new Physics adapter / Actor / learner are absent in contexts where this is no longer true.
- `docs/ARCHITECTURE_MAP.md` still describes the R0 incremental Actor-Critic surface and does not map the landed R1 Round-2 modules through BC-037.
- `docs/R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md` still ends with a creation-state sentence saying `MAIN c373d23 / BC-A NOT YET STARTED`, even though BC-A is now qualified and main has advanced through BC-037.
- prior `docs/README.md` and `AGENTS.md` routing directed implementation agents to BC; the CC planning change replaces that routing.
- `docs/PERFORMANCE_STRATEGY_3700X_1060.md` is design guidance, not an implementation series. CC Thread D converts that design into an explicit hard-cutover implementation package.

Shared management docs are intentionally **not edited by the four implementation sub-agents**. Their implementation-state updates are reserved for the later integration pass to avoid merge conflicts.

### 1.2 Current performance-code mismatch

The repository already contains useful historical performance ideas, but their runtime semantics are not the CC target:

- `gpu_inference_broker.py` uses the old Market64 / Account6 request shape and deterministic direction/risk outputs.
- `multiprocess_trajectory_farm.py` replays precomputed action schedules through `vectorized_physics`; it is not a live stochastic policy-each-decision collector.
- `vectorized_physics.py` carries historical semantics and is not a qualified substitute for the current R1 account/execution lane.
- `market_runtime_cache_r11.py` caches hourly R10.2 payloads in one process; it is useful as a design reference but does not define the CC cross-process/shared-market cache.

**CC does not build compatibility adapters for those performance APIs.** Thread D builds a new CC performance spine and later retires the old performance runtime from the active CC path.

---

## 2. Hard-cutover directive

The user's CC directive is:

> **Performance first. Hard switch. No compatibility mode for legacy performance runtime.**

For CC this means:

- no dual-run production mode between old and new performance paths;
- no compatibility wrapper preserving old `InferenceRequest`, `TrajectoryReplayJob`, or `VectorizedPhysics` API contracts;
- no fallback from CC runtime to old broker/farm/vectorized execution if the new path fails;
- old performance modules may be read as implementation references and used as benchmark baselines, but are not runtime dependencies of the CC lane;
- after the CC fast path is qualified, the active CC route points only to the new high-throughput spine;
- legacy historical/scientific lanes remain preserved as historical evidence; **hard cutover applies to the CC performance/runtime path, not to deletion or rewriting of historical scientific evidence.**

Performance priority does **not** authorize changing the scientific question. The following are correctness constraints of the new fast implementation, not compatibility obligations to old APIs:

- signed account economics and explicit liabilities are preserved;
- actual account consequences reach the next policy decision;
- nominal sampled action and true `log_mu` are preserved even if execution is clamped/rejected;
- failure/terminal facts are not dropped for throughput;
- one logical account remains temporally serialized;
- policy generation / RNG / science identity remain attributable;
- FINAL stays sealed and fresh-data download stays forbidden;
- strategy preferences are not reintroduced as hidden Physics rules;
- arithmetic-return/evaluation meaning is not replaced by log utility because an older vectorized path used it.

Within those constraints Thread D is free to change data layout, process topology, batching, cache design, language, serialization format, worker scheduling, storage tiering, and kernel implementation to maximize measured end-to-end throughput.

---

## 3. How four CC threads remain independent

All four implementation threads start from the exact frozen baseline `89d62bf...`.

### 3.1 Forbidden sibling dependencies

A CC sub-agent must not:

- import a Python module created only on another CC branch;
- cherry-pick another CC branch before completing its own task package;
- wait for another thread's PR to merge before writing or testing its package;
- mutate a file owned by another thread;
- edit shared navigation/state docs during thread execution;
- use a sibling branch's receipt as a prerequisite.

### 3.2 Allowed common inputs

Every thread may rely on:

- the code tree at `main@89d62bf...`;
- the design documents present at that SHA plus the later CC planning/handoff documents;
- this CC master file and its frozen cross-thread payload definitions;
- its own local fixtures/mocks that implement those payload definitions;
- repository-frozen non-FINAL data already allowed for component qualification;
- thread-local synthetic/known-answer data.

### 3.3 Disjoint ownership

Each thread owns a disjoint path family:

| Thread | Primary owned prefix / surfaces | Must not edit |
|---|---|---|
| A | `cb16_local_opt/cc_runtime_*`, `cc_clock_*`, `cc_environment_*`, `cc_account_*`, A-specific tests/receipts | B/C/D modules; shared docs |
| B | `cb16_local_opt/cc_policy_*`, `cc_critic_*`, `cc_vtrace_*`, `cc_learner_*`, B-specific tests/receipts | A/C/D modules; shared docs |
| C | `cb16_local_opt/cc_experience_*`, `cc_replay_*`, `cc_economic_*`, C-specific tests/receipts | A/B/D modules; shared docs |
| D | `cb16_local_opt/cc_fast_*`, optional `native/cc_*`, performance workflows/benchmarks, D-specific tests/receipts | A/B/C modules; shared docs |

Each task package gives a more exact file list.

---

## 4. Frozen cross-thread payload contracts

These contracts are frozen **in documentation** so that sibling branches do not need a shared new Python module during parallel work. Every thread implements thread-local fixtures/serializers around the same field meanings. Final integration may later centralize them into a shared typed module.

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

## 5. CC gap registry at `89d62bf`

The four task packages collectively address the following current gaps.

### CC-G01 — BC management docs are stale

Implementation has advanced through BC-037 while current-state/architecture/BC creation-state prose still describes much earlier code. Shared docs must be corrected only after thread outputs are merged.

### CC-G02 — Stochastic Actor is only partially complete

Direction and conditional risk are present, but joint nominal log probability, RNG serialization/provenance, deterministic evaluation identity, and full Brain integration remain incomplete at the frozen baseline.

### CC-G03 — No qualified continuous R1 runtime spine

There is no finished four-clock policy-each-decision environment/account collector with exact pause/resume and generation switch semantics.

### CC-G04 — No qualified immutable CC experience path

Round-2 transition/sequence schemas, raw-fact storage, replay views, failure persistence, and support-health contracts are not complete.

### CC-G05 — No complete reward/Critic/V-trace learner chain

Arithmetic reward, bootstrap rules, Critic, V-trace, policy loss, deterministic replay sampler, exactly-once learner update, and complete checkpoint recovery remain open.

### CC-G06 — No generation/retention known-answer closure

Fixed behavior collection, account-preserving generation replacement, mixed-generation attribution, replay mixture provenance, and A→B→A retention qualification remain open.

### CC-G07 — No economic evaluator for the new objective

Common cohort/T/capital, B&H/FLAT baselines, complete failures, mixed-generation policy identity and promotion boundary are not implemented as the current Round-2 evaluator.

### CC-G08 — Existing high-throughput code is semantically obsolete for CC

Old broker/farm/vectorized execution uses different request schemas, deterministic actions, historical reward/physics semantics or precomputed action schedules. CC chooses retirement/hard cutover rather than compatibility.

### CC-G09 — No shared-market / many-account performance spine

Frozen market representations are not yet exposed as a new CC shared cache keyed by exact market/preprocessing/organ identity for many concurrent accounts.

### CC-G10 — No CC stochastic batch-inference broker

The old broker does not carry `log_mu`, RNG provenance, stochastic distribution identity, generation identity or current observation schema.

### CC-G11 — No CC live account worker farm

The existing trajectory farm replays an action schedule instead of interleaving real Actor decisions with account evolution.

### CC-G12 — No bounded asynchronous fact writer/backpressure contract

High-throughput collection must not trade correctness for unlimited queue growth or dropped failure records.

### CC-G13 — Storage tiering is design-only

The SSD/RAM/HDD hot/warm/cold plan and ordered IO optimization have not been implemented or benchmarked on the Shanxi workflow.

### CC-G14 — No semantic-throughput benchmark harness

The project lacks a single benchmark that simultaneously reports compliant transitions/s, wall-clock known-answer completion, memory/IO/GPU metrics and semantic-equivalence checks.

### CC-G15 — Native-code escalation is not executable policy

The performance document describes Numba/Rust/Go choices, but there is no profile-driven gate deciding when a hotspot warrants native migration.

### CC-G16 — Thread-safe integration identity is absent

Four independently built components need a later integration gate that verifies wire-schema identity, not Python import lineage.

---

## 6. Four task packages

### Thread A — Runtime / Account / Continuous Interaction

Goal: build a semantically correct single-account/multi-account reference runtime using current R1 execution/account truth, independent of the learner, storage engine, and performance spine.

Detailed tasks: `docs/cc/CC_THREAD_A_RUNTIME_ACCOUNT.md`.

### Thread B — Policy / Critic / Learner / Retention

Goal: complete stochastic policy probability/RNG semantics, Brain/Critic/reward/V-trace/learner/checkpoint logic and known-answer learning using synthetic in-memory W-01/W-03/W-04 fixtures.

Detailed tasks: `docs/cc/CC_THREAD_B_POLICY_LEARNING.md`.

### Thread C — Experience / Replay / Economic Evaluation

Goal: define immutable experience and data views, replay compatibility/support health, raw-fact durability, failure retention, generation attribution and full economic evaluation using synthetic W-02/W-03/W-05 fixtures.

Detailed tasks: `docs/cc/CC_THREAD_C_EXPERIENCE_ECONOMICS.md`.

### Thread D — Performance Hard Cutover / High-Throughput Spine

Goal: implement the new single active CC high-throughput path for 3700X/GTX1060 with shared market reuse, stochastic batching, account-parallel CPU execution, bounded IO and benchmark-driven native optimization.

Detailed tasks: `docs/cc/CC_THREAD_D_PERFORMANCE_HARD_CUTOVER.md`.

---

## 7. Branch and delivery protocol for the four sub-agents

Each sub-agent creates exactly one primary thread branch from `89d62bf...`:

```text
ai/r11-cc-thread-a-runtime-r0
ai/r11-cc-thread-b-learning-r0
ai/r11-cc-thread-c-experience-r0
ai/r11-cc-thread-d-fast-cutover-r0
```

The sub-agent may read CC documents from current main, but implementation code must remain based on the frozen baseline unless explicitly re-frozen.

Within a thread, commits may be incremental, but sibling branches are never dependencies.

Each thread must finish with:

```text
authority/rearchitecture_r11/CB16_R11_CC_THREAD_<A|B|C|D>_RECEIPT_V1.json
```

The receipt must include frozen base/head SHA, owned file list, tests, evidence level, FINAL/fresh firewall, unresolved items, wire deviations, and for Thread D benchmark/hardware identity when measured.

---

## 8. Final integration is a join, not a fifth implementation thread

After all four packages land, a later integration agent performs only these activities:

1. build central typed representations for W-01..W-05 or explicit adapters from each thread's local types;
2. bind Thread B policy to Thread A runtime;
3. bind A transitions to Thread C raw-fact/replay/evaluation path;
4. replace A's reference scheduling/transport with Thread D fast spine;
5. prove D's fast runtime produces CC-semantic outputs equivalent to the reference path within preregistered exact/tolerance rules;
6. run closed-loop known-answer qualification;
7. run the Shanxi performance benchmark;
8. hard-route the CC lane to the fast path only;
9. mark old `gpu_inference_broker.py`, `multiprocess_trajectory_farm.py`, and `vectorized_physics.py` as non-CC runtime paths; no compatibility fallback;
10. update shared management docs (`CURRENT_STATE`, `ARCHITECTURE_MAP`, `README`, `AGENTS`, BC status/history) from actual merged evidence.

A wire mismatch returns `INTEGRATION_BLOCKED` and is sent back to the owning thread scope.

---

## 9. Performance decision hierarchy for CC

For the active CC runtime, use the fastest implementation that passes semantic qualification under the frozen workload. The optimization order is no longer constrained by preserving old performance APIs.

Preferred investigation order:

1. eliminate repeated market/encoder work;
2. contiguous/shared market representation;
3. batch stochastic policy inference by policy version;
4. persistent CPU account workers and account batching;
5. bounded queues / large contiguous fact chunks / minimized serialization;
6. SSD hot working set + HDD cold archive with deterministic indexing;
7. vectorized/compiled numeric loops;
8. Numba or native extension for measured CPU hotspots;
9. Rust coarse-grained extension if it wins end-to-end after boundary cost;
10. Go only for a measured service/orchestration bottleneck, not numeric Actor/Critic replacement.

A faster new layout does **not** need to expose old broker/farm APIs. It needs to satisfy W-01/W-02/W-03 semantics and current R1 science meaning.

### Required performance metrics

```text
compliant_transitions_per_second
policy_decisions_per_second
learner_updates_per_second (when applicable)
wall_clock_to_fixed_known_answer_threshold
CPU per-core utilization
process/thread counts
PSS/cgroup working set
major faults / swap activity
SSD/HDD throughput and IO wait
GPU peak VRAM
GPU active time / transfer time / kernel time where measurable
queue depth in bytes and oldest-item age
batch-size distribution
policy staleness/generation distribution
```

No implementation wins solely because one microkernel is faster.

### Correctness checks that remain mandatory under performance-first policy

- W-01 probability/provenance meaning remains correct;
- signed account economics conserve the defined ledger;
- no account transition reordering;
- no failure fact loss;
- no unauthorized strategy exit rule;
- no FINAL/fresh access;
- no hidden capital reset;
- fixed behavior policy within a registered collection unit;
- batching/reordering does not silently alter per-account RNG identity.

Floating-point execution may use a preregistered tolerance where exact bit identity is neither required nor meaningful; liquidation/min-quantity boundaries need stronger discrete agreement tests.

---

## 10. CC evidence levels

Use:

- `CONTRACT`
- `COMPONENT`
- `CLOSED_LOOP`
- `KNOWN_ANSWER`
- `ECONOMIC`
- `TRANSFER`
- Thread D additionally may report orthogonal `PERFORMANCE_MEASURED`.

High throughput cannot upgrade a scientific claim.

---

## 11. Owner-open questions that do not block parallel CC work

- exact formal market-evaluation horizon/cohort weighting;
- master promotion rule when B&H and FLAT comparisons disagree;
- strategic leverage cap not dictated by resource/exchange mechanics;
- future simulated-account deployment promotion authority;
- whether policy memory beyond current causal observation is ultimately required.

Synthetic/component/known-answer work must not stop on these questions.

---

## 12. Definition of Done for a CC thread

A thread is DONE only if:

- implementation is based on exact `89d62bf...`;
- no sibling-branch dependency exists;
- only owned paths were modified, except thread-specific authority receipts/workflows;
- positive and negative/fail-closed tests exist;
- W-01..W-05 semantics are obeyed where applicable;
- historical authority and FINAL boundary remain intact;
- evidence claim is explicit;
- thread receipt is machine-readable;
- no shared state/navigation doc was opportunistically edited;
- Thread D has no compatibility wrapper/fallback to old performance runtime.

---

## 13. Immediate handoff

Give each sub-agent exactly one thread document plus this master document. No sub-agent needs the other three thread files.

All four implementation branches start from:

```text
89d62bf966f476e598f0e2f5c5e8e03c15a8db51
```

The concurrency model is:

```text
                     ┌─ Thread A: runtime/account ─────────┐
89d62bf frozen base ─┼─ Thread B: policy/learner ─────────┼─> later CC integration join
                     ├─ Thread C: experience/economics ───┤
                     └─ Thread D: performance hard cutover ┘
```

There is **no A→B→C→D execution dependency** inside CC.