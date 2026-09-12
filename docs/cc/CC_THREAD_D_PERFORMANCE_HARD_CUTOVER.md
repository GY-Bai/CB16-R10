# CC Thread D — Performance Hard Cutover / High-Throughput Spine

**Thread:** D  
**Frozen base:** `89d62bf966f476e598f0e2f5c5e8e03c15a8db51`  
**Branch:** `ai/r11-cc-thread-d-fast-cutover-r0`  
**Independence rule:** no imports/cherry-picks/dependencies from CC Threads A/B/C  
**Primary objective:** maximize measured compliant end-to-end throughput on the 3700X / GTX1060 class host  
**Cutover policy:** **CC-only fast path; no legacy performance API compatibility; no fallback**  

Thread D turns `PERFORMANCE_STRATEGY_3700X_1060.md` into executable performance work.

The following modules may be inspected as historical implementation ideas or benchmark baselines only:

- `cb16_local_opt/gpu_inference_broker.py`
- `cb16_local_opt/multiprocess_trajectory_farm.py`
- `cb16_local_opt/vectorized_physics.py`
- `cb16_local_opt/market_runtime_cache_r11.py`

The new CC fast path must **not import or wrap the first three legacy performance runtimes**. There is no compatibility adapter preserving their request/job/result types. If the new fast path cannot run, CC performance execution fails; it does not fall back.

Thread D uses synthetic local policy/account kernels matching the master W-01/W-02/W-03 semantics so it can finish without Threads A/B/C.

---

## D-01 — Freeze performance baseline and retirement registry

**Files**
- `[NEW] authority/rearchitecture_r11/CB16_R11_CC_THREAD_D_BASELINE_V1.json`
- `[NEW] authority/rearchitecture_r11/CB16_R11_CC_LEGACY_PERF_RETIREMENT_V1.json`
- `[NEW] tests/test_cc_thread_d_baseline_r0.py`

Bind:

- frozen base SHA;
- performance design doc blob;
- current R1 Actor/account/science blobs needed only for semantic reference;
- legacy performance module blobs.

Retirement registry marks:

```text
gpu_inference_broker.py        REFERENCE_ONLY_FOR_CC
multiprocess_trajectory_farm.py REFERENCE_ONLY_FOR_CC
vectorized_physics.py          REFERENCE_ONLY_FOR_CC
```

This does not delete historical files. It forbids them from the active CC fast runtime dependency graph.

---

## D-02 — Thread-local fast-path wire types

**Files**
- `[NEW] cb16_local_opt/cc_fast_wire_r0.py`
- `[NEW] tests/test_cc_fast_wire_r0.py`

Implement compact thread-local representations of W-01/W-02/W-03 suitable for arrays/shared memory.

Requirements:

- fixed-width IDs/hashes where practical;
- numeric direction encoding with canonical mapping;
- explicit `log_mu` and risk measure kind;
- generation and RNG provenance;
- signed economics fields;
- boundary code;
- no Python-object graph required on the hot path.

Serialization may differ from other threads; semantic fields may not.

---

## D-03 — CC performance benchmark contract

**Files**
- `[NEW] cb16_local_opt/cc_fast_benchmark_contract_r0.py`
- `[NEW] tests/test_cc_fast_benchmark_contract_r0.py`

Freeze a benchmark report schema containing at minimum:

- workload identity;
- code/science identity;
- process/thread topology;
- account count;
- decision rate;
- batch-size distribution;
- compliant transitions/s;
- policy decisions/s;
- wall-clock duration;
- CPU per-core utilization;
- PSS/cgroup memory;
- page faults/swap;
- disk throughput/IO wait;
- GPU VRAM;
- GPU kernel/transfer time if measurable;
- queue bytes/depth/oldest age;
- correctness checksum/semantic verdict.

A benchmark missing correctness verdict is invalid.

---

## D-04 — Deterministic synthetic performance workload

**Files**
- `[NEW] cb16_local_opt/cc_fast_workload_r0.py`
- `[NEW] tests/test_cc_fast_workload_r0.py`

Create synthetic but realistic-enough workloads for:

- shared market state + many accounts;
- stochastic policy outputs with true log-prob fixtures;
- account open/reduce/close/reversal/liquidation cases;
- variable decision schedules;
- terminal/failure records;
- long sequential runs.

The workload must be deterministic from seed and must not access FINAL/fresh data.

---

## D-05 — New CC shared-market cache identity

**Files**
- `[NEW] cb16_local_opt/cc_fast_market_cache_r0.py`
- `[NEW] tests/test_cc_fast_market_cache_r0.py`

Build a CC-specific cache key including:

```text
market source lineage
visible timestamp/window identity
preprocessing version
frozen organ/encoder identity
normalizer/preprocessor identity where applicable
output dtype/layout version
```

Cache only deterministic market-only products. Never cache trainable Brain/account-dependent hidden state across policy checkpoints.

No requirement to preserve `MarketRuntimeCacheR11` API.

---

## D-06 — Shared/contiguous market representation

**Files**
- `[MODIFY] cb16_local_opt/cc_fast_market_cache_r0.py`
- `[MODIFY] tests/test_cc_fast_market_cache_r0.py`

Expose contiguous read-only market tensors/arrays that many account workers can reference without N copies.

Prefer shared memory/memmap/RAM views based on measured cost.

**PASS:** N accounts referencing one market representation do not materialize N full market copies in the measured fixture.

---

## D-07 — Frozen-organ reuse benchmark

**Files**
- `[NEW] tests/test_cc_fast_market_reuse_benchmark_r0.py`

Compare:

- repeated per-account market/organ computation;
- one market computation reused across N accounts.

Report speed and memory deltas. Do not claim independent market evidence increases with N accounts.

---

## D-08 — CC-only batched policy inference broker

**Files**
- `[NEW] cb16_local_opt/cc_fast_policy_broker_r0.py`
- `[NEW] tests/test_cc_fast_policy_broker_r0.py`

Build a new broker from scratch around CC semantics.

Request must carry:

- policy generation/hash;
- observation tensor reference;
- account/decision identity;
- RNG stream/counter or sampling token;
- required stochastic mode.

Response must carry:

- nominal direction;
- nominal risk;
- true `log_mu`;
- risk measure kind;
- policy identity;
- RNG provenance.

**Forbidden:** deterministic old direction/risk-only response contract.

---

## D-09 — Single CUDA owner

**Files**
- `[MODIFY] cb16_local_opt/cc_fast_policy_broker_r0.py`
- `[MODIFY] tests/test_cc_fast_policy_broker_r0.py`

One process owns CUDA/model state. Worker processes must not initialize independent CUDA contexts for collection.

Use `spawn`/compatible process semantics.

**PASS:** worker-side CUDA initialization detector remains false in multiprocess fixture.

---

## D-10 — Policy-generation partitioned batching

**Files**
- `[MODIFY] cb16_local_opt/cc_fast_policy_broker_r0.py`
- `[MODIFY] tests/test_cc_fast_policy_broker_r0.py`

Only requests for the same exact behavior checkpoint/distribution identity may share one inference batch unless the implementation explicitly supports multiple resident versions without cross-contamination.

A request may never be scored by a different generation because it was convenient to batch.

---

## D-11 — Batching-order RNG independence

**Files**
- `[NEW] tests/test_cc_fast_rng_batching_r0.py`

Submit the same logical account decisions in different cross-account arrival orders/batch groupings.

Required result: each account's declared RNG identity determines its stochastic sample; scheduler order cannot silently change the sample sequence.

This is required for aggressive batching without turning scheduling noise into a new policy.

---

## D-12 — CPU-vs-GPU policy inference benchmark

**Files**
- `[NEW] cb16_local_opt/cc_fast_policy_benchmark_r0.py`
- `[NEW] tests/test_cc_fast_policy_benchmark_r0.py`

Benchmark at least:

- CPU batch=1/small batch;
- GPU micro-batching across ready accounts;
- transfer/queue overhead included.

On Shanxi, choose the faster valid mode for the measured workload. Do not assume GPU wins merely because CUDA exists.

The selected CC fast policy path may be CPU or GPU; compatibility with the old broker is irrelevant.

---

## D-13 — Compact account state SoA layout

**Files**
- `[NEW] cb16_local_opt/cc_fast_account_state_r0.py`
- `[NEW] tests/test_cc_fast_account_state_r0.py`

Represent many accounts in structure-of-arrays or another contiguous layout optimized for CPU cache/vectorization.

Must preserve signed economics, liabilities, position/cost state, lineage/index identity and discrete terminal state.

No forced nonnegative clamp for storage convenience.

---

## D-14 — Deterministic account-kernel reference

**Files**
- `[NEW] cb16_local_opt/cc_fast_account_kernel_r0.py`
- `[NEW] tests/test_cc_fast_account_kernel_r0.py`

Implement the CC synthetic account mechanics needed by D's benchmark, matching the master W-02 semantics:

- target delta application;
- ordered reversal legs;
- fees/funding;
- mark-to-market;
- liquidation/terminal responsibility fixture;
- signed liabilities.

This is a performance-thread local kernel for qualification, not a replacement authority for Thread A.

---

## D-15 — Persistent CPU account worker pool

**Files**
- `[NEW] cb16_local_opt/cc_fast_account_workers_r0.py`
- `[NEW] tests/test_cc_fast_account_workers_r0.py`

Use persistent spawned workers. Initial candidate matrix on the 3700X class host:

```text
workers = 2, 4, 6
per-worker numeric library threads = 1
```

Do not blindly use all 16 logical threads.

Each worker owns batches of account state; same-account updates remain ordered.

---

## D-16 — Cross-account parallel / same-account serial scheduler

**Files**
- `[NEW] cb16_local_opt/cc_fast_scheduler_r0.py`
- `[NEW] tests/test_cc_fast_scheduler_r0.py`

Parallelize across accounts and independent work only.

Prevent:

- account `n+1` before `n`;
- duplicate decision commit;
- account migration without exact ownership handoff;
- generation mix-up.

Scheduler fairness may be optimized for throughput, but must not starve an account beyond a declared bound that changes the environment decision contract.

---

## D-17 — CC fast collector coordinator

**Files**
- `[NEW] cb16_local_opt/cc_fast_collector_r0.py`
- `[NEW] tests/test_cc_fast_collector_r0.py`

Coordinate:

```text
shared market state
→ ready-account batching
→ CC policy broker
→ account workers
→ next ready state
→ fact output
```

Use D's local synthetic policy/account kernels. No dependency on Threads A/B/C.

---

## D-18 — Bounded fact-output queue

**Files**
- `[NEW] cb16_local_opt/cc_fast_fact_queue_r0.py`
- `[NEW] tests/test_cc_fast_fact_queue_r0.py`

Queue limits are by bytes and bounded age, not message count alone.

When the writer cannot keep up:

- apply backpressure;
- do not drop failed/terminal transitions;
- do not grow unbounded RAM;
- do not acknowledge durability before actual durable handoff.

---

## D-19 — Large contiguous chunk writer

**Files**
- `[NEW] cb16_local_opt/cc_fast_writer_r0.py`
- `[NEW] tests/test_cc_fast_writer_r0.py`

Optimize physical writes via contiguous chunks/batched serialization/compression.

Preserve W-03 sequence/account ordering and content checksums.

This writer may later feed Thread C semantics at integration; during D qualification it uses local synthetic fact sinks.

---

## D-20 — SSD hot / HDD cold tier manager

**Files**
- `[NEW] cb16_local_opt/cc_fast_storage_tier_r0.py`
- `[NEW] tests/test_cc_fast_storage_tier_r0.py`

Implement physical tiering policy:

- RAM bounded active buffers;
- SSD active shards/index/recovery state;
- HDD cold immutable archive.

Requirements:

- archive checksum verified before hot copy reclamation;
- no failure-history deletion;
- no random HDD access policy that changes replay sampling distribution;
- capacity limits explicit.

No dependence on Thread C's semantic store is required for D completion.

---

## D-21 — Replay/read IO ordering benchmark

**Files**
- `[NEW] cb16_local_opt/cc_fast_io_benchmark_r0.py`
- `[NEW] tests/test_cc_fast_io_benchmark_r0.py`

Given a preselected list of sample IDs, compare physical read strategies:

- naive random reads;
- grouped/sequential reads;
- SSD active-set reads;
- bounded prefetch.

The logical selected samples and weights must stay identical; only IO order may change.

---

## D-22 — Memory-budget controller

**Files**
- `[NEW] cb16_local_opt/cc_fast_memory_budget_r0.py`
- `[NEW] tests/test_cc_fast_memory_budget_r0.py`

For the 16GB host class, expose explicit budgets for:

- workers;
- shared market cache;
- queues;
- policy broker buffers;
- active replay/IO cache;
- learner overlap allowance if later integrated.

Initial design target may keep total active working set near the performance document's ~10–12GB guidance, but the actual limit must be measured/configurable rather than hard-coded as scientific truth.

Detect swap pressure and fail/scale down gracefully instead of expanding workers blindly.

---

## D-23 — Unified performance instrumentation

**Files**
- `[NEW] cb16_local_opt/cc_fast_metrics_r0.py`
- `[NEW] tests/test_cc_fast_metrics_r0.py`

Collect/report the D-03 metrics with monotonic timestamps and phase breakdown:

- market/cache;
- policy queue/wait/inference;
- account kernel;
- serialization/write;
- IO read;
- idle/backpressure.

GPU asynchronous timing must use proper synchronization/events where relevant.

---

## D-24 — Candidate topology matrix

**Files**
- `[NEW] cb16_local_opt/cc_fast_topology_search_r0.py`
- `[NEW] tests/test_cc_fast_topology_search_r0.py`

Define measured candidates such as:

```text
worker_count: 2 / 4 / 6
active_accounts: 16 / 32 / 64
policy mode: CPU / GPU broker
market reuse: on
writer chunk sizes: bounded candidate set
prefetch depth: bounded candidate set
```

Do not mutate scientific semantics between candidates.

Select by end-to-end compliant transitions/s and wall-clock workload completion, subject to correctness/memory constraints.

---

## D-25 — Numba/native hotspot gate

**Files**
- `[NEW] cb16_local_opt/cc_fast_native_gate_r0.py`
- `[NEW] tests/test_cc_fast_native_gate_r0.py`

Profile first. A hotspot becomes a native-migration candidate only if:

- it is a material fraction of end-to-end wall time;
- faster implementation has meaningful Amdahl impact;
- boundary/serialization cost is included;
- semantic checks are available.

Do not retain a slow Python path merely for compatibility if a qualified compiled implementation wins.

---

## D-26 — Numba compiled candidate

**Files**
- `[NEW/OPTIONAL when gate triggers] cb16_local_opt/cc_fast_numba_kernel_r0.py`
- `[NEW] tests/test_cc_fast_numba_kernel_r0.py`

For numeric CPU loops proven hot, implement a Numba candidate first when technically suitable.

Benchmark cold compile separately from warm steady state.

If it wins end-to-end and passes semantics, it may become the CC fast implementation directly; no requirement to keep the original Python hot loop active in production.

---

## D-27 — Rust coarse-grained candidate

**Files**
- `[NEW/OPTIONAL when gate triggers] native/cc_fast_runtime/*`
- `[NEW/OPTIONAL] cb16_local_opt/cc_fast_rust_binding_r0.py`
- `[NEW] tests/test_cc_fast_rust_binding_r0.py`

Trigger only for a measured remaining CPU hotspot where Rust is plausible after batching/layout/Numba analysis.

Rules:

- cross Python/Rust boundary at coarse batch granularity;
- continuous arrays/state, not per-field per-bar calls;
- include conversion/copy cost in benchmark;
- preserve signed account semantics and discrete boundary outcomes;
- if Rust wins and qualifies, CC may hard-switch to it with no Python compatibility fallback for that hot path.

---

## D-28 — Go service gate

**Files**
- `[NEW] tests/test_cc_fast_go_gate_r0.py`
- `[OPTIONAL only if triggered] native/cc_fast_service/*`

Go is allowed only if profiling shows a persistent scheduling/network/service bottleneck that Python async/process orchestration cannot meet economically.

Do not use Go to rewrite Actor/Critic/autograd numeric code by default.

A non-triggered gate is a valid outcome.

---

## D-29 — Semantic equivalence harness

**Files**
- `[NEW] cb16_local_opt/cc_fast_semantic_harness_r0.py`
- `[NEW] tests/test_cc_fast_semantic_harness_r0.py`

Within Thread D, maintain a deliberately simple scalar/local semantic reference for the synthetic D workload. Compare every optimized candidate against it.

Check:

- nominal decisions/provenance;
- account/equity/liability conservation;
- fees/funding;
- reversal ordering;
- liquidation/terminal outcomes;
- transition count/order;
- failure retention.

Use exact comparisons for discrete identities and preregistered numeric tolerance for floating values.

This local reference does not replace Thread A's later integration oracle.

---

## D-30 — Hostile concurrency/failure matrix

**Files**
- `[NEW] tests/test_cc_fast_hostile_runtime_r0.py`

Inject:

- worker crash;
- writer slowdown;
- queue saturation;
- policy broker delay;
- out-of-order response;
- stale policy generation request;
- duplicate account task;
- shared-memory lifecycle error;
- SSD unavailable/full simulation;
- partial archive write.

Required behavior is bounded/fail-closed/recoverable. No silent fact loss and no legacy fallback.

---

## D-31 — Explicit legacy-import firewall

**Files**
- `[NEW] tests/test_cc_fast_no_legacy_imports_r0.py`

The active Thread-D module dependency graph must not import:

```text
cb16_local_opt.gpu_inference_broker
cb16_local_opt.multiprocess_trajectory_farm
cb16_local_opt.vectorized_physics
```

No adapter/fallback/feature flag may route CC fast execution into them.

Importing `market_runtime_cache_r11` is also discouraged; if any market bytes are reused, the new CC cache must own the active API/identity.

---

## D-32 — CC-only fast runtime router

**Files**
- `[NEW] cb16_local_opt/cc_fast_router_r0.py`
- `[NEW] tests/test_cc_fast_router_r0.py`

Thread-D local route has exactly one active CC performance path:

```text
CC_FAST_R0
```

Unknown/legacy performance route requests fail closed.

There is no `legacy`, `compat`, `fallback`, or `auto fallback on error` mode.

---

## D-33 — Shanxi microbenchmark workflow

**Files**
- `[NEW] .github/workflows/cb16-r11-cc-fast-benchmark-r0.yml`

Target the authorized Shanxi self-hosted runner labels appropriate to the repo.

Workflow should:

1. verify exact code/version identity;
2. run semantic harness first;
3. run bounded warm-up;
4. run bounded candidate benchmark matrix;
5. capture system/GPU/memory/IO metrics;
6. upload compact benchmark artifact/receipt;
7. never access FINAL or download fresh market data;
8. avoid concurrent mutation of canonical account/state roots.

Do not make the workflow auto-run on every unrelated push if that would waste the single runner.

---

## D-34 — Performance winner compiler

**Files**
- `[NEW] cb16_local_opt/cc_fast_selection_r0.py`
- `[NEW] tests/test_cc_fast_selection_r0.py`

Select the fastest valid candidate by preregistered rule:

1. semantic qualification must PASS;
2. memory/swap/queue safety must PASS;
3. compare end-to-end compliant transitions/s and fixed-workload wall clock;
4. use secondary metrics only to diagnose ties/bottlenecks.

A microkernel speedup that does not improve end-to-end workload does not win.

---

## D-35 — Hard-cutover readiness manifest

**Files**
- `[NEW] authority/rearchitecture_r11/CB16_R11_CC_FAST_CUTOVER_V1.json`
- `[NEW] tests/test_cc_fast_cutover_r0.py`

Declare:

- selected fast topology/kernel/language;
- exact benchmark identity;
- no compatibility layer;
- no fallback;
- legacy performance runtime statuses `REFERENCE_ONLY_FOR_CC`;
- current semantic harness result;
- whether Shanxi measurement exists or remains `EXECUTION_BLOCKED/HARDWARE_LIMIT`.

Do not claim performance numbers without an actual measured run.

---

## D-36 — Thread-D qualification compiler

**Files**
- `[NEW] cb16_local_opt/cc_fast_qualification_r0.py`
- `[NEW] tests/test_cc_fast_qualification_r0.py`

Compile D-01..D-35.

Minimum qualification:

- new shared market cache/layout;
- new stochastic batch broker;
- persistent account workers;
- strict same-account ordering;
- bounded writer/backpressure;
- storage tiering;
- benchmark instrumentation;
- semantic equivalence against D local reference;
- no legacy imports/fallback;
- performance-selection logic.

If Shanxi was actually run, add `PERFORMANCE_MEASURED`; otherwise report component readiness only.

---

## D-37 — Thread-D receipt

**Files**
- `[NEW] authority/rearchitecture_r11/CB16_R11_CC_THREAD_D_RECEIPT_V1.json`

Record:

- frozen base/head SHA;
- owned files;
- semantic test results;
- benchmark workload/version;
- measured hardware identity if applicable;
- candidate matrix and winner;
- performance metrics;
- memory/IO/GPU metrics;
- native-language decision and evidence;
- hard-cutover/no-fallback assertion;
- FINAL/fresh-data firewall;
- strongest evidence level plus `PERFORMANCE_MEASURED` tag when justified.

### Thread-D DONE

Thread D is complete when a CC-only fast path exists and is qualified entirely on the D branch without Threads A/B/C. It must contain **no compatibility wrapper and no runtime fallback to the old broker/farm/vectorized path**. Later integration may bind Thread A/B/C semantics to this fast spine, but the performance architecture itself is already the hard-cutover target.