# CB16 R11 Task E — Heterogeneous / Asynchronous Orchestrator R0

Branch: `ai/r11-task-e-async-orchestrator-r0`  
Base: `598b5d8e34f5329af657beaae8315a76703638bd`

This change is orchestration-only. It does not implement or alter Teacher, H72/Trace,
Central Brain training/eval, Physics/Permission, market data, frozen sensory organs, or
storage-engine scientific semantics.

## Semantic boundary

The orchestrator is allowed to change only **WHEN / WHERE** computation runs. Every
scientific object carries generation, Champion, Teacher authority and Physics authority
identity. Fail-closed checks reject future taint, stale policy lineage, generation mixing,
duplicate conflicts, Teacher-to-Student autograd taint declarations and non-atomic
Champion/Challenger commits.

## Stage DAG

1. Frozen input/cache verification
2. Teacher evidence acquisition/lookup
3. Training snapshot seal
4. Champion policy inference
5. On-policy trace execution
6. Matured outcome/evidence materialization
7. Challenger training
8. Validation
9. Tournament
10. Promotion/rejection atomic commit
11. Checkpoint/journal seal
12. Next-generation release

On-policy trace materialization is an asynchronous side lane after stage 5. Once the
training snapshot is sealed, late trace evidence may be persisted but can never enter the
current generation's snapshot. It becomes eligible only to a later generation through its
`source_generation` lineage.

## Hard barriers

### Training Snapshot Barrier

`SnapshotSeal` is immutable, must exactly bind the accepted evidence-id set, is persisted
through `EvidenceStore.seal_snapshot`, and is journaled before the state advances. After
seal, `accept_teacher_evidence` is illegal.

### Generation Ownership Barrier

The Challenger result must name the active generation, the exact sealed `snapshot_id`, and
the exact current `parent_champion_id`. Policy/trace results must name the current Champion.
Evidence used by G(n)'s training snapshot must have `source_generation < n` and a Teacher
generation no later than its source generation. Current-generation on-policy trace evidence
is stored separately and cannot back-pollute G(n).

### Tournament Commit Barrier

The next parent authority is unavailable until `CheckpointStore.atomic_commit` returns an
atomic durable receipt and the checkpoint/journal seal is durable. For `REJECT`, the receipt
must return the unchanged current Champion id/hash; returning the Challenger is rejected as
self-laundering. For `PROMOTE`, it must return the exact validated Challenger id/hash.

## State machine

`PREPARING -> SNAPSHOT_SEALED -> TRACE_RUNNING -> CHALLENGER_TRAINING -> VALIDATING -> TOURNAMENT_PENDING -> COMMITTING -> COMMITTED`

Illegal state transitions raise `IllegalTransition`; lineage mismatches raise
`LineageViolation`. Mainline state advancement is journaled before in-memory state advances,
except the atomic commit store operation itself, whose crash gap is explicitly reconciled by
`read_commit()`.

## Heterogeneous scheduling / backpressure

Worker pools are separated into:

- `CPU_TEACHER`
- `CPU_TRACE`
- `GPU_BRAIN`
- `STORAGE`
- `IO_PREFETCH`

All queues are bounded. Full queues raise `BackpressureRequired`; there is no drop API.
Scientific work is durably journaled before queue admission, so a full queue leaves the work
pending rather than lost. `pump_pending_work()` reconstructs pending jobs as capacity frees.
`claim_any()` allows work stealing only across an explicitly provided compatible pool set.
GPU unavailability does not consume queued work.

Intended storage placement is explicit in the protocol vocabulary:

- SSD: journal, snapshot/lineage/checkpoint metadata
- HDD: immutable scientific payloads behind `EvidenceStore`
- RAM/GPU: ephemeral queues, caches, prefetch buffers and device residency

## Durable vs ephemeral state

Durable authority/state:

- frozen-input verification receipt
- accepted evidence ids + lineage
- immutable snapshot seal
- work schedule and scientific completion identities
- matured evidence materialization receipts
- Challenger/validation/tournament lineage
- atomic PROMOTE/REJECT receipt
- checkpoint seal and generation-release event

Ephemeral/performance-only state:

- in-memory queue image (rebuilt from durable pending work)
- worker leases/process/thread state
- prefetch/pinned-memory buffers
- decompressed/mmap caches
- GPU residency / CUDA Graph capture
- retry timers / scheduler heuristics

## Crash / resume rules

1. Replay the append-only generation journal.
2. Resolve evidence and snapshot ids back through `EvidenceStore`; missing or hash-mismatched
   authority objects fail recovery.
3. Rebuild scheduled work and requeue only work lacking a durable completion identity.
4. A worker crash increments only `attempt`; `work_id`, generation, parent Champion,
   snapshot and payload reference remain unchanged.
5. If atomic tournament commit succeeded before its journal acknowledgement, reconcile it
   with `CheckpointStore.read_commit()`; never run a second scientific decision.
6. If checkpoint sealing succeeded before its journal acknowledgement, reconcile with
   `read_checkpoint()` and advance to `COMMITTED` only when its commit id matches.
7. Performance caches, worker leases and GPU state may be discarded and rebuilt.
8. Exact duplicate completions are idempotent; same `work_id` with a different payload hash
   fails closed.

`RetryMode.IDEMPOTENT_RECOMPUTE` is for work that can be safely recomputed from immutable
inputs. `RetryMode.RESUME_FROM_DURABLE_ENGINE_CHECKPOINT` is provided for engines such as
training where the adapter must resume from its own durable checkpoint rather than invent a
new trajectory.

## Dependency injection API

`runtime_protocols_r11.py` defines structural Protocols only:

- `TeacherEngine`
- `TraceEngine`
- `InferenceEngine`
- `TrainingEngine`
- `ValidationEngine`
- `TournamentEngine`
- `EvidenceStore`
- `EventJournal`
- `CheckpointStore`
- `EngineBundle`

No Task A/B/C/D concrete implementation is imported.

## Fault tests

Pure fake-engine/store tests cover:

- Teacher completions out of order
- hard snapshot seal barrier
- stale policy result
- future evidence / future Teacher taint
- Teacher→Student autograd taint declaration
- poison/future-taint event
- current-Champion trace ownership
- duplicate completion idempotency and conflict
- worker crash/requeue with unchanged scientific work identity
- bounded storage backpressure and later pumping
- work stealing across compatible pools
- GPU pool unavailable
- GPU engine adapter unavailable
- late trace completion while GPU mainline has advanced
- rejected Challenger parent retention
- promoted Challenger parent release
- explicit self-laundering attack on REJECT
- future-generation result injection
- deterministic restart in every generation state
- recovery of durable incomplete queued work
- crash after atomic commit before journal acknowledgement

Qualification run for this branch is dependency-free only: `30 passed`. No real Teacher,
H72/Trace, GPU, market data, or self-hosted/Shanxi workflow is invoked.

## INTEGRATION_REQUIREMENT

1. Task A/B/C/D engine adapters must convert their native requests/results to `WorkItem` /
   `WorkCompletion` and the lineage dataclasses without weakening generation, Champion,
   snapshot, Teacher or Physics identity checks.
2. `TrainingEngine` must prove that a retry marked
   `RESUME_FROM_DURABLE_ENGINE_CHECKPOINT` resumes the same scientific training identity
   (including RNG/optimizer/checkpoint state). Task E does not invent training determinism.
3. `EvidenceStore` must provide atomic/idempotent `put_once` and immutable
   `seal_snapshot/get_snapshot`; concrete SSD metadata vs HDD payload layout belongs to the
   storage task.
4. `CheckpointStore.atomic_commit` must be a compare-and-set/idempotent durable operation.
5. `EventJournal.append_once` must be append-only and durable before acknowledgement.
6. Cross-generation CPU Teacher precomputation may overlap current GPU validation only after
   the Teacher task provides a contract declaring which work is parent-Champion-independent.
   Until then, Task E deliberately refuses to accept future-generation scientific work into
   the active generation.
7. Pre-generation frozen historical evidence must be assigned an explicit pre-generation
   lineage (for example `source_generation=-1`, with matching Teacher lineage) before it
   can seed G0. Task E does not invent that scientific ownership convention.
8. Engine adapters must set poison/autograd-taint metadata from their own tensor/runtime
   boundary checks; Task E can enforce the boundary but cannot inspect a future engine's
   internal autograd graph itself.

## Known limitations

- This is an orchestration/state-machine layer, not a process supervisor or concrete
  multiprocessing implementation.
- One scientific generation is active per `R11Orchestrator` instance. Pre-release future
  generation scientific results are not admitted without the integration contract above.
- Storage durability/atomicity guarantees are Protocol obligations and are unit-tested with
  fakes here; concrete storage implementations require integration qualification on their
  own branches.
- No real hardware throughput claim is made by Task E.
