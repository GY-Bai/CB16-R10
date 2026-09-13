# R21 RC2 Infra Upgrade TODO

> Status: **SOL EXECUTION TODO V2 — OWNER / ASTRA DIRECTION ACCEPTED**  
> Source: `docs/infra/R21_RC2_INFRA_ARCHITECTURE_RESEARCH.md`  
> Existing debt: `docs/infra/INFRA_ENGINEERING_DEBT_TODO.md`  
> Live runner baseline: `docs/infra/SHANXI_DOCKER_RUNNER_CONTRACT.md`  
> PR #102 stays draft/frozen until **RC2-Recovery** is independently qualified.

This TODO authorizes implementation task by task. It does not authorize changing S1 scientific semantics or performing unlisted host changes.

## 1. Program split

### RC2-Recovery

Direct dependency of frozen S1:

```text
R0 recovery contract freeze
-> R1 reproducible runner definition
-> R2 SSD capacity/provision proposal
-> R3 parallel Recovery runner + FAST_HOT
-> R4 real write-path measurement
-> R5 placement-only accepted-runtime canary
-> R6 conditional SQLite batching only if evidence requires it
-> R7 Recovery qualification
-> R8 Sol review + S1 CI-C reauthorization
```

### RC2-Evolution

Not a prerequisite for S1 resume:

- constrained OIDC control plane;
- separate control identity/private Infra repository if feasible;
- Valkey, JetStream or RocksDB challengers only when triggered by measured evidence;
- broader automatic resource control and stronger failure testing.

## 2. Hard invariants

1. S1 manifest/seeds/task/reward/model/optimizer/scientific thresholds/budget/evaluation remain frozen.
2. Infrastructure failure is never relabeled `SCIENTIFIC_FAIL`.
3. Physical write batching must not change account order, replay visibility, learner-update count/order, checkpoint/generation timing, policy-version attribution or provenance.
4. Message de-duplication is not proof of effective learner-update exactly-once.
5. Storage-engine, writer-path or commit-protocol code changes create a new implementation identity and require fresh Sol review.
6. Runtime evidence must come from GitHub Actions -> Shanxi Docker -> durable evidence -> Sol review.

## 3. Recovery performance target V1

These are engineering targets, not scientific gates. They are frozen before storage benchmarking.

- Full frozen S1 CI-C target wall time: **<= 240 minutes**. Existing 360-minute workflow timeout remains only the hard ceiling.
- 8-worker production-shaped canary: no sustained positive durable-write backlog growth over the final 25% of steady state.
- After producers stop: backlog returns to <= one configured physical batch within **60 seconds**.
- Hot durable-boundary latency: p50 <= **50 ms**, p99 <= **250 ms**.
- Process-level recovery: <= **60 seconds**.
- Container-level recovery: <= **180 seconds**.
- Every `FAST_HOT` surface resolves to non-rotational storage; fallback to HDD is a contract failure.
- Zero lost acknowledged logical events, zero duplicate effective learner updates, and required checkpoint/provenance equivalence for the tested fault class.

If a required metric cannot be measured, verdict is `EVIDENCE_INSUFFICIENT`, not PASS.

## 4. Execution authority matrix

`Host session` means an executor with explicit permission to access the Shanxi host. `Change` means host/container configuration changes are part of the task. Exact operations must be proposed in the task-local note before execution.

| Task | Purpose | Suggested branch | GitHub Actions | Host session | Host/container change | Owner approval before change | New runtime implementation identity |
|---|---|---|---|---|---|---|---|
| R0 | Freeze Recovery authority/performance/fault contract | `ai/r21-rc2-r0-recovery-freeze` | no | no | no | no | no |
| R1 | Version/reproduce current runner definition | `ai/r21-rc2-r1-runner-spec` | yes | **read-only** | no | no | no |
| R2 | SSD capacity, reserve, protected list and provision proposal | `ai/r21-rc2-r2-ssd-plan` | no | **read-only** | no | **yes before R3** | no |
| R3 | Build a parallel R21 Recovery runner using approved FAST_HOT profile | `ai/r21-rc2-r3-fast-hot-runner` | yes | **change required** | **yes** | **yes** | no S1 runtime code change |
| R4 | Measure actual S1-shaped write path | `ai/r21-rc2-r4-writepath-inventory` | yes | read-only if host telemetry is needed | no | no | no |
| R5 | Placement-only canary with accepted S1 runtime | `ai/r21-rc2-r5-placement-canary` | **yes** | read-only if telemetry is needed | no | no | no |
| R6 | Conditional SQLite physical batch writer | `ai/r21-rc2-r6-sqlite-batch-r0` | **yes** | normally no | no | no | **yes** |
| R7 | Recovery/restart/provenance qualification | `ai/r21-rc2-r7-recovery-qualification` | **yes** | change access only for the container-level recovery test | limited to task-defined Recovery component | task authorization | depends on R6 |
| R8 | Sol review and S1 reauthorization | reviewer-owned | evidence read | no | no | no | no |

Tasks are sequential authority gates. Do not run R3 host changes before R0/R1 are accepted and the owner approves the exact R2 capacity/provision proposal.

## 5. R0 — Recovery contract freeze

Deliver a machine-readable Recovery spec containing:

- task registry;
- `PERF-RC2-V1` targets above;
- fault classes;
- scientific-vs-implementation identity rules;
- required receipts/artifacts;
- verdict taxonomy.

No host access. No runtime changes.

## 6. R1 — Reproducible runner definition

Use the current live contract/snapshot as input and version:

- runner image/build definition;
- launch/profile definition;
- uid/gid contract;
- mount/storage-class declarations;
- shared-memory declaration;
- runner/image identity;
- sanitized snapshot generator.

**Host session: read-only required.** The authorized DS may inspect the current runner/container/storage metadata needed to prove the repository definition matches reality. R1 must not change the live runner.

Gate: a disposable/recovery runner created from repo definitions reproduces the declared contract and passes Shanxi preflight.

## 7. R2 — SSD capacity/provision proposal

Produce a machine-readable inventory with:

- SSD total/used/free;
- protected/keep list;
- reclaim candidates and sizes;
- explicit proposed cleanup list, if any;
- proposed `FAST_HOT` quota;
- required free-space reserve;
- expected before/after capacity.

**Host session: read-only required.** R2 does not authorize cleanup or configuration changes.

R3 is blocked until the owner approves the concrete capacity/provision proposal.

## 8. R3 — Parallel Recovery runner + FAST_HOT

Provision a **parallel** R21 Recovery runner and retain the current R11 runner as rollback/reference.

The new profile must provide:

- approved SSD-backed `FAST_HOT` storage exposed through a stable container path;
- correct runner write ownership for job-scoped hot data;
- versioned shared-memory profile of **2 GiB**;
- existing protected scientific inputs kept read-only;
- dedicated R21 runner identity/label;
- exact repo-defined image/profile identity.

**Host session: change access required. Container configuration changes are required. Owner approval is required before execution.**

The task-local implementation note must state the exact intended host/container changes before they are applied. Unrelated host services/storage are out of scope.

Gate: GitHub Actions on the new R21 runner proves profile identity, write permissions, shared-memory size, non-rotational FAST_HOT placement and protected-input read-only status.

## 9. R4 — Real write-path measurement

Instrument one bounded S1-shaped run and produce:

```text
surface
bytes/update
writes/update
durable-sync frequency
random vs sequential
SQLite transaction/lock wait where measurable
WAL/checkpoint activity
physical device
consumer
durability requirement
```

Include replay, SQLite, update journal, checkpoints, generation-switch receipts, provenance and artifact staging.

GitHub Actions is required. A read-only host session is allowed only when device telemetry cannot be collected from the runner. No host change and no science-semantic change.

Gate: R5/R6 decisions cite measured surfaces rather than assumptions.

## 10. R5 — Placement-only accepted-runtime canary

Use the already accepted S1 runtime unchanged on the qualified R21 Recovery execution surface.

Required gates:

- semantic/provenance equivalence to the accepted bounded canary;
- applicable `PERF-RC2-V1` targets;
- no FAST_HOT fallback to HDD;
- no sustained backlog growth;
- failure artifacts retained.

Decision:

```text
R5 meets all targets with headroom -> skip R6 and proceed to R7
R5 is correct but misses a target -> Sol reviews R4/R5 before opening R6 or an alternate targeted task
```

## 11. R6 — Conditional SQLite physical batching

R6 is **closed by default**. Sol opens it only if R4/R5 show that SQLite transaction/durability frequency remains a material bottleneck.

Required proof:

- event/order equivalence;
- replay-visibility timing equivalence;
- exactly one effective learner update at every frozen update boundary;
- checkpoint/generation equivalence;
- policy-generation attribution equivalence;
- provenance reconstruction equivalence;
- retry/recovery idempotence.

`NEW_IMPL_ID=YES`. Old runtime code acceptance cannot be inherited. DS must produce a new exact-SHA candidate and return it to Sol for independent review before R7.

## 12. R7 — Recovery qualification

R7 qualifies only the failure classes actually tested.

### Process-level recovery

Run under GitHub Actions and verify restart/replay/idempotence of the durable path. A PASS here proves only process-level recovery.

### Container-level recovery

A host-capable authorized executor is required because the execution component under test cannot independently provide the external restart boundary.

**Host session: task-limited change access required.** Only the R21 Recovery component(s) named in the task-local plan are in scope.

`HOST_REBOOT` and `POWER_LOSS` are not Recovery PASS claims unless separately authorized and tested.

Final R7 gates:

- semantic equivalence PASS;
- effective-update exactly-once PASS;
- artifact-only provenance reconstruction PASS;
- applicable performance/backlog/latency gates PASS;
- storage/profile identity PASS;
- failure evidence retained;
- frozen S1 scientific conditions unchanged.

## 13. R8 — Sol review and S1 CI-C reauthorization

Sol independently reviews repo diff, runner/profile identity, host-change receipts, R4/R5 evidence, any R6 implementation, R7 evidence and the frozen performance gates.

If R6 was skipped, Sol may bind the **already accepted S1 runtime** to the new qualified R21 execution surface.

If R6 changed runtime code, Sol must first accept the new implementation SHA/tree. The scientific manifest remains frozen but old runtime-code acceptance is not reused.

Only after R8 acceptance may PR #102 leave execution-frozen state and formal S1 CI-C be dispatched again.

## 14. RC2-Evolution task triggers

Evolution is not an S1 blocker.

| Task | Trigger | Purpose |
|---|---|---|
| E0 | can run after/alongside late Recovery | Audit available GitHub protection/identity capabilities. |
| E1 | separate owner authorization | Narrow OIDC/profile control plane. |
| E2 | E0 supports it and owner chooses | Optional private Infra repository/control identity. |
| E3-V | SQLite misses frozen target and an ingress/log bottleneck is measured | Valkey challenger. |
| E3-N | SQLite misses frozen target and a durable event-log bottleneck is measured | JetStream challenger. |
| E3-R | SQLite misses frozen target and the embedded transaction/store path is measured as the bottleneck | RocksDB challenger. |

Do not implement Valkey, JetStream and RocksDB in parallel by default.

## 15. Challenger durability requirements

### Valkey

Freeze exact version, persistence mode, AOF policy, durable-ack rule, acknowledgement-count requirements and timeout behavior. Insufficient confirmation cannot be treated as durable success.

### NATS JetStream

Freeze exact server/client version, storage mode, persistence/sync configuration, replication if used, de-dup semantics, publish-ack meaning and tested fault class. `PubAck` cannot be promoted beyond the selected configuration's actual guarantee.

### RocksDB

Freeze exact version, WAL state, synchronous durable-boundary rule, status/error handling, batch semantics, WAL/data placement, background-work budget and tested fault class. WAL enabled alone is not proof of synchronous durable commit.

## 16. DS delivery protocol

Every DS task returns:

```text
READY_FOR_SOL_REVIEW
```

with:

- branch and exact head SHA/tree;
- files changed;
- authority/spec version;
- whether a Shanxi host session was used and whether it was read-only or change-authorized;
- host-change receipt when applicable;
- exact GitHub Actions run/job/artifact ids;
- tests/hostile tests;
- implementation-identity impact;
- unresolved items;
- explicit statement that no S1 scientific constant changed.

DS does not self-certify Recovery acceptance. Sol reviews each gate independently.

## 17. Immediate next tasks

Authorize in order:

```text
R0 -> R1 -> R2
```

R3 waits for owner approval of the exact R2 capacity/provision plan.

No storage challenger is currently authorized.
