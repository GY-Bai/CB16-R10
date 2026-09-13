# R21 RC2 Infra Architecture Research

> Status: **OWNER / ASTRA DIRECTION ACCEPTED — SOL ARCHITECTURE BASELINE**  
> Date: 2026-09-13  
> Implementation authority: **NO** — implementation is authorized task-by-task in `R21_RC2_INFRA_UPGRADE_TODO.md`.  
> S1 relationship: PR #102 remains draft/frozen until the Recovery stage supplies a qualified execution surface.

This document records the accepted architecture direction for R21 RC2. It does not change frozen S1 science and does not by itself authorize host mutation, storage cleanup, Docker reconfiguration, or a storage-backend rewrite.

## 1. Revised objective

S1 qualification exposed two independent infrastructure gaps:

1. the current Shanxi runner is not yet a fully versioned and reproducible execution surface;
2. durable replay/update/provenance traffic ultimately lands on a single mechanical HDD and becomes I/O-bound before CPU or memory are saturated.

The immediate objective is therefore **not** to build a general infrastructure platform. It is to restore the smallest independently qualified execution surface that can run the already-frozen S1 experiment. Broader automation and storage evolution follow only when evidence justifies them.

## 2. Owner/Astra decision freeze

| Decision | Accepted direction | Boundary |
|---|---|---|
| Constrained OIDC control plane | Long-term accepted | Not required before S1 resumes; first version must expose a small allowlisted operation set and cannot redefine its own authorization policy. |
| Separate/private Infra repository | Long-term preferred | Repository privacy alone is not an approval boundary; actual GitHub plan/features and actor identity separation must be audited first. |
| SSD `FAST_HOT` | Accepted; highest Recovery priority | Capacity inventory and explicit owner-approved cleanup/provision plan come first. No implicit authorization to delete, format or repartition storage. |
| Storage candidates | Evidence-triggered competition | SQLite + SSD first. Valkey, JetStream or RocksDB open only when a frozen target is missed and measured evidence identifies a relevant bottleneck. |
| Exactly-once / provenance | Hard invariant | Must cover logical events, replay visibility, effective learning updates, checkpoint/generation state and proof; message de-duplication alone is insufficient. |

Project preference after correctness: when a solution meets the frozen performance targets with explicit headroom, prefer the operationally simpler qualified solution.

## 3. Two-stage architecture

### 3.1 RC2-Recovery — direct S1 dependency

Recovery contains only work required to remove already-demonstrated blockers:

```text
versioned runner contract
-> permission/shared-memory correction
-> SSD capacity inventory
-> bounded SSD FAST_HOT
-> real write-path instrumentation
-> accepted-runtime placement-only canary
-> conditional SQLite physical batching only if needed
-> persistence/restart/provenance qualification
-> Sol review
-> S1 CI-C reauthorization
```

A task belongs in Recovery only when it removes a demonstrated blocker or is necessary to qualify the next Recovery gate.

### 3.2 RC2-Evolution — not an S1 prerequisite

Evolution includes:

- constrained OIDC infrastructure broker;
- separate privileged control identity;
- possible private Infra repository;
- broader automatic Docker/resource orchestration;
- Valkey Streams, NATS JetStream or RocksDB challengers;
- stronger host-level failure qualification;
- long-term ephemeral/JIT science runners.

Once Recovery is qualified, Evolution must not hold frozen S1 science hostage.

## 4. Long-term control-plane direction

Science jobs should remain unprivileged and should not receive direct host Docker authority. Greater Sol flexibility should come from a declarative, policy-constrained interface that accepts versioned resource profiles rather than arbitrary host paths or raw Docker options.

Preferred long-term trust split:

```text
GitHub policy identity
        -> constrained R21 control service
        -> allowlisted runner/service/resource profile
        -> unprivileged science execution surface
```

GitHub OIDC is suitable for short-lived caller authentication because it can bind repository/ref/workflow/run identity without requiring a long-lived host-management credential in the science workflow.

OIDC proves caller identity; it does not by itself make readable data unreadable or prove process isolation.

References:

- GitHub OIDC: https://docs.github.com/en/actions/concepts/security/openid-connect
- Docker authorization: https://docs.docker.com/engine/extend/plugins_authorization
- OPA Docker authorization: https://openpolicyagent.org/docs/docker-authorization

## 5. Secret boundary correction

A credential that is readable by the science process is not “usable but unreadable” merely because it arrived through a mounted file.

If a future operation requires a credential to be usable but not readable by science code, the credential must remain inside a separate controlled service boundary and science code should call a narrow interface instead of receiving the credential value.

Therefore secret-isolation claims must be based on actual process/service permissions, not on OIDC or mount naming alone.

## 6. GitHub approval boundary correction

A private Infra repository remains a useful long-term separation mechanism, but it does not automatically prove independent approval. Before relying on protected environments or required reviewers, RC2-Evolution must audit:

- current GitHub plan/repository visibility support;
- which protection features are actually available;
- whether the human owner and automation use distinguishable identities;
- whether a privileged workflow could alter the policy that authorizes itself.

This audit is an Evolution task, not a Recovery prerequisite.

## 7. Physical storage architecture

Historical R2 already established the physical principle that should be restored:

```text
SSD / NVMe -> small random + WAL/journal/index/fsync-heavy hot state
HDD        -> large immutable sequential/cold payload
one rotating spindle -> one physical sequential payload-writer lane
```

The current Docker layout flattened runner work/temp and Docker volumes onto one HDD. Recovery first restores physical tiering; it does not begin by adding another database service.

## 8. SQLite remains the first candidate

SQLite WAL permits one active writer per database, but that fact alone does not prove SQLite locking is the present dominant bottleneck. Multiple databases and files can collectively saturate one physical HDD even without contending on the same database lock.

Recovery must first measure the real write path: bytes, write frequency, durability calls, WAL/checkpoint behavior, lock wait and physical-device pressure.

### 8.1 Placement-only comes before runtime rewrite

The first Recovery execution test keeps the accepted S1 runtime unchanged and changes only the qualified execution surface:

```text
accepted runtime
+ versioned runner/resource profile
+ compliant shared memory
+ SSD FAST_HOT
```

If this meets the frozen performance, backlog, recovery and semantic gates, no storage rewrite is justified.

### 8.2 SQLite batching is conditional and creates a new implementation identity

If measurement still shows SQLite transaction/durability frequency as the limiting factor, a bounded single-writer/batch design may be implemented.

The firewall is:

```text
physical write batching != semantic event batching != learner-update batching
```

Physical batching may not change account/path ordering, replay eligibility timing, learner-update boundaries, checkpoint/generation timing, policy-version attribution or provenance identity.

Any commit-protocol or writer-path code change creates a new implementation identity and requires fresh implementation review and semantic-equivalence qualification even though the scientific S1 manifest remains frozen.

Reference: https://sqlite.org/wal.html

## 9. Evidence-triggered challengers

Valkey Streams, NATS JetStream and RocksDB remain approved research candidates, but are not implemented in parallel by default.

A challenger opens only after SQLite + SSD is correct but misses a pre-frozen target and the measured bottleneck explains why that challenger is relevant.

### Valkey

Freeze exact version and AOF policy. `appendfsync everysec` is not equivalent to an every-commit durable boundary. If `WAITAOF` participates in the durability boundary, returned acknowledgement counts must satisfy the required local/replica counts; timeout or insufficient counts are not a successful durable acknowledgement.

References: https://valkey.io/topics/persistence/ and https://valkey.io/topics/streams-intro/

### NATS JetStream

A `PubAck` proves only the guarantee supplied by the selected NATS version and persistence configuration. Exact server version, storage/persist mode, sync policy, replication and fault model must be frozen before it can be compared with CB16 durability requirements.

References: https://docs.nats.io/learn/jetstream/publishing , https://docs.nats.io/learn/jetstream/policies , https://docs.nats.io/reference/config/jetstream

### RocksDB

Freeze exact version and write options. WAL enabled with default asynchronous write behavior is not automatically an fsync-at-commit guarantee. The selected authority boundary must state when synchronous WAL persistence is required and all write-status results must be checked.

References: https://github.com/facebook/rocksdb/wiki/Write-Ahead-Log-(WAL) and https://github.com/facebook/rocksdb/wiki/WAL-Performance

## 10. Exactly-once is an end-to-end property

R21 qualification must distinguish at least:

```text
logical event committed exactly once
experience becomes replay-visible at the frozen semantic point
learner update becomes effective exactly once
parent -> child checkpoint transition is unique
child generation adoption is unique
required proof remains reconstructable
```

Message de-duplication alone does not prove an effective learner update happened once.

## 11. Scientific identity vs implementation identity

The S1 scientific manifest remains frozen: seeds, task definitions, reward, model/optimizer scientific settings, budget, evaluation and gates do not change during Recovery.

Execution-surface changes such as mount placement, shared-memory sizing or resource profiles can be qualified without changing the accepted runtime code identity when the runtime itself is untouched.

A storage-engine replacement, commit-protocol rewrite, SQLite batch-writer rewrite or replay-visibility code change creates a new implementation identity and cannot inherit old runtime code acceptance.

## 12. Failure-model taxonomy

Qualification must state what fault class was actually tested:

| Fault class | Scope of evidence |
|---|---|
| `PROCESS_CRASH` | process-level restart/replay/idempotence |
| `CONTAINER_RESTART` | container lifecycle plus persisted-storage recovery |
| `HOST_REBOOT` | controlled host-reboot recovery; separate operational authorization required |
| `POWER_LOSS` | abrupt power-loss durability; not implied by process/container tests |

A PASS for one fault class may not be promoted to a stronger untested class.

Recovery may qualify process and container restart without claiming host-reboot or power-loss proof.

## 13. Selection rule

Storage selection order is:

1. semantic/durability equivalence;
2. effective-update exactly-once and recovery correctness;
3. pre-frozen performance targets with explicit headroom;
4. operational simplicity;
5. additional throughput/latency margin;
6. CPU/RAM/maintenance burden.

If SQLite + SSD meets all required targets with margin, the default is to keep it and defer additional services.

## 14. Evidence rule

All implementation/runtime evidence must flow through:

```text
GitHub Actions -> authorized Shanxi Docker execution surface -> durable artifacts/receipts -> Sol review
```

Host-side inventory or provisioning may be performed only by an executor explicitly authorized for the corresponding RC2 task. Host changes must also be represented by versioned repository definitions and machine-readable before/after receipts; host observations alone are not sufficient runtime qualification evidence.

Failures retain the existing taxonomy: `PASS`, `SCIENTIFIC_FAIL`, `CONTRACT_MISMATCH`, `EXECUTION_BLOCKED`, `HARDWARE_LIMIT`, `EVIDENCE_INSUFFICIENT`.

## 15. Current conclusion

R21 RC2 is governed by one principle:

> Restore the smallest independently qualified execution surface that removes the demonstrated blocker and lets frozen S1 science run again. Continue control-plane and storage evolution only when evidence justifies the additional complexity.

Task ordering, host-access authority and Recovery performance gates are defined in `docs/infra/R21_RC2_INFRA_UPGRADE_TODO.md`.
