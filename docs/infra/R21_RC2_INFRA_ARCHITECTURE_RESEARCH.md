# R21 RC2 Infra Architecture Research

> Status: **RESEARCH / PROPOSED DIRECTION / NOT YET IMPLEMENTATION AUTHORITY**  
> Date: 2026-09-13  
> Owner goal: rebuild CB16 Infra so formal science can use the available CPU/GPU/memory without being serialized by unsafe Docker control or single-HDD random I/O.

This document is a research/architecture input for R21 RC2. It does not modify frozen S1 science, and it does not yet authorize privileged host changes.

## 1. Problem statement

Current S1 qualification exposed two separate Infra gaps:

1. **Control-plane gap** — GitHub Actions can execute inside the runner container but does not safely control the host's Docker lifecycle, mount placement, storage class, shared-memory size, CPU/memory/GPU assignment, or I/O policy. Giving the normal science runner direct access to the Docker daemon would create an unacceptable privilege boundary.
2. **Data-plane gap** — durable replay/update/provenance writes are routed through one mechanical HDD. The workload is dominated by random writes, SQLite transactions and fsync/fdatasync latency rather than compute.

R21 RC2 should solve both. A storage rewrite without a safe control plane will remain operationally fragile; a powerful Docker control plane without storage tiering will just reproduce the same I/O bottleneck faster.

---

# 2. Security finding: do not expose the Docker socket to the science runner

Docker's default daemon access model is effectively all-or-nothing. A process that can freely control a rootful Docker daemon can normally create privileged containers, mount host filesystems and obtain host-equivalent authority. Docker provides authorization plugins because direct daemon access is otherwise too coarse.

GitHub separately warns that persistent self-hosted runners are not clean ephemeral trust boundaries and can be persistently compromised by workflow code. GitHub recommends ephemeral/JIT runners for stronger isolation and advises particular caution for self-hosted runners serving public repositories.

Therefore R21 RC2 should explicitly forbid:

```text
science runner -> /var/run/docker.sock
science workflow -> arbitrary docker CLI against host daemon
science workflow -> arbitrary host bind mounts
science workflow -> arbitrary privileged/container exec
```

Giving Sol more flexibility should mean **more declarative control through a constrained interface**, not raw root access.

References:

- GitHub Secure use reference: https://docs.github.com/en/actions/reference/security/secure-use
- GitHub self-hosted runner reference: https://docs.github.com/en/actions/reference/runners/self-hosted-runners
- Docker authorization plugins: https://docs.docker.com/engine/extend/plugins_authorization
- OPA Docker authorization: https://openpolicyagent.org/docs/docker-authorization

---

# 3. Recommended control-plane shape

## 3.1 Three trust domains

```text
GitHub / policy plane
        |
        | OIDC signed job identity
        v
R21 Infra Control Plane
(root-owned, narrow API, policy-checked)
        |
        | allowlisted declarative operations
        v
Docker / storage / cgroup host authority
        |
        +------------------------------+
        |                              |
        v                              v
unprivileged science runner      infra services
(no Docker socket)               Valkey/NATS/RocksDB/etc.
```

### A. Science plane

Runs model/science code only.

- non-root;
- no host Docker socket;
- no arbitrary host filesystem access;
- receives only approved `/cb16/*` mounts;
- receives CPU/GPU/memory/shm/storage classes decided by a versioned runner profile;
- science jobs cannot mutate their own host privilege profile.

### B. Infra control plane

A small root-owned host service, tentatively `cb16-infra-broker`, is the only component allowed to perform privileged Docker/storage operations.

The broker should expose a **small operation vocabulary**, not arbitrary shell commands or arbitrary Docker JSON.

Allowed operation classes should look like:

```text
inspect_sanitized_inventory(profile)
validate_storage_profile(profile_id)
apply_runner_profile(versioned_profile_id)
start_known_service(service_profile_id)
stop_known_service(service_profile_id)
recreate_science_runner(versioned_profile_id)
set_resource_profile(cpu/memory/pids/shm/gpu/io)
prepare_job_storage(run_id, storage_profile_id)
cleanup_job_storage(run_id)
collect_resource_telemetry(run_id)
```

Forbidden API surface:

```text
run arbitrary shell
arbitrary docker run arguments
arbitrary bind mount source
read arbitrary host file
return secret values
mount Docker socket into a workload
arbitrary docker exec
privileged=true unless a separately frozen profile requires it
```

### C. GitHub policy plane

A workflow requests a GitHub OIDC JWT (`id-token: write`) and sends it to the broker. The broker verifies GitHub's signature and exact claims before accepting any privileged operation.

The broker should bind at least:

- immutable repository identity / repository id;
- exact approved ref;
- protected environment;
- exact reusable workflow identity (`job_workflow_ref` / workflow SHA where available);
- expected audience;
- optionally actor / owner constraints;
- expiry and run identity.

GitHub's OIDC model lets external services issue short-lived authorization based on these signed workflow claims without storing a long-lived host credential in GitHub.

References:

- GitHub OIDC: https://docs.github.com/en/actions/concepts/security/openid-connect
- GitHub OIDC with Vault: https://docs.github.com/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-hashicorp-vault
- HashiCorp validated GitHub Actions + Vault pattern: https://developer.hashicorp.com/validated-patterns/vault/retrieve-vault-secrets-from-github-actions

---

# 4. Strong recommendation: split science runner from privileged control runner

The component that recreates or changes the science runner must not be the same process/container that it is trying to destroy or reconfigure.

Recommended topology:

```text
shanxi-r21-science
  - unprivileged
  - receives science jobs
  - ephemeral/JIT preferred over time
  - no Docker socket

shanxi-r21-infra-control
  - separate runner identity
  - only accepts infra-control workflow
  - no science payload execution
  - talks to cb16-infra-broker

cb16-infra-broker
  - host service
  - actual privileged Docker/storage/cgroup authority
  - validates OIDC/policy
```

Because the CB16 source repository is a science/code surface, a stronger long-term boundary is a **separate private infra-control repository** whose only purpose is to host reviewed R21 control workflows and versioned infra profiles. A privileged workflow should not be writable through normal science PRs.

If a separate private repository is not immediately available, an interim same-repo control workflow must at minimum be `workflow_dispatch` only, reference a protected GitHub Environment with required review/prevent-self-review, be pinned to main/exact reusable workflow identity, and still use the broker rather than the raw Docker socket.

---

# 5. Secret handling: host secrets should stay on the host

The broker must not provide an API such as `get_secret(name)`.

Instead, use **secret profiles by reference**:

```text
workflow requests: secret_profile = SCIENCE_RUNNER_NETWORK_V1
broker resolves host-owned secret files internally
broker mounts/injects only into the approved child container
workflow never receives the value
```

For credentials that must be consumed directly by a workflow, prefer short-lived credentials obtained through GitHub OIDC. HashiCorp Vault is a strong optional candidate because it can validate GitHub OIDC claims and issue minute-scale tokens without a static GitHub secret.

R21 RC2 does not require Vault to exist on day one. The minimum safe architecture is:

- host-owned secret material remains unreadable to science runner;
- OIDC authenticates privileged control requests;
- broker never logs secret values;
- profiles reference secrets by opaque id only;
- no personal SSH key or broad PAT is inserted into the runner.

---

# 6. Docker resource control that R21 RC2 should expose

Docker already supports the resource dimensions CB16 needs. The broker/profile layer should expose a validated subset:

- CPU limit / CPU shares / cpuset;
- memory hard and soft reservations;
- swap/swappiness policy;
- pids limit;
- `/dev/shm` size;
- GPU assignment;
- block-I/O weight;
- per-device read/write BPS or IOPS limits where useful;
- mount/storage profile;
- network profile;
- read-only root filesystem/capability/security options.

Reference: https://docs.docker.com/engine/containers/resource_constraints/

The R21 abstraction should be declarative, for example:

```json
{
  "profile_id": "R21_SCIENCE_QUALIFICATION_RC2",
  "cpu_class": "SCIENCE_FULL_HOST_BOUNDED",
  "memory_class": "SCIENCE_HIGH",
  "gpu_class": "GTX1060_SINGLE",
  "shm_class": "SCIENCE_2G",
  "hot_storage_class": "FAST_HOT_V1",
  "cold_storage_class": "COLD_HDD_V1",
  "io_policy": "HOT_RANDOM_COLD_SEQUENTIAL_V1"
}
```

The workflow selects an approved profile id. It should not supply raw host paths, arbitrary device names or unlimited Docker flags.

---

# 7. Storage architecture: Redis-like front layer is plausible, but only if its authority is explicit

The owner's idea of a Redis-like high-frequency layer in front of SQLite targets a real issue: SQLite still serializes writers. WAL improves reader/writer concurrency and turns writes into a more sequential log, but SQLite documents that a WAL database still has only one writer at a time.

Reference: https://sqlite.org/wal.html

A memory-first service can absorb concurrent producers and let a materializer batch many logical events into fewer SQLite transactions. **But this only helps if it actually reduces synchronous SQLite commit frequency.**

Bad design:

```text
worker -> Redis -> immediately commit same event to SQLite
```

This adds another hop while preserving the same SQLite fsync pressure.

Useful design:

```text
N workers
   -> durable ingress log / queue
   -> bounded batching + idempotent materializer
   -> metadata store
   -> cold sequential payload writer
```

The critical question is what proves a write is durable before the producer is allowed to forget it.

---

# 8. Redis vs Valkey

For a new R21 open-source dependency, **Valkey should be evaluated before Redis**.

Valkey is a Linux Foundation project under the BSD 3-Clause license and remains protocol/command compatible with the Redis OSS lineage. It supports Streams, consumer groups, RDB and AOF persistence.

References:

- https://valkey.io/
- https://valkey.io/topics/streams-intro/
- https://valkey.io/topics/persistence/

## 8.1 Why Valkey Streams fits CB16

Valkey Streams gives:

- ordered append-only event ids;
- multiple producers;
- consumer groups;
- pending-entry tracking;
- replay;
- crash recovery of stream/consumer state when persistence is configured;
- bounded retention controls.

A possible R21 path is:

```text
worker events
   -> Valkey Stream on SSD
   -> single/bounded materializer
   -> SQLite/RocksDB metadata
   -> HDD immutable pack writer
```

## 8.2 Important durability warning

`appendfsync everysec` can lose roughly the most recent second in a severe crash. That does **not** match CB16's current exactly-once/durable-update expectations if the producer treats the Valkey acknowledgment as final authority.

For a semantics-preserving candidate, R21 must benchmark a stronger policy such as:

- AOF with `appendfsync always`, which can still group concurrent commands into fewer fsyncs; and/or
- explicit AOF durability confirmation (`WAITAOF` where applicable);
- stable event ids and idempotent materialization.

If Valkey is only a non-authoritative cache and SQLite remains the durability boundary for each event, it will not solve the current commit bottleneck.

---

# 9. NATS JetStream: strong candidate for the durable ingress-log role

JetStream is closer to a purpose-built event log than a general cache.

Useful properties:

- persistent streams;
- server `PubAck` confirming accepted storage under durable/default persistence mode;
- asynchronous producer batching;
- durable consumers;
- explicit ack/redelivery;
- publication deduplication via stable message ids;
- replay and backpressure.

References:

- https://docs.nats.io/concepts/jetstream
- https://docs.nats.io/learn/jetstream/publishing.md
- https://docs.nats.io/learn/jetstream/delivery-and-acknowledgment

Potential shape:

```text
N science workers
   -> JetStream durable event log on SSD
   -> materializer consumer
   -> RocksDB / SQLite projection
   -> sequential cold pack writer
```

For CB16 this maps naturally to immutable event/provenance production, but it adds a broker service and a second event identity model. Its de-duplication and acknowledged consumption must be mapped carefully onto CB16 update ids and exactly-once semantics.

---

# 10. RocksDB: strongest embedded candidate for hot metadata / high-write state

RocksDB is worth evaluating because its write path is already built around:

- memtables;
- sequential WAL;
- WriteBatch;
- group commit;
- concurrent writer coordination;
- background flush/compaction;
- separate `wal_dir`, allowing WAL to be placed on faster storage.

References:

- https://github.com/facebook/rocksdb/wiki/RocksDB-Overview
- https://github.com/facebook/rocksdb/wiki/Write-Ahead-Log-(WAL)
- https://github.com/facebook/rocksdb/wiki/WAL-Performance
- https://github.com/facebook/rocksdb/wiki/Pipelined-Write

RocksDB can potentially replace the highest-frequency SQLite KV/catalog workload while SQLite remains a derived reporting/query projection.

Candidate shape:

```text
workers
   -> RocksDB WriteBatch / WAL on SSD
   -> memtable
   -> SST / archive placement
   -> optional SQLite reporting projection
```

Important cost: RocksDB moves complexity from SQLite locking to LSM flush/compaction management. Compaction on the HDD could create another I/O problem, so WAL/SST placement and background-job limits must be benchmarked on the real Shanxi topology.

---

# 11. SQLite should remain a baseline, not be assumed obsolete

Before adding a service, R21 should benchmark a simpler baseline:

```text
N workers
 -> one bounded in-process/interprocess writer queue
 -> one SQLite WAL writer on SSD
 -> large transactions / prepared batches
 -> checkpoint managed separately
```

SQLite WAL already provides sequential WAL writes, concurrent readers and fewer fsyncs; its key limit is one active writer. If one dedicated writer on SSD can satisfy the required throughput, this is operationally much simpler than introducing Valkey/NATS.

Therefore R21 RC2 should treat **SQLite-on-SSD with one batched writer** as the baseline that every more complex solution must beat.

---

# 12. Candidate matrix

| Candidate | Role | Main strength | Main risk | Recommended R21 status |
|---|---|---|---|---|
| SQLite WAL + single batch writer | baseline durable metadata | simplest; current semantics closest | one writer; must batch well | **Benchmark first** |
| Valkey Streams + AOF strong durability | high-frequency ingress/event buffer | memory speed, stream consumers, group fsync | must define durable authority; RAM/AOF rewrite | **Strong candidate** |
| NATS JetStream | durable ingress log | explicit pub/ack/replay/dedup/backpressure | new broker + identity mapping | **Strong candidate** |
| RocksDB | hot embedded metadata store | group commit, WAL+memtable, high write throughput | compaction/LSM tuning | **Strong candidate** |
| Redis as volatile cache | cache only | easy, fast | does not remove durability bottleneck | **Not sufficient** |
| direct concurrent SQLite writers on HDD | current anti-pattern | none for this host | seek/fsync contention | **Reject** |

---

# 13. Recommended R21 RC2 data-plane research topology

Do not choose one database prematurely. Build a common benchmark adapter and compare three production-shaped paths:

```text
Path A — Minimum change
workers -> bounded queue -> single SQLite WAL writer on SSD

Path B — Durable log + projection
workers -> Valkey Streams OR JetStream on SSD
        -> materializer -> SQLite metadata on SSD
        -> HDD sequential payload writer

Path C — Embedded high-throughput store
workers -> RocksDB WAL/memtable on SSD
        -> optional SQLite reporting projection
        -> HDD sequential payload writer
```

All paths must retain exactly the same logical update ids, content hashes, replay eligibility, checkpoint/generation facts and artifact proof. Performance cannot be accepted until semantic equivalence passes.

---

# 14. One physical HDD still means one physical cold writer

R21 does not repeal the historical R2 rule.

Even if Valkey, NATS or RocksDB removes hot metadata pressure from the HDD, large cold payload writes should still converge through one bounded sequential writer lane per spindle:

```text
compute/learner workers
        |
        +-> fast durable hot layer on SSD
        |
        +-> bounded cold-payload queue
                      |
                      v
             single HDD pack writer
```

The HDD should become boring: large reads and large sequential append, not transaction metadata.

---

# 15. R21 RC2 observability requirements

A formal run should emit an execution receipt that binds scientific evidence to physical execution conditions.

At minimum collect:

- container profile id/hash;
- image digest;
- CPU/memory/pids/shm/GPU profile;
- hot/cold mount identities and physical storage class;
- disk rotational flag and filesystem;
- free space/inodes before run;
- device util/await/queue depth over time;
- process D-state count;
- CPU user/system/iowait;
- per-service queue lag;
- SQLite WAL/checkpoint metrics or equivalent;
- Valkey/NATS/RocksDB persistence/queue metrics if used;
- cold writer queue depth and bytes;
- artifact/checkpoint/provenance semantic hashes.

A run that silently falls back from SSD hot storage to HDD should fail closed before science starts.

---

# 16. Recommended security decision for RC2

Preferred long-term design:

```text
private infra-control repo
        |
protected manual workflow + GitHub Environment
        |
GitHub OIDC short-lived identity
        |
cb16-infra-broker (root-owned host service)
        |
versioned allowlisted R21 profiles
        |
Docker + storage + cgroups
```

Science workflows remain in CB16 and use only the resulting unprivileged execution surface.

OPA Docker authorization can be used as an additional defense if direct authenticated Docker Engine API access is ever introduced, but it should be **defense in depth**, not the primary reason raw Docker daemon access becomes acceptable. Docker notes authorization plugin limits around upgraded/streaming connections; a narrow broker API is easier to reason about for CB16.

---

# 17. Proposed R21 RC2 decisions to freeze after owner/Astra review

1. No raw Docker socket in science runner.
2. Separate science runner and infra-control identity.
3. Privileged operations go through a narrow broker and versioned profiles.
4. GitHub OIDC is the primary secretless authentication mechanism for control requests.
5. Host secrets are referenced by profile id and never returned to workflow code.
6. Restore SSD hot / HDD cold storage classes.
7. Preserve one physical cold writer lane per rotating HDD.
8. Benchmark SQLite-batched baseline, Valkey/JetStream durable-log paths and RocksDB hot-store path.
9. Redis-like layer is accepted only if its durability boundary is explicit and it reduces downstream commits.
10. No science constant may be changed to compensate for Infra throughput.

These decisions should become an executable R21 RC2 TODO only after the owner/Astra confirms the architecture direction.
