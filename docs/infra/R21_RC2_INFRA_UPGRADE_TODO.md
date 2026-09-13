# R21 RC2 Infra Upgrade TODO

> Status: **DRAFT FOR OWNER / ASTRA ARCHITECTURE REVIEW**  
> Source research: `docs/infra/R21_RC2_INFRA_ARCHITECTURE_RESEARCH.md`  
> Existing debt: `docs/infra/INFRA_ENGINEERING_DEBT_TODO.md`  
> Active dependency: PR #102 remains draft/frozen until RC2 supplies a qualified execution surface.

This TODO is intentionally architecture-first. It does not authorize changing S1 science and does not pre-select Redis/Valkey/NATS/RocksDB before measured qualification.

## RC2-0 — Freeze goals and non-goals

### Goals

R21 RC2 must provide:

1. reproducible, versioned Docker runner definition;
2. safe privileged control plane without exposing host Docker socket to science jobs;
3. secretless or short-lived authenticated control using GitHub OIDC;
4. explicit CPU/memory/pids/shm/GPU/I/O/storage resource profiles;
5. SSD hot / HDD cold physical tiering;
6. high-throughput durable write path for replay/update/provenance;
7. one sequential physical cold writer lane per rotating HDD;
8. production-shaped storage benchmark and semantic equivalence gates;
9. machine-readable execution receipt and I/O telemetry;
10. a qualified execution surface capable of resuming frozen S1 formal qualification.

### Non-goals

RC2 must not:

- alter S1 seeds/reward/model/optimizer/threshold/budget;
- weaken exactly-once/durability/provenance requirements;
- expose `/var/run/docker.sock` to the science runner;
- add arbitrary host shell/root access to GitHub workflow code;
- treat a volatile cache as durable authority without an explicit contract;
- declare a storage winner from microbenchmarks alone.

---

# RC2-1 — Version the runner and host-control contract

## Deliverables

Add version-controlled infrastructure definitions for:

- runner Dockerfile;
- launch/compose-equivalent specification;
- container uid/gid initialization;
- resource profile schema;
- mount/storage class schema;
- shared-memory profile;
- service dependencies;
- cleanup lifecycle;
- sanitized machine snapshot generator.

The live ad-hoc container is reference input only. RC2 must be reproducible from repository artifacts plus host-owned secret/material profiles.

## Gate

On a disposable test instance:

```text
build from repo spec
-> launch
-> exact image digest recorded
-> mount/resource profile matches manifest
-> preflight PASS
-> destroy
-> recreate
-> same declared contract PASS
```

---

# RC2-2 — Build the constrained Infra Control Plane

## Required topology

```text
GitHub infra-control workflow
        |
        | GitHub OIDC JWT
        v
cb16-infra-broker
        |
        | allowlisted profile actions only
        v
Docker / mount / cgroup host authority
```

## Broker requirements

The broker is root-owned on the host and must:

- verify GitHub OIDC signature;
- require explicit expected audience;
- bind immutable repo/workflow/ref/environment identity;
- reject expired/mismatched identities;
- maintain an allowlist of versioned profile ids;
- log operation id, caller identity, profile hash and result;
- never return host secret values;
- fail closed if policy cannot be evaluated.

## Allowed operation surface

Only predefined actions such as:

- sanitized inventory;
- apply/recreate known runner profile;
- start/stop known infra service profile;
- prepare/cleanup run-scoped storage;
- set approved resource profile;
- collect telemetry.

No arbitrary shell, raw Docker request passthrough, arbitrary bind source, arbitrary host-file read or arbitrary `exec`.

## Preferred trust split

Long-term preferred:

- separate unprivileged science runner;
- separate infra-control runner identity;
- private infra-control repository + protected environment;
- broker as the only root authority.

## Gate

Hostile authorization tests must show:

- wrong repo/ref/workflow/audience -> denied;
- stale token -> denied;
- unknown profile -> denied;
- arbitrary host path -> denied;
- secret-read attempt -> denied;
- approved exact profile -> allowed;
- broker unavailable -> fail closed.

---

# RC2-3 — Establish physical storage classes

Freeze machine-readable classes:

```text
FAST_HOT_SCRATCH
FAST_METADATA
COLD_IMMUTABLE_PAYLOAD
READ_ONLY_INPUT
```

## Required host work

Before using the MX500 SSD as hot storage:

1. inventory root filesystem usage;
2. identify reclaimable Docker/cache/log/obsolete files without deleting science authority;
3. establish hard minimum free-space reserve;
4. create bounded SSD-backed hot root;
5. expose it to containers under a stable `/cb16/*` abstraction;
6. record physical device, filesystem, rotational flag, capacity, owner/mode and cleanup owner.

Do not move the whole Docker data-root to the SSD merely to get speed.

## Gate

A preflight must machine-prove the hot class resolves to non-rotational storage and the cold class resolves to the intended HDD.

---

# RC2-4 — Inventory the real write path

Before selecting a database, instrument one bounded S1-shaped run and attribute bytes/write frequency/fsync calls to:

- transition/fact persistence;
- replay metadata;
- replay payload;
- SQLite DB/WAL/journal;
- learner/update journal;
- checkpoints;
- generation-switch receipts;
- provenance JSON/JSONL;
- artifact staging;
- cold immutable payload.

Produce a table:

```text
surface -> bytes/update -> writes/update -> fsyncs/update -> random/sequential -> durability requirement -> current consumer
```

This is mandatory. RC2 must optimize measured hot surfaces, not guessed ones.

---

# RC2-5 — Build a common storage benchmark harness

All candidates must consume the same deterministic synthetic logical event stream and emit the same canonical materialized state/hash.

## Fixed comparison paths

### Path A — SQLite WAL baseline

```text
N producers
 -> bounded queue
 -> one batch writer
 -> SQLite WAL on SSD
 -> controlled checkpoint
```

Purpose: determine how much of the bottleneck disappears from physical placement + batching alone.

### Path B1 — Valkey Streams durable ingress

```text
N producers
 -> Valkey Stream on SSD
 -> durable ack policy
 -> idempotent batch materializer
 -> SQLite projection
```

Qualify strong durability only. `appendfsync everysec` cannot be treated as equivalent to current exactly-once durability without an explicit changed authority, so the semantics-preserving benchmark must include stronger AOF confirmation.

### Path B2 — NATS JetStream durable ingress

```text
N producers
 -> JetStream stream on SSD
 -> PubAck + stable message id
 -> durable consumer/materializer
 -> SQLite projection
```

Map publish de-duplication and consumer ack to CB16 update ids and replay exactly-once requirements.

### Path C — RocksDB hot store

```text
N producers
 -> RocksDB WriteBatch/WAL on SSD
 -> memtable/SST
 -> optional SQLite reporting projection
```

Qualify WAL sync/group commit and compaction behavior. `wal_dir`/data placement must be measured on Shanxi hardware.

## Common metrics

- logical events/s;
- durable acknowledgments/s;
- materialized updates/s;
- p50/p95/p99 producer latency;
- p50/p95/p99 durable-ack latency;
- fsync/fdatasync count and latency;
- bytes written by device;
- write amplification;
- CPU/RSS;
- HDD/SSD util/await/queue depth;
- recovery time;
- duplicate/loss count under retry/crash;
- canonical final-state hash.

---

# RC2-6 — Crash, restart and hostile equivalence

For every candidate run hostile interruption at multiple boundaries:

```text
before durable ingress ack
after ingress ack / before materialization
mid materialization
after materialization / before consumer ack
mid cold-pack append
before/after checkpoint/generation receipt
```

Required result:

- no acknowledged logical event disappears;
- retries do not produce a second logical update;
- recovered state converges to the same canonical hashes;
- provenance remains reconstructable from exported evidence;
- corrupted mandatory evidence fails closed.

A candidate that is fastest but cannot satisfy these invariants is rejected.

---

# RC2-7 — Cold payload physical writer

Restore historical R2 physical-lane discipline:

```text
one rotating HDD -> one physical sequential payload writer
```

Implement/qualify:

- bounded producer queue;
- byte/count backpressure;
- batch/framed pack append;
- one writer process/service per spindle;
- recovery from incomplete trailing record;
- integrity hash/CRC or existing canonical equivalent;
- queue lag telemetry.

Compute/learner parallelism is independent from the number of physical cold writers.

---

# RC2-8 — Resource profiles and Docker control

Define versioned profiles instead of raw flags.

At minimum profiles should specify classes for:

- CPUs / cpuset / shares;
- memory hard + reservation;
- swap policy;
- pids;
- `/dev/shm`;
- GPU;
- hot/cold mounts;
- block-I/O priority/limits;
- network/service dependencies;
- cleanup policy.

Example identity only:

```text
R21_SCIENCE_QUALIFICATION_RC2
R21_INFRA_BENCHMARK_RC2
R21_STORAGE_SERVICE_VALKEY_RC2
R21_STORAGE_SERVICE_NATS_RC2
R21_STORAGE_SERVICE_ROCKSDB_RC2
```

The workflow chooses a profile id. It does not supply arbitrary device/path/privilege arguments.

---

# RC2-9 — Observability and fail-fast gates

Every long run must retain a machine-readable execution receipt binding:

- workflow/ref/SHA;
- runner/profile hash;
- image digest;
- mount/storage class identity;
- CPU/memory/shm/GPU profile;
- storage free-space guard;
- device rotational identity;
- service config hashes;
- I/O telemetry summary;
- science artifact identity where applicable.

Fail before science if:

- FAST_HOT resolves to HDD;
- scratch resolves to 64 MiB shared memory;
- required service persistence is weaker than declared;
- disk reserve is below floor;
- profile identity differs from reviewer authorization;
- cold writer count exceeds physical-lane contract.

---

# RC2-10 — Winner selection

No storage backend is selected solely because it is faster.

Selection order:

1. semantic/durability equivalence;
2. recovery/exactly-once correctness;
3. operational simplicity;
4. throughput/latency;
5. CPU/RAM footprint;
6. observability/maintenance burden.

Likely outcomes to evaluate rather than pre-decide:

- SQLite-on-SSD + single batch writer may be sufficient and simplest;
- Valkey/JetStream may win as a durable ingress log if producer concurrency is the dominant issue;
- RocksDB may win if high-frequency metadata/KV writes dominate and relational SQLite queries are secondary;
- a hybrid may use durable ingress + RocksDB hot projection + SQLite report export + HDD cold packs.

The final architecture decision requires a versioned benchmark receipt and owner/Astra acceptance.

---

# RC2-11 — S1 resume gate

Only after RC2 is qualified should PR #102 leave draft/frozen state.

Resume must use the already frozen S1 science:

- same accepted runtime candidate identity;
- same manifest/seeds/budget/model/optimizer/thresholds;
- no rescue based on prior invalid execution attempts.

RC2 provides a new execution surface, not a new S1 experiment.

Required pre-resume canary:

```text
S1 production-shape bounded canary
-> exact RC2 profile
-> fast-hot storage verified
-> cold writer contract verified
-> provenance/artifact-only audit PASS
-> I/O saturation guard PASS
-> Sol review
-> formal CI-C reauthorization for the RC2 execution surface
```

---

# Owner/Astra decisions needed before implementation branch opens

1. Approve **broker + OIDC + separate control identity** as the privileged-control direction.
2. Decide whether to create a separate private `CB16-Infra-Control` repository for privileged workflows.
3. Approve reclaim/provision of bounded SSD capacity for FAST_HOT.
4. Approve the four-candidate benchmark set: SQLite baseline, Valkey, JetStream, RocksDB.
5. Confirm that storage winner must preserve current exactly-once/provenance semantics rather than weakening durability for throughput.

After these decisions, Sol should convert this document into branch/file-specific DS implementation tasks and independent qualification gates.