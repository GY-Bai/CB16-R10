# CB16 Infra Engineering Debt TODO

> Status: **OPEN / INFRA DEBT / NOT SCIENCE AUTHORITY**  
> Date: 2026-09-13  
> Scope: Shanxi Docker runner storage topology, qualification scratch, durable replay/update/provenance I/O.

This file records infrastructure debt only. It does not authorize changes to S1 seeds, reward, model, optimizer, thresholds, task semantics, budget, or evidence ceiling.

## 1. Current incident: S1 qualification is HDD-I/O bound

The current S1 formal workload is not compute-bound. Observed symptoms:

- CPU utilization stays roughly below 10%;
- memory is not exhausted;
- about 8 qualification workers are active;
- multiple processes enter D-state I/O wait;
- the single HDD reaches about 100% device utilization;
- read/write await rises to roughly 50-70 ms under contention;
- 4K random read/write latency reaches tens to more than 100 ms;
- fdatasync latency is roughly 100 ms class under contention;
- HDD sequential throughput collapses to roughly 10-15 MB/s while the workload is active.

Current physical topology:

- `/dev/sda`: Seagate ST1000DM010 1TB 7200rpm HDD; `/data`, Docker data-root, CB16 Docker volumes, runner work tree, runner temp and current qualification scratch ultimately share this spindle.
- `/dev/sdb`: Crucial MX500 SSD; root/LVM lives here, but the filesystem is already heavily occupied.

The SSD probe is orders of magnitude better for random I/O and fsync latency than the HDD.

**Conclusion:** increasing worker count cannot solve this bottleneck. More workers increase seek/fsync contention on the same spindle.

## 2. Historical solution already existed: R2 storage tiering

This problem was already identified in the historical R2 storage work.

Historical references:

- commit `c01bd1fc8467faf52612edb588f6333c3627644c` — `docs: freeze R2 evidence storage architecture R0`
- historical file `docs/CB16_R2_EVIDENCE_STORAGE_ARCHITECTURE_R0.md`
- branch `ai/r2-evidence-storage-r0`
- implementation commit `ad3f6deb74d47dbc96b10d187f1d55ed0c600097`

R2 froze the following physical rule:

### SSD / NVMe = small random and fsync-heavy hot state

Prefer fast storage for:

- SQLite WAL / journal;
- locator/catalog and replay indexes;
- durable update journal;
- small manifests;
- generation/checkpoint metadata;
- hot event/trace state;
- frequently fsynced small files and transaction state.

### HDD = large sequential immutable payload and cold archive

Prefer HDD for:

- large immutable evidence packs;
- cold evidence/archive data;
- large append-only records.

Historical hard rule:

```text
R2_PHYSICAL_PAYLOAD_LANES = 1
```

One rotating HDD should normally have one sequential physical payload writer. Logical shards or worker count must not be mapped directly to multiple writers on the same spindle.

Historical conclusion:

> Increasing Teacher/H72 workers could not fix this storage bottleneck.

## 3. Regression source: Docker isolation flattened hot and cold storage onto one HDD

The Docker migration correctly improved isolation and capacity management, but the live runner contract now shows that the runner work tree, temp area and Docker volumes are all backed by the same HDD filesystem.

Therefore moving S1 scratch from a tiny shared-memory filesystem to runner temp solved capacity/permission failures but **did not solve the physical-media bottleneck**.

The regression can be summarized as:

```text
R2 intended topology:
SSD = random/fsync hot metadata
HDD = sequential immutable cold payload

current Docker topology:
runner work + temp + Docker volumes -> one HDD

formal S1 workload:
8 workers + durable replay/update/provenance
-> random write/fsync contention
-> D-state + low CPU + HDD ~100% util
```

Docker isolation is not the mistake. Missing hot/cold placement inside the Docker architecture is the debt.

# 4. Engineering debt items

## INFRA-IO-01 — Restore explicit storage classes

Define and version at least:

```text
FAST_HOT_SCRATCH
FAST_METADATA
COLD_IMMUTABLE_PAYLOAD
READ_ONLY_INPUT
```

A path being writable is not sufficient proof that it is suitable for a workload.

Target mapping:

| Workload | Storage class |
|---|---|
| SQLite / WAL / journal | SSD/NVMe hot |
| replay index / locator | SSD/NVMe hot |
| per-update durable journal | SSD/NVMe hot |
| checkpoint metadata / small checkpoints | SSD/NVMe hot |
| provenance hot write set | SSD/NVMe hot |
| formal job scratch | SSD/NVMe hot |
| large immutable packed payload | HDD sequential |
| raw/cold historical input | HDD read-only/cold |
| finalized archive | HDD sequential or external artifact store |

Do not weaken durability, fsync, exactly-once, or provenance semantics for performance.

## INFRA-IO-02 — Provision a bounded SSD-backed runner hot mount

The existing SSD is already heavily occupied, so do not move the entire Docker data-root back to SSD.

First perform capacity cleanup/provisioning, then create one explicit SSD-backed hot mount exposed to the container through a stable `/cb16/*` abstraction.

Requirements:

- live contract and machine snapshot record the physical device and storage class;
- runner uid has write access;
- job-local subdirectories can be safely cleaned;
- hot space has a measured capacity guard;
- existing scientific roots remain separate;
- the host path is not guessed by workflows.

## INFRA-IO-03 — Keep Docker cold capacity on HDD without forcing hot volumes there

Docker images and large cold volumes may remain on the HDD for capacity reasons.

Hot scratch/metadata should use a dedicated SSD-backed bind/local volume instead of inheriting the Docker data-root spindle.

Container code should continue to see stable `/cb16/*` paths and should not know host physical paths.

## INFRA-IO-04 — Reinstate one HDD physical writer lane

When large immutable data is written to HDD:

```text
physical payload writers per HDD spindle = 1
```

Use:

- bounded producer queue;
- one sequential pack writer;
- batching/chunking;
- backpressure;
- crash/recovery semantics;
- queue-depth/bytes telemetry.

Compute workers may remain parallel. Physical disk writers should not mirror worker count.

## INFRA-IO-05 — Separate worker parallelism from storage concurrency

Target shape:

```text
N compute workers
    -> SSD hot durable state
    -> bounded aggregation
    -> one HDD sequential cold writer per spindle
```

Worker/account/shard count must not define physical disk-lane count.

## INFRA-IO-06 — Add storage preflight and saturation fail-fast

Before formal qualification, machine-check:

- scratch physical device;
- rotational/non-rotational identity;
- filesystem and mount source;
- free bytes/inodes;
- runner write probe;
- shared-memory capacity;
- HDD/SSD placement.

During long runs record at least:

- device read/write await;
- utilization and queue depth;
- D-state process count;
- process block I/O;
- CPU user/system/iowait;
- scratch bytes/file count where practical.

`low CPU + high D-state + disk util near 100% + high await` must be classified as an execution/storage bottleneck, not a model/science failure.

## INFRA-IO-07 — Benchmark topology before canonical cutover

Compare at least:

1. current HDD-all-in-one baseline;
2. SSD hot scratch + HDD cold payload;
3. SSD hot metadata + single HDD sequential writer;
4. worker-count scaling with a fixed physical writer lane.

Retain wall time, decisions/s, updates/s, CPU, memory, MB/s, random IOPS, fsync latency, await/util, D-state count and semantic/provenance hashes.

A faster topology becomes canonical only after semantic/provenance equivalence passes.

## INFRA-IO-08 — Version the runner launch/storage definition

The current live runner was reconstructed from the running container rather than from a fully versioned launch definition.

The next Infra upgrade should version:

- Dockerfile;
- compose or equivalent launch spec;
- volume declarations;
- uid/gid ownership initialization;
- shared-memory size;
- hot/cold storage mount contract;
- cleanup lifecycle.

# 5. Forbidden shortcuts

Do not close this debt by:

- lowering S1 scientific budget;
- reducing frozen seeds;
- deleting durable journal/provenance;
- disabling exactly-once persistence;
- weakening fsync/durability contract;
- changing replay sampling;
- changing reward/threshold/model/optimizer;
- declaring success merely because a one-worker run eventually finishes.

Lower worker count is allowed only as a diagnostic/emergency execution workaround, not as the canonical fix.

# 6. Upgrade sequence

```text
A. Freeze current I/O incident evidence
B. Inventory all durable write surfaces
C. Measure per-surface bytes/write frequency/fsync pattern
D. Reclaim or provision bounded SSD hot capacity
E. Add versioned SSD-backed hot mount
F. Route random/fsync hot state to SSD
G. Keep large immutable cold payload on HDD
H. Enforce one HDD physical writer lane
I. Run crash/restart/artifact-only equivalence tests
J. Run worker-scaling + I/O benchmark
K. Adopt only after equivalence + performance gates pass
```

# 7. Definition of done

This debt is closed only when:

- formal qualification hot scratch is not on the rotating HDD;
- SQLite/WAL/journal/random metadata hot paths use SSD/NVMe;
- HDD carries read-mostly cold data or sequential immutable writes;
- one-HDD-one-physical-writer is machine verified;
- runner storage topology is versioned and machine-readable;
- cleanup and free-space guards are explicit;
- an 8-worker campaign no longer shows the current low-CPU/HDD-100%-util/D-state pattern;
- restart/exactly-once/provenance/artifact equivalence remains PASS;
- no frozen scientific constant was changed to obtain the speedup.

# 8. Relationship to current S1

The current S1 storage saturation is an **execution/infrastructure bottleneck**.

Until hot/cold tiering is restored:

- HDD saturation must not be called `SCIENTIFIC_FAIL`;
- adding workers is not a rescue;
- repeatedly launching the same formal campaign on the same spindle is not useful evidence;
- a temporary execution workaround may complete S1 only if Sol explicitly marks it as execution-only; it does not close this Infra debt.
