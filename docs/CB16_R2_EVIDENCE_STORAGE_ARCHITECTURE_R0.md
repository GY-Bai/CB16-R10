# CB16 R2 Evidence Storage Architecture R0

## Status

Infra architecture / qualification branch only.

This architecture is intentionally isolated from the currently running canonical R10.4 source. It must not mutate `/data/cb16_hdd/cb16_runtime/R10_4`.

## Why R2 exists

The legacy R10.2/R10.4 Experience Lake couples three distinct facts into every `EVIDENCE_PACKAGE` object identity:

1. immutable probabilistic teacher evidence content;
2. generation membership;
3. current Champion / policy authority.

As a result, the same teacher evidence is materialized again for every generation. For about 9,714 admitted training evidence objects and 100 generations, the storage layer behaves approximately like O(N * G), even though the teacher evidence itself is compiled once and is unchanged between generations.

R2 separates these authorities.

## R2 authority graph

```text
Probabilistic teacher evidence
        |
        v
Generation-independent canonical payload
        |
        | SHA-256(raw canonical JSON)
        v
Immutable payload pack record
(HDD sequential append)
        |
        v
Immutable Evidence Set Manifest  <---- written once for the stable evidence set
        |
        | evidence_set_hash
        +---------------------------+
                                    |
                          +---------+---------+
                          |                   |
                          v                   v
                    G00 snapshot        G01 snapshot ... G99
                    generation=0        generation=1
                    Champion hash       Champion hash
                    evidence_set_hash   same evidence_set_hash
```

Storage complexity becomes:

```text
O(unique evidence content + evidence-set manifests + generations)
```

instead of:

```text
O(generations * evidence objects)
```

## Scientific / authority boundary

R2 does not claim that evidence has no historical context. The immutable evidence payload still includes the parent context, dependence group, student context object, Operator/Medium/Account inputs, probabilistic direction target, requested-risk target, action laws, admission record, and teacher protocol hash.

R2 deliberately removes only the following from immutable evidence content:

- generation number;
- current Champion / policy hash;
- generation snapshot id.

Those are membership/authority facts. They are represented by the generation snapshot layer.

A stable `evidence_id` may not silently map to new content. Same evidence ID with a different identity raises `R2_EVIDENCE_ID_CONTENT_CONFLICT`.

## Legacy-to-R2 semantic migration gate

R2 does not require legacy object identity to remain identical, because generation and current Champion authority are deliberately removed from immutable evidence content.

Instead, `r2_legacy_migration.py` defines a generation-independent semantic projection:

```text
legacy EVIDENCE_PACKAGE
  - generation
  - legacy schema tag
  - generation-specific object id prefix
  - current policy/Champion authority
        |
        v
CB16_EVIDENCE_SEMANTIC_PROJECTION_V1
        ==
R2 immutable evidence semantic projection
```

The migration qualifier independently preserves and checks:

- parent snapshot hash;
- lineage hash;
- teacher protocol hash;
- evidence ID suffix;
- payload SHA verification before projection.

For two legacy generations containing the same teacher evidence, the required R2 result is:

```text
Gx semantic_projection_aggregate_hash == Gy semantic_projection_aggregate_hash
Gx R2 evidence_set_hash == Gy R2 evidence_set_hash
first generation created_payload_count == N
second generation created_payload_count == 0
```

This is the key proof that R2 removes generation-specific materialization without deleting scientific content.

## Physical SSD / HDD topology

### SSD / NVMe: small random metadata

Prefer SSD for:

- SQLite WAL and locator/catalog index;
- immutable evidence-set manifests;
- tiny generation snapshots;
- hot event/trace journal;
- model/checkpoint metadata where appropriate.

These workloads benefit from low random-access and fsync latency.

### HDD: large sequential immutable payload streams

Prefer HDD for:

- immutable evidence payload packs;
- old/cold evidence archive;
- large append-only records.

R2 does not create one physical file per evidence object. Many logical objects are concatenated into framed `segment_XXXXXXXX.pack` files.

Each record contains:

```text
codec-tag
magic = CB16R2P1
raw SHA-256
raw byte length
stored byte length
CRC32(stored bytes)
compressed payload
```

The SHA-256 defines content identity. CRC32 detects physical record corruption cheaply during sequential recovery scans.

## Physical lanes are devices, not arbitrary shards

If the host has one mechanical HDD, configure exactly one HDD payload root. R2 then performs one sequential append stream.

Do not create four concurrent write lanes merely because the legacy Lake used four logical SQLite shards. Four streams on one spindle can convert sequential writes into seek competition.

If the host has multiple independent physical HDDs/NVMes, one payload root per physical device may be used. Content hashes are deterministically mapped across those physical lanes.

The qualified Shanxi R0 topology is:

```text
SSD: /dev/sdb  CT500MX500SSD1
  -> metadata authority: /var/lib/cb16-r2

HDD: /dev/sda  ST1000DM010-2EP1
  -> immutable payload authority: /data/cb16_r2_store

physical payload lanes = 1
```

## Dynamic event workload separation

Generation-specific `DECISION_EVENT` and `OUTCOME_SAMPLE` records are not immutable teacher evidence and are not written into the HDD evidence pack.

R2 buffers a generation's events and commits them to an SSD-resident hot event journal as one transactional batch. This keeps the mechanical HDD workload sequential and cold while small transactional event metadata stays on SSD.

## Crash model

### Pack fsync before metadata index commit

A process can fail after pack bytes are durable but before SQLite knows their locations.

On open, R2 sequentially scans pack files. Any fully valid framed record missing from SQLite is registered again. Replaying the same evidence therefore converges without creating a new logical evidence object.

### Incomplete trailing record

If a crash leaves a partial final record, recovery truncates only the incomplete trailing bytes back to the last valid record boundary.

A CRC/SHA corruption in a fully framed record is not silently repaired or skipped; it is a hard integrity failure.

## Read-path physical discipline

The production audit path scans pack segments in physical offset order. It must not iterate objects in content-hash order and seek randomly across a mechanical disk.

The same rule applies to recovery: one segment open, monotonically increasing offsets, sequential readahead-friendly access.

## SQLite role

SQLite remains useful but its role changes.

Legacy R10 effectively makes every logical evidence insertion a SQLite transaction plus a small-file CAS operation.

R2 uses SQLite as a compact SSD-resident locator/catalog accelerator. Payload bytes exist independently in immutable packs and evidence-set/snapshot authority exists independently in immutable manifest JSON.

This allows transaction batching and index rebuilding without redefining evidence content.

## Compression

The core implementation supports:

- `zstd` when the optional `zstandard` package is installed;
- automatic fallback to stdlib `zlib`;
- `none` for raw-I/O qualification.

Compression does not participate in scientific identity. Identity is SHA-256 of canonical uncompressed payload bytes.

## Open-source comparison path

R2 also qualifies RocksDB through the maintained Python `rocksdict` binding.

Relevant RocksDB primitives:

- `WriteBatch` / group commit;
- WAL on a dedicated fast path;
- `db_paths` / `DBPath` for newer-data-on-flash and older-data-on-HDD placement;
- background compaction and recovery;
- checksummed SST/WAL storage.

RocksDB is not automatically preferred over the pack/manifest path because CB16 teacher evidence is predominantly write-once immutable content, where append-only packs avoid LSM compaction write amplification. Host qualification will compare the two approaches before any backend is selected as production authority.

## Qualification gates

R2 must prove at least:

1. storage package import works without torch/CUDA;
2. generation-independent content identity;
3. same evidence set reused across 100 generation snapshots;
4. physical payload count remains O(N), not O(N*G);
5. same-ID/different-content conflict detection;
6. crash after pack fsync / before index commit converges;
7. partial-tail recovery converges;
8. deterministic evidence-set and generation snapshot hashes;
9. full sequential payload audit PASS;
10. event journal exactly-once replay and conflict detection;
11. legacy-to-R2 semantic projection equality on real evidence;
12. repeated legacy generations collapse to one R2 evidence-set hash;
13. no canonical R10.4 writes;
14. FINAL 2025-09 remains untouched;
15. SSD/HDD host topology is measured rather than guessed;
16. same-device and split-device benchmark results are retained.

## Current branch

`ai/r2-evidence-storage-r0`

Core modules:

- `cb16_local_opt/r2_evidence_storage.py`
- `cb16_local_opt/r2_event_journal.py`
- `cb16_local_opt/r2_sequential_audit.py`
- `cb16_local_opt/r2_learning_bridge.py`
- `cb16_local_opt/r2_legacy_migration.py`
- `cb16_local_opt/r2_campaign.py`
- `tests/test_r2_evidence_storage.py`
- `tests/test_r2_event_journal.py`
- `tests/test_r2_legacy_migration.py`
- `scripts/benchmark_r2_evidence_storage.py`
- `scripts/qualify_r2_legacy_migration.py`
- `scripts/probe_r2_storage_topology.py`
- `scripts/probe_r2_rocksdb_backend.py`
- `configs/R2_STORAGE_SHANXI_R0.json`
- `requirements-r2-storage.txt`

## Integration sequence

```text
R2 storage semantics
  -> synthetic storage/event qualification
  -> actual Shanxi SSD/HDD topology discovery
  -> real legacy Gx/Gy semantic projection equivalence
  -> one-time 9,714-evidence R2 materialization
  -> 100-generation manifest-only benchmark
  -> RocksDB tiered comparison
  -> R2 campaign replay/recovery equivalence
  -> new versioned campaign authority
```

The running R10.4 campaign remains frozen while these gates execute.
