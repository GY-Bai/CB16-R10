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

## Physical SSD / HDD topology

### SSD / NVMe: small random metadata

Prefer SSD for:

- SQLite WAL and locator/catalog index;
- immutable evidence-set manifests;
- tiny generation snapshots;
- future hot event/trace metadata;
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

This is an important R2 rule.

If the host has one mechanical HDD, configure exactly one HDD payload root. R2 then performs one sequential append stream.

Do not create four concurrent write lanes merely because the legacy Lake used four logical SQLite shards. Four streams on one spindle can convert sequential writes into seek competition.

If the host has multiple independent physical HDDs/NVMes, one payload root per physical device may be used. Content hashes are deterministically mapped across those physical lanes.

## Crash model

### Pack fsync before metadata index commit

A process can fail after pack bytes are durable but before SQLite knows their locations.

On open, R2 sequentially scans pack files. Any fully valid framed record missing from SQLite is registered again. Replaying the same evidence therefore converges without creating a new logical evidence object.

### Incomplete trailing record

If a crash leaves a partial final record, recovery truncates only the incomplete trailing bytes back to the last valid record boundary.

A CRC/SHA corruption in a fully framed record is not silently repaired or skipped; it is a hard integrity failure.

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

The branch includes a capability probe and optional dependencies. RocksDB is not automatically preferred over the pack/manifest path because CB16 teacher evidence is predominantly write-once immutable content, where append-only packs avoid LSM compaction write amplification. Host qualification will compare the two approaches before any backend is selected as production authority.

## Qualification gates

R2 must prove at least:

1. generation-independent content identity;
2. same evidence set reused across 100 generation snapshots;
3. physical payload count remains O(N), not O(N*G);
4. same-ID/different-content conflict detection;
5. crash after pack fsync / before index commit converges;
6. partial-tail recovery converges;
7. deterministic evidence-set and generation snapshot hashes;
8. full payload audit PASS;
9. no canonical R10.4 writes;
10. FINAL 2025-09 remains untouched;
11. SSD/HDD host topology is measured rather than guessed;
12. same-device and split-device benchmark results are retained.

## Current branch

`ai/r2-evidence-storage-r0`

Core modules:

- `cb16_local_opt/r2_evidence_storage.py`
- `cb16_local_opt/r2_learning_bridge.py`
- `tests/test_r2_evidence_storage.py`
- `scripts/benchmark_r2_evidence_storage.py`
- `scripts/probe_r2_storage_topology.py`
- `scripts/probe_r2_rocksdb_backend.py`
- `requirements-r2-storage.txt`

## Integration sequence

R2 integration should proceed in this order:

```text
R2 storage semantics
  -> disposable synthetic qualification
  -> actual Shanxi SSD/HDD topology discovery
  -> real 9,714-evidence materialization benchmark
  -> 100-generation manifest-only benchmark
  -> RocksDB tiered comparison
  -> campaign storage adapter
  -> replay/recovery equivalence
  -> new versioned campaign authority
```

The running R10.4 campaign remains frozen while these gates execute.
