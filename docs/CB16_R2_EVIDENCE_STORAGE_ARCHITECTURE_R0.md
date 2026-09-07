# CB16 R2 Evidence Storage Architecture R0

## Status

Qualification-only structural Infra successor to the legacy generation-specific Experience Lake path.

Current branch: `ai/r2-evidence-storage-r0`.

The running canonical R10.4 campaign remains untouched.

## Core semantic split

R2 separates three authorities that were previously coupled:

1. **Immutable evidence content** — generation-independent teacher evidence, content-addressed by SHA-256.
2. **Evidence-set authority** — immutable manifest naming the exact ordered evidence membership once.
3. **Generation authority** — tiny immutable snapshot binding generation + Champion policy hash to an evidence-set hash.

Dynamic on-policy decision/outcome events are a fourth workload and use a separate SSD hot event journal.

The storage complexity target changes from approximately `O(N * G)` generation-specific evidence materialization to `O(unique evidence content + G tiny snapshots)`.

## Shanxi physical topology binding

Verified host topology:

- SSD: `/dev/sdb` — Crucial CT500MX500SSD1, root filesystem through `/dev/sdb3` + LVM.
- HDD: `/dev/sda` — Seagate ST1000DM010-2EP1, `/data` via `/dev/sda1`.

Qualification/production placement contract:

- SSD (`/var/lib/cb16-r2/...`): SQLite WAL/index, evidence-set manifests, generation snapshots, hot event journal, recovery metadata.
- HDD (`/data/cb16_r2_store`): immutable evidence pack segments.
- Physical payload lanes: **1**, because the host has one mechanical HDD. Logical sharding must not turn one spindle into competing seek streams.
- `/tmp` is scratch only; it must not become persistent metadata authority.

See `configs/R2_STORAGE_SHANXI_R0.json`.

## Payload pack

Teacher payloads are framed into append-only pack segments:

- content SHA-256;
- raw/stored length;
- codec tag;
- CRC32 over stored bytes;
- compressed payload bytes.

Writes are coalesced into a large sequential append and fsync. The SSD locator records segment path + byte offset.

Crash model:

- fully fsync'd payload record but missing locator: recovered by sequential pack scan;
- incomplete trailing record: truncate to the last complete record boundary;
- complete record with CRC/SHA mismatch: hard corruption failure;
- same evidence ID with different identity/content: hard conflict.

## HDD read discipline

Production audit does **not** verify payloads in content-hash order. `r2_sequential_audit.py` scans by physical lane/segment/offset using one open file handle and a sequential readahead hint where supported. This preserves HDD sequentiality for verification as well as writes.

## Hot event journal

`r2_event_journal.py` stores generation-specific `DECISION_EVENT` and `OUTCOME_SAMPLE` records on SSD in SQLite WAL mode.

The compatibility sink buffers one generation and commits the complete event set in one transaction. A crash before flush does not publish the generation receipt; recovery deterministically regenerates the trace. Replaying an already committed generation converges by exactly-once identity.

## R2 campaign orchestration

`r2_campaign.py` is the R2-first orchestrator. It intentionally does not call legacy `persist_generation_snapshot()`.

Flow:

1. build/load historical evidence cache;
2. compile teacher evidence once;
3. materialize immutable training evidence once;
4. enter generation loop;
5. produce on-policy trace and batch-commit hot events to SSD;
6. seal tiny generation snapshot referencing the immutable evidence-set hash;
7. train challenger;
8. adjudicate/persist Champion lineage;
9. repeat without re-materializing teacher payloads.

## Lightweight package initialization

`cb16_local_opt/__init__.py` now lazily resolves ML runtime exports. Storage, diagnostics, recovery and topology tooling can therefore run on a host without importing PyTorch/CUDA.

This is intentional Infra decoupling, not a workaround for a test runner.

## Open-source comparison

RocksDB/RocksDict remains an explicit comparison backend, not a foregone production choice. Host capability confirms support for WriteBatch, DBPath, separate WAL placement and pipelined writes.

RocksDB advantages:

- mature WAL/recovery;
- group/batch commit;
- memtable/SST/compaction;
- hot/cold `db_paths` support;
- separate `wal_dir` suitable for SSD.

Custom pack/manifest advantages for CB16 teacher evidence:

- evidence is predominantly write-once immutable content;
- one HDD sequential append stream has minimal seek and compaction amplification;
- no LSM background compaction is required for payload authority;
- generation reuse is explicit in the authority model.

The host benchmark will decide whether RocksDB is useful as metadata/event backend or whether the custom pack/manifest path should remain the evidence payload authority.

## Qualification gates before canonical adoption

- storage-only import with no torch;
- R2 unittest suite;
- 100-generation physical payload reuse;
- same-ID/different-content conflict;
- pack fsync-before-index recovery;
- incomplete-tail recovery;
- generation snapshot conflict;
- event journal exactly-once batch replay;
- sequential physical audit;
- host same-disk benchmark only after canonical R10.4 I/O contention ends;
- end-to-end R2 campaign replay/equivalence using the real ML runtime;
- FINAL 2025-09 remains locked.
