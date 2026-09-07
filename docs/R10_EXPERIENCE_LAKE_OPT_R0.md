# CB16 R10 Experience Lake Persistence Optimization Qualification R0

Status: **engineering qualification only**. This branch is not wired into the canonical R10.4 campaign and is not scientific/status driving.

## Measured motivation

Live diagnostics on the frozen R10.4 implementation found:

- each generation persists 9,714 `EVIDENCE_PACKAGE` objects;
- G60 training-evidence persistence span: ~943.25 s;
- G61 repeated span: ~864.60 s;
- G60 snapshot-seal -> challenger training receipt: ~1.06 s;
- GPU utilization during the coarse persistence/training stage was effectively zero;
- the snapshot object count exactly matches the 9,714 training evidence objects;
- each generation also has 24 `DECISION_EVENT` + 24 `OUTCOME_SAMPLE` objects, which are not members of the training snapshot.

The current `persist_generation_snapshot()` calls `lake.put()` sequentially for each admitted evidence object. The Lake has four deterministic shards, but the generation persistence loop does not concurrently exploit them. Each current `put()` also uses the existing content-addressed payload fsync path and a SQLite transaction.

## R0 candidates

### `SHARD_PARALLEL_OBJECTWISE`

- bucket objects by the existing deterministic shard function;
- one worker per active shard;
- call the existing `lake.put()` unchanged inside each shard;
- preserve per-object payload fsync and per-object SQLite transaction semantics;
- preserve returned refs in input order;
- snapshot sealing remains the existing implementation.

This is the conservative candidate.

### `SHARD_PARALLEL_BATCHED`

- same deterministic four-shard parallelism;
- keep the existing payload CAS path, compression, file fsync, and atomic replace;
- batch metadata INSERT/check operations into one transaction per shard batch;
- preserve object IDs, payload hashes, identity hashes, shard routing and snapshot hash;
- fault-prefix behavior can differ from one-object-at-a-time commit, so crash/recovery equivalence is mandatory before any adoption.

This is the higher-upside candidate and is **not** automatically preferred.

## Qualification gates

Before any later runtime adoption:

1. baseline/candidate semantic refs byte-equivalent except root-specific payload paths;
2. identical snapshot content hash;
3. full payload audit PASS;
4. replay creates zero new objects;
5. same object ID with different content still fails `EXPERIENCE_ID_CONTENT_CONFLICT`;
6. close/reopen recovery converges;
7. abrupt failure before metadata commit converges after replay;
8. abrupt failure after metadata commit but before caller completion converges after replay;
9. benchmark must run on the same physical filesystem class as canonical R10.4, but in a separate noncanonical directory;
10. only a candidate with material host speedup should proceed to an integration qualification.

## Current branch safety

- no changes to `r102_campaign.py`;
- no changes to `r102_learning.py`;
- no changes to canonical `sharded_experience_lake.py`;
- no changes to worker/runtime authority;
- no access to FINAL 2025-09;
- no writes to `/data/cb16_hdd/cb16_runtime/R10_4`;
- current R10.4 must finish using the frozen implementation.

## Host benchmark

Use `scripts/benchmark_r10_experience_lake_opt_r0.py` with a work/output root outside canonical R10.4. The benchmark creates synthetic immutable evidence objects, runs baseline, objectwise and batched modes, verifies snapshot/ref/replay equivalence, reports objects/sec and speedup, and removes its generated work directories unless `--keep-work` is specified.

Host abrupt-crash qualification remains a separate gate and must be run only against disposable qualification roots.
