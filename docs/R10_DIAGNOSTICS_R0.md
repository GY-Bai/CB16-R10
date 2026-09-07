# CB16 R10 Diagnostics R0

Status: observer-only engineering/scientific diagnostics. This layer is **not status driving** and must not mutate canonical R10 scientific state.

## Safety boundary

- Frozen R10 scientific source authority remains `f056ae6a0722e3e92d71793024a6e6d3fe9af003`.
- Canonical R10.4 root remains `/data/cb16_hdd/cb16_runtime/R10_4`.
- Diagnostics output MUST be outside the canonical root.
- Diagnostics MUST NOT read FINAL 2025-09.
- Diagnostics MUST NOT delete/rewrite cache, generations, Experience Lake, checkpoints, or receipts.
- Diagnostics MUST NOT alter promotion/rejection or scientific verdict.
- Weight/checkpoint bytes stay on the Shanxi host; only compact JSON/hash/count summaries should leave the host.

## Components

### Runtime sidecar

`cb16_diagnostics/runtime_observer.py`

Samples without third-party dependencies:

- host CPU busy and per-core busy;
- CPU iowait;
- RAM/swap availability;
- R10.4 process-tree CPU, RSS, virtual memory, thread count, read/write bytes;
- NVIDIA GPU utilization/memory/power/temperature through `nvidia-smi` when available;
- active generation and an inferred stage;
- appearance of generation receipts/checkpoints.

Default interval is 5 seconds to keep observer overhead small.

The sidecar only writes `runtime_samples.jsonl`, `artifact_events.jsonl`, and `RUNTIME_DIAGNOSTIC_SUMMARY.json` under a separate diagnostics root.

### Stage attribution

`cb16_diagnostics/stage_attribution.py`

Aggregates runtime samples by `(generation, stage)`. The parser consumes the actual RuntimeObserver schema (`wall_time_unix`, `generation.active_generation`, `host.cpu_busy_pct`, `process_tree.cpu_pct_one_core_100`, `gpu.gpus[]`) while retaining legacy aliases. It exposes total/valid/discarded sample counts so schema drift cannot silently become a zero-sample PASS.

Bottleneck labels carry LOW/MEDIUM/HIGH confidence and remain diagnostic only.

### Experience Lake timing probe

`cb16_diagnostics/experience_lake_probe.py`

Read-only timing probe over the local Experience Lake SQLite metadata shards plus generation artifact mtimes. It is intended to split the coarse `SNAPSHOT_OR_CHALLENGER_TRAINING` interval without instrumenting the frozen trainer.

Per generation it reports:

- Experience Lake object count and per-shard counts;
- first/last object `created_at` and observed object insertion span;
- total raw/stored bytes and observed object/storage rate;
- immutable training snapshot seal time and object-count match;
- ON_POLICY receipt -> first Lake object;
- first Lake object -> snapshot seal;
- snapshot seal -> challenger training receipt;
- training receipt -> challenger checkpoint;
- challenger checkpoint -> generation result.

SQLite connections use `mode=ro` plus `PRAGMA query_only=ON`. Payload files are not opened by this probe.

### Model probe

`cb16_diagnostics/model_probe.py`

Reads local PyTorch checkpoints and reports semantic SHA256, global/Brain-group L2 norms, nonfinite/zero counts, parent -> challenger/champion update scale, cosine similarity, changed-parameter fraction, largest relative tensor changes, and optional effective-rank/SVD diagnostics. It never writes model weights.

### Post-run aggregator

`cb16_diagnostics/postrun.py`

Produces compact generation census, runtime/model summaries, post-run diagnostic summary, and SHA256 manifests. It does not replace validation/tournament/adjudication.

## Current qualification state

- Shanxi runtime sidecar canary passed with negligible observer overhead and all safety flags false.
- Stage attribution now consumes 100% of the actual live RuntimeObserver rows in the qualified schema.
- Current runtime evidence shows long `SNAPSHOT_OR_CHALLENGER_TRAINING` windows with near-zero GPU utilization; this is a candidate engineering bottleneck, not yet an optimization authorization.
- The next qualification is the Experience Lake timing probe against live G59/G60 metadata to measure the persistence/training split directly.

No optimization is authorized by diagnostics alone. Any runtime change must be implemented separately and pass scientific identity/equivalence gates before entering a canonical campaign.

## Durable evidence

After post-run completion, sanitize/retain compact JSON and `SHA256SUMS` in `ci-results`. Do not upload model weights/checkpoints, full trajectories, Experience Lake payloads, datasets, secrets/tokens, or FINAL bytes.
