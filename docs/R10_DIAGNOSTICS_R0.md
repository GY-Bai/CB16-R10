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

The sidecar only writes:

- `runtime_samples.jsonl`
- `artifact_events.jsonl`
- `RUNTIME_DIAGNOSTIC_SUMMARY.json`

under a separate diagnostics root.

Stage timings reconstructed from artifact appearance are explicitly approximate and are never scientific authority.

### Model probe

`cb16_diagnostics/model_probe.py`

Reads local PyTorch checkpoints and reports:

- serialization-independent semantic SHA256 compatible with R10 tensor identity;
- global and Brain-group L2 norms;
- zero/nonfinite counts;
- parent → challenger / parent → champion relative update norm;
- cosine similarity;
- changed-parameter fraction;
- tensors with largest relative changes;
- optional 2-D tensor effective-rank/SVD diagnostics.

It never writes model weights.

### Post-run aggregator

`cb16_diagnostics/postrun.py`

Produces compact diagnostics:

- `GENERATION_CENSUS.json`
- `RUNTIME_PHASE_SUMMARY.json`
- `MODEL_EVOLUTION.json`
- `POSTRUN_DIAGNOSTIC_SUMMARY.json`
- `SHA256SUMS`

It checks generation completeness, trace maturation receipts, training receipts, champion/challenger lineage hashes, model update scale, gradient/update-group summaries, existing Experience Lake audit, controls status, and runtime bottleneck summaries.

## CLI

Unified entry point:

```bash
python scripts/run_r10_diagnostics.py observe-runtime \
  --run-root /data/cb16_hdd/cb16_runtime/R10_4 \
  --out /data/cb16_hdd/cb16_diagnostics/R10_4/live \
  --interval 5 \
  --stop-when-complete
```

Read-only snapshot while the campaign is still running:

```bash
python scripts/run_r10_diagnostics.py snapshot \
  --run-root /data/cb16_hdd/cb16_runtime/R10_4 \
  --out /data/cb16_hdd/cb16_diagnostics/R10_4/snapshot \
  --runtime-diagnostics /data/cb16_hdd/cb16_diagnostics/R10_4/live
```

Full post-run diagnostics after 100 completed generations and `FINAL_RESULT_R102.json` exist:

```bash
python scripts/run_r10_diagnostics.py postrun \
  --run-root /data/cb16_hdd/cb16_runtime/R10_4 \
  --out /data/cb16_hdd/cb16_diagnostics/R10_4/postrun \
  --runtime-diagnostics /data/cb16_hdd/cb16_diagnostics/R10_4/live
```

The default R10.4 start checkpoint for lineage comparison is:

`/home/bgy/cb16_ssd/runtime/R10_3/generations/G19/champion_after.pt`

## Interpretation

The runtime bottleneck classifier is heuristic and diagnostic only:

- `MEMORY_PRESSURE`
- `IO_WAIT_BOUND`
- `GPU_BUSY`
- `CPU_BOUND_OR_CPU_FEED_BOUND`
- `SERIAL_BARRIER_OR_WAIT_BOUND`
- `MIXED_OR_NO_CLEAR_BOTTLENECK`

No one of these is a scientific PASS/FAIL.

Likewise model diagnostics describe how weights moved; they do not replace validation/tournament/adjudication.

## Durable evidence

After post-run completion, sanitize/retain compact JSON and `SHA256SUMS` in `ci-results`. Do not upload:

- model weights/checkpoints;
- full trajectories;
- Experience Lake payloads;
- datasets;
- secrets/tokens;
- FINAL bytes.
