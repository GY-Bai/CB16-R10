# CB16 R10 Diagnostics R0 — Shanxi Live Canary 2026-09-07

Status: PASS

Diagnostics source SHA:
`39498ad59431a818f1bb58924e62dc79fc369fce`

Observed canonical R10.4 process:
- PID: `3161816`
- interpreter: `/usr/bin/python3.10`
- current generation: `59`
- inferred stage: `SNAPSHOT_OR_CHALLENGER_TRAINING`
- GPU available: `true`

Canary:
- syntax canary: `PASS`
- runtime canary: `PASS`
- samples: `7`
- candidate bottleneck in 30s window: `SERIAL_BARRIER_OR_WAIT_BOUND`

Observer overhead:
- CPU: `0.7%`
- RSS: `16924 KiB`

Safety boundary:
- writes_to_canonical_run_root: `false`
- scientific_semantics_changed: `false`
- final_holdout_2025_09_accessed: `false`

Interpretation:
The sidecar is qualified for continued live observation. The 30-second bottleneck label is observational only and is not yet a campaign-level performance verdict; longer stage/generation attribution is required before optimization decisions.
