# CB16 Remote CI Profiles and Trigger Contract

This repository uses GitHub push webhooks to create immutable, exact-commit CI jobs on the Shanxi worker.

## Default behavior

A push to `main` or `ai/*` defaults to the `smoke` profile.

The webhook does **not** accept arbitrary shell commands.

## Explicit profile request

Place exactly one marker in the pushed head commit message:

| Marker | Profile | Default timeout on Shanxi |
|---|---|---:|
| `[ci:smoke]` | repository guard + Python smoke | 30 min |
| `[ci:unit]` | smoke + repository pytest | 2 h |
| `[ci:r102]` | unit + R10.2 5-generation qualification | 24 h |
| `[ci:r103]` | unit + R10.3 20-generation expansion | 72 h |
| `[ci:r104]` | unit + R10.4 100-generation research | 7 d |
| `[ci:v63]` | unit + `ci/run_v63_ci.sh` | 72 h |

Examples:

```text
fix: normalize funding timestamps [ci:unit]
qualify: run R10.2 on Shanxi [ci:r102]
infra: V6.3 feedback graph qualification [ci:v63]
```

If no marker is present, the relay uses `smoke`.

If multiple different profile markers occur in the same head commit message, the relay rejects the push for CI purposes.

## V6.3 entrypoint contract

`[ci:v63]` is intentionally a fixed contract rather than an arbitrary command channel.

The exact commit being tested must contain:

```text
ci/run_v63_ci.sh
```

That script is version-controlled, travels inside the exact-SHA bundle, and is therefore auditable together with the code it executes.

## Scientific identity

Every job is keyed by:

```text
repository + exact commit SHA + CI profile
```

The OCI relay creates the source bundle using:

```text
git archive <exact SHA>
```

and records the bundle SHA256 before the Shanxi worker can claim it.

## Long-job recovery

The worker sends heartbeats while a process is alive.

The relay requeues stale `CLAIMED`/`RUNNING` jobs after the configured stale interval. After the configured maximum number of attempts, a stale job becomes `ERROR`.

The Shanxi worker runs each CI process in its own process group. A profile timeout terminates the whole process group and reports a canonical `TIMEOUT` result.

Host overrides:

```text
CB16_JOB_STALE_SECONDS
CB16_MAX_JOB_ATTEMPTS
CB16_MAX_JOB_TIMEOUT_SECONDS
CB16_TIMEOUT_SMOKE_SECONDS
CB16_TIMEOUT_UNIT_SECONDS
CB16_TIMEOUT_R102_SECONDS
CB16_TIMEOUT_R103_SECONDS
CB16_TIMEOUT_R104_SECONDS
CB16_TIMEOUT_V63_SECONDS
```

## Evidence contract

For a normal completed run, `ci/run_shanxi_ci.sh` is the canonical producer of:

```text
ci_output/result.json
ci_output/REPORT.md
ci_output/SHA256SUMS
```

`SHA256SUMS` is generated only after `result.json` and `REPORT.md` are final.

The worker validates `job_id`, exact `commit_sha`, `ci_profile`, and verdict before publishing. It must not rewrite a successful result after evidence hashing.

Full stdout/stderr remain on OCI. Only sanitized evidence is published to the `ci-results` branch.

## Non-Git artifacts

Datasets, replay stores, checkpoints, and model weights remain on Shanxi and are never carried in the Git bundle or returned to GitHub.
