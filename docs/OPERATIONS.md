# CB16 Remote CI Relay V1.1 – Operations

## Components

- GitHub public repo: source authority + `ci-results` branch
- Cloudflare Tunnel: `cb16-ci-relay`
- Cloudflare Access: service auth for `/api/v1/worker/*` (deployment hardening requirement)
- Japan OCI: FastAPI relay listening on `127.0.0.1:18787`
- Shanxi: systemd worker, preferably under a dedicated unprivileged `cb16-ci` account

## Directory layout (OCI)

```text
/home/bgy/cb16-ci/
├── relay/
├── repo/source.git
├── bundles/
├── jobs/
├── results/
├── logs/
└── state/
```

## Directory layout (Shanxi)

```text
/data/cb16_ci/
├── worker/
├── workspaces/
├── results/
└── cache/
```

Datasets, runtime checkpoints, and model weights live outside the Git workspace.

## Start/stop

OCI user units:

```bash
systemctl --user start cb16-ci-relay.service
systemctl --user start cloudflared.service
```

Shanxi:

```bash
systemctl --user start cb16-ci-worker.service
```

For production hardening, migrate the Shanxi worker to a dedicated unprivileged account and system service if it is still running under a broad-privilege login.

## Health

```bash
curl https://ci-speedtest.bayesdesk.com/healthz
# => {"status":"ok"}
```

## CI profiles

See `ci/CI_PROFILES.md`.

A normal push defaults to smoke. Long research must be explicitly requested with a bounded commit-message marker such as:

```text
[ci:r102]
[ci:r103]
[ci:r104]
[ci:v63]
```

## Job lifecycle

```text
PENDING -> CLAIMED -> RUNNING -> PASS|FAIL|ERROR|TIMEOUT
```

A worker heartbeat refreshes the active lease. Stale jobs are requeued up to the configured attempt limit and then become `ERROR`.

## Result branch

```text
ci-results/runs/<commit_sha>/
├── result.json
├── REPORT.md
└── SHA256SUMS
```

`ci-results` pushes are excluded from CI triggering.

## Clock discipline

All machine-readable timestamps are UTC/RFC3339 with a true `Z` suffix. Host display time zones may differ.

The Shanxi worker startup clock canary compares the local clock against the relay/Cloudflare HTTP Date header. Direct GitHub clock checking is optional because Shanxi may not have reliable GitHub connectivity.
