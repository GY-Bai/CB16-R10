# CB16 Remote CI Relay V1 – Operations

## Components

- GitHub public repo: source + `ci-results` branch
- Cloudflare Tunnel: `cb16-ci-relay`
- Cloudflare Access: service auth for `/api/v1/worker/*`
- Japan OCI: FastAPI relay listening on `127.0.0.1:18787`
- Shanxi: systemd worker

## Directory layout (OCI)

```
/home/bgy/cb16-ci/
├── relay/          # FastAPI app
├── repo/source.git # bare mirror
├── bundles/        # <sha>.tar.zst
├── jobs/
├── results/
├── logs/
└── state/          # SQLite
```

## Directory layout (Shanxi)

```
/data/cb16_ci/
├── worker/
├── workspaces/
├── results/
└── cache/
```

## Start/stop

OCI user units:

```
systemctl --user start cb16-ci-relay.service
systemctl --user start cloudflared.service
```

Shanxi:

```
systemctl --user start cb16-ci-worker.service
```

## Health

```
curl https://ci-speedtest.bayesdesk.com/healthz
# => {"status":"ok"}
```

## Result branch

`ci-results/runs/<commit_sha>/{result.json,REPORT.md,SHA256SUMS}`
