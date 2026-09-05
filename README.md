# CB16-R10 + Remote CI Relay V1

Public source authority for CB16 R10 code. This repository contains **code only**:
no model weights, checkpoints, datasets, secrets, or raw Binance data.

- `ci/run_shanxi_ci.sh` – CI entrypoint executed by the Shanxi worker.
- `ci/check_repository_policy.py` – repository secret/binary guard.
- `ci/relay/` – Japan OCI CI relay (FastAPI + SQLite).
- `ci/worker/` – Shanxi long-running worker.
- `ci/deploy/` – systemd/env templates.
- `docs/` – operational documentation.

CI results are written to the `ci-results` branch under `runs/<commit_sha>/`.
ci: remote relay smoke test
ci: remote relay smoke test v2
