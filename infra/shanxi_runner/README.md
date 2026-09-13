# CB16 Shanxi Docker runner definition (R21 RC2 R1)

Status: **DS_PROPOSED_PENDING_SOL_REVIEW**

This directory versions the current live `cb16-runner-r11` container definition.
It is a reconstruction, not the original build definition. The original
Dockerfile, compose file, `docker run` script and systemd unit do not exist in
the repository or in the readable host paths.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Versioned image build contract reconstructed from `docker history`. |
| `build_context_v1.json` | Pins the external `uv` binary by version, size and SHA256. |
| `runner_launch_spec_v1.json` | Versioned launch/profile contract matching the frozen live snapshot. |

## Reproduce (requires a future authorized host change)

1. Place the pinned `uv` binary at `infra/shanxi_runner/uv` and verify it
   against `build_context_v1.json` (version `0.12.6`, SHA256 `d381f115...`).
2. Build the image:
   `docker build --tag cb16-runner-r11:latest infra/shanxi_runner`
3. Create or start the runner from `runner_launch_spec_v1.json` with the
   existing named volumes and the `cb16-net` network.
4. Run `scripts/verify_r21_rc2_runner_definition_v1.py --inspect-json <live
   docker inspect JSON>` and require `status=PASS`.

These steps are **not executed by R1**: the R1 authority matrix forbids host and
container changes. A literal disposable-runner launch remains
`EXECUTION_BLOCKED` until the owner or Sol either authorizes a bounded temporary
launch or accepts definition parity and defers the launch to R3.

## Known divergence (must be reviewed, not silently fixed)

`runner_launch_spec_v1.json` records `R1-DIV-001`:

- live `CB16_PROVISION_ENV=/run/secrets/cb16-provision.env`
- `/run/secrets/cb16-provision.env` does **not** exist
- the mounted file is `/run/secrets/provision.env` and is not readable by the
  runner user
- `/cb16/worker/provision.env` exists and is readable by the runner
- the frozen snapshot omits `CB16_PROVISION_ENV` from its environment values

R1 records the live value and declares `CONTRACT_MISMATCH`. It does not change
the live container, the snapshot or the consumer contract. Sol/owner must
decide which path is authoritative.

## Read-only verification

- `scripts/verify_r21_rc2_runner_definition_v1.py --static` validates the repo
  definition.
- `scripts/verify_r21_rc2_runner_definition_v1.py --self-check` compares the
  current live runner surface with the repo definition without changing it.
- `scripts/verify_r21_rc2_runner_definition_v1.py --inspect-json <file>`
  compares the repo definition with a read-only `docker inspect` capture.
- `.github/workflows/cb16-r21-rc2-r1-runner-spec.yml` runs the static tests and
  the read-only self-check on the Shanxi runner and uploads machine evidence.
