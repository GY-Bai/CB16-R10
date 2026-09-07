# CB16 R2 WSS Runner Storage Exception R0

Status: NONCANONICAL QUALIFICATION INFRA ONLY

This document records the minimum host-side sandbox change required to run R2 evidence-storage qualification on the independent WSS qualification runner. It must not alter R10.4 scientific state, canonical runner authority, model semantics, promotion/rejection, or FINAL 2025-09 access.

## Observed blocker

The WSS qualification runner executes in:

- unit: `cb16-github-runner.service`
- user/group: `cb16-ghrunner`
- cgroup: `/system.slice/cb16-github-runner.service`
- `ProtectSystem=strict`
- current `ProtectHome=yes`
- current `ReadWritePaths=/data/cb16_ci/github-runner-qualification/actions-runner`

Inside that runner, `/data` is read-only unless explicitly allow-listed. Therefore `/data/cb16_hdd/cb16_diagnostics/...` fails with `EROFS` even when ordinary Unix ownership/mode is otherwise valid.

The canonical runner independently proves the intended pattern: it also uses `ProtectSystem=strict`, while specific canonical runtime paths are explicitly listed in `ReadWritePaths`.

## Required R2 native paths

R2 native qualification requires:

- write: `/data/cb16_hdd/cb16_diagnostics/r2_native`
- read-only: `/data/cb16_hdd/cb16_runtime/R10_4`
- read-only: `/home/bgy/cb16_ssd/runtime/R10_3/qualification_r8_3_8w`
- read/execute: `/data/cb16_ci/venvs/b6e3e3c287f5f4e8ab0cb1b80a7af8aba0803a0f66a09d4501e7051a33edf7ba`
- write: the WSS runner's own GitHub Actions workspace (already allowed)

No package download channel is specified by the repository. Package transport remains host-owned. Package identity is direct package name/version/specifier plus import/CUDA canaries; package inventory SHA is not a qualification gate.

## Least-privilege model

The R2 qualification runner should have this view:

```text
/data/cb16_ci/github-runner-qualification/actions-runner   RW  existing
/data/cb16_hdd/cb16_diagnostics/r2_native                 RW  add
/data/cb16_hdd/cb16_runtime/R10_4                         RO  unchanged
/home                                                       hidden tmpfs
  /bgy/cb16_ssd/runtime/R10_3/qualification_r8_3_8w       RO  bind exception
```

Do not make `/data`, `/data/cb16_hdd`, or the R10.4 root writable to the WSS runner.

`ProtectHome=yes` cannot be combined with a nested bind exception under `/home`. For this qualification runner, use `ProtectHome=tmpfs` and expose only the exact R10.3 authority path through `BindReadOnlyPaths`.

## Host change

Create the R2 payload lane before restarting the qualification runner:

```bash
sudo install -d -o cb16-ghrunner -g cb16-ghrunner -m 0750 \
  /data/cb16_hdd/cb16_diagnostics/r2_native
```

Create:

`/etc/systemd/system/cb16-github-runner.service.d/r2-native-qualification.conf`

with:

```ini
[Service]
ProtectHome=tmpfs
ReadWritePaths=/data/cb16_hdd/cb16_diagnostics/r2_native
BindReadOnlyPaths=/home/bgy/cb16_ssd/runtime/R10_3/qualification_r8_3_8w
```

The existing `ReadWritePaths=/data/cb16_ci/github-runner-qualification/actions-runner` must remain present in the merged unit configuration. Do not reset the list with an empty `ReadWritePaths=` assignment.

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl show cb16-github-runner.service \
  -p ProtectSystem -p ProtectHome -p ReadWritePaths \
  -p BindReadOnlyPaths -p DropInPaths --no-pager
sudo systemctl restart cb16-github-runner.service
sudo systemctl is-active cb16-github-runner.service
```

This change applies only to the WSS qualification runner. Do not restart or modify `cb16-github-canonical-runner.service`.

## Mandatory post-change canary

Run GitHub workflow:

`CB16 R2 WSS Storage Exception Canary`

It must prove all of the following:

- runner cgroup is `cb16-github-runner.service`;
- `ProtectSystem=strict` remains enabled;
- `ProtectHome=tmpfs`;
- original WSS runner workspace remains RW;
- only R2 diagnostics payload lane is added RW;
- R10.3 authority root is available through a read-only bind;
- R10.4 remains read-only in the WSS runner view;
- disposable write + sync + delete succeeds under `r2_native`;
- verified Torch/CUDA environment remains functional;
- no write is attempted against R10.3 or R10.4.

Expected terminal markers:

```text
R2_WSS_STORAGE_EXCEPTION_CANARY=PASS
PROTECT_SYSTEM=strict
PROTECT_HOME=tmpfs
R103_AUTHORITY_VIEW=READ_ONLY
R104_AUTHORITY_VIEW=READ_ONLY
DIAGNOSTICS_WRITE=PASS
CANONICAL_CAMPAIGN_MODIFIED=false
SCIENTIFIC_SEMANTICS_CHANGED=false
FINAL_HOLDOUT_2025_09_ACCESSED=false
```

Only after this canary passes may `CB16 R2 Native Campaign Qualification` be dispatched.

## Prohibited changes

- Do not disable or weaken `ProtectSystem=strict`.
- Do not set `ProtectHome=false` or expose all of `/home`.
- Do not remount `/data` globally RW for the runner.
- Do not add `/data/cb16_hdd` wholesale to `ReadWritePaths`.
- Do not grant WSS write access to R10.4 or R10.3.
- Do not modify/restart the canonical runner for R2 qualification.
- Do not chmod/chown canonical R10.4 data.
- Do not access FINAL 2025-09.
- Do not make R2 qualification status-driving for the frozen R10.4 campaign.
