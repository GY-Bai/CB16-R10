# CB16 V6.3 Shanxi Network Recovery Runbook

## Scope

This is a host/network repair only. Do **not** change CB16 scientific semantics, teacher logic, Central Brain logic, frozen model bytes, account physics, data chronology, final holdout policy, or model checkpoints.

Current source branch:

```text
ai/v63-activation-r0
```

The current code already supports host-local Python routing through:

```text
/etc/cb16-ci/provision.env
```

and reads host-local proxy variables without publishing their values.

## Known failure

The Shanxi worker has already failed all three code-side Python routes:

1. configured/private Python index route;
2. public PyPI after private index removal while preserving inherited proxy;
3. direct public PyPI after removing inherited proxy.

The latest classified failure was:

```text
PYTHON_REQUIREMENTS_DIRECT_PUBLIC_FALLBACK_FAILED_PIP_NETWORK_OR_INDEX
```

Therefore do not keep changing CB16 package-resolution semantics. Repair the host egress path or provide an OCI-hosted immutable wheel source.

## Hard safety boundaries

Never print or commit:

- proxy URLs containing credentials;
- tokens, passwords, service-token values, cookies, SSH keys;
- public/private host IP addresses;
- absolute model-weight/checkpoint paths;
- model weights or checkpoints;
- Binance local data contents;
- `/etc/cb16-ci/*.env` contents.

Do not read or open the final holdout.

Do not modify frozen model/checkpoint bytes.

Do not upload model weights to GitHub or OCI.

## Phase A — Diagnose the Shanxi route

On the Shanxi host, check out the exact current branch head and run:

```bash
python3 ci/diagnose_shanxi_network.py
```

The script is designed to emit only sanitized evidence. Preserve its JSON output.

Also inspect, without printing secret values:

```bash
systemctl status cb16-ci-worker.service --no-pager
```

Determine whether a usable local proxy/egress service already exists on Shanxi. Inspect service names, listening sockets, and configuration locally, but do not paste proxy credentials or addresses into GitHub/chat logs.

Required external destinations for V6.3 provisioning are:

```text
pypi.org
download.pytorch.org
huggingface.co
```

## Phase B — Preferred repair: existing Shanxi proxy

If a working host-local proxy already exists, configure the worker provisioning layer by editing:

```text
/etc/cb16-ci/provision.env
```

with the appropriate host-local routing variables, for example one or more of:

```text
HTTP_PROXY=...
HTTPS_PROXY=...
ALL_PROXY=...
NO_PROXY=...
```

Use mode `0600` and root ownership as appropriate. Do not reveal the values.

The repository code at/after the runbook commit explicitly imports these variables into Python provisioning. No scientific-code change is needed.

After configuration, rerun:

```bash
python3 ci/diagnose_shanxi_network.py
```

Acceptance for this phase:

- PyPI HTTPS succeeds in at least one non-direct routing mode;
- PyTorch CUDA wheel index is reachable;
- Hugging Face reachability is preferred for later V6.3 model provisioning.

If only Hugging Face remains unavailable, do not weaken asset integrity. Later V6.3 provisioning may instead use already-present frozen local model directories through the existing host-local hints:

```text
CB16_KRONOS_SMALL_ROOT
CB16_TIMESFM_2P5_ROOT
```

Only use them if the contained `model.safetensors` passes the repository's frozen SHA256/size verification.

## Phase C — Fallback repair: OCI immutable wheel/cache

Use this only if no safe/usable Shanxi proxy route can access public Python package sources.

OCI has the good external network. Build an environment-specific wheelhouse/cache on OCI for the exact repository requirements. Keep it outside GitHub.

Requirements:

- bind the cache to the exact requirements/environment identity;
- preserve the PyTorch `cu126` build required by `requirements-shanxi-pascal.txt`;
- serve/download over an authenticated or otherwise narrowly scoped existing OCI/Cloudflare path;
- verify wheel hashes before Shanxi installation;
- do not store model weights in this wheel cache;
- do not expose an unauthenticated arbitrary file server.

Configure Shanxi's host-local `provision.env` to use the resulting trusted Python package source. Do not commit its URL/credentials if they are sensitive.

## Phase D — Qualification after host repair

Do **not** trigger V6.3 yet.

First trigger exactly one unit CI commit on `ai/v63-activation-r0` with marker:

```text
[ci:unit]
```

The commit may be an empty/no-science-change trigger commit. Record its exact SHA.

Wait for the Shanxi result to appear on `ci-results` and verify:

- provisioning status is `READY`;
- Python import canaries pass;
- repository tests pass;
- frozen asset verifier tests pass;
- no secrets/host addresses/model paths appear in sanitized evidence.

Do not merge anything to `main`.

Do not trigger `[ci:v63]`; the primary ChatGPT executor will review the unit evidence first and trigger V6.3 itself.

## Return contract

Return only the following sanitized information to the primary executor:

```text
NETWORK_RECOVERY_STATUS = PASS | FAIL
PROCESS_ENV_ROUTE = PASS | FAIL
PROVISION_ENV_ROUTE = PASS | FAIL
DIRECT_ROUTE = PASS | FAIL
PYPI = PASS | FAIL
PYTORCH_CU126 = PASS | FAIL
HUGGINGFACE = PASS | FAIL
WORKER_SERVICE = ACTIVE | INACTIVE | ERROR
UNIT_TRIGGER_SHA = <40-char SHA or NONE>
UNIT_CI_STATUS = PASS | FAIL | NOT_RUN
REPAIR_MODE = EXISTING_SHANXI_PROXY | OCI_WHEEL_CACHE | NONE
```

Do not include secret values, proxy URLs, IP addresses, local model/checkpoint paths, or raw private configuration.

## UV adaptation notes

- The Provisioner now uses `uv` when it is present on `PATH`.
- `uv` should be used for `uv venv`, `uv pip install`, and `uv pip check`.
- For the flat PyTorch wheel directory (e.g. Aliyun `pytorch-wheels/cu126/`), use `UV_FIND_LINKS` (not `UV_EXTRA_INDEX_URL`).
- Use `--index-strategy unsafe-best-match` so CUDA companion packages can be resolved across PyPI and the PyTorch wheel mirror.
- Host-local configuration remains in `/etc/cb16-ci/provision.env`; no values are committed to GitHub.
