# UV Installation & Operations

## Install uv (host)

Shanxi/OCI can install uv directly from a public PyPI mirror:

```bash
sudo python3 -m pip install -i https://mirrors.volces.com/pypi/simple/ uv
```

Or follow the upstream installer if GitHub/astral.sh is reachable.

## Static service venvs

Use `ci/deploy/bootstrap_venv.sh` for relay/worker static venvs:

```bash
ci/deploy/bootstrap_venv.sh /home/bgy/cb16-ci/relay/.venv -r ci/relay/requirements.txt
ci/deploy/bootstrap_venv.sh /data/cb16_ci/worker/.venv -r ci/worker/requirements.txt
```

The systemd `ExecStart` paths do not change.

## Provisioner host configuration

Host-local configuration lives in `/etc/cb16-ci/provision.env` (root:cb16-ci, mode 640).

Recommended for mainland China direct egress:

```ini
PIP_INDEX_URL=https://mirrors.volces.com/pypi/simple/
PIP_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cu126/
PIP_NO_CACHE_DIR=1
UV_INDEX_URL=https://mirrors.volces.com/pypi/simple/
UV_EXTRA_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/ https://mirrors.aliyun.com/pypi/simple/
UV_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu126/
UV_NO_CACHE=1
HF_ENDPOINT=https://hf-mirror.com
```

- `UV_EXTRA_INDEX_URL` may contain multiple whitespace-separated PEP-503 mirrors.
- `UV_FIND_LINKS` is for flat wheel directories (e.g. PyTorch cu126).
- `--index-strategy unsafe-best-match` is applied automatically by the Provisioner.
