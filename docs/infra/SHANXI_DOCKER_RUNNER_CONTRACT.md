# Shansi Docker Runner 契约（CB16 `shanxi-docker-r11`）

> 状态：**live-snapshot 契约**（从运行中的容器只读采集，非作者声明）
> 采集时间：2026-09-13T19:44Z（host 本地 2026-09-14T03:44+08:00）
> 采集方式：SSH 到 Shansi 主机后执行只读 `docker inspect` / `docker exec` 查询；
> **未读取任何 secret 值**（`provision.env` 只取 key 名，`.credentials*` 未读，镜像/容器 env 已过滤 token/key）。
> 机器可读版本：`authority/infra/SHANXI_DOCKER_RUNNER_SNAPSHOT_V1.json`
>
> 本文解决 Sol 提出的问题：**不再通过 `df` 和环境变量反推目录语义**。
> 凡本文未列出的路径，一律视为「无契约」；需要新增 scratch 位置时先改本文件再改工作流。

---

## 1. Container identity

| 项 | 值 |
|---|---|
| container name | `cb16-runner-r11` |
| container id | `26fb170773b85ce12f9d54113efcaf56b6838206be84810533a8a08c21cee30f` |
| image | `cb16-runner-r11:latest`（`sha256:dcba97125b2e42f76444cbdf369b82fe8930d017c25032e69b32c8227f0b5a0e`，304.7 MB，构建于 2026-09-09T08:36:54+08:00） |
| base image | `python:3.10-slim-bookworm`（Debian bookworm，CPython 3.10.21，镜像默认 `CMD ["python3"]`） |
| runtime command | `["./run.sh"]`（无 entrypoint），WorkingDir `= /home/cb16-runner/actions-runner` |
| restart policy | `unless-stopped`（**无 systemd unit、无 compose 管理**：容器 labels 为空、`Binds=null`、全部挂载为 named volume） |
| network | `cb16-net` |
| runtime / 权限 | `runc`，`Privileged=false`，`CapAdd=null`，无 memory/cpu 限制（`Memory=0`, `NanoCpus=0`） |
| GPU | `DeviceRequests=[{Count:-1, Capabilities:[["gpu"]]}]`（等价 `--gpus all`，nvidia container toolkit） |
| /dev/shm | `ShmSize=67108864`（64 MiB） |
| 启动时间 | `2026-09-09T13:21:57Z`（running=true，已跨多次 job 存活） |
| runner 版本 | GitHub Actions runner **2.337.0**（`bin/Runner.Listener --version`） |

### 1.1 由 `docker history` 重建的 Dockerfile（**重建物，非原始文件**）

宿主机上**未找到**该镜像的 Dockerfile / docker-compose / `docker run` 脚本 / entrypoint（已做有界搜索：
`/home/bgy`、`/data/cb16_ci`、`/etc/systemd/system` 内含 `cb16-runner-r11` 的文件；`bgy` 无 sudo，无法搜索 root-only 目录）。
以下步骤来自 `docker history --no-trunc`，可作为重建基线：

```dockerfile
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates coreutils curl findutils git grep gawk gzip \
      libgomp1 libicu72 libstdc++6 procps rsync sed tar time util-linux xz-utils \
    && rm -rf /var/lib/apt/lists/*

COPY uv /usr/local/bin/uv

RUN addgroup --gid 1001 cb16-runner \
 && useradd -m -u 1001 -g 1001 cb16-runner \
 && mkdir -p /cb16/raw /cb16/g0 /cb16/venv /cb16/uv-cache /cb16/worker \
             /cb16/runtime/r104 /cb16/r2-native /cb16/package /cb16/parent-r101 /cb16/parents

ENV UV_CACHE_DIR=/cb16/uv-cache
ENV PYTHONUNBUFFERED=1
USER cb16-runner
```

**缺失制品**（Sol 要求的原始文件清单中）：Dockerfile、docker-compose.yml / compose.yaml、runner startup script、
entrypoint —— 均**不存在于仓库或宿主机可读范围**；容器由 `docker run` 式即席创建 + `restart=unless-stopped` 维持。
建议后续补一份版本化的 `docs/infra/` 或 `ci/` 内的 Dockerfile/compose，使该 runner 可复现。

### 1.2 由 `docker inspect` 重建的 `docker run`（**重建物**）

```bash
docker run -d --name cb16-runner-r11 --restart unless-stopped \
  --network cb16-net --gpus all --shm-size 64m \
  -w /home/cb16-runner/actions-runner \
  -e CB16_RAW_ROOT=/cb16/raw -e CB16_G0_ROOT=/cb16/g0 -e CB16_R104_ROOT=/cb16/runtime/r104 \
  -e CB16_VERIFIED_VENV=/cb16/venv -e CB16_UV_CACHE_DIR=/cb16/uv-cache \
  -e CB16_CI_WORKER_ROOT=/cb16/worker -e CB16_PACKAGE_ROOT=/cb16/package \
  -e CB16_PARENT_R101_ROOT=/cb16/parent-r101 -e CB16_PARENT_G0=/cb16/parents/r10_1.pt \
  -e HTTP_PROXY=http://cb16-gost:33128 -e HTTPS_PROXY=http://cb16-gost:33128 \
  -e NO_PROXY=localhost,127.0.0.1 \
  -v cb16-raw:/cb16/raw:ro -v cb16-g0:/cb16/g0 -v cb16-runtime-r104:/cb16/runtime/r104:ro \
  -v cb16-venv:/cb16/venv -v cb16-uv-cache:/cb16/uv-cache -v cb16-worker:/cb16/worker \
  -v cb16-package:/cb16/package:ro -v cb16-parent-r101:/cb16/parent-r101:ro \
  -v cb16-parent-g0:/cb16/parents:ro -v cb16-r2-authority:/cb16/r2-authority:ro \
  -v cb16-r2-native:/cb16/r2-native -v cb16-config:/run/secrets:ro \
  -v cb16-runner-docker:/home/cb16-runner/actions-runner \
  cb16-runner-r11:latest ./run.sh
```

---

## 2. User identity

| 项 | 值 |
|---|---|
| 容器内用户 | `cb16-runner` |
| uid / gid | `1001` / `1001`（`uid=1001(cb16-runner) gid=1001(cb16-runner) groups=1001`） |
| HOME | `/home/cb16-runner` |
| shell | `/bin/sh` |
| runner 进程属主 | `Runner.Listener run` 以 `cb16-runner` 运行（`run.sh` 父进程同为该用户） |
| 目录属主映射 | `/home/cb16-runner/actions-runner` → `1001:1001`；容器内 `/cb16/raw`、`/cb16/package`、`/cb16/parent-r101`、`/cb16/runtime/r104` 显示 `uid=1000 gid=1000`（宿主遗留 uid，**runner 不可写**）；`/cb16/venv` 为 `997:997`（**runner 不可写**） |

---

## 3. Important paths

| 路径 | 语义 |
|---|---|
| `$RUNNER_TEMP` | GitHub 约定 = `<runner-root>/_work/_temp` = `/home/cb16-runner/actions-runner/_work/_temp`；**每个 job 开始时为空**，job 结束后由 runner 清理 |
| `$RUNNER_WORKSPACE` | GitHub 约定 = `<runner-root>/_work/<repo>` = `/home/cb16-runner/actions-runner/_work/CB16-R10`；`GITHUB_WORKSPACE` = `_work/CB16-R10/CB16-R10` |
| runner root | `/home/cb16-runner/actions-runner`（named volume `cb16-runner-docker`，跨 job / 跨重启持久） |
| `_work` 子目录 | `_temp`（临时）、`CB16-R10`（仓库检出）、`_actions`（action 缓存）、`_tool`（工具缓存）、`_PipelineMapping` |
| `/cb16/worker` | `CB16_CI_WORKER_ROOT` / `CB16_PROVISION_ENV=/cb16/worker/provision.env` 所在；**root:root 0755，runner 只读** |
| `/cb16/raw` | `CB16_RAW_ROOT`；只读科学输入 |
| `/cb16/g0` | `CB16_G0_ROOT`；**setgid 2755 + runner 属主**，是当前唯一「runner 可写且持久」的科学根 |
| `/cb16/runtime/r104` | `CB16_R104_ROOT`；只读 |
| `/cb16/venv` | `CB16_VERIFIED_VENV`；只读消费（属主 997，runner 不可写） |
| `/dev/shm` | tmpfs 64 MiB，`nosuid,nodev,noexec`，容器级、重启即清空 |
| `/run/secrets` | named volume `cb16-config`（ro）；含 `provision.env`（root:root 0640，runner 无读权限、无法读取） |
| `/tmp` | overlay（容器可写层）1777；跨 job 保留、无自动清理、容器删除即消失 |
| `_diag` | runner 诊断日志目录，当前 94 MB 且持续增长（宿主机侧负责轮转） |

### 3.1 路径契约总表（Sol 要求的那张表）

| Container path | Writable (runner uid 1001) | Persistent across jobs | Persistent across container restart | Capacity | Intended use |
|---|---|---|---|---|---|
| `/dev/shm` | yes（1777） | **no**（每 job 语义上为空；实际保留到重启） | no | **64 MiB** | IPC / 小信号量 / DataLoader 小共享 |
| `$RUNNER_TEMP` (`_work/_temp`) | yes（1001） | **no**（job 起止清空/重建） | 目录持久、内容不保证 | 与 `/data` 同 FS：**648 GiB free** | 小临时文件；job 内自清理 |
| `$RUNNER_WORKSPACE` (`_work/CB16-R10`) | yes（1001） | **yes**（复用同一目录，checkout 每次清理） | yes | 648 GiB free | 仓库检出 / 构建产物（job 范围内） |
| `_work/_tool`、`_work/_actions` | yes（1001） | yes（工具/action 缓存） | yes | 648 GiB free | runner 工具缓存（不建议业务写入） |
| `/home/cb16-runner/actions-runner`（runner root） | yes（1001） | yes | yes | 648 GiB free（已用 864 MB） | runner 状态（`.runner`/`.credentials`/`_diag`）——**禁删** |
| `/cb16/raw` | **no**（ro, uid 1000） | yes | yes | 648 GiB free（已用 1.3 GB） | 只读原始数据 |
| `/cb16/g0` | **yes**（1001:1001, 2755） | yes | yes | 648 GiB free（已用 15.6 MB） | G0 可写科学根 |
| `/cb16/runtime/r104` | **no**（ro, uid 1000） | yes | yes | 648 GiB free | 只读运行时 |
| `/cb16/venv` | **no**（uid 997, 0755） | yes | yes | 648 GiB free（已用 6.34 GB） | 只读已验证 venv |
| `/cb16/uv-cache` | **no**（root:root 0755） | yes | yes | 648 GiB free（0 B） | 名义 uv 缓存目录（**当前 runner 写不进去**，见 §10） |
| `/cb16/worker` | **no**（root:root 0755） | yes | yes | 648 GiB free（412 B） | `provision.env` 载体（只读） |
| `/cb16/package` | **no**（ro, uid 1000） | yes | yes | 648 GiB free（205 MB） | 只读包/材料 |
| `/cb16/parent-r101` | **no**（ro, uid 1000） | yes | yes | 648 GiB free（200 MB） | 只读父代 |
| `/cb16/parents` | **no**（ro, root） | yes | yes | 648 GiB free（1.5 MB） | 只读父代（`r10_1.pt`） |
| `/cb16/r2-authority` | **no**（ro, root） | yes | yes | 648 GiB free（0 B） | 只读 R2 authority |
| `/cb16/r2-native` | **yes**（1001:1001, 0750） | yes | yes | 648 GiB free（6.9 MB） | R2 native 可写区 |
| `/run/secrets` | **no**（ro, root） | yes | yes | 648 GiB free（412 B） | secrets 卷（runner 无读权限） |
| `/tmp` | yes（1777） | yes（无清理） | yes | 648 GiB free（overlay 上） | 临时；建议仅 job 内自清理 |
| `/cb16`（父目录） | no（root:root 0755） | 随挂载点 | 随挂载点 | overlay | 挂载点容器目录，勿写入 |

> 全部 `/data` 卷与 Docker 数据根同处 **`/dev/sda1`（916 GiB 总 / 648 GiB free）**：表里的 "648 GiB free" 是**共享**容量，
> 不是每个路径各自 648 GiB。

---

## 4. Resources

### 4.1 宿主机

| 项 | 值 |
|---|---|
| hostname | `bgyubuntusever22041` |
| kernel | `5.15.0-191-generic`（Ubuntu） |
| CPU | 16 逻辑核 |
| 内存 | 15 GiB total（采集时 1.4 GiB used，13 GiB available） |
| swap | 33,554,428 KiB（33.5 GiB） |
| GPU | **NVIDIA GeForce GTX 1060 6GB**（单卡，UUID `GPU-82acd8ea-83e0-00e2-44d1-e5496bc81c1c`） |
| Docker root | `/data/cb16_docker`（overlay2） |

### 4.2 存储

| 文件系统 | Size | Used | Avail | Use% | 用途 |
|---|---|---|---|---|---|
| `/dev/sda1` → `/data` | 916 G | 222 G | **648 G** | 26% | Docker data-root + 全部 `cb16-*` volume |
| `/dev/sda1` inodes | 61,054,976 | 1,873,964 | 59,181,012 | 4% | |
| `/dev/mapper/ubuntu--vg-ubuntu--lv` | 423 G | 373 G | **32 G** | 93% | 挂载在 `/usr/bin/nvidia-smi`（nvidia 工具链文件级 bind），与 runner scratch 无关但已近满 |

### 4.3 容器视图

| 项 | 值 |
|---|---|
| `nproc` | 16 |
| 内存限制 | 无（`Memory=0`，容器的 `/proc/meminfo` 直接暴露宿主机 16.28 GB） |
| swap（容器所见） | 33.5 GB |
| `/dev/shm` | 64 MiB（`ShmSize=67108864`） |
| GPU | `nvidia-smi -L` 可见 GTX 1060 6GB |
| volume 用量 | `cb16-venv` 6.34 GB；`cb16-raw` 1.30 GB；`cb16-runner-docker` 864 MB；`cb16-package` 205 MB；`cb16-parent-r101` 200 MB；`cb16-runtime-r104` 174 MB；`cb16-g0` 15.6 MB；`cb16-r2-native` 6.9 MB；`cb16-parent-g0` 1.53 MB；`cb16-worker` 412 B；`cb16-uv-cache` 0 B；`cb16-r2-authority` 0 B；`cb16-config` 412 B；**孤儿卷 `cb16-runner-qual` 679.6 MB（links=0）** |

---

## 5. Docker mount contract

全部为 **named volume**（无 host bind；`Binds=null`），宿主机源路径统一为
`/data/cb16_docker/volumes/<volume>/_data`（Docker root = `/data/cb16_docker`，ext4 `/dev/sda1`）。

| Named volume | Host path | Container path | RW | 容器内 mode / owner |
|---|---|---|---|---|
| `cb16-raw` | `/data/cb16_docker/volumes/cb16-raw/_data` | `/cb16/raw` | **ro** | 0775 uid=1000 |
| `cb16-g0` | `/data/cb16_docker/volumes/cb16-g0/_data` | `/cb16/g0` | rw | **2755 uid=1001（setgid）** |
| `cb16-runtime-r104` | `/data/cb16_docker/volumes/cb16-runtime-r104/_data` | `/cb16/runtime/r104` | **ro** | 0775 uid=1000 |
| `cb16-venv` | `/data/cb16_docker/volumes/cb16-venv/_data` | `/cb16/venv` | rw | 0755 uid=997 |
| `cb16-uv-cache` | `/data/cb16_docker/volumes/cb16-uv-cache/_data` | `/cb16/uv-cache` | rw | 0755 root |
| `cb16-worker` | `/data/cb16_docker/volumes/cb16-worker/_data` | `/cb16/worker` | rw | 0755 root |
| `cb16-package` | `/data/cb16_docker/volumes/cb16-package/_data` | `/cb16/package` | **ro** | 0775 uid=1000 |
| `cb16-parent-r101` | `/data/cb16_docker/volumes/cb16-parent-r101/_data` | `/cb16/parent-r101` | **ro** | 0755 uid=1000 |
| `cb16-parent-g0` | `/data/cb16_docker/volumes/cb16-parent-g0/_data` | `/cb16/parents` | **ro** | 0755 root |
| `cb16-r2-authority` | `/data/cb16_docker/volumes/cb16-r2-authority/_data` | `/cb16/r2-authority` | **ro** | 0755 root |
| `cb16-r2-native` | `/data/cb16_docker/volumes/cb16-r2-native/_data` | `/cb16/r2-native` | rw | 0750 uid=1001 |
| `cb16-config` | `/data/cb16_docker/volumes/cb16-config/_data` | `/run/secrets` | **ro** | 0755 root（内含 `provision.env` 0640 root） |
| `cb16-runner-docker` | `/data/cb16_docker/volumes/cb16-runner-docker/_data` | `/home/cb16-runner/actions-runner` | rw | 0755 uid=1001 |

容器 env（**只列 key**，值已省略/过滤）：`CB16_RAW_ROOT`、`CB16_G0_ROOT`、`CB16_R104_ROOT`、`CB16_VERIFIED_VENV`、
`CB16_UV_CACHE_DIR`、`CB16_CI_WORKER_ROOT`、`CB16_PACKAGE_ROOT`、`CB16_PARENT_R101_ROOT`、`CB16_PARENT_G0`、
`HTTP_PROXY`/`HTTPS_PROXY`（`http://cb16-gost:33128`）、`NO_PROXY`、`HOME`、`PATH`、`LANG`、`PYTHON_VERSION`、
`PYTHON_SHA256`、`UV_CACHE_DIR`、`PYTHONUNBUFFERED`。

`/cb16/worker/provision.env` 的 key（**只列 key**）：`PIP_INDEX_URL`、`PIP_EXTRA_INDEX_URL`、`PIP_NO_CACHE_DIR`、
`UV_INDEX_URL`、`UV_EXTRA_INDEX_URL`、`UV_FIND_LINKS`、`UV_NO_CACHE`、`HF_ENDPOINT`。

---

## 6. Intended scratch contract（**提案，待 Sol 批准后才成为契约**）

现状：**没有任何一条「大科学 scratch」专用挂载**；runner 可写的持久位置只有 `/cb16/g0`、`/cb16/r2-native`
与 runner 自己的 `_work`。以下为建议映射：

| Scratch 类型 | 建议位置 | 依据 / 约束 |
|---|---|---|
| small tmp（< 64 MiB、IPC） | `/dev/shm` | 已是 tmpfs 64 MiB；`nosuid,nodev,noexec`；**不要**放大数据 |
| small tmp（job 内一次性） | `$RUNNER_TEMP` = `_work/_temp` | runner 语义保证「job 开始为空」，勿跨 job 传状态 |
| large scientific scratch（GB～百 GB） | **建议新增 `cb16-scratch` volume → `/cb16/scratch`**（rw、uid 1001、同一 `/dev/sda1`、648 GiB free） | 现行 `/cb16/g0` 是科学根、`/cb16/worker` 是 authority 输入，均不应被临时数据污染 |
| durable artifact staging | 建议 `cb16-scratch/<job_id>/`（或经 provisioner 的 `jobs/<job_id>/`） | 当前 `/cb16/worker` 对 runner **只读**，无法直接落盘 |
| disposable CI workspace | `_work/CB16-R10` + `_work/_temp` | runner/checkout 自带清理语义 |
| 禁止用作 scratch | `/cb16/raw`、`/cb16/runtime/r104`、`/cb16/package`、`/cb16/parents`、`/cb16/parent-r101`、`/cb16/r2-authority`、`/cb16/venv`、`/run/secrets`、runner root | 只读输入/状态；写入会破坏 authority 或 runner 自身 |

---

## 7. Cleanup contract

| 范围 | 谁可以删 | 说明 |
|---|---|---|
| `_work/_temp`、`_work/_actions`、`_work/_tool` | runner 自身 | GitHub runner 生命周期管理；人工清理前先停 runner |
| `_work/CB16-R10`（检出目录） | runner / checkout action / job | 每次 job 由 checkout 重置；跨 job 复用 |
| `/tmp` | job 自行清理（无系统清理） | 会持续占用容器可写层；**约定：job 结束前删自己创建的 `/tmp` 内容** |
| `/cb16/g0` | 由持有 g0 写入职责的 job / provisioner | 是科学可写根；**不**做通用 scratch 清理 |
| `/cb16/uv-cache` | **仅宿主机 root/provisioning**（runner 无写权限） | runner 内 `uv` 无法写入该目录 |
| `/cb16/worker`、`/run/secrets`、`/cb16/package`、`/cb16/raw`、`/cb16/runtime/r104`、`/cb16/venv`、父代卷 | **仅宿主/授权 provisioner** | 任何 `rm -rf` 都属破坏 authority |
| runner root（`.runner`/`.credentials`/`_diag`） | **仅宿主维护者** | 删除 = runner 失联，需重新注册 |
| Docker volume（13 个） | **仅宿主维护者** | `docker volume rm` 不可逆；`cb16-runner-qual` 为孤儿卷（links=0） |

**永不删除清单**：`cb16-raw`、`cb16-runtime-r104`、`cb16-package`、`cb16-parent-r101`、`cb16-parent-g0`、
`cb16-r2-authority`、`cb16-venv`、`cb16-config`、`cb16-runner-docker`、`cb16-worker` 中的 `provision.env`。

---

## 8. 现有 infra docs / scripts 清单

### 仓库内（这些是已版本化的执行契约）

| 路径 | 作用 |
|---|---|
| `ci/docker_preflight.py` | 容器内 fail-closed 路径/GPU 预检（本文 §3/§5 的机器校验版） |
| `ci/check_r11_docker_path_policy.py` | 静态路径/runner-label 契约（禁 `/home/bgy` 等宿主路径） |
| `ci/shanxi_runner_pre_job_gate.py` | 宿主机 job-start gate（**本容器未安装 hook，见 §9 缺口 5**） |
| `ci/run_shanxi_ci.sh` | 自定义 relay 体系的 CI 入口（`ci_output/`、`result.json`、`SHA256SUMS`） |
| `ci/CI_PROFILES.md` | 推送 marker → profile（smoke/unit/r102/r103/r104/v63）契约 |
| `ci/deploy/*.service`、`relay.env.example`、`worker.env.example` | relay/worker 部署模板（systemd user units） |
| `provision/README.md`、`provision/environments/*.json`、`provision/scripts/*` | 依赖/资产 provisioning（`CB16_CI_WORKER_ROOT/jobs/<job_id>/`） |
| `provision/assets/*.json`、`provision/schemas/*` | 逻辑资产与 schema |
| `docs/UV_INSTALLATION.md`、`docs/OPERATIONS.md` | uv 安装与 relay/worker 运维 |
| `.github/workflows/_cb16-shanxi-preflight.yml`、`.github/workflows/cb16-r11-*.yml` | GitHub Actions 侧 runner/label/流程契约 |

### 宿主机（只读采集到的现状）

| 路径 | 作用 |
|---|---|
| `/etc/systemd/system/cb16-ci-worker.service` | 自定义 relay worker（**不驱动本 docker runner**） |
| `/etc/systemd/system/cb16-github-canonical-runner.service`（disabled） | 历史 canonical runner（user `cb16-ci`，`/data/cb16_ci/github-runner-canonical`） |
| `/etc/systemd/system/cb16-gh-runner-proxy.service`（disabled） | 历史 gost runner 代理 |
| `/etc/systemd/system/cb16-generational-views.service`、`cb16-raw-view.service` | 视图/挂载辅助服务 |
| `/data/cb16_ci/worker/worker.py`、`run_shanxi_ci.sh` | relay worker 代码 |
| `/data/cb16_ci/cache/`、`assets/`、`asset_registry/` | relay 资产缓存 |
| `docker container cb16-gost`（image `cb16-gost:latest`，`-C /config/gost.yml`，volume `cb16-gost-config` → `/config`） | 出口代理（runner env 指向 `http://cb16-gost:33128`） |
| **不存在** | 本 docker runner 的 Dockerfile / compose / 启动脚本 / entrypoint / systemd unit |

## 9. 缺口与风险（供 Sol 决策）

1. **不可复现**：`cb16-runner-r11` 没有版本化的 Dockerfile/compose/`docker run` 记录；重装只能靠本文件 §1.1/§1.2 的重建物。
2. **`/cb16/uv-cache` 不可写**：`UV_CACHE_DIR=/cb16/uv-cache` 但目录 `root:root 0755`，runner(1001) 无法写入 → uv 缓存实际不生效（或静默失败）。
3. **`/cb16/worker` 不可写**：`CB16_CI_WORKER_ROOT` 指向它，但 `root:root 0755`，runner 无法落任何 staging 产物；当前仅承载只读 `provision.env`。
4. **`/dev/shm` 只有 64 MiB**：多 worker DataLoader / 多进程 IPC 容易 OOM；工作流的 `shared memory` 需求需显式设计。
5. **job-start gate 未安装**：容器 env 无 `ACTIONS_RUNNER_HOOK_JOB_STARTED`，runner `.env` 亦无 hook → 仓库声明的「宿主机硬边界」在当前容器上不成立（策略侧只剩 GitHub 工作流与 relay 体系）。
6. **无资源上限**：容器 `Memory=0`（宿主仅 15 GiB + 33.5 GiB swap），恶意/失控 job 可拖垮宿主与其上所有 runner/服务。
7. **孤儿卷** `cb16-runner-qual`（679.6 MB，links=0）等待回收决策。
8. **`_diag` 94 MB 且增长** + `/tmp` 无自动清理 → 容器可写层长期膨胀（容器删除才回收）。
9. **`/data` 与宿主其它服务共享**：648 GiB free 是全机共享额度（`/dev/sda1`），大 scratch 需要配额约定，而不是「看到 648G 就写」。
10. **另一块 LV 已 93% 满**（`/dev/mapper/ubuntu--vg-ubuntu--lv`，373G/423G），当前只 bind 了 `/usr/bin/nvidia-smi`；不要误当作 scratch。

---

## 10. 使用建议（给未来 workflow 作者）

1. 需要临时空间 → 用 `$RUNNER_TEMP`（job 内）或 `/dev/shm`（< 64 MiB）。
2. 需要跨 job 持久且可写 → 先按 §6 提案新增 `cb16-scratch`；在此之前**只**使用 `/cb16/g0/<带命名空间+job-id 的子目录>`，并在清单里登记。
3. 任何写入前先对照 §3.1 的 Writable 列；只读卷写失败不是「环境问题」，是本契约的预期行为。
4. 新增/变更挂载点 → 更新本文件与 `authority/infra/SHANXI_DOCKER_RUNNER_SNAPSHOT_V1.json`，再改工作流（先契约后执行）。
