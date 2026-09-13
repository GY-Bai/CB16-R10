# CB16 Main TODO

> 状态：仓库级长期 TODO / 技术债清单。
>
> 本文件记录**具有长期收益、但不应被临时塞进当前科学阶段**的工程与基础设施工作。它不是 scientific authority，也不自动授权修改冻结的 reward、model、seed、budget、threshold、oracle、evaluation 或 FINAL 边界。
>
> 当前科学阶段、工作 PR 与 exact SHA 仍以 [CURRENT_STATE](CURRENT_STATE.md) 和对应 authority/receipt 为准。

## 维护规则

- `TECHNICAL_DEBT` 表示已确认存在、值得长期偿还，但可以与当前 scientific program 解耦的工程债。
- 技术债不得为了“顺手修复”而改变当前冻结实验的科学语义。
- 每项债务在实施前应有独立 task/branch/receipt；涉及 Shanxi runner 的改动必须先更新 infra contract/snapshot，再更新 workflow。
- 已完成项目保留 completion evidence/commit，不直接删除历史。
- 当前 S1 formal qualification 允许使用已验证的 `$RUNNER_TEMP` 作为 **job-local disposable scratch**；这是 execution-only 临时方案，不等价于完成下面的长期 scratch 架构。

---

# TD-SHANXI — Shanxi Docker Runner / Provisioning 技术债

权威现状：

- [SHANXI_DOCKER_RUNNER_CONTRACT](infra/SHANXI_DOCKER_RUNNER_CONTRACT.md)
- `authority/infra/SHANXI_DOCKER_RUNNER_SNAPSHOT_V1.json`
- live snapshot baseline commit：`1349d31778988a3f76308820f60f4a4576bf9ce3`

当前核心事实：runner 为 uid/gid `1001:1001`；`/dev/shm` 仅 64 MiB；`/cb16/worker` 与 `/cb16/uv-cache` 虽是 RW volume，但目录权限使 runner 实际不可写；`$RUNNER_TEMP` 位于 host-backed runner volume，可写且与 `/dev/sda1` 共享大容量。

## TD-SHANXI-01 — 版本化 Runner 构建与启动定义

**状态：TECHNICAL_DEBT / P1**

当前 `cb16-runner-r11` 由 ad-hoc `docker run` + `restart=unless-stopped` 维持，仓库中没有原始 Dockerfile、compose、startup/entrypoint 或 systemd 定义。

目标：

- 增加版本化 Dockerfile；
- 增加 compose 或等价 `docker run` declarative spec；
- 固化 user/uid/gid、mount、GPU、network、shm、restart policy；
- 提供 bootstrap/rebuild/rollback 操作说明；
- 使 live container 可以从 repo authority 可重复重建。

完成条件：干净主机可按 repo 定义重建 runner，并通过同一 preflight/identity receipt。

## TD-SHANXI-02 — 专用大容量 Scientific Scratch

**状态：TECHNICAL_DEBT / P0**

不要再把 durable replay / sqlite / checkpoint campaign 放进 64 MiB `/dev/shm`，也不要污染 `/cb16/g0`。

建议目标：

```text
cb16-scratch named volume
    -> /cb16/scratch
    owner 1001:1001
    rw
```

需要同时定义：

- job namespace：`/cb16/scratch/<stage>/<run_id>/`；
- free-space / quota guard；
- lifecycle 与 cleanup owner；
- artifact upload 前不得删除；
- abnormal termination 后的 orphan cleanup；
- 与 `/cb16/g0` durable science authority 明确隔离。

完成条件：formal multi-worker campaign 不依赖 `/dev/shm` 或 runner internal `_work` 承载大 scratch，并有 fail-closed capacity probe。

## TD-SHANXI-03 — `CB16_CI_WORKER_ROOT` / Provisioner Ownership 漂移

**状态：TECHNICAL_DEBT / P0**

repo 中 `provision_common.py` / `provision/README.md` 把 `CB16_CI_WORKER_ROOT/jobs|venvs|locks|asset_registry` 当作可创建工作区，但 deployed `/cb16/worker` 为 `root:root 0755`，runner uid 1001 无法创建目录。

必须先做语义决策，二选一并统一：

1. `/cb16/worker` 是 **config/provision carrier only**：保持 runner 只读，Provisioner writable roots 改到专用 volume；或
2. `/cb16/worker` 是真正的 writable worker root：修正 ownership/mode，并明确其中 `provision.env` 的保护边界。

禁止通过 workflow 临时 `chmod/chown` 绕过 contract。

完成条件：code、docs、env、live mount ownership 和 preflight 对同一 worker-root 语义一致。

## TD-SHANXI-04 — `UV_CACHE_DIR` 假可写问题

**状态：TECHNICAL_DEBT / P1**

当前 `UV_CACHE_DIR=/cb16/uv-cache`，但 live directory 为 `root:root 0755`，runner 不可写。

目标：

- 要么修正 volume ownership/mode 为 runner 可写；
- 要么把 uv cache 指向合法 writable cache；
- 明确 cache eviction / capacity / rebuild semantics；
- preflight 增加真实 write/unlink canary。

## TD-SHANXI-05 — Docker Preflight 覆盖 declared writable roots

**状态：TECHNICAL_DEBT / P0**

当前 `ci/docker_preflight.py` 会验证 raw read-only 和 g0 writable，但没有验证：

- `CB16_CI_WORKER_ROOT` 实际可写；
- `CB16_UV_CACHE_DIR` 实际可写；
- large scratch contract；
- `$RUNNER_TEMP` / dedicated scratch 的 capacity 与 filesystem class。

目标：让“env 声称可用”与“runner 实际可写”之间不存在盲区。

完成条件：对所有 declared writable roots 执行 create/fsync/unlink canary；权限漂移时 preflight fail closed。

## TD-SHANXI-06 — Job-start Hook / 宿主硬边界恢复

**状态：TECHNICAL_DEBT / P1**

仓库存在 `ci/shanxi_runner_pre_job_gate.py`，但 live Docker runner 没有 `ACTIONS_RUNNER_HOOK_JOB_STARTED`。

目标：

- 明确该 hook 是否仍是 R11 的必需硬边界；
- 若是，版本化安装并验证；
- 若不是，删除/降级过时声明，避免文档声称存在实际上未执行的保护。

## TD-SHANXI-07 — Container Resource Limits 与并发策略

**状态：TECHNICAL_DEBT / P1**

当前容器无 memory/cpu limit；host 约 15 GiB RAM + 33.5 GiB swap，16 logical CPUs，单 GTX 1060 6GB。

目标：

- 明确 runner 可用 CPU/RAM/swap/GPU 预算；
- 为 formal multi-worker campaign 定义 concurrency guard；
- 防止单个失控 job 拖垮 host 上其他 runner/service；
- resource exhaustion 必须分类为 execution/hardware failure，不污染 scientific verdict。

## TD-SHANXI-08 — Runner 临时数据与 Orphan 清理

**状态：TECHNICAL_DEBT / P2**

现状包括：

- `_diag` 持续增长；
- `/tmp` 无自动清理；
- orphan volume `cb16-runner-qual`；
- failed campaign 可能留下 scratch。

目标：版本化 retention/cleanup policy，所有删除必须遵守 never-delete authority list。

## TD-SHANXI-09 — Shared Disk Capacity / Quota Policy

**状态：TECHNICAL_DEBT / P1**

`/data` 的可用空间是 Docker root 与全部 `cb16-*` volume 共享资源，不是每个路径独占。

目标：

- formal campaign 前执行 capacity guard；
- 约定 reserve floor；
- 记录 pre/post disk bytes + inode；
- 大 campaign 估算 peak scratch；
- 必要时为 `cb16-scratch` 建立 quota/soft limit。

## TD-SHANXI-10 — Infra Snapshot Refresh Protocol

**状态：TECHNICAL_DEBT / P1**

`SHANXI_DOCKER_RUNNER_SNAPSHOT_V1.json` 是 live snapshot，不应永久当成不会变化的事实。

目标：

```text
infra change
-> refresh machine snapshot
-> review human contract
-> preflight update
-> workflow adoption
-> exact-host evidence
```

触发条件至少包括：image rebuild、mount/ownership change、runner version change、shm/resource change、new scratch volume、provision root change。

---

# 当前 S1 的临时执行例外

在 TD-SHANXI-02 完成前，S1 formal qualification 可以使用：

```text
$RUNNER_TEMP/s1-qualification-$GITHUB_RUN_ID
```

约束：

- 仅 job-local disposable scratch；
- 不作为跨 job authority；
- 必须位于 writable host-backed filesystem；
- formal artifact 必须在 cleanup 前上传；
- 运行前保持保守 free-space guard；
- 不改变 frozen S1 seeds/model/reward/budget/evaluation/threshold/oracle；
- 该例外完成 S1 后不自动推广成永久 large-scratch contract。

这是一项 **execution-only bridge**，不是对长期 Infra 技术债的关闭。
