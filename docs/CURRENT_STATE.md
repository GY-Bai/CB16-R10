# 当前状态与唯一接手断点

核对日期：2026-09-14 UTC；文档审阅基线 main `b88c152a007bb483be5ed0f4ad6371f13e317558`。这是时点记录；执行前重新核对相应 PR/review/receipt。旧 PR body 中的“待批准”或“READY”不能覆盖后继审核与合并事实。

## 1. 当前目标与新设备决定

**完成 S1 受控端到端可学习性；当前直接依赖为 R21 RC2-Recovery。** 不重开已完成 CC/S0/S0-v2，不把 Evolution 或数据库四路竞争变成共同前置条件。

用户于本轮明确：**起步 CPU-only，先降低复杂度。** Sol 应将该设备决定落实到当前执行 profile/相关检查，复用已验证依赖；不要仅为 CPU 执行重装整套工具链。尚未完成 CPU-only profile 验证，本页不代签授权。模型、种子、奖励、优化器、预算、阈值保持冻结；若设备迁移暴露数值/身份差异，报告并验证，不能改科学门槛救结果。详见 [性能策略](PERFORMANCE_STRATEGY_3700X_1060.md)。

## 2. 已合入 main 与未合入工作

| 对象 | 本次核对状态 | 证据范围 |
|---|---|---|
| CC / PR #99 | 已合入 | 合成闭环、已知答案和指定 Shanxi benchmark；不是市场经济证据 |
| S0 | 已合入、PASS | 后继经济排序/双基准/晋升合同迁移；不再等待 baseline 主次决定 |
| S0-v2 | 已合入、Sol R2 QUALIFIED | durable observation/replay、联合动作学习、恢复与换代基础；不等于 S1 |
| 资格框架 / PR #104 | 已合入，merge `15f6a14efa09c3d72ace9f5f3e77d1d80c3f0129` | 公共 primitive；不自动证明阶段科学声明 |
| S1 / PR #102 | Draft / execution frozen；未合入 | PR 记录 runtime 已接受，CI-A/B/D 证据保留；尚无有效完整 scientific verdict |
| RC2 R0 / PR #106 | 已合入本次 main 基线 | Recovery 合同；不是 Recovery runtime PASS |
| RC2 R1–R4 / PR #107–#110 | 本次仍 open | 作者产物/运行记录不自动等于 Sol 接受或 main 集成完成 |
| S2–S5 | 后续科学路线 | 当前不启动历史训练、扩容或经济确认 |

基础证据：[S0 receipt](../authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json)、[S0-v2 receipt](../authority/rearchitecture_r11/CB16_R11_S0V2_RECEIPT_V1.json)、[S0-v2 Sol R2 review](reviews/S0V2_CODE_REVIEW_2026-09-13_R2.md)、[CC integration receipt](../authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json)。

## 3. S1 的精确身份

- PR：[S1 #102](https://github.com/GY-Bai/CB16-R10/pull/102)，分支 `ai/r11-post-cc-s1-end-to-end-learnability-r1`。
- PR 记录的 accepted runtime：`19c6b7ece5a5f76b048ed2946fa8fb1354765b63`。
- 本次核对的 metadata/authorization head：`a974e2803ccc2693d67a0375e460da35837636a4`；历史授权不解除当前执行冻结。
- 原 parent：S0-v2 accepted merge `fc80b472236e7a4df8094563f8adac826fc42231`。
- [R1 TODO](https://github.com/GY-Bai/CB16-R10/blob/a974e2803ccc2693d67a0375e460da35837636a4/docs/post_cc/S1_R1_IMPLEMENTATION_AND_QUALIFICATION_TODO.md)、[baseline](https://github.com/GY-Bai/CB16-R10/blob/a974e2803ccc2693d67a0375e460da35837636a4/authority/rearchitecture_r11/CB16_R11_POST_CC_S1_BASELINE_V1.json)、[runner](https://github.com/GY-Bai/CB16-R10/blob/a974e2803ccc2693d67a0375e460da35837636a4/scripts/run_r11_post_cc_s1_learnability.py)。

不要执行旧 r0 分支，也不要因 main 暂无 S1 runner 就重写它。main、S1 分支、RC2 叠层分支和已部署容器是不同身份，集成与资格由 Sol 显式绑定。

## 4. RC2 恢复路径与实际断点

当前任务来源：[RC2 TODO](infra/R21_RC2_INFRA_UPGRADE_TODO.md)、[复用与测量规则](infra/R21_RC2_UPSTREAM_REUSE_AND_MEASUREMENT_RULES.md)、[Recovery spec](../authority/infra/R21_RC2_RECOVERY_SPEC_V1.json)。CPU-only 决定由 Sol 对设备相关执行合同作显式后继绑定，不改写历史 receipt。

| 任务 / PR | 本次核对 head | 已提交内容 / 待确认 |
|---|---|---|
| R1 / [#107](https://github.com/GY-Bai/CB16-R10/pull/107) | `bd3153f6051337e7b19bf3ac10da651fd5357cd4` | runner 定义与只读检查；按实际 review 确认接受状态 |
| R2 / [#108](https://github.com/GY-Bai/CB16-R10/pull/108) | `5afec101e9caf4746244c9c597fbe73699d01eeb` | SSD 方案；旧 body 的 owner-pending 必须结合 R3 后继 approval V2 读取，不重复询问旧选项 |
| R3 / [#109](https://github.com/GY-Bai/CB16-R10/pull/109) | `c6c1d9cdab1a3c858e34ad2797294dde74073f76` | 作者报告新 runner、100 GB SSD hot 与 2 GiB shm；本次未独立重验宿主或代签接受 |
| R4 / [#110](https://github.com/GY-Bai/CB16-R10/pull/110) | `59690ddde02061dfd69ecb8b0a521226af38c4ac` | 测量修订已交审；[run 34791392083](https://github.com/GY-Bai/CB16-R10/actions/runs/34791392083) success；不等于完整负载延迟/积压已合格 |

下一步由 Sol 收拢现有审阅状态、绑定 CPU 执行配置，按既有门槛推进 R5 placement-only canary；只有 R4/R5 证明必要才开启 R6。之后 R7 验证适用恢复能力、R8 重新授权 S1 CI-C。文档精炼不跳过现有必要 gate，也不新增一轮通用框架建设。

## 5. 接手只带这些材料

先读 [VISION](VISION.md) → [ARCHITECTURE_MAP](ARCHITECTURE_MAP.md) → 本页；再按 [角色协议](ROLES_AND_REVIEW_PROTOCOL.md) 读取当前任务、直接上游代码和适用合同。完整历史、旧失败与原始科学值按需追溯。

本次是文档与方向对齐，未修改业务代码/历史 authority，未做宿主变更、未运行训练或代替 Sol 接受。CPU-only 已获用户方向授权，具体可执行状态仍需对应证据。
