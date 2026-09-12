# 当前状态与接手断点

核对日期：2026-09-12 UTC。本文件记录文档提交前的代码基线，后续提交必然使 HEAD 改变。接手时先核对 live GitHub，再决定增量；不要把本页静态 SHA 当作强制 checkout 命令。

## 1. 已核对的分支

| 分支 | 本次核对 SHA | 地位 |
|---|---|---|
| main | `47f1693709918978bf1e2119d77b611c01f5991f` | 本轮文档的代码基线；包含 Stage-4 与此前实验代码 |
| ai/r11-demonstration-foundation-r0 | `b299be68a9bd9cc041900db75586ed5667407f8a` | 未合并示范学习、密集清单及 H72 分片生产线 |
| ai/r11-continuous-generation-qualification-r0 | `168978dc632be9c62a11d2748f87c549f137dcfe` | 存在的代际资格分支；不凭分支名推断生产闭环完成 |
| ai/r11-formal-league-contestant-g0-r0 | `b20223cc41164c29c30e4ca6aa29fbf31b8c0eb3` | 前序 contestant 工作 |
| ai/r11-formal-science-league-r0 | `d5200237448fe91a69443639dfca705cc1ea3918` | 前序 league 工作 |
| ai/r11-formal-science-league-season-r0 | `1e5e8ab084d01aee7357e789735df3c0f17fba48` | 旧 preseason 设计；新示范协议明确不再沿用其 50+10 划分 |

本轮不合并这些开发分支。首次文档提交只新增/修改 README、AGENTS 和 docs。

## 2. 能够确认的进度

### 单一示范 corpus 适配

示范分支的 [学习范式修订](https://github.com/GY-Bai/CB16-R10/blob/b299be68a9bd9cc041900db75586ed5667407f8a/authority/rearchitecture_r11/CB16_R11_LEARNING_PARADIGM_CORRECTION_V1.json) 明确允许同 corpus 学习、重叠 episode 和跨代回放；`TRAIN` 是 Teacher 路由标记；同 corpus loss 不是泛化证据。

[资格 receipt](https://github.com/GY-Bai/CB16-R10/blob/b299be68a9bd9cc041900db75586ed5667407f8a/authority/rearchitecture_r11/CB16_R11_DEMONSTRATION_FOUNDATION_QUALIFICATION_R0_FREEZE_RECEIPT_V1.json) 指向 run [34658217339](https://github.com/GY-Bai/CB16-R10/actions/runs/34658217339)。该 receipt 记录适配与继承测试通过、没有 Arena 科学消费。不能据此称完整历史 foundation 已训练或已学会交易。

### 密集清单已冻结，不能把计划数当完成数

[manifest receipt](https://github.com/GY-Bai/CB16-R10/blob/b299be68a9bd9cc041900db75586ed5667407f8a/authority/rearchitecture_r11/CB16_R11_DEMONSTRATION_FOUNDATION_R0_MANIFEST_FREEZE_RECEIPT_V1.json) 指向 run [34658436628](https://github.com/GY-Bai/CB16-R10/actions/runs/34658436628)：2020 年十资产，67,669 个合格母状态，预期 609,021 个九动作分支。清单生成时未加载模型、未编译 Teacher targets、未观察未来 utility。

### 第一个生产分片已验证发布

本轮读取了 run [34661816571](https://github.com/GY-Bai/CB16-R10/actions/runs/34661816571) 的状态、jobs 和 `materialize-btc-shard0` 日志：

- HEAD：`b299be68a9bd9cc041900db75586ed5667407f8a`。
- workflow 与相关 jobs success；20 个 materialization 测试通过。
- BTCUSDT shard 0：1,024 个母状态、9,216 个分支。
- 3 个 S4F 对象封存；再次发布仍是 3 个对象，幂等通过。
- 日志结果明确 `teacher_targets_compiled=false`、`student_training_started=false`。
- 分类为第一块 canonical shard，可计入 71 块计划。

这是日志与配置核验。本轮未重新下载并逐字节校验该 run 的 artifact，不宣称独立重算产物哈希。71 块是 [生产计划](https://github.com/GY-Bai/CB16-R10/blob/b299be68a9bd9cc041900db75586ed5667407f8a/authority/rearchitecture_r11/CB16_R11_DEMONSTRATION_PRODUCTION_MATERIALIZATION_PLAN_R0_V1.json) 的总数；本次证据确认第一块，不证明全部完成。

## 3. 当前示范与最高目标之间的缺口

| 已实现/已记录 | 尚不能等同于 |
|---|---|
| 单 corpus 适配与真实优化器更新能力 | 新策略主动生成并学习新账户轨迹的整个循环 |
| H72 分片母状态为冻结 flat account | 长期持仓、连续暴露下的广泛账户覆盖 |
| 首步候选动作，后续冻结 E6R2 续行 | 每一步重新调用 Brain 的自主交易 |
| 1 块生产分片发布成功 | 71 块全量完成、Teacher 编译或 foundation checkpoint 完成 |
| 原有 72h log utility 与 Teacher loss | 用户现在的长期期望超额收益目标 |
| 生命周期、存储和 binding 代码 | 整条新闭环已在真实市场经验上运行 |

这里的“尚不能”是证据边界，不是全仓库不存在其他实验的断言。

母状态与续行定义来自 [H72 shard spec](https://github.com/GY-Bai/CB16-R10/blob/b299be68a9bd9cc041900db75586ed5667407f8a/authority/rearchitecture_r11/CB16_R11_DEMONSTRATION_H72_SHARD_SPEC_V1.json)：`FROZEN_FLAT_PARENT_STATE_PLUS_ACCOUNT6_AT_DECISION` 和 `FROZEN_E6R2_BRANCH_RUNTIME`。已有数据可作为示范组件，不能默认为满足整个目标的必要且充分路线。

## 4. 下一步如何接手

当前用户授权的是文档建设，不是启动全量生产、改损失或开新科学实验。后续任务先读取 [未决问题](OPEN_QUESTIONS.md) 和 [决策记录](DECISIONS.md)，据用户实际任务选择路线。

工程设计前需要对齐：长期目标的数值定义、失败经验的训练用途、Supervisor/Physics 中手工风险规则的边界，以及示范生产与策略驱动连续账户循环的先后关系。

不要因为某个旧 JSON 写着 `next_gate` 就自动执行它。新任务若已明确授权，应继续完成，不对已有授权重复确认。

## 5. 自动检查与历史文件

main 的 [repo-guard](../.github/workflows/guard.yml) 对 push/PR 运行；[main smoke](../.github/workflows/cb16-r11-main-smoke.yml) 对 main push 运行，检查冻结 blob、Docker/Python 和有界 Stage-4 测试。文档提交可以触发现有这些检查，不需要手工 dispatch 科学训练。

仓库旧文档和旧 verdict 保留原身份。目标更新不会追溯改变已经执行的实验结论，也不会自动开放 FINAL 或 fresh data。

## 6. 本页后续更新必须包括

核对日期、分支与精确 SHA、所读代码/receipt/run、实际完成数量、证据类型、尚未验证的连接、下一项用户授权工作。不要只把计划复制成完成清单。
