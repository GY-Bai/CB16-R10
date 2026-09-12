# 当前状态与接手断点

核对日期：2026-09-12 UTC。**本轮代码快照：main `2da90a0b4577ad7d950641fe0a1bcd81f08d58e2`**。提交前补核对：`7c412ee264c463e8f9f28c707c02eccfa4ab055c` 新增 AC-014 执行适配器，具体增量见下节。main 正由实现 agent 持续推进，接手必须核对 live SHA；以下不是强制 checkout 命令，也不是后续提交的完成证明。

## 1. 当前分工和目标

用户要求本设计对话只负责最高理念和文档；GPT-5.6 sol 负责 TODO 拆解和代码实现。已发布 [理念对齐规则](PRINCIPLE_ALIGNMENT.md) 与 [组件要求](COMPONENT_REQUIREMENTS.md)，实施沿用 [AC-001–AC-058 TODO](R11_ACTOR_CRITIC_CODE_ALIGNMENT_TODO.md)，本轮未修改其任务依赖或实现。

用户已确认保留成功与失败完整经历，优胜示范另筛，失败可参与长期后果学习。主候选为 [V-trace 序列回放 Actor–Critic](TRAINING_ALGORITHM_R0.md)；候选、接口实现和完整训练资格是不同状态。

## 2. 新链已推进到哪些接口

本轮核对 main 历史、新增文件、Supervisor 与数量映射代码，以及快照对应 Actions 状态；没有重新运行每项 AC 专项 gate 或独立下载其 artifact。

| 已观察的实现 | 证据入口 | 尚不能据此断言 |
|---|---|---|
| AC-001–AC-004 对应科学版本、哈希、新旧路由和 legacy sentinel | [science contract](../cb16_local_opt/actor_critic_contract_r0.py)、[router](../cb16_local_opt/actor_critic_runtime_router_r0.py)、[sentinel](../tests/test_actor_critic_legacy_freeze_r0.py) | 新链已有生产级完整调度 |
| AC-005–AC-009 对应目标动作、转换、记录和行为身份 | [action](../cb16_local_opt/action_contract_r0.py)、[execution record](../cb16_local_opt/execution_record_r0.py) | 已有随机 Actor 或真实行为概率采集 |
| AC-010–AC-012 对应持仓调整许可与显式先平后开反转契约 | [Supervisor](../cb16_local_opt/actor_critic_supervisor_r0.py) | 已经成交、反转两腿已完整计账 |
| AC-013 对应 risk 到合法名义额度比例及目标数量映射 | [target exposure](../cb16_local_opt/target_exposure_r0.py)、[commit](https://github.com/GY-Bai/CB16-R10/commit/2da90a0b4577ad7d950641fe0a1bcd81f08d58e2) | 目标数量已经被 Physics 执行 |
| 部分原分支通用组件已整合 | [consolidation commit](https://github.com/GY-Bai/CB16-R10/commit/25d5df8337905232ca76b7d334c390d5deeb1a07) | 所有旧示范分支内容或 71 块生产均已完成 |

此快照中未找到新链计划的 Physics adapter、随机 Actor、连续 Collector 与 V-trace learner 实现文件。因此最高目标尚需真实执行、连续经验、学习更新和下一代使用的连接证据。不能用组件数量或 commit 标题证明这些连接已完成。

旧 `risk_supervisor_r1.py` 与冻结 kernel 的持仓 FORCED_NOOP 仍属于旧链。新链已实现持仓调整许可和反转契约；**不再把旧限制概括为整个项目目前无法表达这些动作**。

### 提交前增量：AC-014 与最高理念的剩余差异

随后 main [`7c412ee`](https://github.com/GY-Bai/CB16-R10/commit/7c412ee264c463e8f9f28c707c02eccfa4ab055c) 新增 [目标仓位执行适配器](https://github.com/GY-Bai/CB16-R10/blob/7c412ee264c463e8f9f28c707c02eccfa4ab055c/cb16_local_opt/actor_critic_physics_adapter_r0.py) 与对应测试；本轮读取了适配器代码。因此上述“尚无 adapter”只适用于 AC-013 快照，不能作为最新 main 的缺口继续转述。

该函数实现目标数量到开仓、增减仓、退出和先平后开反转的仓位操作，并记录实际 legs；函数明确不推进市场 bar。随机 Actor、连续 Collector 与 learner 的完整连接仍需后续证据。本轮未独立复跑 AC-014 专项测试或下载产物。

**理念审阅的具体未闭合项**：模块说明明确将 intrabar SL/TP 与 max-hold 留给旧 `step_account`；开仓继续调用 `_entry_risk_prices`，增仓许可仍检查止损与清算价关系。保留这些行为可以构成版本明确的受限环境实验，但不能宣称已消除人为退出策略、完全实现 P-05。

sol 应在后续执行设计中列出这些规则的来源、策略影响和新链适用范围；沿已有任务记录差异，不改写旧证据。仅仅保持历史文件字节不变，并不要求新链永远继承其交易偏好。具体是否复用同一内核或增加版本化模式属于工程选择，本页不指定代码方案。

另有需对账验证的审阅点：`_partial_reduce` 将 cash 夹到非负。是否可能在该路径抹去经济损失，取决于可达状态及完整账本定义；本轮未运行反例，不先判为已证 bug。实现者应给出守恒证据，不能让数值保护替代真实损失记录。

## 3. 已读取的自动检查

- 算法文档提交 [875ab16](https://github.com/GY-Bai/CB16-R10/commit/875ab16f92c6504a2bdd5fcc12d5ea6172464990)：[repo-guard 34674613980](https://github.com/GY-Bai/CB16-R10/actions/runs/34674613980) 与 [Main Smoke 34674614161](https://github.com/GY-Bai/CB16-R10/actions/runs/34674614161) 均 success；当时 10 个发布文件 blob 与本地内容相符。
- AC-013 快照：[repo-guard 34686788794](https://github.com/GY-Bai/CB16-R10/actions/runs/34686788794) 与 [Main Smoke 34686788897](https://github.com/GY-Bai/CB16-R10/actions/runs/34686788897) 的状态以本页更新时核对值为准，见下行。

AC-013 核对结果：repo-guard 与 Main Smoke 均 completed / success。CI success 不是 Actor–Critic 的科学资格或盈利证明；专项 gate 的独立产物复核不在本轮文档任务内。

## 4. 下一次理念审阅关注什么

沿既有实施计划推进时，优先看三个可观察问题：

1. 许可目标是否真正变成仓位与账本变化，包括减仓、退出、反转失败和费用。
2. 该账户后果是否进入下一次政策决策，并跨切片、恢复、模型换代接续。
3. 普通及失败经历是否进入合规后果学习，更新后的政策是否被实际使用；受控任务与经济比较分别给结论。

其中包括 risk 比例与真实数量的区别、保证金来源、终态负净值以及暂停恢复语义，详见 [组件要求](COMPONENT_REQUIREMENTS.md)。这些是审阅方向，不是本轮已发现 bug 的定论。

精确经济时域、比较人群/权重、基准组合和未来模拟账户晋升规则仍见 [OPEN_QUESTIONS](OPEN_QUESTIONS.md)。常规任务拆解不必因此停下；依赖该科学选择的正式比较须先形成具体方案。不得重复请求失败学习等已确认原则的许可。

## 5. 历史记录与维护

最初的示范分支、manifest、第一块生产分片及当时运行核对保留在 [首次文档阶段历史快照](HISTORICAL_STATE_20260912_INITIAL.md)。其中“未合并”“下一步”是历史状态，不代表当前分支仍存在，也不覆盖后续 consolidation。历史 FAIL、authority 和产物保持原身份。

后续实现 agent 更新本页时写明：精确 SHA、所读代码/receipt/run、证据类型、实际接通处、剩余缺口。只核对代码时不要写成实验通过；只看到 receipt 哈希时不要写成独立重算产物。当前 FINAL 封存、fresh download 与实验权限不因文档更新自动变化。
