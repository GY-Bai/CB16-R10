# 面向最高理念的组件要求

日期：2026-09-12。性质：由 [理念对齐规则](PRINCIPLE_ALIGNMENT.md) 推导的行为要求与验收方向；不规定文件数量、服务数量或固定网络。

**状态更新：** CC R11 integration 已完成并合入 `main`。本页现在用于审阅 canonical CC 与未来 successor 是否继续满足理念，不再作为 AC/BC/CC TODO 的待实施导航。当前实现事实见 [CURRENT_STATE](CURRENT_STATE.md)、[ARCHITECTURE_MAP](ARCHITECTURE_MAP.md) 和 integration spec/receipt。

## 1. 组件边界和可观察后果

| 组件 | 必须承担的责任 | 交给下游的内容 | 关键验收方向 | 理念 |
|---|---|---|---|---|
| 市场环境与感知器官 | 用因果市场前缀形成冻结表示；当前动作不改变市场路径 | 市场观察、可见时点和来源/版本 | 改变未来不能改变过去观察；同一合法输入可复现 | P-01/02/08 |
| 账户真值与观察投影 | 完整账本接续；把决策所需的可比账户信息交给策略 | 完整恢复状态及独立策略观察 | 区分账本与观察；观察不足时报告状态混淆，不以增大训练量掩盖 | P-01/02/04 |
| Actor / Central Brain | 联合市场、账户与必要历史选择名义目标动作 | 方向、风险请求、策略身份、真实采样概率与 RNG 来源 | 同一 policy likelihood 可重算；不把成交值冒充 nominal action 计算概率 | P-02/05/08 |
| Supervisor | 按明确外部合法性/资源约束许可动作，不代替策略预测收益 | 原请求、ACCEPT/CLAMP/REJECT、许可目标、原因和来源 | 机械合法减仓/平仓可达；被拒绝不等于目标已成交 | P-05 |
| 仓位映射与 Physics | 将许可目标变为实际交易量并成交、计费、清算 | 成交差量、完整账本、下一可见账户、终止事实 | 对账守恒；反转两腿和失败有真实后果；不以目标记录替代成交 | P-02/04/05 |
| 连续 Collector | 在实际 policy decision 时点重新调用策略并接续账户 | 有序、有 provenance 的轨迹、切片与恢复边界 | 不是首步-only；chunk/pause/generation 不清空账户 | P-02/03/04/09 |
| 经验存储与回放 | 保留成功失败事实，按兼容性提供训练样本 | raw facts、sequence、来源、behavior identity、采样与使用记录 | 终态不丢失；重试不重复事务；不伪造 `log_mu` | P-03/07/09 |
| Reward / Critic / 信用分配 | 从真实经济后果构造目标，估计后续价值 | 奖励、边界、value target/value estimate 及版本 | 不抹掉先赚后亏；不把失败每一步全标错；估计与已发生收益分开 | P-04/06/07 |
| Learner / checkpoint | 更新授权的可训练参数并保存足够恢复状态 | Actor/Critic、optimizer/RNG/update lineage/checkpoint | 冻结器官保持冻结；运行 policy 不后台突变；恢复不重复 gradient transaction | P-08/09 |
| 评价与晋升 | 明确比较单 checkpoint 还是 generation chain，完整计入经济结果 | cohort、资本、common horizon、baseline、全部结果与结论范围 | 高风险高期望可赢；不用 survivor/loss 冒充经济进步 | P-06/07/10 |
| 调度与状态管理 | 提供持久性、版本连接、恢复与高吞吐 transport | 可追溯 account/policy/task identity 和 boundary | 性能优化不得改变 scientific semantics；same account 不乱序 | P-04/09/10 |

组件可以共享进程或采用高吞吐实现；职责边界比服务数量更重要。

## 2. 动作到真实仓位：固定含义

Canonical CC 必须保持：

- FLAT 目标为零仓位；是否已经空仓、是否被许可、是否实际成交分别记录。
- `requested_target_risk` 是目标暴露请求，不是 confidence、亏损概率、波动率或预计最大回撤。
- 若 risk 是当前合法名义额度比例，相同 risk 在 equity/price/legal envelope 变化后可以映射到不同 target quantity；不能只因 risk 数字相同就提前 NOOP。
- Target position 与 order delta 不同；当前 quantity、price、unit、margin、minimum quantity、reduce-only 例外有明确含义。
- 反方向目标先处理 close，再根据中间 authoritative state 重算 opposite-open 可行性；第二腿失败留下真实中间结果。
- REJECT / NOOP 不会冻结已有持仓的市场 PnL、funding 或机械 liquidation；environment clock 继续推进。
- 完整账本保留清算和可能的负净值/负债；不能用非负 sizing input 反向抹掉经济事实。
- 固定 SL/TP/max-hold/cooldown 不属于 canonical autonomous CC 默认策略选择。

这些语义已经进入 CC synthetic qualification；未来优化仍必须保持。

## 3. 四个时钟与账户身份

| 概念 | 含义 | 不能替代的对象 |
|---|---|---|
| 环境时钟 | 价格、成交、费用、funding、清算推进 | 不等同于每次都调用 Actor |
| 政策决策时钟 | 模型何时选择新的目标动作 | 不等同于一根 K 线或一个 gradient step |
| 学习/计算跨度 | sequence、反传、存盘或 job 的长度 | 不定义账户寿命 |
| 经济评价时域 | 本次比较追踪到哪个共同终点 | 不自动强制平仓，不证明无限期表现 |

账户身份横跨这些时钟。平仓通常是状态变化，不必是账户死亡；数据耗尽、暂停、模型换代、进程故障、objective horizon 与经济责任终止分别记录。

比较中允许新建账户，但新资本必须显式登记；不能多次重开后只统计最后幸存一次。

## 4. 事实保留与训练用途不是同一个池子

| 对象 | 保存/使用目的 | 必须避免 |
|---|---|---|
| 完整 raw facts | 保留普通、成功、失败和终态事实，支持追溯/恢复 | 赢家筛选提前删除失败；压缩改变经济后果 |
| Replay-admissible view | 选择版本、状态、behavior probability 足够的样本 | 声称所有历史经验都可用同一 off-policy loss |
| Demonstration-selected view | 单独研究优秀行为或辅助模仿 | 用幸存者视图估计全策略期望；把幸运轨迹当唯一正确答案 |
| Evaluation cohort | 固定比较分母、资本和时间，记录所有纳入结果 | 借训练采样比例重新加权比赛 |

“完整保留”要求事实可追溯，不要求无限热内存或无限等权训练。存储分层、压缩、读取顺序和 replay sampling 可以优化，但不能静默销毁失败事实或改 sampling meaning。

未来 outcome 可以成为训练反馈，不能成为当时 online input。长远后果能否传回早期行为需要 known-answer/真实科学证据，而不是只凭连续文件存在。

## 5. 当前 CC 对应关系

历史 AC/BC/CC TODO 继续保留用于 provenance；当前组件职责以 integrated CC ownership 为准：

| 需求范围 | 当前 authority owner | 当前 evidence |
|---|---|---|
| 账户/runtime/execution | Thread A semantics + integration binding | synthetic component + integrated closed-loop/hostile qualification |
| stochastic Actor / true `log_mu` / RNG | Thread B | known-answer + integrated provenance qualification |
| Critic / V-trace / learner / checkpoint | Thread B | synthetic known-answer + exactly-once/recovery qualification |
| immutable experience / replay | Thread C | component + integrated sequence/replay qualification |
| economic contracts / B&H / FLAT components | Thread C | contract/component; **not real-market ECONOMIC evidence** |
| high-throughput scheduling/writer | Thread D constrained by A semantics | Shanxi measured performance + reference-fast equivalence |
| W-01..W-05 join / canonical route | Integration | `QUALIFIED_HARD_CUTOVER_PASS` |

Canonical topology：

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

当前最强证据：

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

它不升级成 ECONOMIC 或 TRANSFER。

## 6. 当前仍需 fail closed 的 owner boundary

Buy-and-Hold 与 FLAT component outcomes 冲突时，没有授权 master precedence / master winner。系统可以分别报告 component results，但 promotion/master ranking 不能自行选择更有利 baseline。

除此之外，未来 horizon、cohort、budget、replay mixture、sequence length、model capacity 等通常属于具体科学 run 的预注册参数，不再作为本页的永久 owner-open 原则。

未来 successor 若改变上述组件语义，必须新 science version + 新 qualification；不能借性能、模型扩容或文档更新在现有 CC identity 下静默改义。
