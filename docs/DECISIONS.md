# 设计决策与修订关系

本记录用于防止不同 Agent 从旧聊天、旧 TODO 或历史实验推导出互相冲突的目标。日期：2026-09-12。

当前机器可读 authority：

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

当前 CC integration 已通过资格并由 PR #99 合入 `main`；历史 AC/BC/Stage-4/Teacher/demonstration authority 与 verdict 继续保留原身份，不追溯改写。

## D-01 单资产账户 Trader — 已确认

人决定资产、资金配置和宏观方向；模型负责该账户的交易与风险取舍。模型根据市场、账户与必要历史作动作，执行层掌握真实规格、资金、费用、保证金与合法数量。

## D-02 经验来自闭环与反复练习 — 已确认

历史市场是可重复环境。完整经验来自 Market + Account + nominal action + permission/execution + subsequent account consequence。可以反复学习已有经验，新政策也应产生新的合法账户轨迹。

重复市场不等于增加独立市场未来；示范拟合不等于自主连续 rollout。

## D-03 长期账户，计算边界不是生命边界 — 已确认并由 CC qualification 接线

Chunk、暂停、进程恢复、learner update、checkpoint generation switch 不得自动重置逻辑账户。真实终止、目标时域结束、数据耗尽和计算边界分别记录。

CC integrated closed-loop 已证明同一逻辑账户跨 generation switch 保持连续，child policy 实际继续行动。

## D-04 高风险可因更高算术期望收益获胜 — 已确认

完整计入失败后，如果高爆仓概率策略的长期算术期望收益更高，用户允许其获胜。不得静默替换为最少爆仓、最大 Sharpe、最大 log-growth 或回撤最小化目标。

## D-05 Buy-and-Hold 与 FLAT baseline — 部分明确，master precedence 仍未决

两种 baseline 均正式保留并分别计算。当前唯一 owner-open 科学决策是：当 Buy-and-Hold 与 FLAT component outcomes 冲突时，哪个拥有 master precedence，或采用什么显式组合规则。

在 owner 决定前，系统不得自行制造 master winner。

## D-06 失败经验 — 原则已关闭

成功、普通、失败和终态事实均完整保留。优胜示范可另外筛选，失败可以参与长期后果学习；不能把失败轨迹每一步统一标成错误，也不能用 survivor-only 结果估计整体策略期望。

## D-07 分代更新与历史知识 — 已确认并已接线

运行策略在注册 collection unit 内保持固定；learner 在副本上更新并产生下一代 checkpoint。换代不重置账户。旧兼容经验可复习，新经验可校准当前行为；不通过时间到期或手工周期/共振开关删除历史知识。

## D-08 文档与 authority 分层 — 已确认

用户目标、版本化 authority、代码、run receipt、文档建议回答不同问题。新目标不追溯改写历史 verdict；旧 authority 也不能否认后续已明确的新目标。语义改变必须版本化。

## D-09 V-trace Actor–Critic 路线 — 从设计候选推进为 CC synthetic-qualified implementation

最初 `TRAINING_ALGORITHM_R0.md` 中的序列回放 Actor–Critic + V-trace 是设计候选。CC 已实现 stochastic Actor、true joint `log_mu`、Critic、V-trace learner、exactly-once learner transaction、checkpoint/generation switch，并在 synthetic known-answer closed loop 中通过 qualification。

这不等于已获得真实市场 ECONOMIC 或 TRANSFER 证据。

## D-10 执行语义迁移 — 已由 CC canonical runtime 落地

旧冻结链中的持仓 FORCED_NOOP、固定退出规则和 H72/log-utility 语义继续作为历史协议存在，但不再概括 canonical CC Trader。

Canonical CC 将 nominal action、permission、target sizing、execution 与 account consequence 分离；允许合法减仓/平仓/反转；signed account economics 保留负净值/负债；reject/NOOP 不冻结环境时钟；固定 SL/TP/max-hold/cooldown 不作为隐藏自主策略默认规则。

## D-11 CC 四线程并行实施与最终 join — 已完成

用户明确将后续工作重组为四个独立线程：

- A：Runtime / Account / Continuous Interaction
- B：Policy / Critic / Learner / Retention
- C：Experience / Replay / Economic Evaluation
- D：Performance Hard Cutover / High-Throughput Spine

四线程共同 implementation base 为 `89d62bf966f476e598f0e2f5c5e8e03c15a8db51`，随后由单一 integration branch 完成 W-01..W-05 binding、closed-loop wiring、equivalence、performance selection 和 canonical handoff。

PR #99 已合入 `main`；merge commit 为 `daa889d758ce80c7d1ce73ea37110937e1b146c0`。

## D-12 性能优先 + Hard Cutover — 已确认并完成

用户明确选择：`PERFORMANCE FIRST AMONG SEMANTICALLY QUALIFIED IMPLEMENTATIONS`，不做旧性能 runtime 的兼容层或 fallback。

最终选择：

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

职责仍严格分离：Thread A 保持 runtime/account/execution science authority；Thread D 只负责 performance implementation。Canonical CC 不依赖或回退到：

- `gpu_inference_broker.py`
- `multiprocess_trajectory_farm.py`
- `vectorized_physics.py`

新 fast path 失败时 fail closed。

首次冻结资格 benchmark：16 accounts × 64 market steps，reference/fast 各 7 次交替运行；receipt 记录 reference median 7.535366 transitions/s、fast median 638.765768 transitions/s、median speedup 84.769×。

后续 PR head `5d236e9d2368a79e830bec1971b949de88c032df` 又完整重复 qualification，仍选择同一 topology；重复 run 只作为确认，不覆盖 receipt 中首个冻结 benchmark 数值。

## D-13 CC 最终 evidence ceiling — 已冻结

当前最强证据：

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

它证明 integrated synthetic closed loop、known-answer、recovery、exactly-once update、reference-vs-fast semantic equivalence、hard cutover 和 Shanxi measured performance。

它**不证明**真实历史市场 ECONOMIC edge，也不证明 TRANSFER。FINAL 仍封存，fresh data 未使用。

## 运行语义迁移登记

| 历史材料/字段 | 当前地位 |
|---|---|
| `SEMANTIC_FREEZE_V1` 的 72h net log-equity growth | 历史冻结语义；不等于当前算术期望收益方向 |
| validation Teacher loss 晋升 | 历史协议；不能作为 canonical CC economic promotion master rule |
| 旧 Teacher 对非正净值的有限排除 | 不得扩展为整体策略评价删除失败 |
| 旧固定 SL/TP/max-hold/cooldown | 历史/受限策略行为；不属于 canonical autonomous CC 默认执行策略 |
| AC/BC TODO | 设计与实施 provenance/history；不是当前 execution authority |
| CC 四线程 TODO | 已完成的实施 provenance；当前入口转为 integration receipt/handoff |
| CC integration receipt/spec | 当前 canonical CC authority |

任何后续科学迁移仍需记录旧版本、替代版本、验证方式和证据上限；不得只因分支更新就自动抬高 scientific claim。
