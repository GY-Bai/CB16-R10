# CB16 文档入口

本目录同时保存当前 authority、实施说明与历史阶段资料。**CC R11 integration 已完成并合入 `main`；当前 canonical 入口是 integration receipt + handoff，不再是四线程 TODO。**

PR #99 merge commit：

`daa889d758ce80c7d1ce73ea37110937e1b146c0`

## 当前阅读路线

| 问题 | 当前入口 |
|---|---|
| 当前精确状态与下一断点 | **[CURRENT_STATE](CURRENT_STATE.md)** |
| 当前 S0/S1 可执行总控 TODO | **[R11_POST_CC_S0_S1_TODO](R11_POST_CC_S0_S1_TODO.md)** |
| S0 契约/经济语义迁移任务包 | **[S0_CONTRACT_MIGRATION_TODO](post_cc/S0_CONTRACT_MIGRATION_TODO.md)** |
| S1 端到端可学习性任务包 | **[S1_END_TO_END_LEARNABILITY_TODO](post_cc/S1_END_TO_END_LEARNABILITY_TODO.md)** |
| CC integration 最终交接 | **[CC_INTEGRATION_HANDOFF](CC_INTEGRATION_HANDOFF.md)** |
| 最终机器可读 spec / receipt | `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json` / `CB16_R11_CC_INTEGRATION_RECEIPT_V1.json` |
| 项目最高目标 | [VISION](VISION.md) |
| 当前已确认设计决策 | [DECISIONS](DECISIONS.md) |
| 双基准、模型排序与沙盒冠军晋升 | [ECONOMIC_ORDERING_AND_PROMOTION](ECONOMIC_ORDERING_AND_PROMOTION.md) |
| 后 CC scientific program 总体路线 | **[POST_CC_SCIENTIFIC_PROGRAM_R0](POST_CC_SCIENTIFIC_PROGRAM_R0.md)** |
| 已关闭的问题与将来问题提交边界 | [OPEN_QUESTIONS](OPEN_QUESTIONS.md) |
| 当前组件地图 | [ARCHITECTURE_MAP](ARCHITECTURE_MAP.md) |
| Agent 执行规则 | [根目录 AGENTS](../AGENTS.md) |
| 理念约束 | [PRINCIPLE_ALIGNMENT](PRINCIPLE_ALIGNMENT.md) / [COMPONENT_REQUIREMENTS](COMPONENT_REQUIREMENTS.md) |
| 学习与评价原则 | [LEARNING_CONTRACT](LEARNING_CONTRACT.md) / [EVALUATION_PRINCIPLES](EVALUATION_PRINCIPLES.md) |
| Actor–Critic 算法设计来源 | [TRAINING_ALGORITHM_R0](TRAINING_ALGORITHM_R0.md) / [TRAINING_QUALIFICATION_R0](TRAINING_QUALIFICATION_R0.md) |
| 当前 Shanxi 性能策略与已测结果 | [PERFORMANCE_STRATEGY_3700X_1060](PERFORMANCE_STRATEGY_3700X_1060.md) |
| Central Brain 容量探索建议 | [BRAIN_CAPACITY_ROADMAP_3700X_1060](BRAIN_CAPACITY_ROADMAP_3700X_1060.md) |
| CC 原始并行决策 | [CC_PARALLEL_EXECUTION_DECISION](CC_PARALLEL_EXECUTION_DECISION.md) — 历史实施决策 |
| CC 四线程原始总控 | [R11_CC_PARALLEL_HARD_CUTOVER_TODO](R11_CC_PARALLEL_HARD_CUTOVER_TODO.md) — 已完成的历史执行计划 |
| Thread A 历史任务包 | [Runtime / Account](cc/CC_THREAD_A_RUNTIME_ACCOUNT.md) |
| Thread B 历史任务包 | [Policy / Critic / Learner](cc/CC_THREAD_B_POLICY_LEARNING.md) |
| Thread C 历史任务包 | [Experience / Replay / Economics](cc/CC_THREAD_C_EXPERIENCE_ECONOMICS.md) |
| Thread D 历史任务包 | [Performance Hard Cutover](cc/CC_THREAD_D_PERFORMANCE_HARD_CUTOVER.md) |
| 原 AC/BC 设计历史 | [AC TODO](R11_ACTOR_CRITIC_CODE_ALIGNMENT_TODO.md) / [BC Round 2 TODO](R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md) |

## 当前 CC 结论

四个 CC threads 从冻结 implementation base `89d62bf966f476e598f0e2f5c5e8e03c15a8db51` 独立实现后，由唯一 integration join 完成 W-01..W-05 binding、closed-loop wiring、equivalence、performance selection 与 hard cutover。

Receipt-qualified runtime code head：

`fc7adaae66da5b7735a3ab6aee7a8c7bd3721ed2`

Final PR handoff head：

`5d236e9d2368a79e830bec1971b949de88c032df`

Final head 又完整运行 workflow `34712758444` 并 PASS，确认 authority/docs 更新没有破坏 runtime。

Canonical topology：

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

Thread A 保持账户/runtime/execution science authority；Thread D 只拥有 performance implementation authority。Canonical CC runtime 不兼容、不 import、也不 fallback 到旧 broker/farm/vectorized performance runtime。

Receipt 冻结的 Shanxi benchmark：

- reference median 7.535366 transitions/s；
- fast median 638.765768 transitions/s；
- median speedup 84.769x。

## 证据边界

当前最强证据：

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

它不是 ECONOMIC 或 TRANSFER evidence。FINAL 未开启，fresh market data 未使用；当前没有真实历史市场 edge 或未来盈利性结论。

## 后 CC 设计决定与当前执行入口

双基准不设 master precedence；同合同下完整算术收益决定候选排序。有充分预登记证据的相对改善可晋升沙盒学习冠军；不额外要求同时击败 B&H 和 FLAT。旧 C-20 代码/receipt 保留历史身份，当前 S0 负责显式 successor migration，不改写历史 receipt。

当前直接执行顺序是：

`S0 contract migration -> S0 receipt -> freeze S1 base -> S1 end-to-end learnability -> S1 receipt`

具体 task numbering、hard gates、文件建议和机器可读交付均已写入 `R11_POST_CC_S0_S1_TODO.md` 及两个 `docs/post_cc/` 任务包。Sol 不需要重新拆一套 S0/S1。

S1 通过之前，不进入 S2/S3 historical learning；S4 capacity 与 S5 economic qualification 继续按 `POST_CC_SCIENTIFIC_PROGRAM_R0.md` 的条件推进。

## 历史资料地位

AC、BC、Stage-4、CC 四线程 TODO、旧 Teacher/demonstration、旧 performance runtime 文档和代码继续保留为 provenance/history。

新的 CC authority 不追溯改写历史 scientific verdict，也不把 workflow PASS、性能提升或 synthetic known-answer PASS 写成市场 ECONOMIC evidence。

容量规划、未来 historical science、模型扩容和后续训练均须从当前 CC canonical authority 出发，并在各自任务中重新冻结实验参数与证据范围。
