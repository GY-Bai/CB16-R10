# CB16 agent 工作约定

本文件适用于整个仓库。主要交流语言为中文，代码标识和版本化协议名称保持原样。

## 用户指定的角色与审核链

必须先明确本次角色并读取 [三层对齐与审核协议](docs/ROLES_AND_REVIEW_PROTOCOL.md)：**Astra 用文档向用户对齐；Sol 用 TODO 向文档对齐；DS Flash 用代码向 TODO 对齐。用户审核 Astra 文档，Astra 审核 Sol TODO，Sol 审核 DS Flash 代码产物。** Astra 本角色不负责业务代码实现；代码执行者不能自行代签上游审核。

Sol 必须在 TODO 中标出公式转代码、复杂比较/直接条件分支、mask/空值/边界和 verdict 逻辑，合并 main 前独立审查实际实现与反例。DS Flash 的自检和绿色 CI 不替代 Sol 审核。此要求源于用户项目经验，不是模型能力排名。

运行变更合并 main 前须经 **GitHub Actions → Shanxi Docker** 的相关冒烟/专项测试，复用共享 preflight、`[self-hosted, shanxi-docker-r11]` 与 verified canonical Python。临时沙盒用于读写与静态准备，不在那里运行业务冒烟/训练或推测 Shanxi 的结果。仅 repo-guard 通过不是 runtime PASS；必须核对实际 checkout SHA、job/runner、测试执行及产物。Main Smoke 当前手动触发且只覆盖旧 Stage-4 测试，S1 新代码须有相应覆盖。纯说明性文档按协议做静态检查，不声称已跑 runtime。

## 接手顺序

1. 阅读 `docs/CURRENT_STATE.md`。
2. 阅读当前可执行总控 `docs/R11_POST_CC_S0_S1_TODO.md`。
3. 若任务是 S0，继续读 `docs/post_cc/S0_CONTRACT_MIGRATION_TODO.md`；若任务是 S1，必须先读取已经 PASS 的 S0 receipt，再读 `docs/post_cc/S1_END_TO_END_LEARNABILITY_TODO.md`。
4. 阅读 `docs/CC_INTEGRATION_HANDOFF.md` 与机器可读 CC authority：
   - `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
   - `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`
5. 再按任务需要阅读 `docs/ECONOMIC_ORDERING_AND_PROMOTION.md`、`docs/POST_CC_SCIENTIFIC_PROGRAM_R0.md`、`docs/VISION.md`、`docs/PRINCIPLE_ALIGNMENT.md`、`docs/COMPONENT_REQUIREMENTS.md`、`docs/TRAINING_ALGORITHM_R0.md`、`docs/EVALUATION_PRINCIPLES.md`、`docs/BRAIN_CAPACITY_ROADMAP_3700X_1060.md` 以及历史 AC/BC/Stage-4/CC thread 文档。

**不要重新拆一套 S0/S1。** 当前 task numbering、hard gates、建议 successor surfaces 和 receipts 已由 `R11_POST_CC_S0_S1_TODO.md` 及两个 `docs/post_cc/` 任务包冻结为执行入口。

撰写、修订或审阅 S 系列 TODO 时，必须读取 [TODO 撰写角色原则](docs/S_SERIES_TODO_AUTHORING_PRINCIPLES.md)；当前具体修补位置见 [S0/S1 可执行性审阅](docs/reviews/S0_S1_EXECUTABILITY_REVIEW_2026-09-13.md)。沿用现有编号，区分组件开发就绪与正式科学运行就绪；后者须具体化生成器、控制、判分、预算和种子。此要求不新增 owner 审批，不授权覆盖冻结文件，也不因后续功能待实现而停止已明确的开发。

CC R11 integration 已通过 PR #99 合入 `main`。Merge commit：

`daa889d758ce80c7d1ce73ea37110937e1b146c0`

不要再从移动的 A/B/C/D branch 名恢复当前 authority。四线程 frozen heads、qualified implementation SHAs 和 receipt hashes 已由 integration receipt 冻结。

## 当前 canonical CC

Receipt-qualified integration runtime code head：

`fc7adaae66da5b7735a3ab6aee7a8c7bd3721ed2`

Final PR handoff head：

`5d236e9d2368a79e830bec1971b949de88c032df`

Final PR head 在 authority/docs 更新后又完整执行 workflow `34712758444` 并 PASS，因此后续接手者不需要重新怀疑“最终文档 head 是否经过 runtime qualification”。

四线程原始 implementation base：

`89d62bf966f476e598f0e2f5c5e8e03c15a8db51`

职责：

- Thread A：runtime/account/execution correctness oracle。
- Thread B：stochastic policy、true behavior likelihood、Critic、V-trace learner、checkpoint/retention。
- Thread C：immutable experience、replay、arithmetic-economic evaluation contracts。
- Thread D：performance implementation；不得重新定义科学语义。
- Integration：W-01..W-05 binding、closed-loop wiring、equivalence、performance selection、canonical handoff。

W contracts：

- W-01 `CCPolicyDecisionV1`
- W-02 `CCEnvironmentTransitionV1`
- W-03 `CCExperienceSequenceV1`
- W-04 `CCLearningUpdateV1`
- W-05 `CCEconomicResultV1`

Science identity：`CB16_R11_CC_SCIENCE_SEMANTIC_V1`。

## Canonical runtime 与 hard cutover

Selected topology：

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

Thread A 保持账户/runtime/execution science authority；Thread D 提供 same-account scheduler、bounded fact transport、durable chunk writer 等性能实现。

Canonical CC runtime 不得依赖、兼容或静默回退到历史 performance runtime：

- `gpu_inference_broker.py`
- `multiprocess_trajectory_farm.py`
- `vectorized_physics.py`

这些文件可继续作为历史/reference 材料。新 fast path 失败时按 integration authority fail closed。

## 必须保持的科学含义

- Account continuity；chunk、pause、process restart、checkpoint generation switch 与 economic terminal 分开。
- Signed account economics；negative equity 与 liability 不得被数值 clamp 抹除。
- Nominal action、permission、target sizing、execution 必须分离。
- Stochastic behavior 保存真实 `log_mu` 与 policy/RNG/generation provenance。
- 没有 policy decision 的 mechanical environment advance 可以是 raw fact，但不得伪造 action 或 `log_mu` 进入 V-trace replay。
- Success、ordinary、failure、terminal facts 都必须保留事实身份。
- Frozen market organ 与 trainable Brain ownership 明确。
- Economic orientation 使用 arithmetic expected return；survivor-only、Sharpe、log-growth、drawdown 不能未经 owner 决定替换 master objective。
- FINAL firewall 与 fresh-data firewall 不因实现、容量或性能工作自动解除。
- 固定 SL/TP/max-hold/cooldown 不得偷渡成 canonical autonomous strategy policy。
- Same logical account 的 temporal order 必须串行化。
- 新经验可以校准当前行为，但不要加入复杂手工周期/共振/规则激活系统覆盖模型隐式学习。

## 资格与证据纪律

冻结 integration qualification：workflow `34710702090`，job `103598868269`，artifact `10303497599`。

Joined tests：`155 passed, 1 deselected`。唯一 deselected 是正式 join 后不再适用的 Thread-A independent-branch sibling-isolation assertion，不是科学行为测试。

Closed-loop、provenance、hostile/recovery、exactly-once learner update、reference-fast equivalence、legacy-import firewall、FINAL/fresh-data firewall 均 PASS。

Receipt 冻结 benchmark：

- reference median 7.535366 transitions/s；
- fast median 638.765768 transitions/s；
- median speedup 84.769x。

最终 PR head 的重复 qualification 只用于确认最终 handoff 没有破坏 runtime，不用于事后覆盖 receipt 的首个预注册 benchmark。

当前最强 evidence：

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

不得把 workflow PASS、性能 PASS、loss 下降或 checkpoint 变化升级为 ECONOMIC/TRANSFER evidence。FINAL 未打开，fresh market data 未使用。

## 经济排序与 S0/S1 当前任务

B&H/FLAT 是两个并列 benchmark components，**没有 master precedence，也没有 master baseline winner**。模型间排序、单模型 baseline component 结果、promotion decision 是三个独立对象。此原则已经关闭，不再提交 owner。

旧 `cc_economic_promotion_r0.py` 与旧 Thread-C/integration receipts 仍保留其历史 unresolved 身份；S0 已通过 successor contract/routing 完成迁移并获得 PASS，禁止改写 frozen receipt 或假装历史语义从未存在。

当前执行顺序：

1. `S0_CONTRACT_MIGRATION_TODO.md`：已关闭，receipt 的证据范围为 `POST_CC_CONTRACT_MIGRATION_QUALIFIED`；
2. S1 从带 S0 receipt 的交接 head `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f` 继续；它与 qualified implementation `5760061d6c274e9f8796e6608e86bb173018148f` 分别绑定，不能混为同一 checkout；
3. `S1_END_TO_END_LEARNABILITY_TODO.md`：消灭 rollout-memory learner side channel，建立 durable observation/replay materialization、joint direction+risk learner，并通过多 seed known-answer + negative controls；
4. S1 结束即停，不自动进入 S2/S3/S4/S5。

S1 的核心资格要求是：learner 的训练真值必须可以从持久化 experience + observation facts 重建；不能只从 collector 内存里的 `records` 喂训练。Canonical policy likelihood 必须使用 nominal `direction + target_risk` 与真实 persisted `log_mu`，不能拿 executed action 重构。

## 后续执行

真实 historical science、容量实验或下一阶段训练从 integration receipt 指定的 canonical CC runtime 继续，不重新打开四线程 implementation，不恢复 legacy fallback。

未来 run 的 horizon、cohort、budget、replay mixture、sequence length、model size 等属于版本化实验参数：在相关 run 前预注册并冻结，不需要反复把常规工程选择提交 owner。

`docs/BRAIN_CAPACITY_ROADMAP_3700X_1060.md` 只提供模型容量探索范围，不授权自动扩容、训练或改变 science contract。

如果发现跨组件不兼容：先定位 owning authority 和 W contract。若修复改变已冻结科学含义，开新 science version / 新任务，不得在 integration 层静默重解释。

历史 AC/BC/Stage-4/Teacher/demonstration/CC-thread receipts 与 scientific verdict 继续保留原身份，作为 provenance/history 使用。
