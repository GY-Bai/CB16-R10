# CB16 agent 工作约定

本文件适用于整个仓库。主要交流语言为中文，代码标识和版本化协议名称保持原样。

## 接手顺序

1. 阅读 `docs/CURRENT_STATE.md`。
2. 阅读 `docs/CC_INTEGRATION_HANDOFF.md`。
3. 读取机器可读 authority：
   - `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
   - `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`
4. 再按当前任务需要阅读 `docs/VISION.md`、`PRINCIPLE_ALIGNMENT.md`、`COMPONENT_REQUIREMENTS.md`、`TRAINING_ALGORITHM_R0.md`、`EVALUATION_PRINCIPLES.md`、`docs/BRAIN_CAPACITY_ROADMAP_3700X_1060.md` 以及历史 AC/BC/Stage-4/CC thread 文档。

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

## 当前唯一 owner-open 决策

当 Buy-and-Hold 与 FLAT component outcomes 冲突时，没有授权统一 master winner / precedence。

在 owner 明确之前：分别报告 component outcomes；promotion/master winner 层 fail closed；不得自行选更有利 baseline。见 `docs/OPEN_QUESTIONS.md`。

## 后续执行

真实 historical science、容量实验或下一阶段训练从 integration receipt 指定的 canonical CC runtime 继续，不重新打开四线程 implementation，不恢复 legacy fallback。

未来 run 的 horizon、cohort、budget、replay mixture、sequence length、model size 等属于版本化实验参数：在相关 run 前预注册并冻结，不需要反复把常规工程选择提交 owner。

`docs/BRAIN_CAPACITY_ROADMAP_3700X_1060.md` 只提供模型容量探索范围，不授权自动扩容、训练或改变 science contract。

如果发现跨组件不兼容：先定位 owning authority 和 W contract。若修复改变已冻结科学含义，开新 science version / 新任务，不得在 integration 层静默重解释。

历史 AC/BC/Stage-4/Teacher/demonstration/CC-thread receipts 与 scientific verdict 继续保留原身份，作为 provenance/history 使用。
