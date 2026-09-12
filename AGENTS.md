# CB16 agent 工作约定

本文件适用于整个仓库。主要交流语言为中文，代码标识和版本化协议名称保持原样。

## 接手顺序

1. 阅读 `docs/CURRENT_STATE.md`。
2. 阅读 `docs/CC_INTEGRATION_HANDOFF.md`。
3. 读取机器可读 authority：
   - `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
   - `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`
4. 再按任务需要阅读 `docs/VISION.md`、`PRINCIPLE_ALIGNMENT.md`、`COMPONENT_REQUIREMENTS.md`、`TRAINING_ALGORITHM_R0.md`、`EVALUATION_PRINCIPLES.md` 和历史 AC/BC/Stage-4/CC thread 文档。

不要从移动的 A/B/C/D branch 名恢复当前 authority。四线程 frozen heads、qualified implementation SHAs 和 receipt hashes 已由 integration receipt 冻结。

## 当前 canonical CC

经过完整资格测试的 integration code head：

`fc7adaae66da5b7735a3ab6aee7a8c7bd3721ed2`

四线程原始 implementation base：

`89d62bf966f476e598f0e2f5c5e8e03c15a8db51`

当前职责：

- Thread A：runtime/account/execution correctness oracle。
- Thread B：stochastic policy、true behavior likelihood、Critic、V-trace learner、checkpoint/retention。
- Thread C：immutable experience、replay、arithmetic-economic evaluation contracts。
- Thread D：performance implementation；不得重新定义科学语义。
- Integration：W-01..W-05 binding、closed-loop wiring、equivalence、performance selection、canonical handoff。

当前 W contracts：W-01 `CCPolicyDecisionV1`，W-02 `CCEnvironmentTransitionV1`，W-03 `CCExperienceSequenceV1`，W-04 `CCLearningUpdateV1`，W-05 `CCEconomicResultV1`。Science identity：`CB16_R11_CC_SCIENCE_SEMANTIC_V1`。

## Canonical runtime 与 hard cutover

Selected topology：

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

它的含义是：Thread A 的账户/runtime/execution 语义保持 authority，Thread D 提供 same-account scheduler、bounded fact transport 与 durable chunk writer 的性能实现。

Canonical CC runtime 不得依赖或静默回退到历史 performance runtime：

- `gpu_inference_broker.py`
- `multiprocess_trajectory_farm.py`
- `vectorized_physics.py`

这些文件可保留为历史/reference 材料。新的 fast path 失败时按 integration authority fail closed。

## 必须保持的含义

- account continuity；chunk、pause、checkpoint generation switch 和经济终止不得混同。
- signed account economics；negative equity 与 liability 不得被数值 clamp 抹除。
- nominal action、permission、target sizing、execution 必须分离。
- stochastic behavior 必须保留真实 `log_mu`、policy/RNG/generation provenance。
- 没有 policy decision 的 mechanical environment advance 可以保留为 raw fact，但不得伪造 action 或 `log_mu` 进入 V-trace replay。
- success、ordinary、failure、terminal facts 都必须可保留。
- frozen market organ 与 trainable Brain ownership 必须明确。
- economic orientation 使用 arithmetic expected return；survivor-only 统计不可替代总体期望。
- FINAL firewall 与 fresh-data firewall 不因实现或性能工作自动解除。
- 不允许把固定 SL/TP/max-hold/cooldown 偷渡成隐藏策略 policy。
- same logical account 的 temporal order 必须串行化。
- 新数据可校准当前行为，但不要加入复杂手工周期/共振/规则激活系统去覆盖模型隐式学习。

## 资格与证据纪律

最终 integration qualification：workflow run `34710702090`，job `103598868269`，artifact `10303497599`。

Joined tests 为 `155 passed, 1 deselected`；唯一 deselected 是授权 join 后不再适用的 Thread-A independent-branch sibling-isolation assertion，不是科学行为测试。

Closed-loop、provenance、hostile/recovery、exactly-once learner update、reference-fast equivalence、legacy-import firewall、FINAL/fresh-data firewall 均 PASS。

预注册 Shanxi benchmark 的 median throughput：reference 7.535366 transitions/s，fast 638.765768 transitions/s，fast/reference median ratio 84.769x。

当前最强 evidence：

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

不要把 workflow PASS、性能 PASS、loss 下降或 checkpoint 变化写成 ECONOMIC/TRANSFER evidence。FINAL 没有打开，fresh market data 没有使用。

## 后续执行

真实 historical science 或下一阶段训练应从 integration receipt 指定的 canonical CC runtime 继续，而不是重新打开四线程实现或恢复 legacy fallback。

如果后续发现跨组件不兼容：先定位到 owning authority 和 W contract。若修复会改变已经冻结的科学含义，必须开新版本/新任务，不得在集成层静默重解释。

Thread C 仍保留一个 owner-open 决策：当 buy-and-hold 与 FLAT component outcomes 冲突时，没有授权统一 master winner/precedence。Agent 不得自行发明该规则。

历史 AC/BC/Stage-4/Teacher/demonstration receipts 与 scientific verdict 继续保留原身份，作为 provenance/history 使用。
