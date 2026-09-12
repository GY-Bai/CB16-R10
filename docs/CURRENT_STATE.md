# 当前状态与接手断点

核对日期：2026-09-12 UTC。CC R11 integration 已通过 PR #99 正式合入 `main`；integration merge commit 为：

`daa889d758ce80c7d1ce73ea37110937e1b146c0`

机器可读 canonical authority：

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

Receipt 冻结的完整 runtime qualification code head 为 `fc7adaae66da5b7735a3ab6aee7a8c7bd3721ed2`。PR 最终 handoff head `5d236e9d2368a79e830bec1971b949de88c032df` 在 authority/docs 更新后又完整重跑同一 integration workflow 并 PASS，因此不存在“runtime head 通过但最终 PR head 未复验”的残余风险。

后续纯文档维护提交不改变上述 science identity、qualified runtime 或 receipt 内容。

## 1. CC 已完成什么

CC 四线程已完成独立实现、integration join、closed-loop qualification、reference-vs-fast equivalence、hostile/recovery qualification、Shanxi 性能选择和 hard cutover。

合成资格 canary 已实际接通：

`Market + same Account -> stochastic Actor -> nominal action + true log_mu -> permission -> target sizing -> execution -> signed account consequence -> environment advance -> next Actor decision -> immutable experience -> replay -> Critic/V-trace update -> committed child checkpoint -> generation switch -> SAME LOGICAL ACCOUNT -> child policy acts`

该 canary 保留 policy/RNG/generation provenance、真实 `log_mu`、失败事实和同账户连续性，并验证 exactly-once learner commit。

它证明的是 **synthetic closed-loop / known-answer implementation qualification**，不是市场盈利证据。

## 2. 四线程 frozen authority

所有线程原始 implementation base：

`89d62bf966f476e598f0e2f5c5e8e03c15a8db51`

| Thread | Frozen head | Qualified implementation | Canonical authority |
|---|---|---|---|
| A | `d904fa67f66026fc2bc9c320fe5a888ff5f98db6` | `b1ebbc45482f2b10c49bbd9c521f831020558a04` | runtime/account/execution correctness oracle |
| B | `f06babb485daee0a2f7e978c51d6263b03fc2fea` | `87a37398d75c9fb644de31d94595db66a49fafbc` | stochastic policy, true joint `log_mu`, Critic/V-trace learner, checkpoint/retention |
| C | `626241fc1043e10326e538f93cf08d9cfac75b67` | `2a909fe0ba9a55d0540a5c93c034bae7db131cf4` | immutable experience, replay, arithmetic-economic evaluation contracts |
| D | `26742447af209d52943085e9d37996eae522b93c` | `089031a935e58100f3a0dbfec34c16fa5f272915` | performance implementation only; no new scientific semantics |

W contracts：

- W-01 `CCPolicyDecisionV1`
- W-02 `CCEnvironmentTransitionV1`
- W-03 `CCExperienceSequenceV1`
- W-04 `CCLearningUpdateV1`
- W-05 `CCEconomicResultV1`

Science identity：`CB16_R11_CC_SCIENCE_SEMANTIC_V1`。

不要再从移动的 A/B/C/D branch 名恢复 authority；frozen heads、qualified SHAs 和 receipt hashes 已写入 integration receipt。

## 3. 最终资格结果

### 3.1 冻结资格 run

Shanxi workflow run `34710702090`，qualification job `103598868269`：

- Joined CC tests：**155 passed, 1 deselected**。
- 唯一 deselected：Thread A 独立分支阶段的 `test_no_sibling_cc_module_dependency`；A/B/C/D 被正式授权 join 后该断言不再适用，没有跳过科学行为测试。
- Closed-loop + provenance：PASS。
- Hostile/recovery：PASS，包括 negative equity/liability、REJECT 后 world continuation、reversal second-leg failure、process crash/recovery、failure-fact retention、writer backpressure、same-account generation switch、exactly-once update。
- Reference-vs-fast semantic equivalence：PASS。
- Semantic checksum：`c864052eab1c107ab73a88d96ddb527bfc60819a41165b1495ecc74d56644988`。
- Final-account checksum：`29d33e7639029f40c6edfb1c7fe9fff26e4af3f418cf5109a066a55de13e282e`。
- FINAL sealed；fresh market data 未使用。

冻结资格 artifact：ID `10303497599`，SHA256 `9b0be77738e680de071e92d622f9fc25539373ec4c49a54714f471cafde4b903`。

### 3.2 最终 PR head 重复资格确认

最终 PR head `5d236e9d2368a79e830bec1971b949de88c032df` 又运行 workflow `34712758444`，结果仍为 SUCCESS：

- 155 passed, 1 同一合法 deselection；
- closed-loop / provenance / hostile / recovery / equivalence / hard-cutover / legacy-import / FINAL-fresh firewalls 全部 PASS；
- semantic checksum 与 final-account checksum 均未变化；
- selected topology 未变化。

该重复 run 用于证明最后 authority/docs commits 没有破坏 runtime。它不用于事后替换首次冻结 benchmark 的数字。

## 4. Canonical 性能路径

Receipt 冻结的预注册 workload：16 accounts × 64 market steps = 1024 transitions/run，reference 与 fast 各 7 次，交替执行。

- Reference median：**7.535366 transitions/s**；median wall clock **135.892535 s**。
- Integrated fast median：**638.765768 transitions/s**；median wall clock **1.603092 s**。
- Median speedup：**84.769×**。

按冻结规则：

`PERFORMANCE FIRST AMONG SEMANTICALLY QUALIFIED IMPLEMENTATIONS`

最终 canonical topology：

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

Thread D 的 synthetic account kernel 没有被提升为 science authority。账户/runtime/execution 语义仍由 Thread A 定义；D 负责 performance spine、same-account scheduling、bounded fact transport 和 durable chunk writing。

Hard cutover 已完成。Canonical CC runtime 不得 import、兼容或 fallback 到：

- `gpu_inference_broker.py`
- `multiprocess_trajectory_farm.py`
- `vectorized_physics.py`

新 fast path 失败时 fail closed。

## 5. 证据边界

当前最强证据：

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

不能升级为 ECONOMIC 或 TRANSFER。当前没有证明真实历史市场 edge、未来盈利性或跨时期迁移；FINAL 未打开，fresh market data 未使用。

## 6. 双基准问题已澄清，后继实现待迁移

2026-09-12 用户要求本对话表态并落实：B&H/FLAT 无默认主次，同合同的完整算术收益决定候选排序；满足预登记改善证据的候选可晋升沙盒学习冠军，不以同时击败两基准为额外门槛。详见 [经济排序与晋升](ECONOMIC_ORDERING_AND_PROMOTION.md)。

这是后续训练设计决定，不是已实现或已测经济结论。现有 `cc_economic_promotion_r0.py::assess` 及 frozen receipt 仍反映旧 unresolved 契约；下一阶段 S0 迁移并验证新语义，原 receipt 不改写。不得继续把这项文档已关闭的问题当作需要 owner 再选基准。

## 7. 下一断点

当前后继路线由 [后 CC 科学计划 R0](POST_CC_SCIENTIFIC_PROGRAM_R0.md) 明确：S0 契约/spec → S1 端到端可学习性 → S2 历史执行 canary → S3 小规模历史学习 → 条件性 S4 容量 → S5 经济确认。先完成 S0/S1，不直接全量训练或扩大 Brain。本轮只落实文档，尚无这些新阶段的运行结果。

后续执行继续遵守：

1. 以 integration receipt/spec 指定的 canonical CC runtime 为起点；
2. 不重新打开 A/B/C/D 四线程 implementation；
3. 不恢复 legacy performance fallback；
4. 在每个新科学 run 前冻结数据 authority、cohort、horizon、budget、seeds、replay/update 配置和 gate；
5. 保持 synthetic qualification 与真实 market economic evidence 分层。

`docs/BRAIN_CAPACITY_ROADMAP_3700X_1060.md` 是容量规划建议，不自动授权模型扩容或训练；其候选范围需要后续单独的容量实验验证。

历史 AC/BC TODO、Stage-4、Teacher/demonstration 与 CC 四线程 TODO 均作为 provenance/history 保留，不再是当前 execution authority。
