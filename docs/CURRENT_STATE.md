# 当前状态与接手断点

核对日期：2026-09-13 UTC；本次文档审阅基于 `main@8d37166f9fcc87d61affaf1319afe225d8754d1b`。CC R11 integration 已通过 PR #99 正式合入 `main`；integration merge commit 为：

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

## 6. 双基准原则与 S0 实现迁移均已关闭

B&H 和 FLAT 是两个并列 benchmark components，**没有 master precedence，也没有 master baseline winner**。同合同下的模型间完整算术收益排序、单模型 B&H/FLAT component 结果、promotion decision 是不同对象。详见 [经济排序与晋升](ECONOMIC_ORDERING_AND_PROMOTION.md)。

现有 `cc_economic_promotion_r0.py` 与旧 Thread-C/integration receipts 仍反映历史 unresolved 契约。它们不被改写。S0 已创建 successor ordering/component/promotion contracts、显式 adapter/migration/router；receipt 记录 post-CC canonical path 不再把 B&H/FLAT disagreement 解释成 owner uncertainty。

因此不要再询问 owner “B&H 和 FLAT 谁优先”。S0 receipt 的证据范围为 `POST_CC_CONTRACT_MIGRATION_QUALIFIED`，不能提升为 S1 可学习性或市场经济证据。

## 7. 当前精确断点：S0 PASS / S1 ACTIVE

宽泛路线仍由 [后 CC 科学计划 R0](POST_CC_SCIENTIFIC_PROGRAM_R0.md) 定义；**当前直接执行 authority 已进一步拆成：**

- 总控：`docs/R11_POST_CC_S0_S1_TODO.md`
- S0：`docs/post_cc/S0_CONTRACT_MIGRATION_TODO.md`
- S1：`docs/post_cc/S1_END_TO_END_LEARNABILITY_TODO.md`

规划冻结基线：

`main@fc7102442e91a1c27cf705487c6d06bd64b8ea09`

执行顺序严格为：

```text
S0 contract/economic semantic migration
    -> S0 qualification receipt
    -> bind S0 qualified identity and receipt-bearing S1 working base separately
    -> S1 durable end-to-end learnability qualification
    -> S1 receipt/verdict
    -> STOP
```

### 7.1 S0 已关闭的范围

S0 已完成以下迁移/合同资格：

- 旧 promotion code 的 `UNRESOLVED_OWNER_DECISION` 迁移为 successor semantics；
- B&H/FLAT 作为 parallel components；
- model ordering 与 promotion rule 分离；
- old receipts 保持原样并由 migration authority 说明历史/当前边界；
- 预冻结 S1 observation sidecar、joint replay sample、task registry、run spec、seed/no-rescue/evidence rules。

Qualified implementation：`5760061d6c274e9f8796e6608e86bb173018148f`；workflow `34721332483` success，receipt 记录 15 个 focused tests 通过。带 receipt 的交接与 S1 working base 为 `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f`，不因前一 SHA 是合格代码身份而回退 checkout。

上述冻结及 PASS 覆盖当时实际检查，不代表 S1 生成器、控制和运行参数已全部具体化。当前缺口及原编号下的修补见 [S0/S1 可执行性审阅](reviews/S0_S1_EXECUTABILITY_REVIEW_2026-09-13.md)；TODO 维护者遵守 [撰写角色原则](S_SERIES_TODO_AUTHORING_PRINCIPLES.md)。原 registry/run spec/receipt 保持原字节，具体化以显式补充或适用的后继版本交付。

本次所查 S1 分支 `ai/r11-post-cc-s1-end-to-end-learnability-r0` 的 head 为 `9d9911432d1da419b9957d857867cf3c5c8d9d3d`，相对交接基线的净差异仅 S1 baseline JSON；不能据此声称主体实现或学习资格已完成。该快照的下一实现入口是 S1-002，S1-A/B 开发与具体任务/oracle 定义可继续推进。正式运行须先落实 manifest 和判定器。

### 7.2 S1 要证明什么

当前 CC canary 已证明“一次 update 能接通”，但仍不是 robust learnability evidence。主要对齐缺口是：

- integration canary 的训练张量可来自 rollout 内存 `records`，尚未证明 learner 只依靠 durable replay 即可重建训练真值；
- persistent experience 有 observation identity/hash，但需要 Brain-ready `market/account/execution` observation payload 的 durable materialization；
- generic learner batch/API 主要是 categorical action，而 canonical Actor 是 `direction + conditional continuous risk` 联合分布；
- 现有 component toys 不等于 full persistent loop 的多轮收敛证据。

S1 必须建立：

`durable observation fact -> persistent replay materialization -> joint action batch -> true log_mu / joint log_pi -> Critic/V-trace/Actor update -> exactly-once child checkpoint -> generation switch -> actual child behavior -> preregistered known-answer improvement`

并通过多 seed、no-signal、shuffled-credit 等 negative controls。Loss 下降、checkpoint 改变、child 输出不同动作都不能单独构成 S1 PASS。

## 8. S1 之后的路线边界

S1 结束即停。S2 historical execution canary、S3 小规模历史学习、条件性 S4 capacity、S5 economic confirmation 仍由 `POST_CC_SCIENTIFIC_PROGRAM_R0.md` 定义，但不能由 S0/S1 executor 自动启动。

S4 不是必须阶段；是否扩容由 S3 证据决定。`BRAIN_CAPACITY_ROADMAP_3700X_1060.md` 只是容量规划建议，不是自动扩容授权。

后续所有真实 historical runs 继续要求：数据 authority、cohort、horizon、budget、seeds、replay/update 配置与 gate 在运行前冻结，并保持 synthetic qualification 与 market economic evidence 分层。

历史 AC/BC TODO、Stage-4、Teacher/demonstration 与 CC 四线程 TODO 均作为 provenance/history 保留，不再是当前 execution authority。
