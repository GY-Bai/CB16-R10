# 当前状态与接手断点

核对基线：2026-09-13 UTC；本次执行重排以 `main@392063881a6ef0dd1776ac579f1a290a134fc49e` 为科学/代码基线。后续 S0-v2 task-authoring/navigation commits 只改变执行路由，不追溯改变既有 CC / S0 scientific identity。

## 1. 已冻结并完成的历史 authority

CC R11 integration 已通过 PR #99 合入 `main`。Canonical authority：

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

CC strongest evidence：

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

它不是 ECONOMIC 或 TRANSFER evidence。FINAL 未开启，fresh market data 未使用。

历史 post-CC S0 contract/economic migration 也已 PASS：

- frozen implementation base: `fc7102442e91a1c27cf705487c6d06bd64b8ea09`
- qualified implementation identity: `5760061d6c274e9f8796e6608e86bb173018148f`
- receipt-bearing head: `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f`
- evidence: `POST_CC_CONTRACT_MIGRATION_QUALIFIED`

Historical S0 receipt remains immutable. B&H/FLAT are parallel benchmark components with no master precedence; model ordering, benchmark components and promotion decision remain separate.

## 2. 当前协作与审核链

当前 owner 指定流程由 `docs/ROLES_AND_REVIEW_PROTOCOL.md` 约束：

- 文档/原则先与 owner 对齐；
- Sol 维护可执行 TODO，并对实际代码负责审核；
- 一个 implementation Agent 按 TODO 在指定 branch 写代码；
- runtime/code behavior 变更合并 main 前必须通过 GitHub Actions 调度的 Shanxi Docker 相关测试；
- implementer 自检、repo-guard 或绿色 CI 都不能替代 Sol 对实际 diff、公式、比较、裸分支、mask、空值和边界的审核；
- implementer 不自行合并。

## 3. 当前精确断点：S0-v2 ACTIVE

旧 S1 包把 durable data-plane、joint learner、recovery 与最终 known-answer science 放在同一大阶段，导致执行 Agent 多次停在“尚不能签最终 receipt”的状态。根据最新 executability review 和 owner 的单-Agent执行安排，当前真实工程依赖被重新封装为一个**新的 versioned S0**：

`CB16_R11_S0V2_DURABLE_LEARNABILITY_FOUNDATION_R0`

当前任务 authority：

`docs/post_cc/S0_V2_DURABLE_LEARNABILITY_FOUNDATION_TODO.md`

唯一 implementation branch：

`ai/r11-s0v2-durable-learnability-foundation-r0`

S0-v2 scientific/code baseline：

`392063881a6ef0dd1776ac579f1a290a134fc49e`

S0-v2 不是历史 S0 的重跑，不替代旧 receipt。它是新的 successor engineering qualification。

## 4. S0-v2 必须完成的链路

一个 Agent 在同一 branch 内实现完整基础链：

```text
canonical Brain observation
 -> immutable content-addressed observation fact
 -> authoritative transition/experience linkage
 -> persistent replay selection
 -> restart-safe replay materialization
 -> joint nominal direction+risk batch
 -> persisted true behavior log_mu
 -> target joint log_pi on nominal action
 -> separate Critic + boundary-aware bootstrap + V-trace
 -> joint Actor/Critic update
 -> durable exactly-once update provenance
 -> child checkpoint
 -> generation switch
 -> SAME LOGICAL ACCOUNT continuation
```

关键硬约束：

- qualification/training truth 不能来自 collector-private live `records`；
- observation/hash mismatch fail closed；
- executed action 不得替代 nominal action；
- `log_mu` 不得事后重构；
- FLAT risk 必须精确为 0；
- 删除 live rollout Python 对象并重启后必须从 durable state 重建相同 replay identity；
- child generation 不能用新 flat account 伪造 continuity；
- exactly-once update/recovery 必须有 hostile interruption tests；
- FINAL/fresh-data firewall 保持关闭。

S0-v2 最大证据：

`POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`

这不等于最终 known-answer learnability PASS。

## 5. 单 Agent 的结束状态

Implementer 必须持续推进全部 S0-v2 tasks，普通缺功能不是 blocker。完成后把 branch 留在：

`READY_FOR_SOL_REVIEW`

并提供 exact head SHA、diff inventory、测试结果、Shanxi workflow/run/job/artifact identities 以及 candidate machine-readable record。

Implementer 不 merge，不自签 Sol reviewer acceptance，也不自动进入 successor S1。

合法提前停止仅限：

- `CONTRACT_MISMATCH`
- `EXECUTION_BLOCKED`
- `HARDWARE_LIMIT`
- current authority 无法解决的全新 owner scientific choice

## 6. Sol 审核与 merge gate

Sol 接到 branch 后必须独立核对：

1. branch lineage / exact base / diff scope；
2. frozen CC 与 historical S0 receipts 未被改写；
3. observation/replay durable truth 与 restart reconstruction；
4. canonical joint policy likelihood、risk density、log-Jacobian、direction indexing；
5. `log_pi - log_mu` / V-trace ratio 与 boundary/bootstrap masks；
6. nominal/permission/execution routing；
7. gradient ownership；
8. exactly-once update/checkpoint recovery；
9. generation switch / same-account continuity；
10. exact reviewed SHA 的 GitHub Actions -> Shanxi Docker evidence。

有问题则要求 implementer 在同一 branch 修复；满足后由 Sol merge。

## 7. S0-v2 之后

只有 S0-v2 经 Sol review/merge 后，才重新路由 successor S1 scientific qualification。后续 S1 应集中验证五类 known-answer tasks 与 negative controls：

1. `ACCOUNT_DEPENDENT_ACTION`
2. `DELAYED_CONSEQUENCE_CREDIT`
3. `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION`
4. `OFF_POLICY_VTRACE_CORRECTION`
5. `A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION`

S0-v2 结束不自动授权 historical S2/S3、容量扩张、FINAL 或经济资格。

## 8. 仍必须保持的最高科学原则

- `Truth != Belief != Decision != Permission != Execution`；
- requested target risk != confidence；
- one logical account is continuous across chunk/restart/checkpoint/generation boundaries；
- signed account economics / liability / failure facts remain real；
- historical information is not expired merely because it is old；
- recurrence由模型权重/replay隐式表达，不增加手工 cycle/regime/resonance 激活系统；
- arithmetic expected return remains the current economic orientation；
- no hidden strategic SL/TP/max-hold/cooldown；
- FINAL sealed / fresh data forbidden until separately authorized。

历史 AC/BC/Stage-4/Teacher/CC-thread/S0/S1 documents remain provenance/history unless the current routing documents explicitly name them as active authority.