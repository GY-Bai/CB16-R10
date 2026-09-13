# CB16 agent 工作约定

本文件是仓库入口；详细规则以所列 role/framework 文档和当前 stage authority 为准。

## 1. 必读顺序

1. `docs/ROLES_AND_REVIEW_PROTOCOL.md`
2. `docs/SCIENTIFIC_QUALIFICATION_FRAMEWORK.md`
3. Sol 读 `docs/SOL_ROLE_PRINCIPLES.md`；implementation agent 读 `docs/DS_FLASH_ROLE_PRINCIPLES.md`
4. 当前 stage 的 TODO、Qualification Profile、authority、receipt、candidate/review artifacts

责任链：

```text
owner/user
 -> Astra documents/principles
 -> Sol scientific gap + qualification contract + TODO + independent review
 -> DS code/tests/exact-SHA CI evidence
```

绿色 CI、自检或 repo-guard 都不能替代 Sol review；implementation agent 不自行 merge 或签发 reviewer receipt。

## 2. Qualification Framework

资格审查只规定“能力声明如何被证明”，不得反向修改 task、seed、reward、model、threshold、sampling、budget、oracle 或 evidence ceiling。

复杂 stage 在交给 DS 前，Sol 必须先完成 Stage Qualification Profile，至少覆盖：claims/proof obligations、semantic trace、identity/RNG、edge cases、negative controls、machine gates、artifact proof、exact-SHA CI 和 reviewer gate。

字段存在不等于 link 被证明；proof 名称不得强于机器实际验证内容。通用 verdict 只使用：`PASS / SCIENTIFIC_FAIL / CONTRACT_MISMATCH / EXECUTION_BLOCKED / HARDWARE_LIMIT / EVIDENCE_INSUFFICIENT`。

## 3. Sol / DS 工作方式

Sol 按以下顺序工作：

```text
Scientific Gap Freeze
-> Relevant Infra Inventory
-> Interface Map
-> Gap-to-Code Matrix
-> Stage Qualification Profile
-> Executable TODO
-> Adversarial TODO Review
```

mandatory proof obligation 没有 consumer、gate 或 artifact proof 时，TODO 不得交给 DS 猜测。Sol-Reviewer 必须独立检查 actual diff 和 machine evidence。

DS 必须 reuse-first；不得用局部 fallback 改 science，包括改 sampling、强塞 sample、跳过 frozen update、silent clamp、修改 reward/threshold/seed/model/optimizer/budget、扩大 replay eligibility、偷换 behavior/target/evaluation 或 nominal/executed identity、用旧 CI 为新 SHA 背书。未定义的 HIGH-risk semantic choice 交回 Sol。

## 4. 当前边界

S0-v2 已 qualified 并合入 main。S1 End-to-End Learnability 仍通过 PR #102 独立 review；bounded smoke 不是 scientific qualification。

本框架不授权在 S1 active PR 中顺手做公共重构。公共 qualification primitives 应在独立 non-scientific extraction 工作中实现，并证明 behavior/semantic equivalence 后再供后续 stage 复用。

## 5. 不可破坏的基础语义

`Truth != Belief != Decision != Permission != Execution`；nominal 与 executed action 分离；requested risk 不是 confidence；FLAT risk=0；true behavior `log_mu` 在 decision time 持久化；required account continuity 跨 restart/checkpoint/generation 保持；失败/负权益/terminal facts 保持事实；truncation 与 terminal 分离；禁止 handcrafted regime activation；历史 receipts 不回写；FINAL sealed，fresh market data 未经明确授权禁止。

## 6. CI / evidence

runtime/code behavior 合并 main 前必须由 GitHub Actions -> Shanxi Docker 覆盖相关变更，并绑定 exact checkout SHA/tree、shared preflight、verified Python、focused tests/regressions、machine-readable artifact 与 firewall。临时 sandbox 只做静态准备；repo-guard alone 不是 runtime evidence。

任何 stage 只能声明其 frozen authority 允许的 evidence ceiling；synthetic qualification 不自动提升为 economic、transfer、historical profitability 或 production readiness。
