# CB16 agent 工作约定

本文件适用于整个仓库。主要交流语言为中文，代码标识和版本化协议名称保持原样。

## 1. 先确定自己的角色

所有 agent 必须先读：

1. `docs/ROLES_AND_REVIEW_PROTOCOL.md`
2. `docs/SCIENTIFIC_QUALIFICATION_FRAMEWORK.md`
3. 与自己角色对应的 principle：
   - Sol：`docs/SOL_ROLE_PRINCIPLES.md`
   - DS Flash / implementation agent：`docs/DS_FLASH_ROLE_PRINCIPLES.md`
4. 当前 stage 的 TODO、Stage Qualification Profile、authority、receipt 和 candidate/review artifacts。

当前 owner 指定的责任链：

```text
owner/user
  ↓ supervises
Astra — documents/principles aligned to owner
  ↓ supervises
Sol — scientific gap → Stage Qualification Profile → architecture-aware TODO → independent code review
  ↓ supervises
DS Flash — TODO → code/tests/exact-SHA CI evidence
```

implementation agent 自检、repo-guard 或绿色 CI 不能替代 Sol review；implementation agent 不自行 merge，不自行签发 reviewer receipt。

临时 ChatGPT sandbox 可用于静态准备/阅读，不替代 Shanxi runtime evidence。

## 2. Sol 的仓库工作方式

Sol 不得一边无边界扫描仓库、一边直接写 TODO。复杂 S-series authoring 按 `docs/SOL_ROLE_PRINCIPLES.md` 的 pipeline，并在 Gap-to-Code 与 executable TODO 之间加入 Stage Qualification Profile：

```text
Scientific Gap Freeze
→ Relevant Infrastructure Inventory
→ Upstream/Downstream Interface Map
→ Gap-to-Code Matrix
→ Stage Qualification Profile
→ Executable TODO
→ Adversarial TODO Review
```

Sol 只调查 task-local architecture slice。TODO 必须明确：可复用 infra、要新增/修改的 surface、上下游 producer/consumer、禁止重造路径、HIGH-risk identity/RNG/edge cases、formal gate、CI/artifact evidence。

复杂 stage 的 Qualification Profile 至少覆盖 capability claims / proof obligations、semantic trace、identity/RNG、edge-case decisions、negative controls / hostile counterexamples、machine gates、artifact proof、exact-SHA CI 与 reviewer gate。mandatory proof obligation 没有 consumer、gate 或 artifact proof 时，不得交给 DS 猜测。

Sol-Reviewer 必须独立检查 actual diff 和 machine evidence，不默认相信 Sol-Author 对 gate 的命名或 proof 强度。

## 3. DS Flash / implementation agent 的仓库工作方式

DS 必须 reuse-first。TODO 标为 `REUSE_AS_IS / DO_NOT_TOUCH / LEGACY_REFERENCE_ONLY` 的模块不能自行重写或旁路。

DS 不得用局部 fallback 改写 scientific semantics，包括但不限于：

- 改 sampling distribution；
- 跳过冻结要求的 update；
- silent clamp；
- 改 reward/threshold/seed/model/optimizer/budget；
- 扩大 replay eligibility；
- behavior/target/evaluation 身份偷换；
- nominal/executed action 偷换；
- 用旧 CI 为新 SHA 背书。

遇到 TODO/Profile 未定义的 HIGH-risk semantic choice，报告给 Sol，不自行决定。

DS terminal handoff state 是 `READY_FOR_SOL_REVIEW`。

## 4. Scientific Qualification Framework

资格审查只回答“一个已定义的能力声明需要怎样被证明”，不得反向修改 frozen science。

通用 verdict 仅使用：

- `PASS`
- `SCIENTIFIC_FAIL`
- `CONTRACT_MISMATCH`
- `EXECUTION_BLOCKED`
- `HARDWARE_LIMIT`
- `EVIDENCE_INSUFFICIENT`

只有执行/identity/provenance/evidence 合同完整有效时，冻结科学判据失败才可称 `SCIENTIFIC_FAIL`。字段存在不等于 link 被证明；proof 名称不得强于机器实际验证内容。

若 stage 声称 durable/restart-safe/artifact-only，应能删除原 scratch/run root 后仅靠导出 artifact 复核 mandatory lineage，并对中间 corruption fail closed。

当前 qualification framework 的公共 primitive 属于独立 non-scientific infrastructure；不得在 active authority-sensitive stage PR 中顺手重构科学 runtime。

## 5. 当前 stage

S0-v2 Durable Learnability Foundation 已被 Sol 验收并合入 main；其最高 evidence 为：

`POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`

当前 active implementation/review stage 是 **S1 End-to-End Learnability**，工作通过 PR #102 在指定 S1 branch 上多轮修订。S1 正式 5-seed qualification 只有在 Sol 明确签发 `READY_FOR_S1_QUALIFICATION` authorization 后才能运行。

当前 implementation agent 不得重开 S0-v2 foundation，也不得把 S1 bounded smoke 当 scientific qualification。本 qualification framework 不授权修改 S1 frozen science 或顺手抽取/重构 PR #102 runtime。

## 6. Historical authority 必须保持不可变

CC R11 integration 已关闭。Canonical CC authority：

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

Historical post-CC S0 contract migration 已冻结：

- `authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json`

S0-v2 final authority：

- `authority/rearchitecture_r11/CB16_R11_S0V2_RECEIPT_V1.json`

不得编辑历史 receipts 来满足新任务。需要修正 authority 时使用 versioned successor artifact，并由对应 reviewer/owner 权限冻结。

## 7. Scientific semantics that must not change

- `Truth != Belief != Decision != Permission != Execution`.
- Nominal sampled action remains separate from permitted/executed action.
- Requested target risk is not confidence.
- Actor is categorical direction + conditional continuous target risk.
- FLAT risk is exactly 0.
- True behavior `log_mu` is computed/persisted at decision time and never reconstructed from execution.
- Same logical account remains continuous across chunk/restart/checkpoint/generation boundaries where required.
- Negative equity, liabilities, failures and terminal facts remain factual.
- Compute truncation, chunk boundary, task horizon and mechanical/economic terminal are distinct.
- Frozen market organ gradient ownership remains frozen.
- Arithmetic expected return remains the current economic orientation.
- B&H and FLAT are parallel benchmark components; no master precedence.
- No handcrafted cycle/regime/resonance activation system.
- No hidden strategic SL/TP/max-hold/cooldown.
- FINAL remains sealed; fresh market data remains forbidden unless explicitly authorized.

## 8. Canonical runtime and legacy firewall

Selected CC topology remains:

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

Thread A remains runtime/account/execution science authority; Thread D remains performance implementation authority。

Canonical path must not silently fall back to historical performance runtime：

- `gpu_inference_broker.py`
- `multiprocess_trajectory_farm.py`
- `vectorized_physics.py`

这些 legacy modules 可作 reference，不能在没有新 authority 的情况下恢复成 canonical fallback。

## 9. Pre-merge CI requirement

runtime/code behavior 变更在合并 `main` 前必须通过 GitHub Actions → Shanxi Docker 的相关专项/冒烟测试，至少绑定：

- exact checkout SHA/tree；
- shared preflight / current runner contract；
- verified canonical Python；
- focused tests + required regressions；
- machine-readable artifacts；
- FINAL/fresh firewall。

Repo-guard alone is insufficient for runtime evidence。

纯说明性文档变更可只运行 repo-guard/static checks；不得把文档 CI 当 runtime evidence。

## 10. HIGH-risk implementation surfaces

Sol/DS 都应按 role principle 特别处理：

- SHORT/FLAT/LONG mapping；
- FLAT point mass / non-FLAT density；
- risk transform/log-Jacobian；
- `log_pi - log_mu` / V-trace；
- boundary/bootstrap/terminal masks；
- nominal vs execution routing；
- behavior/target/evaluation identity；
- observation/content hash；
- exactly-once retry；
- account lineage / generation switch；
- RNG ownership / initialization order；
- authorization SHA/tree/manifest binding；
- provenance chain；
- threshold equality / empty / missing / NaN / Inf；
- any fallback that could change science or hide failure。

## 11. Evidence ceilings

Historical CC strongest evidence：

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

Historical S0 maximum evidence：

`POST_CC_CONTRACT_MIGRATION_QUALIFIED`

S0-v2 maximum evidence：

`POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`

S1 如果最终通过，其 evidence ceiling 只能按 S1 frozen TODO/authority 声称；不得自动提升为 ECONOMIC、TRANSFER、historical profitability 或 production readiness。
