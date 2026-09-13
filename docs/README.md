# CB16 文档入口

本目录同时保存当前 authority、实施说明与历史阶段资料。CC R11 integration 与历史 post-CC S0 已完成；**当前 active implementation entrypoint 是 S0-v2 Durable Learnability Foundation。**

## 当前阅读路线

| 问题 | 当前入口 |
|---|---|
| 当前精确状态与下一断点 | **[CURRENT_STATE](CURRENT_STATE.md)** |
| 角色、审核与 Shanxi CI | **[ROLES_AND_REVIEW_PROTOCOL](ROLES_AND_REVIEW_PROTOCOL.md)** |
| 当前 post-CC 路由 | **[R11_POST_CC_S0_S1_TODO](R11_POST_CC_S0_S1_TODO.md)** |
| 当前单-Agent S0-v2 执行包 | **[S0_V2_DURABLE_LEARNABILITY_FOUNDATION_TODO](post_cc/S0_V2_DURABLE_LEARNABILITY_FOUNDATION_TODO.md)** |
| S 系列 TODO 撰写原则 | **[S_SERIES_TODO_AUTHORING_PRINCIPLES](S_SERIES_TODO_AUTHORING_PRINCIPLES.md)** |
| 旧 S0/S1 可执行性审阅 | [2026-09-13 executability review](reviews/S0_S1_EXECUTABILITY_REVIEW_2026-09-13.md) |
| 历史 S0 contract migration | [S0_CONTRACT_MIGRATION_TODO](post_cc/S0_CONTRACT_MIGRATION_TODO.md) — PASS / frozen provenance |
| 历史/后续 S1 learnability design | [S1_END_TO_END_LEARNABILITY_TODO](post_cc/S1_END_TO_END_LEARNABILITY_TODO.md) — not current implementation entrypoint |
| CC integration 最终交接 | [CC_INTEGRATION_HANDOFF](CC_INTEGRATION_HANDOFF.md) |
| CC machine-readable authority | `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json` / `CB16_R11_CC_INTEGRATION_RECEIPT_V1.json` |
| 历史 S0 machine-readable receipt | `authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json` |
| 项目最高目标 | [VISION](VISION.md) |
| 当前设计决策 | [DECISIONS](DECISIONS.md) |
| 原则约束 | [PRINCIPLE_ALIGNMENT](PRINCIPLE_ALIGNMENT.md) / [COMPONENT_REQUIREMENTS](COMPONENT_REQUIREMENTS.md) |
| 学习/评价原则 | [LEARNING_CONTRACT](LEARNING_CONTRACT.md) / [EVALUATION_PRINCIPLES](EVALUATION_PRINCIPLES.md) |
| Actor-Critic 设计 | [TRAINING_ALGORITHM_R0](TRAINING_ALGORITHM_R0.md) / [TRAINING_QUALIFICATION_R0](TRAINING_QUALIFICATION_R0.md) |
| 后 CC 总体科学路线 | [POST_CC_SCIENTIFIC_PROGRAM_R0](POST_CC_SCIENTIFIC_PROGRAM_R0.md) |
| 性能与容量规划 | [PERFORMANCE_STRATEGY_3700X_1060](PERFORMANCE_STRATEGY_3700X_1060.md) / [BRAIN_CAPACITY_ROADMAP_3700X_1060](BRAIN_CAPACITY_ROADMAP_3700X_1060.md) |
| Agent 执行规则 | [根目录 AGENTS](../AGENTS.md) |

## 当前 active stage

Owner 当前只安排一个 implementation Agent 执行 S0-v2，统一写入：

`ai/r11-s0v2-durable-learnability-foundation-r0`

Scientific/code baseline：

`main@392063881a6ef0dd1776ac579f1a290a134fc49e`

S0-v2 负责把 durable observation/replay、joint direction+risk learner、restart reconstruction、exactly-once recovery 与 same-account generation continuity 做成一个可审核的基础资格包。

Implementer 完成后停在 `READY_FOR_SOL_REVIEW`。Sol 审核实际 diff 与 exact-SHA GitHub Actions -> Shanxi Docker evidence，要求修复或接受并 merge。Implementer 不自行 merge。

S0-v2 最大证据：

`POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`

它不是最终 known-answer learnability、ECONOMIC 或 TRANSFER evidence。

## 历史 authority

CC R11 integration PR #99 已完成。Canonical topology：

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

CC strongest evidence：

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

历史 post-CC S0 contract/economic migration 也已 PASS，证据：

`POST_CC_CONTRACT_MIGRATION_QUALIFIED`

B&H 和 FLAT 继续是并列 benchmark components，没有 master precedence；模型排序、baseline components 和 promotion decision 相互独立。历史 receipts 不追溯改写。

## S0-v2 后续边界

S0-v2 经 Sol review/merge 后，才重新生成/路由 successor S1 scientific qualification，集中跑五类 known-answer tasks、multi-seed threshold 与 negative controls。

S0-v2 不自动授权 S1、historical S2/S3、FINAL、capacity expansion 或 economic qualification。

AC/BC/Stage-4/CC threads、历史 S0/S1 文档继续作为 provenance/history 保留。