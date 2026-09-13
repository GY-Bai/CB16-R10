# CB16 文档入口

**当前阶段与工作 PR 统一见 [CURRENT_STATE](CURRENT_STATE.md)。** 文档目录同时保存当前规范和历史材料，不能从文件名或旧 ACTIVE 标签判断哪一个任务仍待执行。

## 按问题阅读

| 问题 | 入口 |
|---|---|
| 当前做到哪里，接哪个 PR | **[CURRENT_STATE](CURRENT_STATE.md)** |
| 文档如何分层，谁维护什么 | **[DOCUMENTATION_MAP](DOCUMENTATION_MAP.md)** |
| Astra / Sol / DS 的职责与审核 | **[ROLES_AND_REVIEW_PROTOCOL](ROLES_AND_REVIEW_PROTOCOL.md)** |
| Sol 怎样写任务并审核代码 | **[SOL_ROLE_PRINCIPLES](SOL_ROLE_PRINCIPLES.md)** |
| DS 怎样实现、修复与交付 | **[DS_FLASH_ROLE_PRINCIPLES](DS_FLASH_ROLE_PRINCIPLES.md)** |
| post-CC 阶段材料如何继承 | [R11_POST_CC_S0_S1_TODO](R11_POST_CC_S0_S1_TODO.md) |
| 项目最高目标与已确定取舍 | [VISION](VISION.md)、[DECISIONS](DECISIONS.md)、[PRINCIPLE_ALIGNMENT](PRINCIPLE_ALIGNMENT.md) |
| 组件职责与架构 | [COMPONENT_REQUIREMENTS](COMPONENT_REQUIREMENTS.md)、[ARCHITECTURE_MAP](ARCHITECTURE_MAP.md)；历史快照须结合当前代码/receipt |
| 学习与评价原则 | [LEARNING_CONTRACT](LEARNING_CONTRACT.md)、[EVALUATION_PRINCIPLES](EVALUATION_PRINCIPLES.md) |
| 算法设计与资格思路 | [TRAINING_ALGORITHM_R0](TRAINING_ALGORITHM_R0.md)、[TRAINING_QUALIFICATION_R0](TRAINING_QUALIFICATION_R0.md)；具体实现结论查阶段证据 |
| 双基准与策略晋升 | [ECONOMIC_ORDERING_AND_PROMOTION](ECONOMIC_ORDERING_AND_PROMOTION.md) |
| 后续科学路线 | [POST_CC_SCIENTIFIC_PROGRAM_R0](POST_CC_SCIENTIFIC_PROGRAM_R0.md) |
| 性能/容量规划 | [PERFORMANCE_STRATEGY_3700X_1060](PERFORMANCE_STRATEGY_3700X_1060.md)、[BRAIN_CAPACITY_ROADMAP_3700X_1060](BRAIN_CAPACITY_ROADMAP_3700X_1060.md) |
| 仓库工作约定 | [AGENTS](../AGENTS.md) |

## 当前专题审阅

- [PR #104 Astra 审阅](reviews/PR104_ASTRA_REVIEW_2026-09-13.md)：框架设计方向、具体修订项、合并前条件和组织建议。框架正文/代码尚在 [候选 PR](https://github.com/GY-Bai/CB16-R10/pull/104)，不能当作 main 已生效实现。
- [S1 DS Flash 复盘](reviews/S1_DS_FLASH_RELIABILITY_RETROSPECTIVE_2026-09-13.md)：历史问题用于改善任务与审阅，不代替最新 PR 的实际状态。

## 历史与兼容入口

- [S_SERIES_TODO_AUTHORING_PRINCIPLES](S_SERIES_TODO_AUTHORING_PRINCIPLES.md)：按与 Sol principle 已声明的关系继续兼容。
- [S0/S1 初次可执行性审阅](reviews/S0_S1_EXECUTABILITY_REVIEW_2026-09-13.md)：其断点属于当时 SHA。
- [S0 migration TODO](post_cc/S0_CONTRACT_MIGRATION_TODO.md)、[S0-v2 foundation TODO](post_cc/S0_V2_DURABLE_LEARNABILITY_FOUNDATION_TODO.md)、[旧 S1 TODO](post_cc/S1_END_TO_END_LEARNABILITY_TODO.md)：按当前路由理解继承范围。
- [CC integration handoff](CC_INTEGRATION_HANDOFF.md)：已合格历史基础；AC/BC/Stage-4/CC 四线程任务不再是默认待实施入口。

历史 receipt 保持不可变。文档完整、CI success、基础设施合格、可学习性与经济有效性分别需要自己的证据；导航不抬高任何 evidence ceiling。
