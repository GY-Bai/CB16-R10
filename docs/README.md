# CB16 文档入口

默认只读 **[VISION](VISION.md) → [ARCHITECTURE_MAP](ARCHITECTURE_MAP.md) → [CURRENT_STATE](CURRENT_STATE.md)**：目标、科学要求到技术实现的对应、当前执行断点。起步 CPU-only，具体设备决定见 [性能策略](PERFORMANCE_STRATEGY_3700X_1060.md)。

## 按任务补充阅读

| 要回答的问题 | 唯一主入口 / 配套解释 |
|---|---|
| 谁产出、谁审阅、谁负责交付 | [角色协议](ROLES_AND_REVIEW_PROTOCOL.md)；[Sol](SOL_ROLE_PRINCIPLES.md)、[DS](DS_FLASH_ROLE_PRINCIPLES.md) 操作细则 |
| 经验、账户、失败、回放是什么 | [LEARNING_CONTRACT](LEARNING_CONTRACT.md) |
| 收益、双基准、晋升如何区分 | [EVALUATION_PRINCIPLES](EVALUATION_PRINCIPLES.md)、[经济排序](ECONOMIC_ORDERING_AND_PROMOTION.md) |
| 组件应保持哪些行为 | [PRINCIPLE_ALIGNMENT](PRINCIPLE_ALIGNMENT.md)、[COMPONENT_REQUIREMENTS](COMPONENT_REQUIREMENTS.md) |
| 当前算法与早期提案有何差别 | [ARCHITECTURE_MAP](ARCHITECTURE_MAP.md)；[算法 R0](TRAINING_ALGORITHM_R0.md) 为历史推导，非当前超参数表 |
| 如何证明阶段声明 | [资格框架](SCIENTIFIC_QUALIFICATION_FRAMEWORK.md)，再读当前 stage 合同 |
| 后续科学顺序 | [POST_CC_SCIENTIFIC_PROGRAM_R0](POST_CC_SCIENTIFIC_PROGRAM_R0.md)；阶段起点由 CURRENT_STATE 更新 |
| 当前 Infra 恢复 / 将来技术债 | CURRENT_STATE 指定的 RC2 任务 / [MAIN_TODO](MAIN_TODO.md) |
| 设备、并行、语言、未来容量 | [性能策略](PERFORMANCE_STRATEGY_3700X_1060.md)、[容量规划](BRAIN_CAPACITY_ROADMAP_3700X_1060.md)；后者是未来规划 |
| 决定、未决事项与文档维护 | [DECISIONS](DECISIONS.md)、[OPEN_QUESTIONS](OPEN_QUESTIONS.md)、[DOCUMENTATION_MAP](DOCUMENTATION_MAP.md) |

## 历史按需读取

AC/BC/CC、Stage-4、Teacher/H72、旧 S1 r0、原算法/资格 R0 保留为历史与设计来源。它们不组成新 agent 的默认待实施清单；后继替代范围看当前合同，旧 receipt 保持原字节。

[S0/S1 总控](R11_POST_CC_S0_S1_TODO.md) 与各 stage TODO 保留任务来源；已关闭 foundation 不因旧文字重新建设。[CC handoff](CC_INTEGRATION_HANDOFF.md)、[PR #104 当时审阅](reviews/PR104_ASTRA_REVIEW_2026-09-13.md)、[S1 实现复盘](reviews/S1_DS_FLASH_RELIABILITY_RETROSPECTIVE_2026-09-13.md) 按各自 SHA 解释，当前状态不在这些历史页维护。
