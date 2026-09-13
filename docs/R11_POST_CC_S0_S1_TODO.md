# CB16 R11 Post-CC — 阶段路由入口

本页负责指路，不再复制一份易过时的阶段状态。**当前断点统一见 [CURRENT_STATE](CURRENT_STATE.md)。** 本次路由核对日期：2026-09-13；不改写历史 task 编号、科学参数或 receipt。

## 接手当前任务

1. 先读 [角色与审核协议](ROLES_AND_REVIEW_PROTOCOL.md) 和自己角色的 principle。
2. 从 CURRENT_STATE 找到已合格基础、当前工作 PR 和精确任务来源。
3. 核对该 PR 最新 head、review、candidate、authority 和实际 CI，再继续本角色工作。
4. 执行者不得依据旧任务正文的 ACTIVE 或“尚未实现”重开已经 receipt-qualified 的基础；审阅者也不得把新候选误当成 main 已通过能力。

## 历史阶段与当前入口的关系

| 材料 | 作用 |
|---|---|
| [历史 S0 contract migration TODO](post_cc/S0_CONTRACT_MIGRATION_TODO.md) | 已关闭的历史要求；原 receipt 不变 |
| [S0-v2 foundation TODO](post_cc/S0_V2_DURABLE_LEARNABILITY_FOUNDATION_TODO.md) | 已合格基础的实现要求，供复用与历史追溯 |
| [S0-v2 review protocol](post_cc/S0_V2_REVIEW_AND_HANDOFF_PROTOCOL.md) | 该阶段的交接纪律，保留其适用范围 |
| [旧 S1 learnability TODO](post_cc/S1_END_TO_END_LEARNABILITY_TODO.md) | 科学任务与历史要求；不能孤立采用旧 working base 和缺口描述 |
| post-S0-v2 S1 R1 TODO 与 base handoff | 当前 PR 分支中的执行补充；精确链接由 CURRENT_STATE 提供 |
| [PR #104](https://github.com/GY-Bai/CB16-R10/pull/104) | 公共资格框架候选；是否接收及限制见 CURRENT_STATE 和 Astra 审阅 |

原路由把 S0-v2 标为 ACTIVE 的全文保留在 [此前提交](https://github.com/GY-Bai/CB16-R10/blob/b19d3473a87fe36ec3394be53ad1cc6220f324c6/docs/R11_POST_CC_S0_S1_TODO.md) 中。本次只纠正接手去向，不撤销当时的执行计划或改写运行结果。

## 后续权限

S1 formal qualification、阶段结束和合并仍按自己的 Sol review/authorization 执行。公开框架、文档导航或绿色 smoke 均不自动授权历史 S2/S3、容量扩张、FINAL 或经济资格。

总体方向继续参考 [POST_CC_SCIENTIFIC_PROGRAM_R0](POST_CC_SCIENTIFIC_PROGRAM_R0.md)；不要从总体路线直接跳过当前任务与资格。文档组织和避免重复正文的规则见 [DOCUMENTATION_MAP](DOCUMENTATION_MAP.md)。
