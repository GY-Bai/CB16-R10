# CB16 文档入口

本目录同时保存当前项目说明和历史运维/阶段文档。新接手者不必从聊天记录恢复整套项目，也不必把所有旧文档读入上下文。

## 阅读路线

| 你要了解的问题 | 入口 |
|---|---|
| 项目所有者想培养怎样的 Trader？ | [VISION](VISION.md) |
| 实施者应按哪些理念审阅和拆解任务？ | [PRINCIPLE_ALIGNMENT](PRINCIPLE_ALIGNMENT.md) |
| 各组件必须传递什么、如何判断对齐？ | [COMPONENT_REQUIREMENTS](COMPONENT_REQUIREMENTS.md) |
| **当前 Round-2 实施任务在哪里？** | **[BC-001–BC-105 Round 2 TODO](R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md)**；它以 `main@c373d23` 为审计基线，自包含当前返修、闭环、学习、评价和收尾任务 |
| AC-001–AC-058 现在是什么地位？ | [AC TODO](R11_ACTOR_CRITIC_CODE_ALIGNMENT_TODO.md) 保留为第一轮设计/任务历史与追溯材料；新的 BC Agent 不需要逐项恢复 AC 历史才能执行 |
| 经验怎样生成、复习、更新模型？ | [LEARNING_CONTRACT](LEARNING_CONTRACT.md) |
| 哪个具体算法、如何奖励、回放和归因？ | [TRAINING_ALGORITHM_R0](TRAINING_ALGORITHM_R0.md) |
| 如何验证算法与接口？ | [TRAINING_QUALIFICATION_R0](TRAINING_QUALIFICATION_R0.md) |
| 超额收益、风险、排行榜意味着什么？ | [EVALUATION_PRINCIPLES](EVALUATION_PRINCIPLES.md) |
| 每个组件负责什么，现有实现在哪里？ | [ARCHITECTURE_MAP](ARCHITECTURE_MAP.md) |
| main 与开发分支分别做到哪里？ | [CURRENT_STATE](CURRENT_STATE.md) |
| 哪些理念被修订，旧协议还在哪里？ | [DECISIONS](DECISIONS.md) / [PRINCIPLE_ALIGNMENT](PRINCIPLE_ALIGNMENT.md) |
| 哪些问题尚未得到用户决定？ | [OPEN_QUESTIONS](OPEN_QUESTIONS.md) |
| Agent 如何执行、核验和交接？ | [根目录 AGENTS](../AGENTS.md) |

## 文档地位

这些说明记录 2026-09-12 设计对话和当时核实的仓库状态。明确区分：**用户已确认目标、现有实现、为目标推导出的要求、待确认建议**。说明文档不伪造已经完成的协议迁移或科学结果。

用户当前指令定义本次任务；执行语义查适用的版本化 authority；实际实现和运行分别查精确 SHA 的代码与产物。出现冲突时记录具体差异，不能用其中一种材料代替全部事实。

当前实现计划以 **BC Round 2** 为入口。BC 把已经进入 main 的 AC R0 实现当作待审计/可复用输入，并允许通过版本化 successor 修复不足；它不会追溯改写旧 AC/R10/R11 证据。AC TODO 继续保留用于理解第一轮设计来源，但不是新 BC Agent 的逐项依赖清单。

## 历史资料如何使用

- [LOCAL_AGENT_EXECUTION_R10_2](LOCAL_AGENT_EXECUTION_R10_2.md) 是 R10.2 的精确执行说明。其中 Teacher-loss tournament、H72 和 split 不能自动代表 2026-09-12 的最终目标。
- `CB16_R11_STAGE4_*` 记录 R11 基础设施分工与集成。查当前 runtime 时，从 [组件地图](ARCHITECTURE_MAP.md) 沿链路进入。
- [OPERATIONS](OPERATIONS.md) 记录 Remote CI Relay 版本；不把其中旧命令当作所有 R11 科学任务的通用入口。
- 历史 FAIL 及其适用范围保留。新目标不追溯把旧 FAIL 改成 PASS，也不能把局部失败扩展成项目不可能成功。

## 更新约定

理念变化更新 VISION / DECISIONS；学习或评价语义变化更新对应规则；实现与运行推进更新 CURRENT_STATE / ARCHITECTURE_MAP。未决项确认后注明来源与日期，再调整状态，避免不同 agent 各自猜测。

理念要求使用 P 编号供任务引用；**BC 编号管理当前 Round-2 实施**。AC 编号保留历史追溯。新实施 Agent 以 BC TODO 的依赖和 Gate 为准，并在每项开始时现场核对 live main，而不是把文档快照当实时数据库。