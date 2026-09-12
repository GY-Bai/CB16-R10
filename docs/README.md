# CB16 文档入口

本目录同时保存当前项目说明和历史运维/阶段文档。新接手者不必从聊天记录恢复整套项目，也不必把所有旧文档读入上下文。

## 阅读路线

| 你要了解的问题 | 入口 |
|---|---|
| 项目所有者想培养怎样的 Trader？ | [VISION](VISION.md) |
| 最新任务组织和性能决策是什么？ | **[CC 并行执行决策](CC_PARALLEL_EXECUTION_DECISION.md)** |
| 当前实施总控在哪里？ | **[CC 四线程并行 Hard-Cutover TODO](R11_CC_PARALLEL_HARD_CUTOVER_TODO.md)**；四线程冻结实现基线为 `main@89d62bf966f476e598f0e2f5c5e8e03c15a8db51` |
| Thread A 做什么？ | [Runtime / Account / Continuous Interaction](cc/CC_THREAD_A_RUNTIME_ACCOUNT.md) |
| Thread B 做什么？ | [Policy / Critic / Learner / Retention](cc/CC_THREAD_B_POLICY_LEARNING.md) |
| Thread C 做什么？ | [Experience / Replay / Economic Evaluation](cc/CC_THREAD_C_EXPERIENCE_ECONOMICS.md) |
| Thread D 做什么？ | [Performance Hard Cutover / High-Throughput Spine](cc/CC_THREAD_D_PERFORMANCE_HARD_CUTOVER.md) |
| 实施者应按哪些理念审阅任务？ | [PRINCIPLE_ALIGNMENT](PRINCIPLE_ALIGNMENT.md)；其中旧 AC/BC 任务路由语句已被 CC 决策替代，科学 P-01–P-10 仍有效 |
| 各组件必须传递什么、如何判断对齐？ | [COMPONENT_REQUIREMENTS](COMPONENT_REQUIREMENTS.md)；旧 AC 导航仅作历史映射 |
| BC-001–BC-105 现在是什么地位？ | [BC Round 2 TODO](R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md) 保留为 Round-2 串行设计/落地历史；CC 不要求 sub-agent 逐项等待 BC Gate |
| AC-001–AC-058 现在是什么地位？ | [AC TODO](R11_ACTOR_CRITIC_CODE_ALIGNMENT_TODO.md) 保留为第一轮设计与任务历史 |
| 经验怎样生成、复习、更新模型？ | [LEARNING_CONTRACT](LEARNING_CONTRACT.md) |
| 哪个具体算法、如何奖励、回放和归因？ | [TRAINING_ALGORITHM_R0](TRAINING_ALGORITHM_R0.md) |
| 3700X / 1060 上怎样优化性能？ | [PERFORMANCE_STRATEGY_3700X_1060](PERFORMANCE_STRATEGY_3700X_1060.md)；CC 的最新 owner decision 是性能优先、硬切换、无旧性能 fallback，具体执行见 Thread D |
| 如何验证算法与接口？ | [TRAINING_QUALIFICATION_R0](TRAINING_QUALIFICATION_R0.md) |
| 超额收益、风险、排行榜意味着什么？ | [EVALUATION_PRINCIPLES](EVALUATION_PRINCIPLES.md) |
| 每个组件历史/现有职责在哪里？ | [ARCHITECTURE_MAP](ARCHITECTURE_MAP.md)；注意其当前实现快照落后于 CC 冻结基线 |
| 旧 current-state 快照是什么？ | [CURRENT_STATE](CURRENT_STATE.md)；其中部分 AC/BC 状态描述已过时，CC master 已登记为 integration 文档修复项 |
| 哪些理念被修订，旧协议还在哪里？ | [DECISIONS](DECISIONS.md) / [PRINCIPLE_ALIGNMENT](PRINCIPLE_ALIGNMENT.md) / [CC 决策](CC_PARALLEL_EXECUTION_DECISION.md) |
| 哪些问题尚未得到用户决定？ | [OPEN_QUESTIONS](OPEN_QUESTIONS.md) |
| Agent 如何执行、核验和交接？ | [根目录 AGENTS](../AGENTS.md) |

## 当前实施组织

CC 把后续工作拆成四个真正可以同时启动的 task package。四个 sub-agent 的**实现代码**都从冻结 SHA `89d62bf...` 开工，只共享 CC master 中的 W-01..W-05 wire 语义，不依赖 sibling branch。

CC 任务文档本身提交在 `89d62bf` 之后，以便直接从当前 main 阅读；这不改变四线程的代码基线。正确做法是从当前 main/交接上下文读取任务文档，再从 `89d62bf` 建 implementation branch。

四线程结束后再做一次 integration join。integration 统一 typed interface、连接 A/B/C，并将 D 的 fast spine 作为 CC 唯一性能运行路径。旧性能 broker/farm/vectorized runtime 不做 CC compatibility/fallback。

## 文档地位

明确区分：**用户已确认目标、实现事实、由目标推导的要求、算法候选、运行证据和历史任务管理文字**。最新用户指令可以替代旧的任务组织方式，但不会追溯改写历史 scientific verdict 或 frozen authority。

CC 的性能硬切换只替代旧性能 API/运行拓扑，不授权改变账户连续性、signed economics、真实 `log_mu`、失败记账、算术期望收益目标或 FINAL/fresh-data 边界。

## 历史资料如何使用

- AC TODO：第一轮 Actor-Critic 对齐设计。
- BC TODO：Round-2 返修/闭环的串行规划以及已经推进的 BC-001..037 历史来源。
- `CB16_R11_STAGE4_*`：基础设施/authority/persistence/fencing 的历史与复用依据。
- [LOCAL_AGENT_EXECUTION_R10_2](LOCAL_AGENT_EXECUTION_R10_2.md)：旧 R10.2 精确执行说明；Teacher loss、H72、log utility 不能自动代表当前 CC 目标。
- [OPERATIONS](OPERATIONS.md)：历史运维说明，不是所有 CC thread 的统一 runtime API。

历史 FAIL 及适用范围保留。新目标不追溯把旧 FAIL 改成 PASS，也不能把局部失败扩展成整个项目不可能成功。

## 更新约定

四个 CC thread 在执行过程中不修改共享 `CURRENT_STATE`、`ARCHITECTURE_MAP`、`README`、`AGENTS`，避免并发冲突。各 thread 只交自己 receipt/status。四线程合并后由 integration 根据实际代码/receipt 一次性更新共享管理文档。

科学理念继续使用 P 编号追溯；当前实现任务使用 CC Thread A/B/C/D 编号。