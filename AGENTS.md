# CB16 agent 工作约定

本文件适用于整个仓库。主要交流语言为中文，代码标识和已有协议名称保持原样。

## 接手顺序

1. 阅读 [README](README.md) 与 [文档入口](docs/README.md)。
2. 阅读 [最高理念](docs/VISION.md)、[学习规则](docs/LEARNING_CONTRACT.md)、[评价规则](docs/EVALUATION_PRINCIPLES.md)。
3. 阅读 [CC 并行执行决策](docs/CC_PARALLEL_EXECUTION_DECISION.md) 与 [CC 四线程总控 TODO](docs/R11_CC_PARALLEL_HARD_CUTOVER_TODO.md)。
4. 只读取分配给你的 CC thread 包：
   - [Thread A — Runtime / Account](docs/cc/CC_THREAD_A_RUNTIME_ACCOUNT.md)
   - [Thread B — Policy / Learning](docs/cc/CC_THREAD_B_POLICY_LEARNING.md)
   - [Thread C — Experience / Economics](docs/cc/CC_THREAD_C_EXPERIENCE_ECONOMICS.md)
   - [Thread D — Performance Hard Cutover](docs/cc/CC_THREAD_D_PERFORMANCE_HARD_CUTOVER.md)
5. 按任务需要再读 [理念对齐规则](docs/PRINCIPLE_ALIGNMENT.md)、[组件要求](docs/COMPONENT_REQUIREMENTS.md)、[算法候选](docs/TRAINING_ALGORITHM_R0.md)、[资格计划](docs/TRAINING_QUALIFICATION_R0.md)、[性能策略](docs/PERFORMANCE_STRATEGY_3700X_1060.md)、[决策](docs/DECISIONS.md)、[未决问题](docs/OPEN_QUESTIONS.md)。
6. `CURRENT_STATE.md` 与 `ARCHITECTURE_MAP.md` 在 CC 冻结点仍含较早 AC/BC 状态描述；不要把其中的旧 SHA/“尚无某组件”句子当实时 authority。CC 总控文档已经把该文档漂移登记为待 integration 修复项。

## 当前任务组织

2026-09-12 用户明确建立 **CC 四线程并行执行**。冻结实现基线是：

`main@89d62bf966f476e598f0e2f5c5e8e03c15a8db51`

CC 规划/导航文档为了方便交接会在该 SHA 之后提交到 main；这不改变四线程的代码基线。Sub-agent 应从当前 main/交接上下文读取 CC 文档，但从 `89d62bf...` 创建 implementation branch，除非用户或 integration owner 明确重新冻结新的代码基线。

四个 sub-agent 不互相等待、不依赖 sibling branch、不互相 import、不 cherry-pick 对方代码。跨线程只共享 CC 总控文档冻结的 W-01..W-05 wire 语义。

AC/BC/R10/R11 历史代码、authority、receipt 和 scientific verdict 仍保留原身份。AC/BC TODO 现在是设计/实现历史和追溯材料，不是 CC sub-agent 必须串行执行的任务清单。

## CC branch 纪律

四个主分支固定建议为：

```text
ai/r11-cc-thread-a-runtime-r0
ai/r11-cc-thread-b-learning-r0
ai/r11-cc-thread-c-experience-r0
ai/r11-cc-thread-d-fast-cutover-r0
```

每个 thread：

- implementation code 从 `89d62bf...` 开始；
- 只修改自己 thread 文档声明的文件族；
- 不修改其他 thread 模块；
- 不修改共享 `CURRENT_STATE` / `ARCHITECTURE_MAP` / `README` / `AGENTS`；这些由四线程结束后的 integration 更新；
- thread receipt 必须记录 base/head SHA、测试、证据等级、owned files、FINAL/fresh firewall 和未决项；
- sibling branch 永远不是 authority。

## 性能硬切换

用户最新明确：**性能优先，CC 性能路径硬切换，不做旧性能运行时兼容，不保留 fallback。**

Thread D 的新 CC fast path 不得 import / wrap / fallback 到：

```text
cb16_local_opt.gpu_inference_broker
cb16_local_opt.multiprocess_trajectory_farm
cb16_local_opt.vectorized_physics
```

这些文件可作为历史参考或 benchmark baseline，但不是 CC runtime dependency。新的数据布局、缓存、批处理、CPU worker、IO、Numba/Rust/Go 局部实现不需要维持旧性能 API。

性能优先不能通过改变科学问题获得：账户连续、signed economics、真实 log_mu、失败记录、时序、FINAL/fresh firewall、当前 R1 action/economic semantics 仍必须守住。

## 必须保持的项目含义

- Trader 管理单资产账户，根据 Market、Account 和必要因果历史行动；不是纯市场预测器。
- 历史 Market 是可重复环境；完整经验来自动作、执行、账户演化和后续反馈。
- 允许反复学习同一批经验，也要求新政策继续产生新账户轨迹；复习不等于增加独立市场证据。
- 账户承担后续后果。chunk、暂停、checkpoint 换代和账户终止是不同事件。
- 风险取舍由模型学习。完整计入失败后，高爆仓但更高算术期望收益的策略允许获胜。
- 成功、普通、失败和终态事实都保留；优胜示范另筛；不得用幸存者池估计整体策略期望。
- 冻结市场器官与可训练 Brain stems 不同；梯度归属必须实际验证。
- Truth != Belief != Decision != Permission != Execution；requested_risk != confidence；观察投影 != 完整账本。
- 新经验可校准行为，但不能仅按年龄清除历史知识；禁止为此加入复杂手工周期/共振/规则激活系统。

## 目标、实现、证据与权限

| 材料 | 用途 |
|---|---|
| 用户当前明确指令 | 当前目标与授权，后续明确修订优先于旧任务管理文字 |
| 版本化 authority | 当前执行语义、数据范围和准入条件 |
| 精确 SHA 的代码 | 实际实现 |
| 同版本运行日志/receipt | 实际运行证据 |
| 设计建议/未决问题 | 不得冒充已授权科学结果 |

CC thread 的 CONTRACT/COMPONENT/CLOSED_LOOP/KNOWN_ANSWER/ECONOMIC/TRANSFER 证据必须分开。Thread D 另可报告 `PERFORMANCE_MEASURED`，但吞吐不是更高科学证据层级。

## 执行与证据纪律

- 开始时说明 thread、冻结 base、目标、owned files、预期证据。
- 发现冻结基线之后 main 又有别的 agent 更新，不自动吸收；CC thread 仍按 frozen base 执行，除非用户/集成负责人重新冻结基线。
- 不碰 FINAL，不下载 fresh market data。
- 不因性能删除失败、缩短后果窗口、重置账户、伪造 log_mu 或改变样本分布。
- 科学 run 开始后不通过改 T、seed、预算、sampling、E_ref、执行语义来救结果；新方法开新版本。
- 区分 `PASS`、`SCIENTIFIC_FAIL`、`EXECUTION_BLOCKED`、`HARDWARE_LIMIT`、`UNRESOLVED_OWNER_DECISION`。
- workflow success、loss 下降、权重变化或高 GPU 利用率均不能单独证明经济能力。
- 崩溃恢复幂等与跨代计划内重复学习是不同概念。

## 四线程后的 integration

四个 thread receipt 全部进入 main 后才进行 integration join。integration 负责：

- 统一 W-01..W-05 typed contract；
- B policy → A runtime；
- A facts → C store/replay/evaluator；
- 用 D fast spine 替换 reference transport/scheduling；
- 对照 A semantic oracle 证明 D fast path 正确；
- 运行 closed-loop/known-answer/performance qualification；
- 将 CC lane 硬路由到 fast path；
- 更新 `CURRENT_STATE`、`ARCHITECTURE_MAP`、`README`、`AGENTS`、BC 历史状态。

若 wire 语义冲突，integration 返回 `INTEGRATION_BLOCKED`，不能静默重解释某个 thread 的已通过结果。