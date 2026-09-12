# CC 并行执行与性能硬切换决策

日期：2026-09-12  
冻结基线：`main@89d62bf966f476e598f0e2f5c5e8e03c15a8db51`

## 1. 用户最新明确决定

用户要求把后续实现重新组织为 **CC 系列**，并把任务拆成四个可以同时执行的独立任务包，由四个 sub-agent 分别从同一冻结基线开始工作。

用户随后进一步明确性能方向：

- **性能优先**；
- **硬切换（hard cutover）**；
- **不做旧性能运行时兼容层**；
- **不保留 legacy performance fallback**。

这意味着 CC Thread D 可以建设新的数据布局、批处理、共享市场缓存、账户 worker、存储与原生热点实现，并在通过当前科学语义资格后直接成为 CC 的唯一高吞吐运行路径。旧 `gpu_inference_broker.py`、`multiprocess_trajectory_farm.py`、`vectorized_physics.py` 可以作为历史参考/benchmark baseline，但不是 CC runtime dependency。

## 2. 对旧任务管理与性能文字的替代范围

本决定只替代旧文档中的**任务路由/实施组织和性能迁移方式**说明，不追溯修改历史实验、旧 authority 或已记录科学结果。

下列旧表述从本决定起不再作为当前实施约束：

- “继续沿 AC TODO，不建立新的平行任务编号”；
- “BC Round 2 是当前唯一实施入口”；
- `PERFORMANCE_STRATEGY_3700X_1060.md` 中“性能工作不重开另一套并行实施体系”的旧路由说明；
- 将旧高吞吐模块作为需要兼容/适配后才能使用的默认路线；
- 以 AC/BC 串行 Gate 作为新 sub-agent 必须逐项等待的唯一执行方式。

`PERFORMANCE_STRATEGY_3700X_1060.md` 的测量方法、硬件约束、缓存/批量/并行/存储/Numba/Rust/Go 取舍原则继续有效；但其旧的 BC 路由与“不得新开并行系列”表述由本决定替代。Thread D 是这些性能原则的当前实施入口，并采用 hard cutover 而非兼容迁移。

当前实现入口改为：

- [CC 四线程总控 TODO](R11_CC_PARALLEL_HARD_CUTOVER_TODO.md)
- [Thread A — Runtime / Account](cc/CC_THREAD_A_RUNTIME_ACCOUNT.md)
- [Thread B — Policy / Learning](cc/CC_THREAD_B_POLICY_LEARNING.md)
- [Thread C — Experience / Economics](cc/CC_THREAD_C_EXPERIENCE_ECONOMICS.md)
- [Thread D — Performance Hard Cutover](cc/CC_THREAD_D_PERFORMANCE_HARD_CUTOVER.md)

AC/BC TODO 保留为设计来源、历史任务与已落地代码的追溯材料。

## 3. 没有被性能优先推翻的科学含义

“性能优先”不等于允许通过改变问题来获得吞吐。CC fast path 仍必须保持：

- 同一逻辑账户的时序连续和后果接续；
- signed account economics、负净值/负债和失败事实；
- 名义动作、许可和实际执行的分离；
- 真实 `log_mu` 与行为策略/RNG/代际来源；
- 失败/终态不因吞吐优化被丢弃；
- Frozen organs / trainable Brain 的梯度归属；
- 算术期望收益取向，不偷换成历史 log utility；
- FINAL 封存与禁止 fresh download；
- 不把固定 SL/TP/max-hold/cooldown 重新藏进执行器；
- 同一账户不乱序并发。

因此 CC 的“无兼容”指不兼容旧性能 API/运行拓扑，不指允许破坏当前已确认科学语义。

## 4. 四线程并行纪律

四个 sub-agent 都从 `89d62bf...` 开始，不互相等待、不互相 import、不 cherry-pick sibling branch。跨线程只共享 CC 总控文档冻结的 W-01..W-05 wire 语义。

四线程结束后才进行一次 integration join。integration 可以集中共享类型并把 A/B/C 接到 D fast spine，但不得在集成阶段静默改变某线程已经资格通过的科学语义。

本文件是 2026-09-12 用户最新任务组织和性能选择的记录。若与较早设计文档中的 AC/BC 路由语句冲突，以本文件和用户最新明确指令为准。