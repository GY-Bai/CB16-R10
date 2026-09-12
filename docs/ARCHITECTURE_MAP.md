# 组件职责与实现地图

核对基线：main 代码 `47f1693709918978bf1e2119d77b611c01f5991f`，示范分支 `b299be68a9bd9cc041900db75586ed5667407f8a`。本文的目标职责不意味着所有连接已在生产路径接通。

## 1. 目标闭环

市场观察和账户观察进入策略；策略输出方向与风险请求；执行系统处理许可和成交，更新账户；新的市场与账户再进入策略。过程中积累轨迹，形成学习反馈，训练下一代。运行中的策略按用户要求保持 checkpoint 固定。

| 组件/职责 | 应负责 | 不应越界 |
|---|---|---|
| Market / 感知器官 | 形成决策当时可见的市场表示 | 不给策略泄漏未来，不拥有交易许可 |
| Account / Physics | 账本真值、成交后状态、合法恢复与终止 | 不为了方便训练伪造账户状态 |
| Central Brain | 结合市场、账户和可用历史输出动作 | 不把 requested_risk 解释为置信度 |
| Supervisor / 执行许可 | 将名义动作映射到实际允许的动作 | 不与策略意图、执行真值混为一体 |
| 轨迹生成与经验记录 | 记录状态、动作、结果与策略来源 | 不只留赢家而丢掉评价分母 |
| Teacher / 学习反馈 | 根据证据形成学习目标或价值反馈 | 不将一次 realized winner 视为唯一正确标签 |
| Student / 优化器 | 在明确梯度归属下更新可训练权重 | 不因冻结器官而误冻结 Brain stems |
| 比赛 / 评价 | 按明确人群、时域和目标比较策略 | 不将优化 loss 改善等同于经济目标改善 |
| Orchestrator / 存储 | 调度、幂等恢复、状态和版本连接 | 不通过重试重复更新或隐形补资 |

这些职责不要求新增同名服务。优先理解已有能力再决定组件修改。

## 2. 历史链中可直接定位的组件

| 路径 | 当前用途与阅读重点 |
|---|---|
| [SEMANTIC_FREEZE_V1](../authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json) | 冻结数据、器官、Tier-1 Brain、Teacher、Physics 和旧 utility 的具体语义 |
| [frozen_sensory_stack_r10.py](../cb16_local_opt/frozen_sensory_stack_r10.py) | 冻结感知栈 |
| [minute_sensory_adapter_r1.py](../cb16_local_opt/minute_sensory_adapter_r1.py) | 分钟输入与可见市场前缀 |
| [typed_central_brain_r10.py](../cb16_local_opt/typed_central_brain_r10.py) | 有类型的市场/账户输入、Brain stems、共享决策部分及输出 heads |
| [minute_physics_binding_r2.py](../cb16_local_opt/minute_physics_binding_r2.py) | `MinutePhysicsSessionR2`、账户观察、许可与执行、快照恢复 |
| [risk_supervisor_r1.py](../authority/control_plane_r1/risk_supervisor_r1.py) | 既有执行许可和风险限制 |
| [teacher_runtime_r11.py](../cb16_local_opt/teacher_runtime_r11.py) | Teacher 编译入口；按适用配置使用证据 |
| [training_runtime_r11.py](../cb16_local_opt/training_runtime_r11.py) | PreparedEvidence、损失、12 epochs 等当前配置、snapshot 幂等与训练 receipt |
| [training_integration_r11.py](../cb16_local_opt/training_integration_r11.py) | `IntegratedTrainingRuntimeR11` 与设备运行集成 |
| [stage4_canonical_runtime_r11.py](../cb16_local_opt/stage4_canonical_runtime_r11.py) | 生命周期与 provider 边界 |
| [stage4_runtime_spine_integration_r11.py](../cb16_local_opt/stage4_runtime_spine_integration_r11.py) | Stage-4 runtime 集成 |
| [stage4_state_roots_r11.py](../cb16_local_opt/stage4_state_roots_r11.py) | 持久状态身份与存储职责 |
| [stage4_execution_integration_r11.py](../cb16_local_opt/stage4_execution_integration_r11.py) | 许可和执行集成 |
| [stage4_recovery_integration_r11.py](../cb16_local_opt/stage4_recovery_integration_r11.py) | 恢复集成 |

### 器官与 Brain 的具体区别

当前 nominal 输入是 Operator48、Medium48、AccountState6。器官和账户编码没有梯度权限；输入之后的 Operator/Medium/Account stems 属于 Brain，是可训练参数。冻结协议的 Tier-1 网络共 189,052 参数，含六个训练归属组；它不是仅凭“decoder”一词即可推定的 Transformer。

Ordered4H30 是 task-gated shadow sidecar，不是 nominal Brain 输入；Remote OFF。AccountState6 是观察投影，不是完整账户，也不能未经证明就宣称足以表示所有策略历史。StrategyMemory 的具体表示和恢复仍见 [未决问题](OPEN_QUESTIONS.md)。

### 对旧示范续行的准确描述

[r11_longtraj_e6_feedback_binding_r2.py](../scripts/r11_longtraj_e6_feedback_binding_r2.py) 的 `_simulate_branch` 在首步使用候选动作，后续传入 FLAT/0。这个过程使用真实 Physics，但不等于每一步都重新调用学习策略。

FLAT/0 是名义输入，不能自动翻译为“立即平仓”；持仓情形下的许可、NOOP、退出由冻结执行语义决定。参见 [R10.2 执行说明](LOCAL_AGENT_EXECUTION_R10_2.md) 和实际 binding。

## 3. 原示范分支的历史快照

以下链接固定在 `b299be68a9bd9cc041900db75586ed5667407f8a`，不是 main 相对路径。随后部分通用模块已整合进入 main，不能从此表推断分支仍存在或其所有内容均已合并：

| 文件 | 已有能力及边界 |
|---|---|
| [demonstration_learning_r11.py](https://github.com/GY-Bai/CB16-R10/blob/b299be68a9bd9cc041900db75586ed5667407f8a/cb16_local_opt/demonstration_learning_r11.py) | 单一示范 corpus；train/validation 字段别名到同一 tensors；同 corpus 前后诊断。模块自身不生成新策略轨迹 |
| [continuous_generation_binding_r0.py](https://github.com/GY-Bai/CB16-R10/blob/b299be68a9bd9cc041900db75586ed5667407f8a/cb16_local_opt/continuous_generation_binding_r0.py) | 代际策略、账户与经验关系约束；有 binding 不等于生产循环完成 |
| [r11_demonstration_foundation_r0_h72_shard.py](https://github.com/GY-Bai/CB16-R10/blob/b299be68a9bd9cc041900db75586ed5667407f8a/scripts/r11_demonstration_foundation_r0_h72_shard.py) | 分片生成冻结 flat 母状态及九动作 H72 后果 |
| [r11_demonstration_production_materialize_r0.py](https://github.com/GY-Bai/CB16-R10/blob/b299be68a9bd9cc041900db75586ed5667407f8a/scripts/r11_demonstration_production_materialize_r0.py) | 校验并通过 S4F 发布精确字节，不晋升模型、不训练 |

## 4. 与目标对齐时应验证的行为

以下是目标推导出的未来验收方向，不是本轮授权执行的新实验：

- 相同市场、不同合法账户处境均可进入策略，输入和轨迹记录能证明差异被保留。
- 模型连续作多个动作，其账户后果影响后续观察；不能只看一次候选动作的 H72 分支。
- 计算边界和恢复不清空持仓、账户历史或策略所需记忆。
- 评价包含失败，不以优胜示范池代替全部策略结果。
- 相同 corpus 能复习，计划内新经历能加入，checkpoint 变化和下一代实际使用都有证据。
- 新长期目标与旧 utility 的差异被显式迁移，不通过改名掩盖。

实际已验证项和剩余缺口见 [CURRENT_STATE](CURRENT_STATE.md)。

## 5. 新 Actor–Critic 链的增量定位

核对快照：`2da90a0b4577ad7d950641fe0a1bcd81f08d58e2`。以上旧链的参数量、FLAT/NOOP 与 Teacher 语义不能不加区分地套用到以下新链。组件应满足的目标要求见 [COMPONENT_REQUIREMENTS](COMPONENT_REQUIREMENTS.md)。

| 路径 | 已有职责；不等同于完整闭环 |
|---|---|
| [actor_critic_contract_r0.py](../cb16_local_opt/actor_critic_contract_r0.py) | 科学版本、身份与语义哈希 |
| [actor_critic_runtime_router_r0.py](../cb16_local_opt/actor_critic_runtime_router_r0.py) | 新旧运行协议显式区分 |
| [action_contract_r0.py](../cb16_local_opt/action_contract_r0.py) | 目标方向/风险、转换表示与行为来源 |
| [execution_record_r0.py](../cb16_local_opt/execution_record_r0.py) | 名义请求、许可、执行记录分离 |
| [actor_critic_supervisor_r0.py](../cb16_local_opt/actor_critic_supervisor_r0.py) | 持仓调整许可、先平后开反转契约 |
| [target_exposure_r0.py](../cb16_local_opt/target_exposure_r0.py) | 合法名义额度比例到目标数量的机械映射 |
| [continuous_generation_binding_r0.py](../cb16_local_opt/continuous_generation_binding_r0.py) | 复用的代际/账户来源约束 |

此快照尚无新 Physics adapter、随机 Actor、连续政策 Collector 和 V-trace learner 对应实现文件。这里说明新链集成缺口，不否认历史链已有其他执行/训练模块。代码、组件测试、真实接线和经济结果分别认定。

提交前补核对：main [`7c412ee`](https://github.com/GY-Bai/CB16-R10/commit/7c412ee264c463e8f9f28c707c02eccfa4ab055c) 已新增 [AC-014 adapter](https://github.com/GY-Bai/CB16-R10/blob/7c412ee264c463e8f9f28c707c02eccfa4ab055c/cb16_local_opt/actor_critic_physics_adapter_r0.py)，补上目标仓位操作；其函数不推进 bar，且保留旧 SL/TP 与 max-hold 生命周期。上述缺少 adapter 的描述仅适用于 AC-013 快照；剩余理念差异见 CURRENT_STATE，不把主动仓位操作误当成完整无人工交易偏好的闭环。
