# 未决问题与建议

截至 2026-09-12，CC R11 integration 已以 PR #99 合入 `main`。当前机器可读 authority 为：

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

本页只保留**仍需要项目 owner 决定、且不能由实现 Agent 自行推导**的问题。已经被 CC 合同、qualification 或工程实现关闭的旧问题，不再继续列作 owner-open。

## O-01 Buy-and-Hold 与 FLAT 的 master precedence — 设计关闭，实施待迁移

已明确：

- 经济方向仍以长期算术期望收益为主；完整失败进入分母。
- Buy-and-Hold 与全程 FLAT 都是正式 component baseline。
- Thread C 已实现各 component outcome 的独立计算与报告。
- Integration 原始 receipt 保留当时的未决状态；它是历史事实，不因本轮后继设计被改写。

2026-09-12 用户要求本对话表态并落实，确定后续沙盒设计：**不设 baseline 优先级；同合同按完整算术收益排名；满足预登记证据要求的候选相对改善可晋升沙盒学习冠军，不以同时胜过双基准为额外门槛。** 详见 [经济排序与晋升](ECONOMIC_ORDERING_AND_PROMOTION.md)。

这项选择由既有目标及本次设计委托落实，不把助手选定的统计超参数伪称为用户逐项指定。双基准数值/符号、模型排名、晋升分别报告；不把 sandbox champion 变成部署许可。

当前没有因此需要再次提交 owner 的原则选择。旧 `assess` 仍返回 unresolved 的行为属于 S0 后继语义迁移任务；不得靠重问用户代替修补。精确参数由 [后 CC 科学计划 R0](POST_CC_SCIENTIFIC_PROGRAM_R0.md) 的执行 spec 在开跑前冻结。

## 已不再属于 owner-open 的旧问题

下列内容仍可能在未来实验中成为**预注册参数或版本化工程选择**，但不再是当前 CC canonical 运行的未决 owner 原则：

- 失败轨迹是否允许学习：已关闭，允许；完整事实保留，失败可参与长期后果学习。
- 持仓后能否主动减仓/退出/反转：CC action/runtime 已实现并完成 synthetic qualification。
- 固定 SL/TP/max-hold/cooldown 是否作为 canonical autonomous policy：已关闭为否；不能作为隐藏策略偏好重新进入 canonical CC。
- 账户是否跨 chunk/pause/checkpoint generation 持续：已关闭为是，并已通过 closed-loop/recovery qualification。
- stochastic Actor、true `log_mu`、Critic/V-trace、exactly-once learner update 是否已接线：已完成 synthetic closed-loop qualification。
- 四线程与性能 hard cutover 的组织方式：已完成并合入 main；legacy broker/farm/vectorized runtime 不属于 canonical CC fallback。
- 精确历史经济实验的 horizon `T`、cohort weighting、训练 budget、replay mixture、sequence length 等：这些在相应科学 run 前按该 run 预注册，不在本页提前固定，也不因此阻塞当前 canonical runtime。
- policy memory 是否最终需要：由后续 state-sufficiency / representation evidence 驱动；不是当前必须由 owner 主观选择的原则问题。
- 真实模拟账户的更新频率、晋升、回滚和部署权限：属于未来阶段 authority；当前 CC integration 不自动授权 fresh data、FINAL 或实盘。

## 问题提交纪律

Agent 只有在某个选择会改变用户已经确认的目标、经济 master ranking 含义或未来部署权限时，才应把问题重新提交给 owner。

普通接口编码、性能拓扑、模型容量、batch、worker、训练预算等工程/实验参数，在已有语义边界内由任务自己预注册和验证；不得把常规实现选择伪装成需要 owner 反复批准的问题。
