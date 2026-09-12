# 未决问题与建议

截至 2026-09-12，CC R11 integration 已以 PR #99 合入 `main`。当前机器可读 authority 为：

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

本页只保留**仍需要项目 owner 决定、且不能由实现 Agent 自行推导**的问题。已经被 CC 合同、qualification 或工程实现关闭的旧问题，不再继续列作 owner-open。

## O-01 Buy-and-Hold 与 FLAT 冲突时的 master precedence — 未决

已明确：

- 经济方向仍以长期算术期望收益为主；完整失败进入分母。
- Buy-and-Hold 与全程 FLAT 都是正式 component baseline。
- Thread C 已实现各 component outcome 的独立计算与报告。
- Integration 明确禁止在两个 baseline component 发生冲突时自行制造统一 master winner。

仍待 owner 决定：

> 当 Buy-and-Hold 与 FLAT component outcomes 给出互相冲突的胜负时，哪个 baseline 拥有 master precedence，或者是否采用另一个显式组合规则？

在此决定出现前：

- evaluator 可以分别给出 B&H delta 与 FLAT delta；
- component 级 measurement 可以完成；
- promotion/master-winner 层必须返回 `UNRESOLVED_OWNER_DECISION` 或等价 fail-closed 状态；
- 不得事后选择更容易获胜的 baseline；
- 不得用 Sharpe、log-growth、drawdown 或 survival 替代这项 owner decision。

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
