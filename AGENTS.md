# CB16 agent 工作约定

本文件适用于整个仓库。先理解项目目标和当前断点，再执行用户本次指定的任务。主要交流语言为中文，代码标识和已有协议名称保持原样。

## 接手顺序

1. 阅读 [README](README.md) 与 [文档入口](docs/README.md)。
2. 阅读 [最高理念](docs/VISION.md)、[学习规则](docs/LEARNING_CONTRACT.md)、[评价规则](docs/EVALUATION_PRINCIPLES.md)。
3. 阅读 [当前状态](docs/CURRENT_STATE.md)；现场核对目标分支 SHA、相关 Actions、authority 和 receipt。状态文档是带日期的快照，不是实时数据库。
4. 按本次任务阅读 [组件职责](docs/ARCHITECTURE_MAP.md)、[决策记录](docs/DECISIONS.md)、[未决问题](docs/OPEN_QUESTIONS.md)，再查对应实现。训练设计另读 [算法候选 R0](docs/TRAINING_ALGORITHM_R0.md) 和 [资格计划](docs/TRAINING_QUALIFICATION_R0.md)。
5. 当前 Round-2 实施对齐时，阅读 [理念对齐规则](docs/PRINCIPLE_ALIGNMENT.md) 与 [组件要求](docs/COMPONENT_REQUIREMENTS.md)，然后进入 **[BC Round 2 TODO](docs/R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md)**。该文档是当前自包含实施计划；[AC TODO](docs/R11_ACTOR_CRITIC_CODE_ALIGNMENT_TODO.md) 保留为第一轮设计/任务历史和追溯资料，不是新的 BC Agent 必须逐项恢复的依赖清单。
6. 不要求通读全部历史文件。沿当前 BC 任务的调用链、理念 P 编号和证据索引读取必要材料。

## 当前协作与任务命名

2026-09-12 用户明确建立 **BC Round 2** 作为新的实施任务系列。BC 以 live main 已落地的 AC R0 代码作为待审计、可复用、也允许被版本化修复的输入；新执行 Agent 可以按 BC 自己的 Gate 和依赖工作，不需要先掌握 AC-001–AC-058 每个任务的历史细节。

AC/R10/R11 的历史代码、authority、receipt 和 scientific verdict 仍保持其原身份。BC 不追溯把旧结果改成 PASS，也不能为了实现方便静默改变旧 science identity。若当前理念要求与 R0 实现冲突，优先通过版本化 successor / compatibility adapter / 新 qualification 解决，并保留 R0 regression。

理念要求、算法候选、实现事实和运行证据分别记录。模型名称或对话分工不自动授予权限；用户当前任务授权和适用 authority 仍是工作边界。

## 必须保持的项目含义

- Trader 在单资产账户上根据 Market、Account 和必要历史作出动作。Account 是核心状态，不能把任务缩成纯市场预测。
- Market 历史是可重放环境；完整经验通过动作、执行和账户演化产生。既要反复学习已有经验，也要让行为产生新的连续轨迹。
- 同一批经验反复训练、重复历史市场、重叠片段，均可属于学习。不能擅自用静态训练/测试划分取代整个学习闭环。
- 复习不等于已覆盖全部可达账户状态；拟合固定示范不等于实现自主连续 rollout。复用经验也不增加独立市场证据数量。
- 账户承担后续后果。计算片段、checkpoint 换代和账户终止是不同事件；72 小时不是用户规定的账户寿命。
- 风险取舍由模型学习。用户接受在完整计入失败后、长期期望收益更高的高风险策略获胜。不得私自改成“最少爆仓优先”、Sharpe 优先或对数效用优先。
- 已发生的失败必须计入所声称的整体策略评价。用户已批准保留成功与失败完整经历、优胜示范另筛，并允许失败参与长期后果学习；不要重复请求这一原则的确认。具体算法候选不等于已执行或已资格认定。
- 冻结器官和可训练 Brain stems 不同。当前 nominal Brain 的结构以版本化合同为准；“central brain / decoder”不意味着代码必然是 Transformer decoder。
- Truth != Belief != Decision != Permission != Execution。requested_risk != confidence。观察投影不是完整账本；一次实现收益不是正确动作标签。
- 新经验可以校准当前行为，但不能通过简单的时间过期规则永久抹除仍有用的历史知识；也不得为此加入复杂手工周期/共振/规则激活系统。

## 目标、实现、证据与权限

这几类材料回答不同问题，不能机械地以“最新文件胜出”处理：

| 材料 | 用途 |
|---|---|
| 用户当前明确指令和已确认理念 | 本次工作范围、目标与偏好；明确的后续修订优先于旧的项目意图描述 |
| 对当前任务适用的版本化 authority | 当前执行语义、冻结参数、数据范围和准入条件 |
| 精确 SHA 下的代码 | 实际实现了什么 |
| 同一版本的运行日志、receipt 和产物 | 实际运行了什么、证明了什么 |
| 文档中的建议和未决项 | 供后续设计讨论，不能冒充已授权实现 |

发现用户新目标与旧 authority 冲突时，具体记录字段、版本和影响，按已授权任务推进。不要静默改冻结文件来让检查变绿，也不要用旧协议否认用户的新目标。新实验如需改奖励、时域、Teacher/Critic、执行语义或晋升规则，应有明确版本与验证范围。

用户已授权的工作继续完成；不要对同一授权反复请求确认。尚未决定的科学选择写入 OPEN_QUESTIONS，只在它实际影响当前任务时提问。不得将本文件解释为额外的通用审批流程。

## BC Round 2 执行纪律

- 每个 BC 任务开始时现场核对 live main；不得把 `c373d23` 或任何文档快照永久当作 HEAD。
- 推荐分支 `ai/r11-bc-<NNN>-<short-name>-r0`；只依赖已合并到 main 的显式前置 BC 任务。
- 未合并 AC/BC sibling branch 不是 authority，也不得作为隐藏依赖。可以参考候选实现，但必须重新审查并绑定当前 main。
- 语义变化必须 bump science identity 或使用显式版本化 successor；不要在同一 semantic version 下悄悄换含义。
- 当前 BC TODO 把组件/closed-loop/known-answer/economic/transfer 证据分层；不得把单元测试 PASS 或 workflow success 直接写成经济能力改善。
- BC 的第一阻塞目标是正确执行与完整账户经济后果；在 BC-A 未通过前，不生成或承认 canonical Round-2 replay 经验。

## 执行与证据纪律

- 开始时简要说明：目标、分支/SHA、已知完成项、缺口、本轮范围、完成证据。
- 精确区分 main、未合并开发分支、计划和已运行产物；发现其他 agent 更新时先核对再写，保留其无关改动。
- 只提交本任务涉及的内容，不为无关任务顺手合并其他分支或重写冻结 authority。
- 代码仓库保存源代码、文档和脱敏小型证据；不提交数据、权重、checkpoint、密钥或运行缓存。遵循现有仓库检查。
- 当前 FINAL 自 2025-09 起的封存及运行时禁止 fresh download 的边界继续适用；用户设想的未来模拟账户阶段不自动开放这些边界。
- 冻结实验的问题、输入、指标和判据后再运行。发现新方法开新版本，保留原结果，不用改阈值挽救已观察的 FAIL。
- 区分执行失败、有效执行下的科学失败、硬件限制、未决 owner decision 和未验证。workflow success、训练 loss 下降、权重变化均不单独证明交易能力。
- 崩溃恢复幂等与跨代计划内回放是不同要求；不得用“允许复习”为重复提交同一事务辩护。
- 不在缺少产物时声称已下载或独立验证哈希。引用 receipt 的哈希与自行验证字节应分别表述。

## 文档维护与完成条件

- 行为或状态发生实质变化时，同步更新相关文档；CURRENT_STATE 写检查日期、代码 SHA、证据和剩余缺口。
- 新决定记录确认来源和替代范围；未得到确认的建议保留状态。历史 authority 和 verdict 保持可追溯。
- 文件名、内部链接与实际路径一致。既有目录为 `docs/`，不另建重复的 `doc/` 导航树。
- 文档改动检查链接、措辞、diff 和仓库策略；不为低风险文字修改编写镜像测试或启动科学训练。
- 最终报告实际改动、提交/分支、验证结果、未完成事项。给出代码或文档链接；不只交付计划。

## 性能改动的理念边界

涉及性能阻塞时参考 [3700X / 1060 性能策略](docs/PERFORMANCE_STRATEGY_3700X_1060.md)：先测量，再按重复计算、批量/缓存、并行和编译热点逐项处理。不能以提速为由改变账户连续性、失败记账、动作概率、后果时域或当前科学版本。Rust/Go 是有实测依据时的局部实现选择，不是默认全仓库重写路线；性能文档本身不启动新的实施系列。
