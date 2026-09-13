# S0/S1 TODO 可执行性审阅

审阅日期：2026-09-13。审阅 main：`8d37166f9fcc87d61affaf1319afe225d8754d1b`。只读核对 S1 分支：`9d9911432d1da419b9957d857867cf3c5c8d9d3d`。角色：理念/文档审阅；未运行 S1 学习、未修改实现或冻结 authority。

## 1. 结论

**S0 的迁移实现及资格已有证据；S1-A/B 的开发拆解具备可执行基础。S1-C/D 尚有科学定义与运行配置缺口，不能仅凭 S0 的 FROZEN/PASS 标签认定正式多种子资格已完全可复现。**

无需重拆 S0/S1，也不能把后面的 learner/runner 尚未实现当成全项目阻塞。当前应继续 S1-002 起的开发，同时在读取正式学习结果前落实下表指出的任务具体化。角色规范见 [S 系列 TODO 撰写原则](../S_SERIES_TODO_AUTHORING_PRINCIPLES.md)。

本审阅不撤销 S0 历史 PASS；它区分了“冻结文件/合同通过现有检查”和“所有科学运行自由度已具体冻结”。补充工作沿用原任务编号，通过显式后继 manifest/兼容性说明完成，不能偷偷改写旧 registry、run spec 或 receipt。

## 2. 实际核对的证据

- S0 qualified implementation：`5760061d6c274e9f8796e6608e86bb173018148f`。
- S0 receipt-bearing handoff / S1 working base：`ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f`。
- [S0 workflow 34721332483](https://github.com/GY-Bai/CB16-R10/actions/runs/34721332483) 为 success，head 对应上述 qualified implementation；receipt 记录 15 个 focused tests 通过。
- 核对 S0 inventory 与 receipt 引用的十项当前文件 blob，包括旧 CC receipt、经济材料、S1 registry/run spec，均与登记身份一致；未发现本次所查冻结文件被改写。
- [S1 分支相对交接基线的比较](https://github.com/GY-Bai/CB16-R10/compare/ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f...9d9911432d1da419b9957d857867cf3c5c8d9d3d) 虽有七次提交，净差异仅新增 S1 baseline JSON。所查分支 Actions 是 repo-guard；没有可据此确认的 S1 主体实现或学习资格结果。

因此“当前 ACTIVE”表示已授权工作，不表示 durable replay、joint learner 或科学任务已完成。此结论限于上述 SHA 与所查分支，不声称检查了其他未公开工作区。

## 3. 拆解中应保留的正确方向

S1-A 持久数据 → S1-B 联合学习 → S1-C 受控任务 → S1-D 运行 → S1-E 收尾，依赖主线清楚。区分 rollout 私有内存与可独立恢复的持久经验、名义联合动作与真实 log_mu、完整账户接续与换代，均直接对应当前科学缺口。

“待实现不是 EXECUTION_BLOCKED”“有效科学 FAIL 可以作为阶段结论”“S1 结束不自动启动历史阶段”也应保留。需要修补的是具体合同与判定的充分性，不是另造一套架构或更强措辞要求执行。

## 4. 发现与修补位置

### R-01：任务 ID 已冻结，但环境与验收尚未充分数值化

**影响：正式科学运行前必须补齐；不阻塞独立的数据/learner 开发。**

S0-012 要求每任务冻结 generator、horizon、budget、evaluation schedule、success、controls。实际 `CB16_R11_POST_CC_S1_TASK_REGISTRY_V1.json` 提供任务名、版本名及自然语言语义，但没有足够信息独立生成账户环境与评价：例如转移/概率参数、episode/phase 长度、可达账户集合、输入字段顺序/归一化、控制间差异阈值等。

具体示例：

- `positive_learning_must_exceed_shuffled_credit_control=true` 未定义比较幅度、抽样/聚合与容差。
- A→B→A 的 `returning_A_must_improve_over_pretraining_baseline=true` 未固定立即 retention 还是有预算的 relearning，也未给完整 phase/evaluation 日程。
- 通用 oracle-gap 0.5 未完全定义每任务 oracle 的约束、分母与零分母处理。

`post_cc_s0_qualification_v1.py::compile_s0_gates` 对 registry 主要检查 status、任务集合、seed/4-of-5 等；negative-control gate 检查 control ID 存在；这些检查通过不等于生成器与科学判据已完整。

**落点：**在 S1-013..020 中形成具体任务 manifest 与独立 oracle/控制验证，并在 S1-021 正式运行前绑定。兼容既有冻结语义的补充明确引用父 hash；若改变已冻结科学值，则发布后继版本，保留旧结果，不覆盖 S0 receipt。

### R-02：run spec 更接近框架约束，尚不足以限定训练成本与重复性

**影响：S1-B 实现需明确 consumer 含义，S1-D 运行前需完整预算。**

当前 spec 固定 SGD/0.01、模型与五个 seed、每任务每种子最多 100,000 次策略决策，却未明确 batch/sequence、每次收集后的更新次数、replay 选择及复用上限、评价日程、具体 wall-clock/RAM/VRAM/存储边界。五任务×五种子仅正任务采集上限就达 2,500,000 次决策，尚未计对照、评价与反复梯度更新。

另外，初始模型明确固定 seed 1701 与同一个参数 hash；五个 seed 的环境、动作、replay、评价分工没有完整说明，不能一边随机初始化五次，一边要求全部匹配同一个 checkpoint hash。

**落点：**S1-007..011 明确 replay probability 与 loss weight 的角色、bootstrap/序列语义；S1-021..023 补齐解析后的运行 manifest 和种子派生规则。未知资源可先有界工程 canary，不依据收益选择预算。初始化与冻结值保持一致，若改变须新版本。

### R-03：负控的名称不保证它检验的关系成立

**影响：可能制造假 FAIL 或假 PASS，必须在正式训练前验证控制设计。**

S1-018..020 及 registry 使用“no-signal or zero-reward”“materially outperform shuffled”“random/impossible target where applicable”。需进一步明确：

- 零奖励对照不能直接用自身为零的 oracle gap 作改善分母；需说明在什么固定评分条件下测虚假成功。
- shuffle 必须实际破坏所检验的条件信用关系；保留了行动信息的打乱不必然使学习失败。
- 有限随机标签若反复训练、原样评价，可以被背会；不能把这一现象自动判作泄漏或要求禁止复习。
- hash/log_mu 破坏是完整性攻击，和学不到信号的统计负控分别判定。

**落点：**保留现有三类 controls，具体化训练/评价分布、变换、适用条件、预期差异与随机误报处理。不要临时强加“每个负控所有种子都绝不偶然成功”这一未登记要求。

### R-04：已知答案必须适配冻结的小模型与观察信息

**影响：可学性题目可能与其输入/政策类不相容，不能归咎工程执行。**

S0 冻结的 Actor 为 241 个可训练参数、16 个冻结参数，Critic 为 73 个参数；`CCCentralBrain` 是无 recurrent state 的前馈类。这不是用户口头的约 16K，也不是本轮允许做容量搜索的理由。

尤其 A→B→A：若 A/B 输入完全相同、最优动作相反，又要求无需再学习立即识别返回 A，则冻结 Actor 无从判断。当前 registry 尚未证明采用了这种不可解定义，也尚未给出足够细节排除它。账户消融还需检查 Execution 等通道有无等价答案泄漏。

**落点：**S1-013..017 先证明 oracle 在给定因果观察和联合动作空间中的适用性；明确 retention 或 relearning 的判据和预算。自然可见状态信息可以使用，手工 regime detector/激活规则不能偷渡。不能结果失败后才加 task ID、记忆或更大网络。

### R-05：正文与机器字段、当前状态仍有交接歧义

**影响：可能使执行者错误回退基线或重复做 S0。**

- 总控/S1 正文明确 working base 为 `ec30185...`，run spec 的 `s1_base_rule` 却写 `EXACT_S0_QUALIFIED_HEAD_FROM_...`。若 consumer 把它解释为 checkout 位置，会退回 `5760061...`。两者可通过“合格身份”和“工作交接身份”分别绑定解决，不应由同一字段暗指两个对象。
- 本次 main 的 S0 TODO 顶部仍 `OPEN`，`CURRENT_STATE` 仍有 S0 待迁移措辞，而总控已 `S0 PASS / S1 ACTIVE`。

**落点：**只修当前导航状态；冻结 run spec 不改字节。S1 baseline 已分别记录两 SHA，后续 runner 使用它们并验证谱系、receipt/hash；不得仅凭旧字段退回。若需改变机器字段语义，显式补充后继绑定，不假称旧文件早已统一。

### R-06：S1 新文件建议容易被执行成重复定义已有合同

**影响：可增加两套同名类型/哈希语义，不需要为此停工。**

S0 已实现 `post_cc_observation_contract_v1.py::PostCCObservationFactV1` 和 `post_cc_joint_replay_contract_v1.py::PostCCJointReplaySampleV1`；S1-002/005 又要求在新的建议 surface 中“implement”同名对象。应明确复用合同类，新增 store/materializer/严格验证层；不能自动创建同名但字段、codec 或 support 校验不同的合同。

**落点：**S1-002..007 在实现记录里说明复用/适配位置。现有 `sampling_probability_or_weight` 等模糊 consumer 语义也须明确，不能由存储端和 learner 各猜一个解释。

### R-07：判定器与正式运行存在先后歧义

**影响：若严格逐波顺排，可能看完结果才实现 gate。**

S1-D 要执行正式 workload，S1-E/S1-032 才列 gate compiler。判据在文字中冻结仍不足以保证运行后不会由实现细节重新解释。

**落点：**保持 task 编号，但把 S1-032 的 predicate 实现及预构造结果验收作为正式运行前依赖；真实结果汇总、artifact 与 receipt 仍在 S1-E。不是让所有后续工作提前完成，而是先确定怎样判分。

### R-08：进度要求应看净产物，不能看 commit 次数

**影响：过强措辞不能替代缺失合同，也可能把无效提交当进度。**

S1 的七个 ahead commits 净变化只有 baseline。不能由“已经 commit/push”推断 S1-A 已推进；也不能把合格的数学/文档审阅任务强行要求为代码 commit。执行角色在明确任务下应继续实现，审阅角色则产出可落实的修订和证据。

**落点：**沿用 S1-001..035，通过“净实现 + 相关测试/产物 + 最早未完成项”交接。当前所查 head 下一实际实现仍是 S1-002..007。

## 5. 建议执行顺序，不另开平行任务体系

1. 继承 `ec30185...` 及 S0 PASS；继续已有 S1 分支，不回滚、不重复迁移 S0。
2. S1-A/B 复用 S0 类和 canonical CC，补持久化/物化/联合 learner；独立的具体任务、oracle 和负控定义可同步完善。
3. 在正式学习结果被用于选择之前，发布具体化 manifest、兼容性与预算/种子/判分绑定。若已经看过结果，则按实际历史标为开发探索，使用新版本确认，不能洗成“尚未观察”。
4. 正式多种子运行前固定判定器；运行后按同一判据输出完整结果。出现有效 FAIL 可完成该次科学任务，不做原地 rescue。

这些缺口目前不需要 owner 再选择交易目标。它们属于 TODO 作者/执行者应完成的科学具体化与工程约定；实质改变冻结科学含义时才按后继版本处理。本文未赋予审阅者篡改旧 authority 的权限。

## 6. 本轮交付与边界

新增角色原则与本审阅，修正相关导航的 S0/S1 状态和引用。没有修改冻结 registry/run spec、历史 receipt、学习代码或 S1 分支，没有触发 Shanxi 训练。

审阅结论限于可执行性和抽查的契约消费者；不是全面代码正确性审计，不保证任务具体化之后一定学习成功。
