# S1 Phase-A DS Flash Reliability Retrospective

日期：2026-09-13。

适用范围：CB16 R11 S 系列后续 TODO 撰写、DS Flash 实现、Sol 合并前审核。本文是对 PR #102 Phase-A R1→R3 的工程复盘，不修改 S1 科学合同、S0-v2 receipt 或任何正式科学结果。

## 1. 结论

S1 Phase-A 的实现难度显著高于普通模块开发。原因不是单个函数复杂，而是同一资格链同时包含：

- Actor/Critic/V-trace 与概率密度语义；
- behavior/target/evaluation policy 三种身份；
- nominal action / executed action；
- durable replay、restart、exactly-once 与 generation switch；
- task/oracle/control；
- multi-seed gate compiler；
- code SHA/tree、manifest、candidate record、review authorization；
- GitHub Actions → Shanxi Docker 的 exact-SHA evidence；
- artifact-only provenance 与最终 evidence ceiling。

任一局部实现“看起来合理”但跨层含义错误，都可能让绿色测试对应到错误的科学证明。

三轮返修并不说明 DS Flash 无法实现此类系统。实际观察是：**局部、明确、机械化的实现完成度较高；跨模块语义一致性、证据充分性、稀有边界与 authority 生命周期是主要失误来源。**

## 2. R1→R3 实际缺陷分类

### 2.1 局部补丁改变全局科学合同

代表问题：

- 为避免全-FLAT replay batch 的 risk-head 零梯度，先在 uniform sampling 后强制塞入 non-FLAT sample，破坏了 frozen uniform replay；
- 修复该问题后又改成“全-FLAT 则跳过 update”，又违反“每个完整 collection unit 后 1 次 durable update”的冻结日程。

模式：实现者看见局部异常后，优先让当前函数继续工作，而没有回到上游科学合同检查“这个 fallback 是否改变分布、预算或学习日程”。

### 2.2 producer / consumer 语义错位

代表问题：

- A→B→A retention gate 把 B 阶段新采集的 sequence IDs 当作 replay retention evidence，而非证明 A1 durable facts 仍在 generic replay eligible pool；
- formal gate compiler 读取 `<task>|OBJECTIVE_FIREWALL`，runner 实际提供 `OBJECTIVE_FIREWALL`；
- target actor 真正在训练，但 OFF_POLICY INITIAL/FINAL evaluation 却评估 frozen behavior actor。

模式：单个模块内部字段和类型都合法，但上下游对“这个字段/对象代表什么”理解不同。

### 2.3 “有字段”被误当成“已经证明”

代表问题：

- B6 provenance 初版只检查 journal 中是否存在 sample hashes、weights、losses 等字段；
- 后续虽然遍历 every committed update，但 `checkpoint_chain_verified` 一度只证明 optimizer step 连续和最终 child bytes，并未证明中间 `child_N -> parent_(N+1)`；
- artifact audit 曾经返回 false，但最终 classification 仍可能保持 PASS。

模式：对 evidence code 来说，结构存在性、局部一致性和最终证明是三个不同层级；DS Flash 容易在第二层就给第三层名称。

### 2.4 身份与 authority 生命周期错误

代表问题：

- qualification authorization 最初只看 status / reviewer_role，没有绑定 reviewed runtime SHA/tree + manifest hash；
- candidate record 一度滞后于最新 runtime/CI head；
- initial checkpoint authority 的旧 hash 无法在声明 codec 下重现，不能直接用新 hash 静默覆盖历史 authority。

模式：当代码、record、review、CI、merge 各自有不同 SHA/tree 时，单一 `head` 概念不足以表达资格链。

### 2.5 RNG / 初始化顺序 / policy 身份

代表问题：

- distinct behavior actor 的初始化曾影响全局 torch RNG，而 target Critic 在其后创建，导致 target initial state 可能依 task/seed 改变；
- evaluation policy 身份未与 target/behavior 清晰分离。

模式：随机性不是普通“参数”，而是状态机的一部分；构造顺序本身会改变实验。

### 2.6 terminal / failure / boundary 语义不完整

代表问题：

- HIGH_BANKRUPTCY 早期只 `force_liquidate=True`，但 durable transition 没有形成完整的 `mechanical_terminal=True / ECONOMIC_TERMINAL` failure fact。

模式：动作发生了不等于 authority fact 已经持久化；物理执行、经济终止和 replay 事实必须同时闭合。

### 2.7 formal-only 路径覆盖不足

代表问题：

- isolated gate fixtures 通过，但真实 manifest + runner audit mapping 的 full compiler 会失败；
- bounded smoke 通过，但 qualification-only classification / authorization / artifact fail-closed 路径仍存在错误。

模式：开发测试常覆盖“组件正确”，却没覆盖“生产形状的正式组合”。

## 3. DS Flash 在什么工况下相对可靠

这里仅总结本项目观察，不作为对模型产品的一般能力排名。

相对可靠的工况：

1. **局部输入/输出合同完整**：函数输入字段、输出 schema、边界和失败方式已经写死。
2. **有现成 canonical pattern 可复用**：例如 S0-v2 durable store、generation switch、exactly-once 事务已有实现，只需适配。
3. **确定性机械代码**：serialization、hash、manifest、文件布局、GitHub Actions wiring、adapter、数据结构转换。
4. **expected answer 可独立写成短 fixture**：明确的 PASS/FAIL/等号边界、解析式标量例子。
5. **单一 owner 的局部状态机**：一个模块内的 validate / encode / decode / persist / reopen。
6. **review blocker 被精确指出后的小范围修复**：R2/R3 中多数具体 blocker 在给出 counterexample 和不可改变项后能较快正确修复。

## 4. 什么工况下不可靠或必须按高风险处理

1. **跨 3 个以上模块的同一语义对象**：collection → replay → learner → evaluator → gate → artifact。
2. **局部 fallback 会改变全局实验分布/预算/日程**：补 sample、跳 update、silent clamp、扩大 eligible set。
3. **同一概念有多个身份**：behavior/target/eval policy，nominal/executed action，runtime/candidate/record/merge SHA。
4. **evidence-of-evidence 代码**：provenance auditor、qualification authorization、receipt/gate compiler。这里“字段存在”与“证明充分”最容易混淆。
5. **RNG 状态与构造顺序敏感**：seed isolation、fork_rng、多个网络依次初始化。
6. **高密度数学和直接条件逻辑**：V-trace、log density/Jacobian、mask、terminal/bootstrap、4/5 / 1/5 aggregation、NaN/Inf/zero denominator。
7. **formal-only 路径**：smoke 不会执行的 qualification branch、artifact deletion 后审计、review authorization。
8. **自然语言中含多个否定约束**：例如“uniform，但不能强制非-FLAT；每 unit 必须 update；全-FLAT 又允许 risk gradient 为零”。如果只写散文，执行模型容易遗漏一个条件。
9. **gate 名称需要对应严格证明**：`checkpoint_chain_verified`、`retention_proven`、`artifact_trace_complete` 这类名称尤其危险。

## 5. 为什么三轮修改是合理但可以减少

三轮并不主要来自“同一个低级 bug 修三次”。它更像逐层暴露：

- R1：发现局部语义和 authority 绑定问题；
- R2：修复后暴露 formal-path 组合错误和新 fallback 改义；
- R3：主体运行逻辑基本闭合，只剩 evidence proof-strength 不足。

这说明原 TODO 已经比早期 S1 mega-TODO 好很多，但仍偏“规定结果”，没有把所有高风险跨层 invariant 预先写成**机器可追踪的 producer→consumer→gate→counterexample 表**。

因此目标不是承诺“以后一次必过”，而是把最容易遗漏的错误提前变成 TODO 的显式输入，从而减少 R1 后才发现的结构性问题。

## 6. 后续 TODO 应新增的强制防错机制

### 6.1 Reliability risk class

每个 implementation task 标注：

- `LOW`: 局部 schema/serialization/wiring；
- `MEDIUM`: 单模块状态机、持久化、恢复；
- `HIGH`: 数学、RNG、cross-module identity、gate、authority、provenance、formal verdict。

`HIGH` task 必须附 independent counterexample，不允许只写“加测试”。

### 6.2 Semantic trace matrix

凡跨模块 invariant，TODO 必须列：

```text
semantic fact
→ producer
→ durable field/hash
→ transformer/materializer
→ learner/evaluator consumer
→ formal gate
→ artifact proof
→ one corruption/counterexample that must fail
```

若某一格为空，不能声称该 invariant 已闭环。

### 6.3 Identity ledger

在任务开始前列清：

```text
scientific baseline
runtime implementation SHA/tree
record-binding head
manifest hash
behavior policy identity
target policy identity
evaluation policy identity
parent/child checkpoint
reviewed SHA/tree
merged SHA/tree
successor base
```

禁止用一个泛化的 `head` 字段代替。

### 6.4 RNG ledger

为每个随机流写明 owner、seed derivation、是否跨 seed 改变、是否必须与其他流隔离、是否允许改变全局 RNG。

网络初始化必须明确构造顺序及 hash codec；不能只写 `torch.manual_seed()`。

### 6.5 Edge-case decision table

对 HIGH task，提前列出：

- empty set；
- all-FLAT / all-one-class；
- exact threshold equality；
- NaN/Inf；
- missing durable record；
- terminal vs truncation；
- stale SHA/manifest；
- retry after partial mutation。

每种情况明确：`PROCESS / ZERO-CONTRIBUTION / FAIL-CLOSED / SCIENTIFIC_FAIL` 中哪一个。DS 不得自行发明第五种 fallback。

### 6.6 No semantic fallback rule

如果局部异常不能在冻结合同下处理：

- 不准改变 sampling distribution；
- 不准跳过冻结要求的 update；
- 不准 silent clamp；
- 不准换 evaluation object；
- 不准用旧 CI 证明新代码；
- 不准把不完整 proof 命名为 complete/verified。

应 `CONTRACT_MISMATCH` 或报告上游 authority defect。

### 6.7 Production-shape formal canary

在正式结果前，至少一次使用：

- 真实 execution manifest shape；
- 真实 runner audit mapping；
- 真实 gate compiler；
- 人工构造的 synthetic PASS/FAIL records；

执行完整 formal composition，但**不运行正式科学训练结果**。这样可以在不污染 preregistration 的情况下发现 key mismatch / missing gate / fail-closed 错误。

### 6.8 Artifact-destruction test

只要 TODO 声称 restart-safe / durable / artifact-only provenance：

1. 运行并导出；
2. 移除原 scratch/run root；
3. 仅从 artifact 重建审计；
4. 移除一个中间 link，最终必须 fail closed。

### 6.9 Proof-name adequacy

任何 `*_verified / *_complete / *_proven` gate 必须在 TODO 中枚举它真正验证的子命题。

例如 `checkpoint_chain_verified` 至少应包含：

- initial parent matches；
- every update has parent+child；
- `child_N == parent_(N+1)` 或等价状态内容绑定；
- optimizer step 连续；
- final child bytes 与 journal 匹配。

只验证其中两项不能使用完整名称。

### 6.10 Fix impact matrix

每次 `CHANGES_REQUIRED` 修复前，DS 必须写：

```text
changed invariant
unchanged frozen contracts
possible collateral effects
new counterexample
required rerun scope
```

用于防止“修 sampler → 改 update schedule”这种二次改义。

## 7. 对 Sol TODO 作者的直接要求

Sol 不应期待 DS Flash 自己发现所有跨层证明义务。对于 HIGH task，TODO 必须尽量把审查者脑中的反例提前外显。

减少返修轮数的关键不是让 TODO 更长，而是让它更结构化：

- 少写抽象形容词；
- 多写 identity/trace/edge tables；
- 对 formal gate 给 production-shape fixture；
- 对 provenance 给 destroy-and-audit；
- 对数学给独立标量答案；
- 对 fallback 给禁止清单；
- 对 authority 给 exact-SHA lifecycle。

即使采用这些原则，Sol 独立 review 仍然不可取消。目标是把 review 从“发现架构语义错误”更多地转成“确认执行者按已明确合同实现”。
