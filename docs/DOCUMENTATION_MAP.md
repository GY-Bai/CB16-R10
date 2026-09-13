# 文档职责地图与维护规则

日期：2026-09-13。由 Astra 根据用户要求收敛文档组织。本文负责说明“去哪里找哪种答案”，不定义新的交易目标、算法、实验阈值或运行权限。

## 1. 六类文档，各自回答一个问题

| 类别 | 回答什么 | 维护/审核 | 当前入口 |
|---|---|---|---|
| 最高理念与设计决定 | 为什么做、目标和取舍是什么 | Astra 维护，用户审核 | [VISION](VISION.md)、[DECISIONS](DECISIONS.md)、[PRINCIPLE_ALIGNMENT](PRINCIPLE_ALIGNMENT.md) |
| 角色与协作规则 | 谁产出、谁审核、谁有权限 | Astra 向用户对齐 | [ROLES_AND_REVIEW_PROTOCOL](ROLES_AND_REVIEW_PROTOCOL.md)、[SOL_ROLE_PRINCIPLES](SOL_ROLE_PRINCIPLES.md)、[DS_FLASH_ROLE_PRINCIPLES](DS_FLASH_ROLE_PRINCIPLES.md) |
| 通用资格纪律与工具 | 已授权的声明需要什么证据 | Astra 审边界；Sol 定义具体证明义务并审核实现 | 既有角色规范；[PR #104 候选框架](https://github.com/GY-Bai/CB16-R10/pull/104)，当前接收状态见 CURRENT_STATE |
| 阶段任务与实验合同 | 这次具体做什么、怎样判定 | Sol author；Astra 审 TODO 对齐；DS 实现 | CURRENT_STATE 指定的 stage TODO、baseline、manifest、适用 profile/authority |
| 实现与证据 | 哪个版本实际验证了什么 | DS 交付，Sol 独立审核 | 实际代码、GitHub Actions、artifact、review、receipt；引用精确身份 |
| 当前导航与历史 | 现在接哪个断点、哪些只是历史 | 随阶段收尾同步；Astra 维护组织 | [CURRENT_STATE](CURRENT_STATE.md)、[docs/README](README.md)、历史冻结提交 |

框架是 Sol/DS 使用的工具与合同格式，不是第四种科学决策角色。`Constitution` 若出现在资格框架中，只指上游理念约束下的证明纪律，不能成为第二份最高理念。

## 2. 最短接手路线

先读 [AGENTS](../AGENTS.md) 确定角色，再读 CURRENT_STATE 找到当前工作 PR/精确版本；按角色读取相关原则和本次任务局部材料。无需每次重读全部历史。

- Astra：上游目标/最新用户修订 → 角色规则 → 当前 stage 的理念差异与 TODO → 必要的代码/证据抽查。
- Sol-Author：科学缺口 → 相关 infra/接口 → 需求到实现映射 → 适用证明义务/profile → 可执行 TODO → 对抗审阅。
- DS：已审阅 TODO/合同 → 指定复用代码 → 实现与反例 → Shanxi CI → READY_FOR_SOL_REVIEW。
- Sol-Reviewer：任务和冻结义务 → 实际 diff → 对应 CI/产物 → 独立接受或退回。

Sol-Author 与 Sol-Reviewer 是职责分离，不增加审批层。profile 是阶段任务包的一部分；通用检查器不能代替 reviewer 判断科学声明是否被证据支持。

## 3. 一种信息只维护一份当前正文

- **当前阶段状态统一写 CURRENT_STATE**，README/AGENTS/路由页只指向它，尽量不复制易过时的 SHA、测试数和进度。AGENTS 已有的状态段暂保留；后续触及时改为引用，避免为整理而与正在审查的 PR 同时改同一段。
- **角色职责以 ROLES_AND_REVIEW_PROTOCOL 为总规则**；Sol、DS principle 写各自操作细节。旧 S_SERIES TODO principle 作为兼容入口，遵守其与 Sol principle 已声明的继承关系，不再新建平行角色规范。
- **阶段科学值以该阶段适用的冻结合同为准**。TODO 解释并引用，不能复制出另一套奖励/阈值。错误要通过显式后继版本修正；导航更新不改 receipt。
- **结果写在实际 review/receipt/artifact 中**，状态页只准确摘要其证据范围，不把作者的计划当结果。
- **新通用框架每条要求注明实现层**：规则已定义、结构已校验、语义已验证、独立审阅已完成分别记录。模板非空不代表实验定义完整。

如果两个文档冲突，指出具体条款、版本和覆盖关系；不要按“文件更新较新”自动覆盖科学 authority，也不要把旧导航当成否决新 receipt 的理由。

## 4. 对 PR #104 的文档接收要求

[Astra 审阅](reviews/PR104_ASTRA_REVIEW_2026-09-13.md) 支持公共资格层方向，但当前提交仍需修订并获得 Sol 审核。

接收时采用现有 proposed 路径，不再另造一套框架文档：

- `SCIENTIFIC_QUALIFICATION_FRAMEWORK.md`：短入口、适用范围、主规范/实现状态/模板链接。
- `qualification/QUALIFICATION_CONSTITUTION.md`：唯一的通用证明纪律正文。
- `QUALIFICATION_TODO_AUTHORING_ADDENDUM.md`：说明如何插入 Sol authoring，引用共同规则，不复制第二套纪律。
- `qualification/IMPLEMENTATION_STATUS.md`：逐 primitive 列真实保证、未实现部分和所依赖的 stage adapter。
- `contracts/qualification/PROFILE_TEMPLATE_V1.json`：最小结构模板，明确它不是完整 stage preregistration。

这些路径在 PR 接收前仅是候选，不是 main 可运行入口。即使接收，也不自动改变 active S1 的冻结合同。低风险局部修改可引用现成合同，不为每个小修补机械生成四张新表。

## 5. 维护范围与停止条件

本轮进行逻辑分层和导航收敛，不移动历史文档、不批量重命名、不重写全部 authority，也不建立庞大文档数据库。

一次维护做到：当前断点清楚、任务来源准确、角色不混淆、未接收候选与已合格事实分开、旧入口有明确去向，即应结束整理并回到科学主线。后续 stage 收尾时只更新必要路由与证据索引。
