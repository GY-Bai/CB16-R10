# PR #104：Astra 架构与文档审阅

日期：2026-09-13。审阅 main：`b19d3473a87fe36ec3394be53ad1cc6220f324c6`；PR head：`4583c39738c8b901c53308598685492d1f069db2`；tree：`f17c071de5137ffbd5d55039b7438e50edd25a53`。

## 1. 结论与角色边界

**方向合理，建议保留这个小规模公共层；当前提交暂不建议合并。Astra 审阅结论为需要修订，不是 Sol code acceptance，也不是 scientific verdict。**

本次阅读 22 个变更文件、当前角色协议与阶段导航，核对实际 GitHub Actions 日志、artifact 元数据和 PR 审核记录。未在临时沙盒运行框架测试，没有替 DS 修代码、代 Sol 审批或合并 PR。

## 2. 应保留的设计

- 将 claim、proof obligation、identity、edge case 与 artifact 要求显式化，符合用户要求的“Sol 把科学含义讲清楚，DS 严格实现”。
- 公共层不包含具体奖励、种子、模型、学习策略或晋升目标；实际 diff 没有修改 frozen authority 或 S1 runtime。
- 四个 primitive 共约 429 行；身份检查明确只做相等性，artifact 检查明确只做字节完整性，没有声称万能语义证明。
- Sol-Author 和 Sol-Reviewer 是同一角色的两种工作职责，不是新增权限层。Reviewer 必须独立检查作者的假设与代码，不能默认作者正确。
- 原科学链应继续：S1 的真实可学习性验证仍是主线，框架只是支撑。公共层不应成为无限扩建 Infra 的理由。

## 3. 实际 CI 与审核状态

[Shanxi run 34770356260](https://github.com/GY-Bai/CB16-R10/actions/runs/34770356260) 的 job `103758721997` 为 success，runner 为 `shanxi-docker-r11`。日志实际记录上述 checkout SHA/tree、`17 passed in 0.43s`、零失败/错误/跳过及 profile template validation。

Artifact 元数据：ID `10321339503`，3273 bytes，digest `sha256:154033a0f66b8e577cb978712dba9e6ec05a08f307437eb00b61315d86451af2`。本次核对日志与 GitHub 元数据，未下载 ZIP 重新计算摘要，不把 DS 的“独立检查 ZIP”声明算成本次自己的操作。

这证明现有测试在指定环境完成，证据类别仍为 `NON_SCIENTIFIC_QUALIFICATION_INFRA_VALIDATION`。PR 当前仅有 [DS 交付说明](https://github.com/GY-Bai/CB16-R10/pull/104#issuecomment-5654819622)，状态 `READY_FOR_SOL_REVIEW`；所查 PR reviews 为空，没有发现 Sol acceptance。GitHub 的 mergeable/clean 不等于可以按项目协议合并。

## 4. 必须修订：基础输入校验允许 null 冒充有效身份/引用

这不是要求 R0 实现完整 EvidenceGraph，而是当前 primitive 自己承诺的基础结构约束存在漏洞。

### A104-01：两个缺失身份可被判断为匹配

[identity 实现](https://github.com/GY-Bai/CB16-R10/blob/4583c39738c8b901c53308598685492d1f069db2/cb16_local_opt/scientific_qualification_identity_v1.py) 的 `ExactIdentityBindingV1.validate` 使用 `str(value).strip()` 检查非空；后续直接比较 `observed == expected`。

最小输入：

```json
{
  "identity_id": "implementation_sha",
  "expected": null,
  "observed": null,
  "expected_source_ref": "review",
  "observed_source_ref": "checkout",
  "required": true
}
```

**静态路径推导：** Python 中 `str(None)` 非空，而 `None == None` 成立；因此这条必需身份会进入 matched 集合，并可返回 `all_required_exact_bindings_match=true`。类型注解不负责运行时拒绝 JSON null。此例是代码审阅推导，尚未作为本次新增 CI 反例执行。

要求 Sol 明确非空字符串等输入域，DS 拒绝 null/错误类型，再用真实 Shanxi CI 验证反例与合法相等/不等输入。源引用字段也应遵守自己的真实类型合同，不能只靠字符串化判断。

### A104-02：null 证据引用可以支撑 passed

[evidence 实现](https://github.com/GY-Bai/CB16-R10/blob/4583c39738c8b901c53308598685492d1f069db2/cb16_local_opt/scientific_qualification_evidence_v1.py) 的 `ProofEvidenceV1.validate` 同样把列表元素字符串化。

对一个合法声明 mandatory P1 的 profile，以下输入应被拒绝：

```json
{"obligation_id": "P1", "passed": true, "evidence_refs": [null]}
```

**静态路径推导：** 列表非空，`str(None)` 也非空，记录被接受并进入 passed；该 profile 若仅有 P1，则 coverage 可返回 true。现有 17 个测试没有 null identity/null reference 反例。

要求严格校验引用容器与元素类型，并检查 profile 中同类 `str(value)` 校验。用独立预期测试错误输入，不能以“调用者应该先验证”掩盖当前 validator 明确接收 Mapping/JSON 的边界。这里只要求结构有效；引用解析/文件内容证明仍可由 stage adapter 负责。

## 5. 必须澄清：可开工、可正式运行与证明强度

### A104-03：避免把证明的设计与已实现证明混成一个开工条件

PR 的 `QUALIFICATION_TODO_AUTHORING_ADDENDUM.md` 要求十项闭合才能交 DS，其中包括 canary 使用、删除 scratch 后复核。文字没有清楚区分“先定义验收”与“实现后实际通过验收”，容易形成尚未写 verifier 就要求先通过 verifier 的循环。

应明确三个时点：

1. **交付实现任务前**：claim、输入/输出、禁止行为、证明规则、反例、producer task 和预算清楚；尚不存在的 verifier/canary 正是要实现的任务。
2. **正式资格前**：实现完成，判定器及有界 canary 已实际验证，运行自由度按合同冻结。
3. **接受/合并前**：绑定实际 diff、CI 与 Sol 审核，按该 PR 的阶段资格要求收尾。

要求按具体任务包划定未闭合依赖；可独立的低风险工作继续。已有 S1 使用自己的冻结边界，新框架不得追溯强制 S1 重新设计。

### A104-04：最小 profile 不等于完整的 stage 资格协议

当前结构 validator 仅校验 schema/stage/claims/obligations/edge decisions。semantic trace、RNG、完整科学 gate 集合、实际证据来源与 artifact resolution 没有由它普遍执行验证。此范围可以作为 R0 被接受，但必须在入口和模板旁清楚列出由谁补全。

具体区分：

| 当前 primitive | 已有强度 | stage/审阅仍须承担 |
|---|---|---|
| profile structure | 基础字段和 ID 关联 | 任务是否充分定义、哪些义务必需、阶段扩展字段与引用 |
| proof coverage | 收到的 obligation 状态/引用覆盖 | 状态由哪个 verifier 产生、是否绑定正确 artifact/gate、语义是否成立 |
| exact identity | 已提供的具名值相等 | 必需身份集合是否齐全、值来源、谱系/权限 |
| artifact bytes | 给定 manifest 条目的大小/摘要 | manifest 自身的可信绑定、必需文件集合、语义链 |
| verdict kernel | 对传入标志和 gate 的分类 | 预登记 gate 集合完整、未知/缺失 gate 拒绝、全部必需审计接入 |

例如空 identity 列表的“全部相等”是空集合上的结果，不能被上游当作“所需身份都已经提供”；只传一个绿色 scientific gate 也不证明整个预登记集合齐全。可以通过 stage adapter 和一份小型完整调用示例落实，不要求 R0 扩建万能 verifier。

合同完整性义务失败与科学目标未达成必须分开：前者阻止科学解释，后者在合法运行后应保留 `SCIENTIFIC_FAIL`。不能把所有科学能力负结果都塞进 integrity obligation 并统称 contract mismatch。

## 6. 文档组织结论

需要收敛入口与职责，**不需要全仓库搬迁、改名或重建最高理念**。主问题是多个页面复制当前状态且更新不同步：

- 根 README 仍引导 Sol 沿用 AC TODO；
- docs/README 与 CURRENT_STATE 仍将 S0-v2 写成 ACTIVE；
- AGENTS、main 中 S0-v2 receipt 已表明它合格，PR #102 正推进 S1；
- PR #104 修改 AGENTS 引入 profile，但既有 Sol 角色正文仍是旧六阶段路线，接收时须明确补充的插入点和文档覆盖关系。

本轮在 main 增加 [文档职责地图](../DOCUMENTATION_MAP.md)，修正入口和阶段路由。PR #104 自己的文档/代码仍留在原 PR，由 Sol/DS 修订。尚未接收的框架只作为候选链接，不能伪装成 main 已实现模块。

## 7. 收尾条件

1. DS 修复 A104-01/02，Sol 核对相关输入域与独立边界测试；
2. 在原 PR 澄清 A104-03/04，保持 R0 小范围；
3. 修订后的实际代码完成 GitHub Actions → Shanxi Docker CI，证据绑定最新待审提交；
4. Sol 独立审阅并明确接受，之后按既有合并流程决定。

满足上述条件后，Astra 支持将其作为有限的 qualification infrastructure R0 接收；不因此启动 S1 正式运行、历史训练、扩容或全系统文档重写。
