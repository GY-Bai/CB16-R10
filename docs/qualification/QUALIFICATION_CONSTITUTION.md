# CB16 Qualification Constitution

日期：2026-09-13。

## 1. 位置与边界

资格审查不决定模型“应该学什么”，只决定一个已经由上游 science / architecture 定义的 capability claim 需要什么证据才能进入下一层 authority。

```text
owner principles
-> Astra science/architecture docs
-> Scientific Qualification Framework
-> Sol Stage Qualification Profile + TODO
-> DS code/tests/exact-SHA CI
-> Sol independent review
```

本层不得反向修改已冻结的 task、seed、reward、model、threshold、sampling、budget、oracle 或 evidence ceiling。

## 2. 四层模型

1. **Constitution**：跨 stage 通用证明纪律。
2. **Contract**：每个 stage 的 machine-readable Qualification Profile。
3. **Enforcement**：公共 qualification primitive + stage adapter，只执行合同，不创造 science。
4. **Judgment**：Sol reviewer 独立检查 actual diff 与 machine evidence。

文档存在不等于 enforcement 完成；绿色 CI 不等于 reviewer acceptance；reviewer 说明文字不能补上 mandatory machine proof。

## 3. Claim -> Proof Obligations

任何宽泛 capability claim 必须拆成子命题。禁止用一个 `*_verified=true` 支撑没有展开的复杂能力。

例如 durable repeated learning 可能需要证明：collection truth persisted、replay only from durable truth、behavior identity/true log_mu preserved、learner consumed intended samples、exactly-one update committed、child state authentic、child adopted、account continuity preserved、later decisions came from child、artifact-only reconstruction succeeds。

“字段存在”不等于“link 被证明”。

## 4. No Semantic Fallback

局部工程异常不得改变 scientific contract。默认禁止：

- 改 sampling distribution 或强塞 sample；
- 跳过 frozen update cadence；
- silent clamp / 扩大 eligible set；
- behavior / target / evaluation identity 偷换；
- nominal / executed action 偷换；
- 修改 reward、seed、model、optimizer、threshold、budget；
- 用旧 CI 证明新代码；
- 把缺失 evidence 解释为 PASS。

冻结合同若对合法边界无定义，优先 fail closed 并升级到 Sol/versioning 判断。

## 5. 通用 verdict taxonomy

只使用：

- `PASS`
- `SCIENTIFIC_FAIL`
- `CONTRACT_MISMATCH`
- `EXECUTION_BLOCKED`
- `HARDWARE_LIMIT`
- `EVIDENCE_INSUFFICIENT`

`SCIENTIFIC_FAIL` 只表示执行与证据合同有效、但冻结能力判据失败；implementation/provenance/identity 错误必须归入 contract-invalid 路径，不能污染科学结论。

不得增加模糊 `PARTIAL_PASS` 掩盖 mandatory gate failure。

## 6. Stage Profile requirement

复杂 stage 在交给 implementation agent 前必须先发布 Stage Qualification Profile，至少覆盖 claims、proof obligations、semantic trace、identity/RNG、edge cases、negative controls、machine gates、artifact evidence 和 exact-SHA CI。Profile 不完整时，不得要求 DS 自行补科学语义。
