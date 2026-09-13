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

只使用：`PASS`、`SCIENTIFIC_FAIL`、`CONTRACT_MISMATCH`、`EXECUTION_BLOCKED`、`HARDWARE_LIMIT`、`EVIDENCE_INSUFFICIENT`。

`SCIENTIFIC_FAIL` 只表示执行与证据合同有效、但冻结能力判据失败；implementation/provenance/identity 错误必须归入 contract-invalid 路径，不能污染科学结论。不得增加模糊 `PARTIAL_PASS` 掩盖 mandatory gate failure。

## 6. Stage Profile requirement

复杂 stage 在交给 implementation agent 前必须先发布 Stage Qualification Profile，至少覆盖 claims、proof obligations、semantic trace、identity/RNG、edge cases、negative controls、machine gates、artifact evidence 和 exact-SHA CI。Profile 不完整时，不得要求 DS 自行补科学语义。

Profile 的 TODO 映射必须形成：

```text
claim/proof obligation
-> reusable producer
-> missing adapter/function
-> consumer
-> hostile counterexample
-> formal gate
-> artifact proof
-> required CI
```

mandatory obligation 没有 consumer、gate 或 artifact proof 时，TODO 尚未完成。

## 7. Evidence Graph

跨 collection/replay/learner/checkpoint/generation/evaluator/gate/artifact 的能力，默认按 Evidence Graph 建模：

```text
Scientific Authority
-> Task / Manifest
-> Observation
-> Behavior Policy + nominal action + true log_mu
-> Execution / Account Consequence
-> Durable Transition / Raw Fact
-> Replay Materialization
-> Batch
-> Learner Update
-> Parent/Child Checkpoint
-> Generation Switch
-> Next Runtime / Behavior
-> Evaluation
-> Formal Gate
-> Artifact Evidence
```

每条 mandatory edge 至少声明 producer、consumer、identity/hash、required fields、verification rule、artifact representation、corruption that must fail。

## 8. Identity / RNG

禁止用模糊 `head`、`policy`、`checkpoint` 代表多个身份。按适用范围显式区分 scientific baseline、manifest、implementation SHA/tree、record head、reviewed SHA/tree、qualification checkout、behavior/target/evaluation policy、parent/child checkpoint、account lineage、artifact bundle。

每个随机流声明 owner、seed/derivation、stream identity、isolation boundary、global RNG policy、construction-order dependency，以及需要重放时的 counter/position。

初始化 identity 若被冻结，至少同时冻结 seed、module construction order、RNG isolation、checkpoint codec、semantic state hash。

## 9. Artifact-only proof

凡 stage 声称 durable、restart-safe、replayable 或 artifact-only auditable，默认要求：

1. 导出 canonical evidence；
2. 删除 scratch/run root；
3. 只从 artifact 审计；
4. 破坏至少一个中间 mandatory link；
5. verifier 必须 fail closed；
6. final machine verdict 不得继续为 PASS。

只保存 summary 或 final checkpoint 不足以证明完整 durable lineage。

## 10. Author / Reviewer separation

Sol 逻辑上分成两个阶段：

```text
Sol-Author -> Stage Qualification Profile + TODO
DS -> implementation/tests/CI
Sol-Reviewer -> independent actual-diff + machine-evidence review
```

Reviewer 不默认相信 Author 对 gate 的命名或 proof 强度，必须反向检查机器 gate 实际证明的子命题是否足以支撑 claim。

## 11. Formal canary

若 stage 含 qualification runner、gate compiler、review gate、record binding 或 artifact auditor，在正式 scientific qualification 前必须用真实 manifest/runner/gate shape + 人工 PASS/FAIL/boundary records 做 production-shape canary，且不能提前产生正式科学结果。

## 12. Fix-impact Matrix

每次 review 要求修改后、代码修改前记录：changed invariant、unchanged frozen contracts、possible collateral effects、new counterexample、required rerun scope、versioning impact。

若修复触及 sampling、budget、schedule、reward、model、threshold、seed、oracle、task generator 或 evidence ceiling，停止普通 bugfix，进入 versioned science/authority 判断。

## 13. Public primitive extraction rule

未来公共层只允许实现通用 enforcement，例如 Profile validator、IdentityBundle、EvidenceGraph verifier、classification kernel、artifact verifier、record-binding helpers；不得硬编码某个 stage 的 reward、oracle、threshold、seed 或 rescue logic。

已进入 multi-round review / exact-SHA boundary 的 active stage 不顺手抽公共 framework。先完成并冻结该 stage，再开独立 non-scientific extraction 工作并证明 behavior/semantic equivalence。当前 S1/PR #102 因此不在本框架分支中被重构。
