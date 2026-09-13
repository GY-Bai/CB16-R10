# Qualification TODO Authoring Addendum

本文件补充 `SOL_ROLE_PRINCIPLES.md` 与历史 `S_SERIES_TODO_AUTHORING_PRINCIPLES.md`。

## 1. IMPLEMENTATION_READY：交给 DS 前先把“怎样验收”定义清楚

复杂 S-series TODO 发布前，Sol 必须先完成 Stage Qualification Profile，并逐项定义：

1. 每个 capability claim 已拆成 machine-checkable proof obligations；
2. 每个 mandatory obligation 的 producer、consumer、gate、artifact proof 与 hostile counterexample；
3. behavior/target/evaluation、runtime/review/record/merge、parent/child checkpoint 等 identity 的名称、来源与预期关系；
4. RNG owner、seed derivation、isolation、construction order；
5. 合法 edge case 的 `PROCESS / ZERO_CONTRIBUTION / FAIL_CLOSED / SCIENTIFIC_FAIL`；
6. formal-only runner/gate 需要怎样的 production-shape canary、测试输入与预期输出；
7. durable/restart-safe claim 需要怎样执行“persist -> delete scratch -> artifact-only reconstruct -> corrupt -> fail closed”；
8. TODO 没有把尚未定义的科学选择留给 DS Flash；
9. Fix-impact Matrix 能区分普通 bugfix 与需要 versioned authority 的 science change；
10. exact-SHA GitHub Actions -> Shanxi Docker 的 runner、artifact、receipt 与 evidence binding 路径。

这里要求的是 **验收设计闭合**。尚未存在的 verifier、canary producer、artifact auditor 或 adapter 可以正是 TODO 要求 DS 实现的对象；不得把“实现后必须通过”误写成“开工前必须已经通过”。依赖未闭合时只阻止依赖它的高风险 task，能够独立证明边界的低风险工作不必被整个 stage 一刀切阻塞。

## 2. QUALIFICATION_READY：实现完成后再实际通过验收

进入正式资格运行前，TODO/Profile 要求的 verifier、bounded/production-shape canary、negative controls、artifact-only audit（若适用）和 hostile corruption 必须已经在授权环境实际执行，且运行自由度按冻结合同约束。此时才能把 machine evidence 交给 reviewer；不能用“设计过 canary”代替“canary 已通过”。

## 3. REVIEW_ACCEPTED / MERGE_READY：机器证据不能代替 Sol

接受或合并前必须绑定 latest actual diff、exact-SHA Shanxi CI、对应 artifact/receipt，并由 Sol-Reviewer 独立检查 proof strength 与 claim 是否匹配。`READY_FOR_SOL_REVIEW` 不是 reviewer acceptance。

现有 S1 使用自己的 frozen authority/review boundary；本 addendum 不追溯要求 active S1 重新设计 profile。
