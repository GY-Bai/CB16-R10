# CB16 Scientific Qualification Framework

本文是 CB16 跨 stage 科学资格审查的入口。它只规定能力声明需要怎样被证明，不修改任何已冻结科学参数或历史 receipt。

核心结构：Constitution -> Stage Contract -> Machine Enforcement -> Reviewer Judgment。

规范正文：`docs/qualification/QUALIFICATION_CONSTITUTION.md`。

复杂 S-series stage 在 Sol TODO 之前必须先有 Stage Qualification Profile。这里的“先有”是 **先定义验收合同**，不是要求尚未实现的 verifier/canary 在开工前已经通过。implementation agent 按 profile 实现缺失的 producer/verifier/counterexample/gate/artifact path；实现完成后才进入 canary、artifact-only audit 和 exact-SHA CI 的资格验证。

三个时点必须分开：

1. **IMPLEMENTATION_READY**：交给 DS 前，claim、输入/输出、禁止行为、proof rule、counterexample、producer/consumer、需要实现的 verifier/canary 与 CI 路径已经定义；尚未存在的检查器可以正是 TODO 的实现对象。
2. **QUALIFICATION_READY**：实现完成后，要求的 bounded/production-shape canary、artifact-only reconstruction（若 claim 需要）、hostile corruption 和 exact-SHA runtime evidence 已实际执行并通过合同检查；此状态仍不等于 Sol 接受。
3. **REVIEW_ACCEPTED / MERGE_READY**：Sol-Reviewer 绑定 actual diff、latest exact-SHA evidence 与 proof strength 做独立审核后，才可进入既有合并授权流程。

R0 公共 primitive 只提供最小结构/覆盖/相等性/字节完整性与 verdict precedence。它们不自动证明 stage profile 完整、identity 集合齐全、Git ancestry/authorization、EvidenceGraph 语义、artifact 必需文件集合或科学 gate 集合充分；这些由 stage adapter + Sol reviewer 补全。空 identity audit 的集合论结果不得被上游解释为“所需 identity 已全部提供”。

当前 S1 / PR #102 继续按自己的 frozen review boundary 推进；本框架不追溯要求 S1 重做 profile，也不修改 S1 frozen science 或 S1 runtime。
