# CB16 Scientific Qualification Framework

本文是 CB16 跨 stage 科学资格审查的入口。它只规定能力声明需要怎样被证明，不修改任何已冻结科学参数或历史 receipt。

核心结构：Constitution -> Stage Contract -> Machine Enforcement -> Reviewer Judgment。

规范正文：`docs/qualification/QUALIFICATION_CONSTITUTION.md`。

复杂 S-series stage 在 Sol TODO 之前必须先有 Stage Qualification Profile。implementation agent 按 profile 实现 proof obligations、counterexamples、machine gates 和 artifact evidence；Sol reviewer 独立检查 actual diff 与机器证明强度。

当前 S1 / PR #102 继续按自己的 review boundary 推进；本框架分支不修改 S1 frozen science 或 S1 runtime。
