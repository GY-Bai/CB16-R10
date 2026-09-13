# Scientific Qualification Framework — R0 Implementation Status

日期：2026-09-13。

## 已实现的公共 primitive

- `scientific_qualification_contract_v1.py`
  - Stage Profile 基础结构校验；
  - 统一 verdict taxonomy / precedence；
  - malformed scientific gate 不得落入 `SCIENTIFIC_FAIL`。
- `scientific_qualification_evidence_v1.py`
  - 按 profile 审计 mandatory proof obligation 是否有明确 evidence；
  - missing/failed mandatory proof 输出 contract violation；
  - 不替 stage 判断科学 success criterion。
- `scientific_qualification_identity_v1.py`
  - 对具名 identity 做 exact expected/observed binding；
  - 不声称推导 Git ancestry 或 semantic equivalence。
- `scientific_qualification_artifact_v1.py`
  - 构建并校验 artifact byte SHA256/size manifest；
  - 能发现 missing/tampered bytes；
  - 明确不把“字节完整”命名成“provenance complete”。
- `contracts/qualification/PROFILE_TEMPLATE_V1.json`
  - 最小 machine-readable profile 示例。
- `.github/workflows/cb16-r11-scientific-qualification-framework.yml`
  - shared preflight + Shanxi Docker + verified Python；
  - focused qualification framework tests；
  - machine-readable non-scientific validation receipt。

## 本轮明确没有实现 / 没有声称

- 没有修改 S1 / PR #102 runtime；
- 没有运行 S1 formal qualification；
- 没有修改任何 frozen science authority / historical receipt；
- 没有建立“万能 EvidenceGraph semantic verifier”；
- byte-manifest PASS 不等于 provenance/learning/science PASS；
- exact identity equality 不等于 Git ancestry、authorization validity 或 semantic equivalence；
- 没有把任何 Stage-specific reward、seed、threshold、oracle、task generator 放进公共层。

## 后续只在具体需求成立时抽取

- Git ancestry / reviewed-runtime / record-head 通用 binding helper；
- authorization verifier；
- typed EvidenceGraph edge registry；
- artifact-only semantic reconstruction helpers；
- stage adapters。

上述能力不得因为“框架看起来完整”而提前造假式实现。只有能定义真实 producer/consumer、明确 proof strength、独立 hostile counterexample 时才进入公共层。
