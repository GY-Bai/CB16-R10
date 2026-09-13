# CB16 agent 工作约定

本文件适用于整个仓库。主要交流语言为中文，代码标识和版本化协议名称保持原样。

## 1. 当前角色与审核链

必须先读 `docs/ROLES_AND_REVIEW_PROTOCOL.md`。

当前 owner 指定工作流：

- 上游文档/原则先与 owner 对齐；
- Sol 根据当前文档维护可执行 TODO；
- 一个 implementation Agent 在 owner/Sol 指定的 branch 上按 TODO 写代码；
- Sol 独立审核实际 diff、公式、比较、直接条件分支、mask、空值/非有限值、边界与 verdict 逻辑；
- runtime/code behavior 变更在合并 `main` 前必须通过 GitHub Actions -> Shanxi Docker 的相关专项/冒烟测试；
- implementer 自检、repo-guard 或绿色 CI 不能替代 Sol review；
- implementer 不自行 merge。

临时 ChatGPT sandbox 可用于静态准备/阅读，不替代 Shanxi runtime evidence。

## 2. 当前接手顺序

当前 active implementation stage 是 **S0-v2 Durable Learnability Foundation**。

依次阅读：

1. `docs/CURRENT_STATE.md`
2. `docs/ROLES_AND_REVIEW_PROTOCOL.md`
3. `docs/S_SERIES_TODO_AUTHORING_PRINCIPLES.md`
4. `docs/R11_POST_CC_S0_S1_TODO.md`
5. **`docs/post_cc/S0_V2_DURABLE_LEARNABILITY_FOUNDATION_TODO.md`**
6. `docs/reviews/S0_S1_EXECUTABILITY_REVIEW_2026-09-13.md`
7. `docs/CC_INTEGRATION_HANDOFF.md`
8. `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
9. `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`
10. `authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json`
11. task-relevant `PRINCIPLE_ALIGNMENT / COMPONENT_REQUIREMENTS / TRAINING_ALGORITHM_R0 / EVALUATION_PRINCIPLES` and code.

Historical `docs/post_cc/S0_CONTRACT_MIGRATION_TODO.md` is frozen provenance, not the current S0 execution package. Historical `docs/post_cc/S1_END_TO_END_LEARNABILITY_TODO.md` is retained as provenance/design input but is not the current implementation entrypoint while S0-v2 is active.

Do not create another competing S0/S1 decomposition.

## 3. Current implementation branch

The one authorized S0-v2 implementation branch is:

`ai/r11-s0v2-durable-learnability-foundation-r0`

Scientific/code baseline:

`main@392063881a6ef0dd1776ac579f1a290a134fc49e`

Task-authoring/navigation commits after that baseline are allowed to be included in the branch handoff but do not silently change the frozen science/code baseline.

The implementation Agent must work only on this branch for S0-v2. Do not create sibling S0-v2 branches or parallel implementations unless the owner explicitly changes this rule.

## 4. Historical authority that must remain immutable

CC R11 integration is closed. PR #99 merge commit:

`daa889d758ce80c7d1ce73ea37110937e1b146c0`

Canonical CC authority:

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

Historical post-CC S0 contract migration is PASS/frozen:

- frozen implementation base `fc7102442e91a1c27cf705487c6d06bd64b8ea09`
- qualified implementation `5760061d6c274e9f8796e6608e86bb173018148f`
- receipt-bearing head `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f`
- receipt `authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json`

Do not edit historical receipts to satisfy new work. Use versioned successor artifacts.

## 5. S0-v2 purpose

S0-v2 must build and qualify the durable joint-learning foundation:

```text
canonical Brain observation
 -> durable content-addressed observation fact
 -> immutable experience linkage
 -> persistent replay
 -> restart-safe materialization
 -> nominal direction+risk batch
 -> persisted true log_mu
 -> target joint log_pi
 -> Critic / bootstrap / V-trace
 -> joint Actor/Critic update
 -> exactly-once durable update
 -> child checkpoint
 -> generation switch
 -> same logical account continuation
```

Missing functionality in this chain is the task. It is not a reason to stop after a status report.

Implementer terminal state is `READY_FOR_SOL_REVIEW`, not self-merge.

## 6. Scientific semantics that must not change

- `Truth != Belief != Decision != Permission != Execution`.
- Nominal sampled action remains separate from permitted/executed action.
- Requested target risk is not confidence.
- Actor is categorical direction + conditional continuous target risk.
- FLAT risk is exactly 0.
- True behavior `log_mu` is computed/persisted at decision time and never reconstructed from execution.
- Same logical account remains continuous across chunk/restart/checkpoint/generation boundaries where required.
- Negative equity, liabilities, failures and terminal facts remain factual.
- Compute truncation, chunk boundary, task horizon and mechanical/economic terminal are distinct.
- Frozen market organ gradient ownership remains frozen.
- Arithmetic expected return remains the current economic orientation.
- B&H and FLAT are parallel benchmark components; no master precedence.
- No handcrafted cycle/regime/resonance activation system.
- No hidden strategic SL/TP/max-hold/cooldown.
- FINAL remains sealed; fresh market data remains forbidden.

## 7. Canonical runtime and legacy firewall

Selected CC topology remains:

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

Thread A remains runtime/account/execution science authority; Thread D remains performance implementation authority.

Canonical path must not silently fall back to historical performance runtime:

- `gpu_inference_broker.py`
- `multiprocess_trajectory_farm.py`
- `vectorized_physics.py`

## 8. Pre-merge CI requirement

For S0-v2 runtime/code behavior changes, the implementer must provide a dedicated GitHub Actions workflow using the current Shanxi runner protocol, including shared preflight, `[self-hosted, shanxi-docker-r11]`, verified canonical Python, exact checkout SHA, focused S0-v2 tests, relevant regressions and machine-readable artifacts.

Repo-guard alone is insufficient.

Sol reviews CI evidence for the exact candidate SHA before merge.

## 9. High-risk implementation review points

Sol will explicitly inspect and require counterexample tests for:

- SHORT/FLAT/LONG index mapping;
- FLAT point mass and non-FLAT density;
- risk transform/log-Jacobian and endpoint support;
- `log_pi - log_mu` and V-trace ratio direction;
- boundary/bootstrap masks;
- nominal versus execution routing;
- observation/content hash comparisons;
- exactly-once commit/retry predicates;
- account-lineage preservation across generation switch;
- `<`, `<=`, `>`, `>=` threshold boundaries;
- empty/missing/non-finite inputs and masks;
- any fallback branch that could hide a failure.

## 10. Evidence ceiling

Historical CC strongest evidence remains:

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

Historical S0 maximum evidence remains:

`POST_CC_CONTRACT_MIGRATION_QUALIFIED`

S0-v2 maximum evidence after Sol review/merge is:

`POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`

None of these is ECONOMIC or TRANSFER evidence.

Successor S1 known-answer science is routed only after accepted S0-v2 evidence. S0-v2 completion does not automatically authorize historical training, FINAL, capacity scaling or economic qualification.