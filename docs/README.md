# CB16 文档入口

本目录同时保存当前 authority、实施说明与历史阶段资料。**当前 CC canonical 入口已从“四线程待集成”切换为完成后的 integration handoff。**

## 当前阅读路线

| 问题 | 当前入口 |
|---|---|
| 当前精确状态与下一断点 | [CURRENT_STATE](CURRENT_STATE.md) |
| CC integration 最终交接 | **[CC_INTEGRATION_HANDOFF](CC_INTEGRATION_HANDOFF.md)** |
| 最终机器可读 spec / receipt | `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json` / `CB16_R11_CC_INTEGRATION_RECEIPT_V1.json` |
| 项目最高目标 | [VISION](VISION.md) |
| CC 原始并行决策 | [CC_PARALLEL_EXECUTION_DECISION](CC_PARALLEL_EXECUTION_DECISION.md) |
| CC 四线程原始总控 | [R11_CC_PARALLEL_HARD_CUTOVER_TODO](R11_CC_PARALLEL_HARD_CUTOVER_TODO.md) |
| Thread A 历史任务包 | [Runtime / Account](cc/CC_THREAD_A_RUNTIME_ACCOUNT.md) |
| Thread B 历史任务包 | [Policy / Critic / Learner](cc/CC_THREAD_B_POLICY_LEARNING.md) |
| Thread C 历史任务包 | [Experience / Replay / Economics](cc/CC_THREAD_C_EXPERIENCE_ECONOMICS.md) |
| Thread D 历史任务包 | [Performance Hard Cutover](cc/CC_THREAD_D_PERFORMANCE_HARD_CUTOVER.md) |
| 原 AC/BC 设计历史 | [AC TODO](R11_ACTOR_CRITIC_CODE_ALIGNMENT_TODO.md) / [BC Round 2 TODO](R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md) |
| 理念约束 | [PRINCIPLE_ALIGNMENT](PRINCIPLE_ALIGNMENT.md) / [COMPONENT_REQUIREMENTS](COMPONENT_REQUIREMENTS.md) |
| 算法与资格原则 | [TRAINING_ALGORITHM_R0](TRAINING_ALGORITHM_R0.md) / [TRAINING_QUALIFICATION_R0](TRAINING_QUALIFICATION_R0.md) |
| 经济评价原则 | [EVALUATION_PRINCIPLES](EVALUATION_PRINCIPLES.md) |
| 历史/当前组件地图 | [ARCHITECTURE_MAP](ARCHITECTURE_MAP.md) |
| Owner-open 问题 | [OPEN_QUESTIONS](OPEN_QUESTIONS.md) |
| Agent 规则 | [根目录 AGENTS](../AGENTS.md) |

## 当前 CC 结论

四个 CC threads 均从冻结 implementation base `89d62bf966f476e598f0e2f5c5e8e03c15a8db51` 独立完成后，在 `ai/r11-cc-integration-r0` 做了唯一 integration join。经过完整 Shanxi qualification 的 code head 是 `fc7adaae66da5b7735a3ab6aee7a8c7bd3721ed2`。

Integration 已验证完整 synthetic closed loop、hostile/recovery、reference-vs-fast semantic equivalence、FINAL/fresh-data firewall 和 hard cutover。最终性能选择为：

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

Thread A 保持账户/runtime/execution correctness authority；Thread D 只拥有 performance implementation authority。Canonical CC runtime 不恢复历史 broker/farm/vectorized performance fallback。

最终 Shanxi 预注册 benchmark 的 reference median 为 7.535366 transitions/s，fast median 为 638.765768 transitions/s，median speedup 84.769x。

## 证据边界

当前最强证据是 `INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`。它不是 ECONOMIC 或 TRANSFER evidence。FINAL 没有开启，fresh market data 没有使用。

Thread C 留下的 owner-open 决策仍有效：当 buy-and-hold 与 FLAT component outcomes 冲突时，integration 不自行发明统一 master winner/precedence。

## 历史资料地位

AC、BC、Stage-4、旧 Teacher/demonstration、旧 performance runtime 文档与代码继续保留为 provenance/history。新的 CC authority 不追溯改写历史 scientific verdict，也不把历史 workflow PASS 当成经济证据。
