# 组件职责与实现地图

核对日期：2026-09-12 UTC。当前 architecture authority 已由 CC integration 收口；详细机器可读定义见：

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

## 当前 canonical closed loop

Market + Account -> Thread B stochastic Actor -> W-01 `CCPolicyDecisionV1` -> Thread A permission / sizing / execution -> W-02 `CCEnvironmentTransitionV1` -> signed account consequence + environment advance -> Thread C immutable experience/replay -> W-03 `CCExperienceSequenceV1` -> Thread B Critic/V-trace learner -> W-04 `CCLearningUpdateV1` -> child checkpoint -> Thread A generation switch -> same logical account -> child policy acts.

W-05 `CCEconomicResultV1` 属于 Thread C 的经济评价接口。本轮 integration 没有产生 ECONOMIC verdict。

## Authority map

| Layer | 当前 authority | 主要实现 |
|---|---|---|
| Account / signed economics | Thread A | `account_economics_r0.py`, `cc_runtime_account_loop_r0.py`, `cc_environment_advance_r0.py` |
| Runtime clocks / recovery / generation switch | Thread A | `cc_clock_r0.py`, `cc_account_recovery_r0.py`, `cc_runtime_generation_switch_r0.py` |
| Stochastic Actor / true `log_mu` / RNG provenance | Thread B | `cc_policy_brain_r0.py`, `cc_policy_distribution_r0.py`, `cc_policy_rng_r0.py` |
| Critic / V-trace / learner transaction | Thread B | `cc_critic_value_r0.py`, `cc_vtrace_r0.py`, `cc_learner_transaction_r0.py` |
| Immutable experience / replay | Thread C | `cc_experience_*_r0.py`, `cc_replay_*_r0.py` |
| Arithmetic economic evaluation contracts | Thread C | `cc_economic_*_r0.py` |
| Cross-thread binding / canary | Integration | `cc_integration_contracts_r0.py`, `cc_integration_runtime_r0.py` |
| Canonical fast spine | Thread D under Thread-A semantics | `cc_integration_fast_path_r0.py`, `cc_fast_scheduler_r0.py`, `cc_fast_fact_queue_r0.py`, `cc_fast_writer_r0.py` |
| Equivalence / benchmark | Integration | `cc_integration_benchmark_r0.py`, `scripts/run_r11_cc_integration_benchmark.py` |

Thread D 的性能选择不会把它的 thread-local synthetic account kernel 升格为科学 account authority。Integrated fast path 传输和调度 Thread-A-authoritative facts。

## Selected topology

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

Shanxi 预注册 benchmark：16 accounts × 64 market steps，reference 与 fast 各 7 次交替执行。Reference median 7.535366 transitions/s；fast median 638.765768 transitions/s；median speedup 84.769x。

Canonical CC runtime 没有历史 performance fallback。`gpu_inference_broker.py`、`multiprocess_trajectory_farm.py`、`vectorized_physics.py` 可以保留为 history/reference，但不属于当前 canonical CC dependency。

## Semantic firewalls

Integration 必须继续保留：account continuity；signed economics；negative equity/liability；nominal action 与 permission/execution 分离；true `log_mu`；policy/RNG/generation provenance；failure/terminal fact retention；frozen-organ/trainable-Brain ownership；arithmetic expected-return orientation；FINAL/fresh-data firewall；same-account temporal serialization。

没有 policy decision 的 mechanical advance 可保留为 raw fact，但不得伪造 action 或 `log_mu` 进入 V-trace replay。

## Evidence boundary

Final qualification run `34710702090` / job `103598868269`：155 tests passed，1 个仅用于独立 Thread-A branch isolation 的 assertion 在授权 join 后被 deselect；closed-loop、hostile/recovery、reference-fast equivalence、hard cutover 与 firewalls 全部 PASS。

Semantic checksum：`c864052eab1c107ab73a88d96ddb527bfc60819a41165b1495ecc74d56644988`。
Final-account checksum：`29d33e7639029f40c6edfb1c7fe9fff26e4af3f418cf5109a066a55de13e282e`。

Strongest evidence：`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`。不是 ECONOMIC/TRANSFER evidence；FINAL 未开启，fresh market data 未使用。

## Historical map

Stage-4、AC/BC、Teacher/demonstration 和旧 runtime/performance 文档仍保留为 provenance/history。若它们与 integration spec/receipt 的当前 routing 冲突，以 integration authority 为准；历史 scientific verdict 不被追溯改写。
