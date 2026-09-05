# CB16 Account Physics & Simulator State R0 Report

**Final status:** `ACCOUNT_PHYSICS_RUNTIME_RECOVERED_AND_FROZEN`  
**Central Brain training:** NOT STARTED  
**Fresh market cohort / 2025-09 / final holdout:** NOT ACCESSED

## 1. Blocker resolution

Training Infrastructure Preflight R0 was correct that `AccountStatePacketV1` is only a 6D Brain-visible observation and cannot be used as the replay-side account ledger. This R0 now establishes the missing separation:

```text
Recovered V5.5 FrozenTradingKernel
        +
SimulatorStateSnapshotV1 (authoritative continuation state)
        |
        +-- step_account(snapshot, action, execution_input, physics_contract)
        |      -> next snapshot
        |
        +-- project_observation(snapshot, current_mark, contract)
               -> AccountStatePacketV1 (6D, unchanged)
```

The 6D packet was not expanded and is never accepted as restore state.

## 2. Historical runtime recovery

The finite authority-oriented audit directly found the original `v5.5_clean_rebuild_phase1.tar.gz` in Library. Archive SHA256 is `6da734c5eb8dbc43f86f2091b0a9ad73f1c0f9416c6a63f465a8dd64fc34e88b` and matches its historical sidecar exactly. All 8 files under `src/v55/kernel/` match the Phase-1 source manifest byte-for-byte; aggregate kernel identity is `518f90f3f0db790b0e6e49ec05def3b98344d92984cf2956d9dd9a2cc5701d87`. The recovered parity verifier was rerun in the current environment and again returned **88 steps, mismatch_count=0, max_abs_diff=0.0**.

Therefore this run does **not** create `AccountPhysicsVNext`; it reuses the historical physics authority and adds only a versioned state/replay adapter.

## 3. What the authoritative snapshot contains

`SimulatorStateSnapshotV1` serializes every field in recovered `AccountState`, including cash/margin/position/cost basis, realized accounting, path-dependent peak/drawdown data, holding/trade-age state, stop/take-profit state, liquidation/terminal flags, ATR/previous-close accumulator, cooldown and trade statistics. It also saves the historical hidden continuation state already named by the old `KernelSnapshot`: `last_bar_time`, symbol binding and `last_carry_cost`.

The recovered physics has no stochastic RNG, so `rng_state=null` is explicit rather than fabricated. SimConfig bytes are not duplicated in every snapshot; the immutable physics-contract SHA is referenced.

## 4. Frozen 6D observation projection

`project_observation(...)` calls the exact frozen Account State R0 encoder source. It is pure and non-mutating. Position, cost basis, holding horizon and drawdown are derived from the authoritative ledger at the current mark. Margin capacity is deterministically defined from recovered margin bookkeeping as `margin_used + available_margin()` at that mark.

The frozen field `risk_budget_remaining_fraction` is explicitly **externally authoritative**. No matching historical V5.5 budget-depletion rule exists in the recovered kernel. R0 therefore saves synchronized `risk_budget_remaining` and `risk_budget_capacity` in the composite simulator snapshot and preserves them unchanged through account-physics steps. This avoids inventing economic semantics while keeping AccountStatePacketV1 valid. A future risk supervisor may update those authoritative values explicitly; physics does not secretly infer them from drawdown.

## 5. Action and transition semantics

The historical action contract is recovered: `SHORT=0`, `FLAT=1`, `LONG=2`, with optional `risk_multiplier in [0,1]`. Policy entry is only actionable while flat and outside cooldown. Once positioned, the environment owns stop/take-profit, liquidation, risk-limit and max-holding exits; the policy cannot resize/reverse/manually close the position.

`TransitionRecordV2` references content-addressed snapshots before and after the step. It does not duplicate snapshot bytes or shared MarketSensory payloads per account transition.

## 6. Termination / truncation

R0 no longer compresses all endings into `done`:

- `LIQUIDATED`
- `MAX_HOLDING_FORCED_EXIT`
- `TERMINATED`
- `TRUNCATED`
- `NORMAL_CONTINUE`

`step_account` never autoresets. Terminal/truncated snapshots and their 6D observations are returned first and remain immutable. A subsequent step on a terminal/truncated snapshot fails closed until an explicit reset.

## 7. Frozen gates

All requested correctness gates pass:

- one-step snapshot -> serialize/reload -> same action: exact
- 10-step deterministic replay: exact
- 100-step deterministic replay: exact
- same-snapshot branch A/B isolation: PASS
- native JSON vs portable NPZ snapshot: exact semantic identity
- observation projection repeat/non-mutation identity: PASS
- fee/slippage accounting check: max abs diff `1.46e-11`
- funding check: `abs(equity_delta + carry_cost) = 3.04e-12`
- recovered max-holding forced exit: PASS
- recovered margin/liquidation path: PASS
- terminal and truncation pre-reset state preservation: PASS
- finite/no impossible negative required ledger state over canary trajectory: PASS
- historical 88-step parity rerun: PASS

## 8. Multi-account compatibility

The same immutable synthetic market execution input was shared across 1, 100, 1,000 and 10,000 independent account snapshots. No account state object is shared, MarketSensory is not recomputed inside account physics, and forward/reverse account-processing order produces identical per-account snapshot hashes. `MULTI_ACCOUNT_PHYSICS_STRESS.csv` reports local scalar-reference timings only; it is not a GPU throughput claim.

## 9. Training boundary

This R0 freezes account physics state/replay infrastructure only. It does not start RL, Central Brain training, reward learning, profitability optimization, or any fresh market qualification. A later training launch must bind its exact immutable SimConfig and synchronized risk-budget authority before producing replay records.
