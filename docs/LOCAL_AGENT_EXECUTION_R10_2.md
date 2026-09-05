# CB16 R10.2 local execution authority

## 1. Do not reinterpret the architecture

The active production path is fixed:

```text
1m Binance archive
  -> exact UTC 1h aggregation
  -> Frozen Operator48: Kronos D45(L5-L4) Micro24 + Macro24
  -> Frozen Medium48: TimesFM Layer3 -> frozen Nonlinear48
  -> typed Operator48 + Medium48

SimulatorStateSnapshotV1 -> deterministic AccountState6

Operator48 / Medium48 / AccountState6
  -> TRAINABLE Central Brain-owned Operator/Medium/Account stems
  -> Shared Decision Core
  -> Direction Head + direction-conditioned Requested-Risk Head
  -> ActionIntentV1
  -> Frozen Risk Supervisor R1
  -> ExecutableActionV1
  -> exact recovered V5.5 FrozenTradingKernel
```

The historical checkpoint names `operator_encoder`, `medium_encoder`, and `account_encoder` are **Central Brain stems**, not frozen organ encoders. All 189,052 G0 parameters are trainable. The values entering those stems are detached; Frozen Operator48, Medium48 and AccountState6 have no gradient authority.

`Ordered4H30` remains a task-gated shadow risk sidecar. It is stored in the cache but is **not** passed to the nominal Central Brain. Remote remains OFF.

## 2. Scientific boundaries that code enforces

- `Truth != Belief != Decision != Permission`.
- A realized H72 path is a stochastic economic sample, not `BEST_ACTION` or `CORRECT_DIRECTION`.
- Teacher emits utility distributions / soft direction probability / continuous requested-risk target.
- All account scenarios and all 9 counterfactual actions at the same `(symbol, decision_time)` share one dependence-group ID.
- 1,000 account replicas would still be one future support group if they share that market path.
- Brain proposes. Supervisor permits/clamps/rejects. Physics owns the ledger and exits.
- Funding is an event. Only the exact 00/08/16-style funding timestamp bar receives the archived rate. No forward fill.
- Separate historical mark/index archives are absent, so the already-frozen adapter semantics use 1h close explicitly as mark/index proxy.
- H72: decision at start after closed t-bar; first execution bar begins at decision timestamp; exactly 72 future hourly bars are consumed.
- Train/validation boundary includes a 128h purge after H72 maturity.
- Any attempt to read an archive month `2025-09` fails closed. R10.2/R10.3/R10.4 have no legal FINAL-opening path.

## 3. R10.2 frozen protocol

Canonical status-driving assets:

`BTCUSDT ETHUSDT BNBUSDT SOLUSDT`

All ten downloaded assets are checked for ingest compatibility, but the first five-generation qualification keeps the four-asset scientific lineage stable.

Each market anchor is globally aligned every 512h. At each anchor six deterministic account histories are created with exact Physics:

- CLEAN_FLAT_FULL
- CLEAN_FLAT_LOW_ENVELOPE
- PRIOR_LONG_R025
- PRIOR_SHORT_R025
- PRIOR_LONG_R075
- PRIOR_SHORT_R075

The prior-position scenarios create different realized account histories but are flat/actionable again at the decision anchor. Invalid/terminal parents are retained diagnostically but are excluded from Teacher evidence.

Each eligible parent has exactly nine H72 branches:

```text
FLAT  0
SHORT .25 .50 .75 1.00
LONG  .25 .50 .75 1.00
```

Branch 1 goes through Risk Supervisor + exact Physics on the first future bar. On bars 2..72 nominal action is FLAT/0; an open position therefore produces `FORCED_NOOP`, while stop/take/liquidation/time-stop remain Physics-owned. If an unfinished position somehow survives H72, the byte-exact kernel `finalize()` is used only as an explicitly receipted **evaluation-end** close. It is not a new trading rule.

Nonpositive initial/terminal equity is not clipped. Such a parent is censored from the finite Teacher because the inherited Teacher forbids nonfinite utility. The receipt records the censoring.

## 4. Teacher and training

Training Teacher: inherited dependence-aware R6, blocked cross-fit, fixed 5 folds, one-group embargo, k=64, minimum 32 train future groups, minimum effective dependence N=12. No tuning after seeing R10.2 output.

Validation Teacher: prequential, and its eligible support is restricted to TRAIN dependence groups.

A training snapshot contains only admitted evidence. Snapshot ID and content hash are frozen before optimizer execution. Snapshot consumption is exactly once. Crash recovery uses a durable trained-Challenger checkpoint rather than rerunning an already-consumed snapshot.

Optimizer: FP32 AdamW, AMP OFF, 12 epochs, batch 512, lr 3e-4. The six Brain groups must all receive nonzero real-evidence gradients and nonzero updates.

## 5. Tournament

Tournament is deliberately not a realized-profit contest. It uses fixed validation probabilistic-Teacher target loss:

- Challenger must lower validation loss.
- Relative improvement must be at least 0.1%.
- If it fails, `REJECT`; the unchanged Champion is the parent of the next attempt.
- If it passes, `PROMOTE`; Challenger becomes next Champion.

Five generation attempts are executed even if some/all are rejected. Pipeline PASS does not require a promotion.

## 6. F0/F1/F2/F3

The package separately evaluates group-weighted discrete qCRPS:

- F0 climatology
- F1 Account only
- F2 true Market + Account
- F3 deterministic dependence-group shuffled Market + Account

`F3 - F2 > 0` means true market is better because qCRPS is lower. A negative value is stored as a negative scientific result. It is **not** allowed to block a mechanistically correct pipeline and it is **never** post-hoc rescued.

## 7. Exact commands

```bash
source /home/bgy/m3-infra/.venv-r8-pascal/bin/activate
cd /home/bgy/m3-infra/CB16_SHANXI_R10_2_REAL_HISTORICAL_G0_LEARNING_V1
python scripts/verify_package_r102.py
python scripts/run_r102_pipeline.py
```

Status at any point:

```bash
python scripts/r102_status.py
```

Rerunning `run_r102_pipeline.py` is legal after interruption. Completed generation results are reused. A completed snapshot cannot be consumed twice.

## 8. Autonomous continuation already implemented

After and only after R10.2 `...PIPELINE_PASS`:

```bash
python scripts/run_r103_expansion.py
```

R10.3 uses all ten assets, 20 generation attempts, same H72/Teacher/Physics/final-lock semantics, stride 512h. It starts from R10.2 final Champion.

After and only after R10.3 `...PASS`:

```bash
python scripts/run_r104_long_research.py
```

R10.4 uses all ten assets, 100 research generation attempts, stride 256h. Default run root is on HDD to reduce SSD pressure. A hard stop triggers below 10 GiB free space. It remains research, not FINAL.

There is no automatic R10.5 FINAL. `scripts/final_holdout_LOCKED.py` always aborts. A separately versioned manual opening adjudication is required before 2025-09 can ever be read.

## 9. What to return after a run

R10.2 creates:

`/home/bgy/cb16_ssd/runtime/R10_2/CB16_R10_2_RETURN_RECEIPTS_R0.tar.gz`

Return that small file. It contains receipts, F0-F3 result, generation lineage and compact Champion checkpoints, not raw data or frozen foundation tensors.

If R10.3/R10.4 are run autonomously, their run roots create `CB16_R10_3_RETURN_RECEIPTS_R0.tar.gz` and `CB16_R10_4_RETURN_RECEIPTS_R0.tar.gz` respectively.

## 10. Forbidden local-Agent improvisations

Do not:

- change layer taps;
- fine-tune Kronos/TimesFM;
- refit reducers/Nonlinear48;
- put Ordered4H30 into nominal Brain input;
- enable Remote;
- reinterpret requested risk as confidence;
- use one realized winner as a class label;
- count counterfactual/account branches as independent market samples;
- change Teacher thresholds because evidence admission is low;
- change tournament threshold because promotions are rare;
- open 2025-09;
- use a different backtester because exact V5.5 Physics is slower;
- replace a scientific FAIL with a new threshold after observing it.
