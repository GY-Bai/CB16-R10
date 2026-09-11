# H5.17 Three-Arm Forward Representation Experiment — Exploratory Execution Spec V3

**Status:** FROZEN FOR THIS EXPLORATORY RUN ONLY — NOT SCIENTIFIC AUTHORITY.

Base: `bb072a1eb8d6dbf1b6228a69539c5f4ca143cc5b`.

This file resolves the four issues identified in the 2026-09-11 V2 preregistration review before any new market execution. It does not modify H5.16, the frozen market-information verdict, canonical Teacher/Student/Physics/Supervisor, or any authority file.

## 1. Scope and data boundary

- Use only the already-consumed R10.4 TRAIN support loaded by the existing train-only loader.
- FINAL remains sealed; no fresh market data.
- Use the five frozen R6 outer-fold **train → eval** pairs as the five evaluation blocks.
- Each fold must satisfy a minimum **256 h** train-to-eval clock gap. The frozen utility horizon is **72 h**; failure of the observed gap assertion blocks execution.
- All transformations that learn from data are fitted on that fold's training side only.

## 2. Fixed target and loss

Target is unchanged from H5.16: per-row centered 9-action realized-utility profile.

Primary loss is unchanged: mean over actions, then scenarios within future group, then future groups within decision clock, then equal-weight mean over decision clocks.

No realized winner becomes a correct-action label.

## 3. Three arms

**A — Full102.** Raw Operator48 + Medium48 + Account6, with the exact H5.16 fixed RandomForestRegressor.

**B — blind compression.** Fit Market96 mean/std on the training side only; apply a fixed orthonormal Gaussian projection `96→8`, seed `51708`; concatenate raw Account6; fit the same H5.16 RF.

**C — task-supervised linear bottleneck.** Fit Market96 mean/std on training only. Fit one deterministic weighted joint ridge regression from standardized Market96 + standardized Account6 to centered Utility9, `alpha=1.0`. Take the top 8 left singular vectors of the market coefficient block as the linear Market96 encoder. Fit a second weighted linear auxiliary head on bottleneck8 + standardized Account6 for a diagnostic training loss. Freeze the encoder, discard the auxiliary head, concatenate bottleneck8 + raw Account6, and fit the same H5.16 RF.

Dimension is fixed at 8. No dimension, seed, loss, model, or ridge-alpha scan is allowed.

Because RF `max_features=10` means a different effective feature fraction in 102D vs 14D, this experiment tests **pipeline improvement**, not a pure causal representation effect.

## 4. Negative controls

For C, use the exact five H5.16 whole-future-group state shifts: `1, 7, 13, 23, 31`.

For every shift and every fold:
1. shift the complete six-scenario training state bundle;
2. keep Utility9 targets fixed;
3. fit the C encoder **from scratch** on the shifted binding;
4. freeze that encoder;
5. fit a fresh fixed RF;
6. evaluate the unshifted eval side.

No encoder trained on the real binding may be reused by a misalignment control.

## 5. Forward positive and null controls

Before market evaluation, run the exact three-arm forward implementation on synthetic data with the real six-scenario/group structure.

Positive control: one fixed latent `W` is shared across all five train/eval blocks; only samples/noise change. It must satisfy the same C>A, C>B, constant-baseline, and from-scratch-shift gate.

Null control: Utility9 is independent centered noise. It must **not** satisfy the complete C success gate.

These are pathway checks, not a universal power claim and not a market R² detection floor.

## 6. Pairwise questions and aggregation

All comparisons use identical eval clocks and clock-equal loss. Define positive `delta` as comparator loss minus candidate loss.

- **B vs A:** does blind compression improve the existing pipeline?
- **C vs A:** does supervised compression improve the existing pipeline?
- **C vs B:** does task supervision add value beyond blind compression?
- **C vs constant:** does C beat a train-side constant Utility9 baseline?
- **C vs shifted C:** does the real binding beat from-scratch broken-binding controls?

For B>A, C>A, C>B, and C>constant, a comparison passes only if:
- delta is positive in at least 4/5 evaluation folds;
- folds 4 and 5 are both positive;
- pooled clock-equal delta across all five eval folds is positive.

For C vs shifted C, require the same fold/late/pooled rule against the median of five shift losses **and** at least 20/25 fold×shift wins.

`task_supervised_representation_improves_pipeline = C>A PASS AND C>B PASS AND C>constant PASS AND C>shift PASS`.

B>A is reported independently and is not required for C success.

## 7. Corrected level/shape diagnostic

For normalized clock/group/scenario weights `w`, define 9D block means

`mu_P = Σ_i w_i P_i`, `mu_Y = Σ_i w_i Y_i`.

Then

`MSE_w(P,Y) = mean_9((mu_P-mu_Y)^2) + MSE_w(P-mu_P, Y-mu_Y)`.

Known-answer tests must pass for perfect prediction, pure 9D profile offset `[1,-1,0,...]`, and pure zero-mean shape error. Any use of eval `Y` here is post-hoc error decomposition only; it is never fed back into a deployable model.

## 8. Support diagnostic — explicitly non-gating

Support is reported separately in:
- Market96 at future-group level;
- Account6 using actual scenario rows;
- joint Full102 using actual scenario rows.

Standardization is fitted on the reference side only. Reference vs calibration is split by complete decision clocks, so a group cannot appear on both sides. Thresholds are calibrated from calibration-query→reference distances, then applied unchanged to eval-query→reference distances.

Coverage is an empirical distance diagnostic only. It may be low, high, or uninterpretable; it cannot veto a real predictive improvement and cannot establish mathematical common support.

## 9. Interpretation limits

A failure means only: **this fixed linear 8D supervised bottleneck + fixed RF pipeline did not meet the exploratory improvement gate.**

It does **not** establish:
- compact representations are useless;
- model capacity is not a bottleneck;
- a single-time state is necessarily insufficient;
- sequence/hidden-state models are required;
- more compute cannot help.

A success remains exploratory because all market intervals are already consumed by prior diagnosis/design. It does not reopen `DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`.

## 10. Required durable output

Save `RESULT.json`, the execution spec, SHA256 sums, runner preflight, and Python-environment receipt as one Actions artifact. No authority/adjudication write is authorized by this run.
