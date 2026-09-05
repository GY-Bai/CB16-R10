# CB16 Ordered4H30 Predictive Risk Advisor Shadow Candidate R2

## Final status

**`ORDERED4H30_ADVISOR_SHADOW_CANDIDATE_FROZEN`**

本轮只冻结一个 **shadow-only** predictor/runtime candidate。没有打开 fresh cohort，没有 prospective qualification，没有 active intervention，也没有训练 Central Brain。

## Frozen binding

- Runtime market input: **Ordered4H30 / 30 floats only** (`738ed11011a7749b0ddf182ed5538d2c6c0b118ab24bb2bbf206c9a65e4a5048`).
- Raw24: scientific ceiling/reference only; **not a runtime input**.
- Predictor family: inherited LightGBM distributional readout family (`n_estimators=120`, `num_leaves=15`, fixed seed 24680); no architecture tournament.
- Primary event: `Z64 = 1{max_margin_utilization_excess_H64 >= 0.20}`.
- Continuous companion: conditional `max_margin_utilization_excess_H64` q10/q50/q90.
- Pairwise relief: `rho(A->B|I_t)=P(Z64(A)=1, Z64(B)=0 | I_t)` from the inherited two-stage discordance/direction evaluator.

## Supported concrete edges

| Edge | Family | Level |
|---|---|---:|
| LONG .75 -> .25 | risk_multiplier_long | 1 |
| SHORT .75 -> .25 | risk_multiplier_short | 1 |
| SHORT .25 -> FLAT | flat_vs_short | 2 |
| SHORT .75 -> FLAT | flat_vs_short | 2 |

`flat_vs_long`, `LONG->FLAT`, and `long_vs_short` remain unsupported. Parallel Task B cannot alter this R2 object.

## Development-only calibration diagnostics

- Action event ECE: **0.034148** (diagnostic ceiling 0.10).
- Max-margin q10/q50/q90 mean absolute coverage error: **0.033519** (diagnostic ceiling 0.08).
- Max supported-edge rho ECE: **0.097537** (diagnostic ceiling 0.10).
- Diagnostic calibration pass: **True**.

These values are **DEVELOPMENT_ONLY**. They are not a CRC certificate and do not support `QUALIFIED`, `ACTIVE`, or `FORMALLY_SAFE`.

## Runtime behavior

`PredictiveRiskAdvisorCandidateV1` validates representation/account/action/support lineage before inference. Supported rows receive risk and relief predictions in the shadow log. The candidate deliberately leaves `suggested_intervention_level` unset because R1 did not freeze a formally calibrated operating threshold/lambda. The monotone R1 intervention graph is replayed only on a diagnostic lambda grid to verify ordering. Original `ActionIntentV1` is never mutated.

NO_ADVICE is deterministic for representation mismatch, unsupported action family/risk anchor, positioned account, nonfinite input, low/unsupported envelope, or lineage/support mismatch.

## Canaries

- Reload exact: **True**; portable reload tolerance: **True**.
- OOD/NO_ADVICE canaries: **True**.
- Hard non-override canaries: **True**.
- Frozen Risk Supervisor source SHA unchanged: **True**.
- Shadow lambda-path monotonicity: **True**.

## Stopping condition

This bundle is a serializable/replayable **shadow predictor candidate**, not a safety qualification. The next lawful step is a separately authorized prospective statistical-control qualification using this frozen object; no active intervention is opened here.
