# VOID NOTICE: EXP-2b and EXP-6

Two experiments in this directory are **void**. Their conclusions are withdrawn.
See `H5_17_DIAGNOSTIC_REPORT_V2.md` sections 0-5 for the corrected analysis.

## EXP-2b (`exp2_power.json`, `h517_power2.py`) -- DOWNGRADED

Claim withdrawn: "the original H5.16 criterion was exactly power-calibrated".

Defects:
- D4b: the criterion was REIMPLEMENTED (mean of per-block relative gains,
  majority vote over blocks, rounded mean of block shift wins) instead of
  calling the frozen `run_local_arm_h516`. The two are not equivalent.
- D4: within one replicate every fold used the same RNG stream, and folds 2-5
  all have 28 clocks, so folds 2-5 were generated from IDENTICAL data.
  `folds_2_to_5_identical = true` in `h517_bugcheck.py`.
  The repeated tail in the v1 summary `[8.84, 8.93, 8.93, 8.93, 8.93]` is
  that artefact.

Replacement: **EXP-2c** (`exp2c_power.json`, `h517_revised.py --mode power`)
calls the frozen helper, applies the frozen boolean expression verbatim, and
uses a per-fold independent stream (`distinct_per_fold_streams = true`).

## EXP-6 (`exp6_mechanism.json`, `h517_mechanism.py`) -- VOID

Claims withdrawn:
- "zero support overlap; every evaluation point lies outside the training support"
- "removing the level explains only 2.3pp"

Defects:
- D1 SELF-MATCH: `nn.fit(tr_x[sub]); nn.kneighbors(tr_x[sub])` does not exclude
  the query point from its own reference set, so the reference distance
  degenerates to floating-point noise (max 3.77e-07, p95 threshold 2.92e-07 in
  a 102-d reproduction). Landing on exactly 0 makes `d_eval < 0` always false
  and the coverage identically 0.
- D2 MASK UNITS: `in_support` is already `(groups, 6)` and was passed through
  `np.repeat(..., 6)`, producing a mask 6x longer than the array it indexes
  (1440 vs 240). Any non-empty mask raises
  `IndexError: boolean index did not match indexed array along axis 0`.
  The all-False mask silently took the fallback branch, which is why the defect
  was not visible.
- D3 LEVEL CORRECTION: the evaluation target was centred but the prediction was
  not, so a PERFECT prediction is penalised (raw MSE 0.0 -> "corrected" 0.00255).
- Additionally the `direction_corr_insupport` field is misnamed: when the mask is
  empty the code falls back to ALL rows, so the value is an all-row,
  row-equal-weighted nine-action utility correlation.

Replacement: **EXP-6S** (`exp6s_state.json`, `h517_state_structure.py`) which
measures the actual state structure and redoes the support question in the
correct unit (future group, not row).

Independent reproduction of all four defects: `h517_bugcheck.py`.
