#!/usr/bin/env python3
"""
CB16 H5.17-DIAGNOSTIC (container-only) -- EXP-6: mechanism of the temporal
transfer failure.

EXP-5 established that state->centred-utility9 transfer is negative in 29 of 30
cross-block train/eval pairs. That failure has three distinguishable mechanisms,
and they demand completely different follow-ups:

  M1 INPUT DRIFT / EXTRAPOLATION
     The learned relation is usable, but evaluation states land off the support
     of what the training block ever saw, so the forest extrapolates badly.
     Test: compare transfer error on IN-SUPPORT evaluation rows (nearest
     training neighbour below the training 95th percentile of distances) versus
     OUT-OF-SUPPORT rows. Also test a nearest-neighbour-only baseline.

  M2 LEVEL SHIFT ONLY
     The relation transports up to a per-block offset in the utility profile.
     Test: subtract the evaluation block's OWN clock-equal mean utility before
     scoring (a legitimately available, non-cheating correction: the baseline
     itself is exactly this), i.e. evaluate the model on *centred* evaluation
     targets and compare with a zero predictor.

  M3 RELATION DIRECTION CHANGE
     Even in-support, even after level removal, the sign of the state->utility
     relation differs between blocks.
     Test: per-block Pearson correlation between predicted and actual centred
     profiles, in-support only, after removing the evaluation-block level.

Diagnostic only. Already-consumed R10.4 TRAIN cache. No new data, no FINAL,
no authority change.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import (
    H516_DIM,
    H516_UTILITY_DIM,
    clock_equal_profile_mse_h516,
    clock_equal_target_mean_h516,
)
from h517_common import fit_predict
from h517_curve2 import materialise_clocks, window

N_JOBS = int(os.environ.get("H517_JOBS", "6"))


def block_of(ts: np.ndarray, nb: int = 6) -> list[np.ndarray]:
    clocks = np.unique(ts)
    q, r = divmod(len(clocks), nb)
    out, start = [], 0
    for i in range(nb):
        n = q + (1 if i < r else 0)
        out.append(clocks[start:start + n])
        start += n
    return out


def clock_equal_centred_target(y: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Remove each clock's equal-weighted mean utility9, then each row's own mean.

    This is the 'level-corrected' evaluation target: it asks whether the model
    gets the SHAPE right once the block's average level is granted for free.
    """
    mu = clock_equal_target_mean_h516(y, ts)
    out = y - mu[None, None, :]
    return out - out.mean(axis=2, keepdims=True)


def block_cell(d, i: int, j: int, blocks, k_support: int = 200) -> dict[str, Any]:
    tr = window(d, int(blocks[i].min()), int(blocks[i].max()))
    ev = window(d, int(blocks[j].min()), int(blocks[j].max()))
    tr_x = tr["x"].reshape(-1, H516_DIM)
    ev_x = ev["x"].reshape(-1, H516_DIM)
    pred = fit_predict(tr["x"], tr["y"], tr["ts"], ev["x"])          # (n,6,9)
    ev_y = ev["y"]
    pred_flat = pred.reshape(-1, H516_UTILITY_DIM)
    ev_flat = ev_y.reshape(-1, H516_UTILITY_DIM)

    # M1: support analysis. Distance from each eval row to the nearest eval-visible
    # point of the training block, in the raw 102-d state space.
    rs = np.random.default_rng(20260911 + 31 * i + j)
    n_tr = len(tr_x)
    sub = rs.choice(n_tr, size=min(k_support, n_tr), replace=False)
    nn = NearestNeighbors(n_neighbors=1).fit(tr_x[sub])
    d_eval, _ = nn.kneighbors(ev_x)
    d_train_ref, _ = nn.kneighbors(tr_x[sub])
    thr = float(np.quantile(d_train_ref, 0.95))
    in_support = d_eval.reshape(-1, 6) < thr

    def mse(mask_rows: np.ndarray) -> float | None:
        m = mask_rows.reshape(-1, 6)
        keep = m.any(axis=1)
        if not keep.any():
            return None
        sel = np.repeat(keep, 6)
        return float(np.mean(np.mean((pred_flat[sel] - ev_flat[sel]) ** 2, axis=1)))

    # constant / zero predictors in the same geometry
    mu = clock_equal_target_mean_h516(ev["y"], ev["ts"])
    zero_pred = np.broadcast_to(mu, ev["y"].shape)
    base_mse = clock_equal_profile_mse_h516(zero_pred, ev_y, ev["ts"])
    model_mse = clock_equal_profile_mse_h516(pred, ev_y, ev["ts"])
    rows_in = np.repeat(in_support, 6)

    # M2: level-corrected evaluation (remove evaluation block's own clock-equal level)
    ev_c = clock_equal_centred_target(ev_y, ev["ts"])
    zero_c = np.broadcast_to(
        clock_equal_target_mean_h516(ev_c, ev["ts"]), ev_c.shape
    )
    base_c = clock_equal_profile_mse_h516(zero_c, ev_c, ev["ts"])
    model_c = clock_equal_profile_mse_h516(pred, ev_c, ev["ts"])

    # M3: direction agreement, in-support rows only, after level removal
    sel = rows_in if rows_in.any() else np.ones(len(ev_flat), dtype=bool)
    p = pred_flat[sel] - pred_flat[sel].mean(axis=0, keepdims=True)
    t = ev_c.reshape(-1, H516_UTILITY_DIM)[sel]
    t = t - t.mean(axis=0, keepdims=True)
    denom = float(np.sqrt(np.sum(p * p) * np.sum(t * t)))
    corr = float(np.sum(p * t) / denom) if denom > 0 else float("nan")

    return {
        "train_block": i, "eval_block": j, "gap": j - i,
        "baseline_mse": base_mse, "model_mse": model_mse,
        "relative_gain": float((base_mse - model_mse) / base_mse),
        "in_support_fraction": float(in_support.mean()),
        "mse_model_insupport": mse(in_support),
        "mse_model_outsupport": mse(~in_support),
        "baseline_mse_levelcorrected": base_c,
        "model_mse_levelcorrected": model_c,
        "levelcorrected_relative_gain": float((base_c - model_c) / base_c),
        "direction_corr_insupport": corr,
        "pred_mean_offset": [float(v) for v in (pred_flat.mean(axis=0) - ev_flat.mean(axis=0))],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=5)
    args = ap.parse_args()

    from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support
    from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6

    parents, samples, support_receipt = load_train_only_support(Path("/cb16/runtime/r104"))
    folds = build_outer_folds_r6(parents)
    f = [x for x in folds if int(x["fold"]) == args.fold][0]
    all_clocks = tuple(sorted(set(int(c) for c in f["train_clocks"]) | set(int(c) for c in f["eval_clocks"])))
    d = materialise_clocks(parents, samples, all_clocks)
    ts = np.asarray(d["timestamps_ms"], dtype=np.int64)
    clocks = np.unique(ts)
    q, r = divmod(len(clocks), 6)
    blocks, start = [], 0
    for i in range(6):
        n = q + (1 if i < r else 0)
        blocks.append(clocks[start:start + n])
        start += n

    tasks = [(i, j) for i in range(6) for j in range(6) if i != j]
    res = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(block_cell)(d, i, j, blocks) for i, j in tasks)

    rep = {
        "schema": "CB16_H517_DIAGNOSTIC_TRANSFER_FAILURE_MECHANISM_V1",
        "framing": ("DIAGNOSTIC_ONLY__ALREADY_CONSUMED_R10_4_TRAIN_CACHE__NO_NEW_DATA__"
                    "NO_FINAL__NO_NEW_VERDICT__NO_AUTHORITY_CHANGE"),
        "support_receipt": support_receipt,
        "block_sizes": [int(len(b)) for b in blocks],
        "cells": res,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=float) + "\n", encoding="utf-8")

    print("=== M1 support: in-support vs out-of-support transfer error ===")
    ins = [c["mse_model_insupport"] for c in res if c["mse_model_insupport"] is not None]
    outs = [c["mse_model_outsupport"] for c in res if c["mse_model_outsupport"] is not None]
    print(f"  mean in-support MSE  = {np.mean(ins):.4e}  (n={len(ins)})")
    print(f"  mean out-support MSE = {np.mean(outs):.4e}  (n={len(outs)})")
    print(f"  mean in-support fraction = {np.mean([c['in_support_fraction'] for c in res]):.3f}")
    basel = np.mean([c["baseline_mse"] for c in res])
    print(f"  mean constant-baseline MSE = {basel:.4e}")
    print()
    print("=== M2 level: raw vs level-corrected relative gain ===")
    print(f"  mean raw rel gain              = {100*np.mean([c['relative_gain'] for c in res]):+.2f}%")
    print(f"  mean level-corrected rel gain  = {100*np.mean([c['levelcorrected_relative_gain'] for c in res]):+.2f}%")
    pos_corr = sum(1 for c in res if c["levelcorrected_relative_gain"] > 0)
    print(f"  level-corrected positive cells = {pos_corr}/30")
    print()
    print("=== M3 direction: correlation(pred, actual) in-support after level removal ===")
    cors = [c["direction_corr_insupport"] for c in res]
    print(f"  mean corr = {np.mean(cors):+.4f}   positive cells = {sum(1 for v in cors if v>0)}/30")
    print()
    print("=== per-cell detail: gap | raw% | lvlcorr% | corr | insupport% ===")
    for c in sorted(res, key=lambda z: (z["gap"], z["train_block"])):
        print(f"  B{c['train_block']}->B{c['eval_block']} gap{c['gap']}: raw {100*c['relative_gain']:+7.2f}%  "
              f"lvl {100*c['levelcorrected_relative_gain']:+7.2f}%  corr {c['direction_corr_insupport']:+.4f}  "
              f"insup {100*c['in_support_fraction']:.1f}%")
    print(f"WROTE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
