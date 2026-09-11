#!/usr/bin/env python3
"""
CB16 H5.17-DIAGNOSTIC (container-only) -- EXP-4: control-null calibration on real rows.

The frozen H5.16 local arm judges the true binding against FIVE CYCLIC SHIFTS of
the training state bundle. A cyclic shift preserves the cyclic neighbour
structure: with n groups and shift s, every destination keeps the source that sat
s positions away, and a full rotation makes the multiset of (destination, source)
pairs invariant under relabelling of the cycle.

If the state geometry is temporally smooth, a small shift moves a state bundle
only a little in time, so the shift control is a STRONGER null than "X carries no
information about Y". This experiment quantifies that directly on the real rows:

  * repeat the H5.16 fold-internal crossfit,
  * replace the 5 cyclic shifts with R independent uniform permutations of the
    training state bundle (a strictly stronger disruption, no temporally local
    survivors),
  * report where the true binding gain ranks inside the permutation null.

If the true gain sits INSIDE the permutation null, the H5.16 negative is
corroborated: the frozen representation carries no detectable forward
state->utility9 signal at this support. If the true gain sits ABOVE the
permutation null while at/below the shift null, then the preregistered criterion
is conservative for this representation and its negative must be read more
carefully.

Diagnostic only. Already-consumed R10.4 TRAIN cache. No new data, no FINAL,
no authority change.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import (
    H516_SHIFTS,
    build_fold_dataset_h516,
    clock_blocks_h516,
    clock_equal_profile_mse_h516,
    clock_equal_target_mean_h516,
    run_local_arm_h516,
    shift_group_state_h516,
)
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from h517_common import fit_predict

R104_ROOT = Path("/cb16/runtime/r104")
N_JOBS = 10


def permutation_control_gain(x, y, ts, gids, eval_x, eval_y, eval_ts, rng) -> float:
    """Uniform (non-cyclic) permutation of the training state bundle."""
    n = x.shape[0]
    perm = rng.permutation(n)
    while np.array_equal(perm, np.arange(n)):
        perm = rng.permutation(n)
    px = np.ascontiguousarray(x[perm])
    return float(
        base_loss(eval_y, eval_ts)
        - clock_equal_profile_mse_h516(fit_predict(px, y, ts, eval_x), eval_y, eval_ts)
    )


def base_loss(y, ts) -> float:
    mu = clock_equal_target_mean_h516(y, ts)
    return float(clock_equal_profile_mse_h516(np.broadcast_to(mu, y.shape).copy(), y, ts))


def fold_task(payload: dict[str, Any]) -> dict[str, Any]:
    d = payload["dataset"]
    n_perm = payload["n_perm"]
    seed = payload["seed"]
    fold = int(d["fold"])
    x = np.asarray(d["x"], dtype=np.float64)
    y = np.asarray(d["y"], dtype=np.float64)
    ts = np.asarray(d["timestamps_ms"], dtype=np.int64)
    gids = tuple(str(g) for g in d["future_group_ids"])
    blocks = clock_blocks_h516(ts)
    rng = np.random.default_rng(seed)
    per_block = []
    for bi, blk in enumerate(blocks):
        emask = np.isin(ts, blk)
        tmask = ~emask
        tr_x, tr_y, tr_ts = x[tmask], y[tmask], ts[tmask]
        tr_gids = tuple(g for g, m in zip(gids, tmask.tolist()) if m)
        ev_x, ev_y, ev_ts = x[emask], y[emask], ts[emask]
        bl = base_loss(ev_y, ev_ts)
        true_gain = float(bl - clock_equal_profile_mse_h516(fit_predict(tr_x, tr_y, tr_ts, ev_x), ev_y, ev_ts))
        # frozen cyclic-shift control (as adjudicated)
        cyc = {}
        for s in H516_SHIFTS:
            sx = shift_group_state_h516(tr_x, tr_ts, tr_gids, int(s))
            cyc[int(s)] = float(bl - clock_equal_profile_mse_h516(fit_predict(sx, tr_y, tr_ts, ev_x), ev_y, ev_ts))
        # uniform-permutation control (stronger disruption)
        perms = []
        for r in range(int(n_perm)):
            rr = np.random.default_rng(seed * 1000 + bi * 137 + r)
            pm = rr.permutation(len(tr_ts))
            while np.array_equal(pm, np.arange(len(tr_ts))):
                pm = rr.permutation(len(tr_ts))
            px = np.ascontiguousarray(tr_x[pm])
            perms.append(float(bl - clock_equal_profile_mse_h516(fit_predict(px, tr_y, tr_ts, ev_x), ev_y, ev_ts)))
        per_block.append({
            "block": bi + 1,
            "baseline_loss": bl,
            "true_gain": true_gain,
            "cyclic_gains": cyc,
            "median_cyclic_gain": float(statistics.median(cyc.values())),
            "perm_gains": perms,
            "median_perm_gain": float(np.median(perms)),
            "perm_p_value_upper": float((1 + sum(1 for g in perms if g >= true_gain)) / (1 + len(perms))),
            "true_above_perm_median": bool(true_gain > float(np.median(perms))),
        })
    rel = [b["true_gain"] / b["baseline_loss"] for b in per_block]
    return {
        "fold": fold,
        "blocks": per_block,
        "true_relative_gain": [float(v) for v in rel],
        "mean_true_relative_gain": float(np.mean(rel)),
        "mean_perm_relative_gain": float(np.mean([b["median_perm_gain"] / b["baseline_loss"] for b in per_block])),
        "mean_cyclic_relative_gain": float(np.mean([b["median_cyclic_gain"] / b["baseline_loss"] for b in per_block])),
        "blocks_true_above_perm_median": int(sum(1 for b in per_block if b["true_above_perm_median"])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()

    from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

    parents, samples, support_receipt = load_train_only_support(R104_ROOT)
    folds = build_outer_folds_r6(parents)
    datasets = [
        build_fold_dataset_h516(fold_spec=f, all_train_parents=parents, all_train_samples=samples) for f in folds
    ]
    out = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(fold_task)({"dataset": d, "n_perm": args.n_perm, "seed": args.seed + 13 * int(d["fold"])})
        for d in datasets
    )
    report = {
        "schema": "CB16_H517_DIAGNOSTIC_CONTROL_NULL_CALIBRATION_V1",
        "framing": ("DIAGNOSTIC_ONLY__REAL_TRAIN_ROWS__PURE_UNIFORM_PERMUTATION_NULL_VS_FROZEN_CYCLIC_SHIFT_NULL__"
                    "NO_NEW_DATA__NO_FINAL__NO_AUTHORITY_CHANGE"),
        "support_receipt": support_receipt,
        "n_permutations": args.n_perm,
        "folds": out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({
        "per_fold": [
            {
                "fold": r["fold"],
                "true_rel%": round(100 * r["mean_true_relative_gain"], 3),
                "cyclic_median_rel%": round(100 * r["mean_cyclic_relative_gain"], 3),
                "perm_median_rel%": round(100 * r["mean_perm_relative_gain"], 3),
                "blocks_true_above_perm_median": r["blocks_true_above_perm_median"],
            }
            for r in out
        ]
    }, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
