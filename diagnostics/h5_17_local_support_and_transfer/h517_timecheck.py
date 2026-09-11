#!/usr/bin/env python3
"""Time and memory calibration for one frozen-RF fit at each real support size."""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import numpy as np

from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import (
    H516_DIM, H516_UTILITY_DIM, build_fold_dataset_h516,
)
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6
from h517_common import fit_predict


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r104", type=Path, default=Path("/cb16/runtime/r104"))
    ap.add_argument("--folds", default="1,3,5")
    args = ap.parse_args()

    from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support

    print(f"start rss={rss_mb():.0f}MB", flush=True)
    t0 = time.time()
    parents, samples, _ = load_train_only_support(args.r104)
    print(f"loaded parents={len(parents)} samples={len(samples)} in {time.time()-t0:.1f}s rss={rss_mb():.0f}MB", flush=True)

    folds = build_outer_folds_r6(parents)
    want = {int(v) for v in str(args.folds).split(",") if v.strip()}
    for f in folds:
        fold = int(f["fold"])
        if fold not in want:
            continue
        t0 = time.time()
        d = build_fold_dataset_h516(fold_spec=f, all_train_parents=parents, all_train_samples=samples)
        x = np.asarray(d["x"], dtype=np.float64)
        y = np.asarray(d["y"], dtype=np.float64)
        print(f"fold {fold}: clocks={d['unique_decision_clock_count']} groups={d['future_group_count']} "
              f"x={x.shape} y={y.shape} build={time.time()-t0:.1f}s rss={rss_mb():.0f}MB", flush=True)
        n = x.shape[0]
        n_train = int(n * 0.8)
        tr_x, tr_y, tr_ts = x[:n_train], y[:n_train], np.asarray(d["timestamps_ms"])[:n_train]
        ev_x = x[n_train:]
        t0 = time.time()
        _ = fit_predict(tr_x, tr_y, tr_ts, ev_x)
        dt = time.time() - t0
        print(f"  ONE_FIT train_rows={n_train*6} time={dt:.2f}s rss={rss_mb():.0f}MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
