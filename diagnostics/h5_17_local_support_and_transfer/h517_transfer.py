#!/usr/bin/env python3
"""
CB16 H5.17-DIAGNOSTIC (container-only) -- EXP-5: fit-vs-generalisation and the
temporal transfer matrix.

Two measurements on the real frozen TRAIN payload, both diagnostic:

A. UNDERFIT vs OVERFIT
   For a fixed training window, report the frozen RF's IN-SAMPLE profile MSE and
   its OUT-OF-SAMPLE profile MSE on a later evaluation block. If in-sample error
   is also high, the estimator/target are not being fitted (capacity problem).
   If in-sample error is low and out-of-sample high, the relation does not
   transport (non-stationarity / regime problem).

B. TEMPORAL TRANSFER MATRIX
   Using the 141 distinct TRAIN decision clocks split into 6 contiguous blocks,
   train a frozen RF on every single block and evaluate on every OTHER block.
   The resulting 6x6 matrix measures how well state->centred-utility9 transfers
   as a function of temporal separation. It answers directly whether the
   representation is time-invariant in the sense the whole H5.x chain assumes.

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

from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import (
    H516_SHIFTS,
    clock_equal_profile_mse_h516,
    clock_equal_target_mean_h516,
    shift_group_state_h516,
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


def cell(d, tr_blk, ev_blk) -> dict[str, Any]:
    tr = window(d, int(tr_blk.min()), int(tr_blk.max()))
    ev = window(d, int(ev_blk.min()), int(ev_blk.max()))
    mu = clock_equal_target_mean_h516(tr["y"], tr["ts"])
    base_loss = clock_equal_profile_mse_h516(np.broadcast_to(mu, ev["y"].shape).copy(), ev["y"], ev["ts"])
    pred = fit_predict(tr["x"], tr["y"], tr["ts"], ev["x"])
    model_loss = clock_equal_profile_mse_h516(pred, ev["y"], ev["ts"])
    train_pred = fit_predict(tr["x"], tr["y"], tr["ts"], tr["x"])
    train_loss = clock_equal_profile_mse_h516(train_pred, tr["y"], tr["ts"])
    return {
        "train_clocks": tr["n_clocks"], "train_groups": tr["n_groups"], "eval_clocks": ev["n_clocks"],
        "baseline_loss": base_loss, "model_loss": model_loss, "in_sample_loss": train_loss,
        "relative_gain": float((base_loss - model_loss) / base_loss),
        "train_to_test_ratio": float(train_loss / model_loss) if model_loss > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=5, help="cumulative fold used for the fit-vs-gen part")
    args = ap.parse_args()

    from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support
    from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6

    parents, samples, support_receipt = load_train_only_support(Path("/cb16/runtime/r104"))
    folds = build_outer_folds_r6(parents)
    f5 = [f for f in folds if int(f["fold"]) == args.fold][0]
    all_clocks = tuple(sorted(set(int(c) for c in f5["train_clocks"]) | set(int(c) for c in f5["eval_clocks"])))
    d = materialise_clocks(parents, samples, all_clocks)
    ts = np.asarray(d["timestamps_ms"], dtype=np.int64)
    blocks = block_of(ts, 6)

    # A. fit vs generalisation at several training sizes on the fixed eval block
    eval_blk = blocks[5]
    fitgen = []
    past = np.concatenate(blocks[:5])
    for k in (12, 24, 48, 96, len(past)):
        k = min(k, len(past))
        tr_blk = np.asarray(sorted(past)[-k:], dtype=np.int64)
        r = cell(d, tr_blk, eval_blk)
        r["train_clock_count_requested"] = int(k)
        fitgen.append(r)
        print(json.dumps({"stage": "fitgen", "k": int(k), "in_sample": r["in_sample_loss"],
                          "oos": r["model_loss"], "base": r["baseline_loss"],
                          "ratio": r["train_to_test_ratio"], "rel%": round(100 * r["relative_gain"], 2)}), flush=True)

    # B. full 6x6 temporal transfer matrix
    cells = []
    tasks = []
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            tasks.append((i, j))
    res = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(cell)(d, blocks[i], blocks[j]) for i, j in tasks
    )
    matrix = [[None] * 6 for _ in range(6)]
    for (i, j), r in zip(tasks, res):
        matrix[i][j] = r
        cells.append({"train_block": i, "eval_block": j, "temporal_gap_blocks": j - i, **r})

    rep = {
        "schema": "CB16_H517_DIAGNOSTIC_FITGEN_AND_TEMPORAL_TRANSFER_V1",
        "framing": ("DIAGNOSTIC_ONLY__ALREADY_CONSUMED_R10_4_TRAIN_CACHE__NO_NEW_DATA__"
                    "NO_FINAL__NO_NEW_VERDICT__NO_AUTHORITY_CHANGE"),
        "support_receipt": support_receipt,
        "block_sizes": [int(len(b)) for b in blocks],
        "fit_vs_generalisation": fitgen,
        "transfer_matrix": matrix,
        "transfer_cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=float) + "\n", encoding="utf-8")
    print("\n=== TRANSFER MATRIX: relative gain % (rows=train block, cols=eval block) ===", flush=True)
    hdr = "        " + "".join(f"B{j:<9}" for j in range(6))
    print(hdr, flush=True)
    for i in range(6):
        row = f"B{i:<7}"
        for j in range(6):
            if i == j:
                row += "   --     "
            else:
                row += f"{100*matrix[i][j]['relative_gain']:>+8.2f}%"
        print(row, flush=True)
    print("\n=== in-sample/out-of-sample ratio ===", flush=True)
    for i in range(6):
        row = f"B{i:<7}"
        for j in range(6):
            if i == j:
                row += "   --     "
            else:
                row += f"{matrix[i][j]['train_to_test_ratio']:>9.2f}"
        print(row, flush=True)
    print(f"WROTE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
