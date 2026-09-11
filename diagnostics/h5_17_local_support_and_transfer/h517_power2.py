#!/usr/bin/env python3
"""
CB16 H5.17-DIAGNOSTIC (container-only) -- EXP-2b: detection power of the frozen
H5.16 gate on the REAL fold geometry with a KNOWN signal.

Same question as EXP-2, rewritten with process-level thread pinning and one
fold per job (the first version died with SIGSEGV from OpenMP oversubscription,
not from compute: a single frozen-RF fit costs only ~0.37 s).

Frozen H5.16 gate:
  local_map_supported := positive_gain in >=4/5 folds
                         AND positive in folds 4 AND 5
                         AND beats the median of 5 cyclic-shift controls in >=4/5
                         AND beats it in folds 4 AND 5
                         AND >=20 of 25 fold x shift wins

Reproducing the real H5.16 local arm yields 1/5, 2/5, 9/25 -> not supported.
Here we inject a synthetic state -> centred-utility9 relation of known R^2 onto
the real clock/group layout and ask how strong a signal the gate requires.

Diagnostic only. No market rows enter the model. No new data, no FINAL, no
authority change.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import (
    H516_DIM,
    H516_SHIFTS,
    H516_UTILITY_DIM,
    clock_equal_profile_mse_h516,
    clock_equal_target_mean_h516,
    shift_group_state_h516,
)
from h517_common import fit_predict

N_JOBS = int(os.environ.get("H517_JOBS", "6"))
SCEN = 6
LATENT = 8


def synth_fold(rng, n_clocks: int, gpc: int, r2: float) -> dict[str, Any]:
    n_groups = n_clocks * gpc
    ts = np.repeat(np.arange(n_clocks, dtype=np.int64) * 3_600_000 + 1_600_000_000_000, gpc)
    z = rng.normal(size=(n_groups, LATENT))
    sig = z @ rng.normal(size=(LATENT, H516_UTILITY_DIM))
    sig = sig - sig.mean(axis=0, keepdims=True)
    var_sig = max(float(np.var(sig)), 1e-12)
    resid_sd = float(np.sqrt(max(var_sig / max(r2, 1e-9) - var_sig, 0.0)))
    y = np.repeat(sig[:, None, :], SCEN, axis=1) + rng.normal(size=(n_groups, SCEN, H516_UTILITY_DIM)) * resid_sd
    y = y - y.mean(axis=2, keepdims=True)
    x = np.empty((n_groups, SCEN, H516_DIM), dtype=np.float64)
    x[:, :, :LATENT] = np.repeat(z[:, None, :], SCEN, axis=1) + 0.15 * rng.normal(size=(n_groups, SCEN, LATENT))
    x[:, :, LATENT:] = rng.normal(size=(n_groups, SCEN, H516_DIM - LATENT))
    return {"x": x, "y": y, "ts": ts, "gids": tuple(f"g{i:06d}" for i in range(n_groups))}


def _blocks(ts: np.ndarray, nb: int = 5) -> list[np.ndarray]:
    clocks = np.unique(ts)
    q, r = divmod(len(clocks), nb)
    out, start = [], 0
    for i in range(nb):
        n = q + (1 if i < r else 0)
        out.append(clocks[start:start + n])
        start += n
    return out


def fold_job(payload: dict[str, Any]) -> dict[str, Any]:
    """One evaluation environment: 5 sub-blocks, each one frozen-RF fit + controls."""
    n_clocks, gpc, r2, seed, fold = payload["n_clocks"], payload["gpc"], payload["r2"], payload["seed"], payload["fold"]
    rng = np.random.default_rng(seed)
    spec = synth_fold(rng, n_clocks, gpc, r2)
    x, y, ts, gids = spec["x"], spec["y"], spec["ts"], spec["gids"]
    per_block = []
    for bi, blk in enumerate(_blocks(ts)):
        emask = np.isin(ts, blk)
        tmask = ~emask
        tr = {"x": x[tmask], "y": y[tmask], "ts": ts[tmask],
              "gids": tuple(g for g, m in zip(gids, tmask.tolist()) if m)}
        ev = {"x": x[emask], "y": y[emask], "ts": ts[emask]}
        base = clock_equal_target_mean_h516(tr["y"], tr["ts"])
        bl = clock_equal_profile_mse_h516(np.broadcast_to(base, ev["y"].shape).copy(), ev["y"], ev["ts"])
        ml = clock_equal_profile_mse_h516(fit_predict(tr["x"], tr["y"], tr["ts"], ev["x"]), ev["y"], ev["ts"])
        gain = float(bl - ml)
        ctrl = {}
        for s in H516_SHIFTS:
            sx = shift_group_state_h516(tr["x"], tr["ts"], tr["gids"], int(s))
            ctrl[int(s)] = float(bl - clock_equal_profile_mse_h516(fit_predict(sx, tr["y"], tr["ts"], ev["x"]), ev["y"], ev["ts"]))
        med = float(statistics.median(ctrl.values()))
        per_block.append({"relative_gain": gain / bl, "positive_gain": bool(gain > 0),
                          "beats_shift_median": bool(gain > med),
                          "beats_each_shift_count": int(sum(gain > v for v in ctrl.values())),
                          "median_shift_gain": med})
    rel = float(np.mean([b["relative_gain"] for b in per_block]))
    return {
        "fold": fold, "r2": r2, "seed": seed, "relative_gain": rel,
        "positive_gain": bool(rel > 0),
        "beats_shift_median": bool(sum(1 for b in per_block if b["beats_shift_median"]) >= 3),
        "beats_each_shift_count": int(round(np.mean([b["beats_each_shift_count"] for b in per_block]))),
        "per_block_relative_gain": [b["relative_gain"] for b in per_block],
        "mean_median_shift_relative": float(np.mean([b["median_shift_gain"] / 1.0 for b in per_block])),
    }


def gate_from_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(folds, key=lambda r: r["fold"])
    pos = sum(1 for r in rows if r["positive_gain"])
    med = sum(1 for r in rows if r["beats_shift_median"])
    pw = sum(int(r["beats_each_shift_count"]) for r in rows)
    lp = all(rows[i - 1]["positive_gain"] for i in (4, 5))
    lm = all(rows[i - 1]["beats_shift_median"] for i in (4, 5))
    return {"local_map_supported": bool(pos >= 4 and lp and med >= 4 and lm and pw >= 20),
            "positive_gain_folds": pos, "beats_shift_median_folds": med, "pairwise_wins_of_25": pw,
            "both_late_positive": bool(lp), "both_late_beats_median": bool(lm)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--r2", default="0.05,0.10,0.20,0.40,0.70")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--gpc", type=int, default=10)
    args = ap.parse_args()

    from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support
    from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6

    parents, samples, support_receipt = load_train_only_support(Path("/cb16/runtime/r104"))
    folds = build_outer_folds_r6(parents)
    geometry = [(int(f["eval_clock_count"]), int(f["fold"])) for f in folds]

    r2s = [float(v) for v in str(args.r2).split(",") if v.strip()]
    jobs = []
    for r2 in r2s:
        for rep in range(args.reps):
            seed = 555_000 + rep * 7919 + int(r2 * 1e6)
            for n_clocks, fold in geometry:
                jobs.append({"n_clocks": n_clocks, "gpc": args.gpc, "r2": r2, "seed": seed, "fold": fold})
    res = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(fold_job)(j) for j in jobs)

    out: dict[str, Any] = {
        "schema": "CB16_H517_DIAGNOSTIC_H516_GATE_POWER_V2",
        "framing": ("SYNTHETIC_SIGNAL_ON_REAL_CLOCK_GEOMETRY__CALIBRATES_THE_FROZEN_H5_16_GATE__"
                    "NOT_A_MARKET_VERDICT__NO_NEW_DATA__NO_AUTHORITY_CHANGE"),
        "support_receipt": support_receipt,
        "fold_geometry": [{"eval_clocks": n, "fold": f} for n, f in geometry],
        "summary": {},
        "trials": res,
    }
    for r2 in r2s:
        rows = [r for r in res if r["r2"] == r2]
        reps_ok = []
        for rep in sorted({r["seed"] for r in rows}):
            sub = [r for r in rows if r["seed"] == rep]
            if len(sub) == 5:
                reps_ok.append(gate_from_folds(sub))
        out["summary"][f"r2={r2}"] = {
            "replicates": len(reps_ok),
            "gate_pass_rate": float(np.mean([g["local_map_supported"] for g in reps_ok])) if reps_ok else None,
            "median_positive_fold_count": float(np.median([g["positive_gain_folds"] for g in reps_ok])) if reps_ok else None,
            "median_beats_median_fold_count": float(np.median([g["beats_shift_median_folds"] for g in reps_ok])) if reps_ok else None,
            "median_pairwise_wins": float(np.median([g["pairwise_wins_of_25"] for g in reps_ok])) if reps_ok else None,
            "mean_relative_gain_per_fold": [
                float(np.mean([r["relative_gain"] for r in rows if r["fold"] == f])) for _, f in geometry
            ],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
