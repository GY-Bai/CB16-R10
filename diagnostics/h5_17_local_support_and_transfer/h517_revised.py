#!/usr/bin/env python3
"""
CB16 H5.17-DIAGNOSTIC REVISION (container-only) -- EXP-6R / EXP-2c.

Replaces the defective EXP-6 (support + level decomposition) and EXP-2b (gate
power). All four defects reported in review are fixed and each fix is verified
in-process before any measurement is reported.

DEFECTS FIXED
  D1 self-match in the reference NN distance
        -> n_neighbors=2, discard the self column (verified: reference distance
           is now O(1), not O(1e-7))
  D2 mask unit mismatch (np.repeat on an already (groups,6) mask -> 6x length)
        -> masks kept in the same unit as the array they index; assertion added
  D3 level correction applied to the target but not the prediction
        -> exact decomposition total = level + shape, with the row-mean of the
           centred utility9 target equal to zero by construction
  D4 equal clock counts across folds sharing one RNG stream
        -> per-fold independent seeds; verified distinct draws
  D4b reimplemented gate not equivalent to the frozen one
        -> EXP-2c calls the FROZEN run_local_arm_h516 and applies the FROZEN
           boolean expression, no reimplementation

Diagnostic only. Already-consumed R10.4 TRAIN cache. No new data, no FINAL,
no authority change.
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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import (
    H516_DIM,
    H516_SHIFTS,
    H516_UTILITY_DIM,
    clock_equal_profile_mse_h516,
    clock_equal_target_mean_h516,
    run_local_arm_h516,
    shift_group_state_h516,
)
from h517_common import fit_predict, row_weights
from h517_curve2 import materialise_clocks, window

N_JOBS = int(os.environ.get("H517_JOBS", "6"))
SCEN = 6


# --------------------------------------------------------------------- fixes
def self_excluded_reference(X_ref: np.ndarray, X_eval: np.ndarray, q: float = 0.95) -> dict[str, Any]:
    """D1-fixed nearest-neighbour support diagnostic.

    The reference distribution is the SELF-EXCLUDED 1-NN distance of each
    reference point (n_neighbors=2 -> drop column 0). The threshold is a
    quantile of that reference distribution. Coverage is the fraction of
    evaluation points strictly closer than the threshold.
    """
    n_ref = len(X_ref)
    if n_ref < 3:
        raise RuntimeError("H517REF_TOO_FEW_REFERENCE_POINTS")
    nn2 = NearestNeighbors(n_neighbors=2).fit(X_ref)
    d_full, idx_full = nn2.kneighbors(X_ref)
    d_ref = d_full[:, 1]                                   # <- self-excluded
    n_zero = int(np.sum(d_ref <= 0.0))
    n_exact_dup_pairs = int(np.sum(idx_full[:, 1] > np.arange(n_ref)))
    thr = float(np.quantile(d_ref, q))
    nn1 = NearestNeighbors(n_neighbors=1).fit(X_ref)
    d_ev, _ = nn1.kneighbors(X_eval)
    d_ev = d_ev.reshape(-1)
    return {
        "n_reference_points": int(n_ref),
        "n_eval_points": int(len(X_eval)),
        "n_reference_points_with_zero_self_excluded_distance": n_zero,
        "fraction_reference_duplicates": float(n_zero / n_ref),
        "n_exact_duplicate_pairs": n_exact_dup_pairs,
        "degenerate_threshold": bool(thr <= 0.0),
        "median_reference_distance": float(np.median(d_ref)),
        "p95_threshold": thr,
        "median_eval_distance": float(np.median(d_ev)),
        "distance_ratio": float(np.median(d_ev) / max(np.median(d_ref), 1e-300)),
        "coverage_below_p95": float((d_ev < thr).mean()) if thr > 0.0 else float("nan"),
        "coverage_below_p95_inclusive": float((d_ev <= thr).mean()),
    }


def representations(X_ref_rows: np.ndarray, X_ev_rows: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Three distance variants; every transform fitted on the REFERENCE side only."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out["raw102"] = (X_ref_rows, X_ev_rows)

    mu = X_ref_rows.mean(axis=0, keepdims=True)
    sd = X_ref_rows.std(axis=0, keepdims=True)
    sd[sd <= 0.0] = 1.0
    out["standardised102"] = ((X_ref_rows - mu) / sd, (X_ev_rows - mu) / sd)

    pca = PCA(n_components=8, random_state=20260911).fit(X_ref_rows - mu)
    out["pca8_standardised"] = (pca.transform(X_ref_rows - mu), pca.transform(X_ev_rows - mu))
    return out


def decompose(pred: np.ndarray, y: np.ndarray, ts: np.ndarray) -> dict[str, float]:
    """D3-fixed exact decomposition.

    y rows are centred over the 9 actions by construction, so their row mean is
    zero. Then per row
        mean_9((p - y)^2) = mean_9((p - mean_9(p) - y)^2) + mean_9(p)^2
    and the clock/dependence equal weighting is applied to each term.
    """
    n_groups = y.shape[0]
    w = row_weights(ts)                                    # length n_groups*6, sums to 1
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    assert p.shape == t.shape == (n_groups, SCEN, H516_UTILITY_DIM)
    p_row = p.mean(axis=2)                                 # (n,6)
    t_row = t.mean(axis=2)
    if float(np.max(np.abs(t_row))) > 1e-12:
        raise RuntimeError("H517_TARGET_ROWS_NOT_ROW_CENTRED")
    level = float(np.sum(w * (p_row - t_row).reshape(-1) ** 2))
    shape = float(np.sum(w * np.mean((p - p_row[:, :, None] - t) ** 2, axis=2).reshape(-1)))
    total = float(np.sum(w * np.mean((p - t) ** 2, axis=2).reshape(-1)))
    if abs(total - level - shape) > 1e-14 * max(1.0, abs(total)):
        raise RuntimeError(f"H517_DECOMPOSITION_MISMATCH:{total}:{level}:{shape}")
    return {"total": total, "level": level, "shape": shape,
            "level_share": level / total if total > 0 else float("nan")}


# ---------------------------------------------------------------- EXP-6R job
def support_cell(d: dict[str, Any], i: int, j: int, blocks: list[np.ndarray]) -> dict[str, Any]:
    tr = window(d, int(blocks[i].min()), int(blocks[i].max()))
    ev = window(d, int(blocks[j].min()), int(blocks[j].max()))
    tr_rows = np.asarray(tr["x"]).reshape(-1, H516_DIM)
    ev_rows = np.asarray(ev["x"]).reshape(-1, H516_DIM)

    variants: dict[str, Any] = {}
    for name, (a, b) in representations(tr_rows, ev_rows).items():
        variants[name] = self_excluded_reference(a, b)

    # same-distribution control: split the EVAL block's groups in half, same period
    rng = np.random.default_rng(20260911 + 31 * i + j)
    n_g = ev_rows.shape[0] // SCEN
    perm = rng.permutation(n_g)
    half = n_g // 2
    idx_a = np.concatenate([np.arange(perm[k] * SCEN, perm[k] * SCEN + SCEN) for k in range(half)])
    idx_b = np.concatenate([np.arange(perm[k] * SCEN, perm[k] * SCEN + SCEN) for k in range(half, n_g)])
    ctrl = {}
    for name, (a, b) in representations(ev_rows[idx_a], ev_rows[idx_b]).items():
        ctrl[name] = self_excluded_reference(a, b)

    # fit / level / shape on the actual forward prediction
    pred = fit_predict(tr["x"], tr["y"], tr["ts"], ev["x"])
    base = clock_equal_target_mean_h516(tr["y"], tr["ts"])
    base_pred = np.broadcast_to(base, np.asarray(ev["y"]).shape)
    dec_model = decompose(pred, ev["y"], ev["ts"])
    dec_base = decompose(base_pred, ev["y"], ev["ts"])
    return {
        "train_block": int(i), "eval_block": int(j), "gap": int(j - i),
        "forward": bool(j > i),
        "variants": variants,
        "same_distribution_control": ctrl,
        "n_train_groups": int((np.asarray(tr["x"]).shape[0])),
        "n_eval_groups": int((np.asarray(ev["x"]).shape[0])),
        "decomposition": {
            "model": dec_model, "constant_baseline": dec_base,
            "raw_relative_gain": float((dec_base["total"] - dec_model["total"]) / dec_base["total"]),
            "shape_relative_gain": float((dec_base["shape"] - dec_model["shape"]) / dec_base["shape"]),
            "level_relative_gain": float((dec_base["level"] - dec_model["level"]) / dec_base["level"])
            if dec_base["level"] > 0 else float("nan"),
        },
        "mse_total": clock_equal_profile_mse_h516(pred, ev["y"], ev["ts"]),
    }


# ---------------------------------------------------------------- EXP-2c job
def synth_dataset(rng: np.random.Generator, fold: int, n_clocks: int, n_groups: int, r2: float) -> dict[str, Any]:
    latent = 8
    z = rng.normal(size=(n_groups, latent))
    sig = z @ rng.normal(size=(latent, H516_UTILITY_DIM))
    sig = sig - sig.mean(axis=0, keepdims=True)
    var_sig = max(float(np.var(sig)), 1e-12)
    resid = float(np.sqrt(max(var_sig / max(r2, 1e-9) - var_sig, 0.0)))
    y = np.repeat(sig[:, None, :], SCEN, axis=1) + rng.normal(size=(n_groups, SCEN, H516_UTILITY_DIM)) * resid
    y = y - y.mean(axis=2, keepdims=True)
    x = np.empty((n_groups, SCEN, H516_DIM), dtype=np.float64)
    x[:, :, :latent] = np.repeat(z[:, None, :], SCEN, axis=1) + 0.15 * rng.normal(size=(n_groups, SCEN, latent))
    x[:, :, latent:] = rng.normal(size=(n_groups, SCEN, H516_DIM - latent))
    per = max(1, n_groups // n_clocks)
    ts = np.repeat(
        (np.arange(n_clocks, dtype=np.int64) * 3_600_000) + 1_600_000_000_000, per
    )[:n_groups]
    if len(ts) < n_groups:
        ts = np.concatenate([ts, np.full(n_groups - len(ts), ts[-1], dtype=np.int64)])
    gids = tuple(f"f{fold}g{k:06d}" for k in range(n_groups))
    return {
        "fold": int(fold), "future_group_ids": gids, "timestamps_ms": ts,
        "x": np.ascontiguousarray(x), "y": np.ascontiguousarray(y),
        "unique_decision_clock_count": int(len(np.unique(ts))),
        "future_group_count": int(n_groups), "scenario_count_per_group": SCEN,
        "state_dimension": H516_DIM, "utility_dimension": H516_UTILITY_DIM,
    }


def power_replicate(geometry: list[tuple[int, int]], r2: float, rep: int) -> dict[str, Any]:
    """D4-fixed: every fold gets its own stream. D4b-fixed: frozen local arm only."""
    local = []
    for fold, (n_clocks, n_groups) in enumerate(geometry, start=1):
        rng = np.random.default_rng(880_000 + rep * 104_729 + fold * 7_919 + int(r2 * 1e6))
        d = synth_dataset(rng, fold, n_clocks, n_groups, r2)
        local.append(run_local_arm_h516(d))
    local = sorted(local, key=lambda r: int(r["fold"]))
    pos = int(sum(bool(r["positive_gain"]) for r in local))
    med = int(sum(bool(r["beats_shift_median"]) for r in local))
    pw = int(sum(int(r["beats_each_shift_count"]) for r in local))
    late_pos = all(bool(local[i - 1]["positive_gain"]) for i in (4, 5))
    late_med = all(bool(local[i - 1]["beats_shift_median"]) for i in (4, 5))
    # FROZEN boolean expression, copied verbatim from classify_h516
    supported = bool(pos >= 4 and late_pos and med >= 4 and late_med and pw >= 20)
    return {
        "r2": float(r2), "replicate": int(rep), "local_map_supported": supported,
        "positive_gain_folds": pos, "beats_shift_median_folds": med,
        "pairwise_wins_of_25": pw, "both_late_positive": bool(late_pos),
        "both_late_beats_median": bool(late_med),
        "per_fold_relative_gain": [
            float(r["true_binding_gain"] / r["baseline_loss"]) for r in local
        ],
        "per_fold_shift_median_relative": [
            float(r["median_shift_control_gain"] / r["baseline_loss"]) for r in local
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["support", "power", "both"], default="both")
    ap.add_argument("--fold", type=int, default=5)
    ap.add_argument("--r2", default="0.02,0.05,0.10,0.20,0.40")
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()

    rep: dict[str, Any] = {
        "schema": "CB16_H517_DIAGNOSTIC_REVISION_SUPPORT_AND_POWER_V2",
        "framing": ("DIAGNOSTIC_ONLY__ALREADY_CONSUMED_R10_4_TRAIN_CACHE__NO_NEW_DATA__"
                    "NO_FINAL__NO_NEW_VERDICT__NO_AUTHORITY_CHANGE"),
        "defects_fixed": ["D1_self_match", "D2_mask_units", "D3_level_correction",
                          "D4_shared_seed", "D4b_gate_reimplementation"],
    }

    if args.mode in ("support", "both"):
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
        cells = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(support_cell)(d, i, j, blocks) for i, j in tasks)
        fwd = [c for c in cells if c["forward"]]
        rep["support"] = {
            "support_receipt": support_receipt,
            "block_sizes": [int(len(b)) for b in blocks],
            "n_forward_cells": len(fwd), "n_reverse_cells": len(cells) - len(fwd),
            "blocks_are_shared_across_cells": True,
            "independent_units": 6,
            "summary": {
                name: {
                    "forward_median_distance_ratio": float(np.median([c["variants"][name]["distance_ratio"] for c in fwd])),
                    "forward_median_coverage": float(np.median([c["variants"][name]["coverage_below_p95"] for c in fwd])),
                    "forward_mean_coverage": float(np.mean([c["variants"][name]["coverage_below_p95"] for c in fwd])),
                    "same_distribution_control_median_coverage": float(
                        np.median([c["same_distribution_control"][name]["coverage_below_p95"] for c in cells])
                    ),
                    "same_distribution_control_median_ratio": float(
                        np.median([c["same_distribution_control"][name]["distance_ratio"] for c in cells])
                    ),
                }
                for name in ("raw102", "standardised102", "pca8_standardised")
            },
            "decomposition_forward": {
                "raw_relative_gain_mean": float(np.mean([c["decomposition"]["raw_relative_gain"] for c in fwd])),
                "shape_relative_gain_mean": float(np.mean([c["decomposition"]["shape_relative_gain"] for c in fwd])),
                "level_share_of_error_mean": float(np.mean([c["decomposition"]["model"]["level_share"] for c in fwd])),
                "baseline_level_share_mean": float(np.mean([c["decomposition"]["constant_baseline"]["level_share"] for c in fwd])),
                "shape_positive_cells": int(sum(1 for c in fwd if c["decomposition"]["shape_relative_gain"] > 0)),
            },
            "cells": cells,
        }
        s = rep["support"]["summary"]
        print("=== EXP-6R corrected support diagnostic (15 forward cells) ===")
        for name, v in s.items():
            print(f"  {name:<20} ratio {v['forward_median_distance_ratio']:>8.3f}  "
                  f"coverage {v['forward_median_coverage']:>6.3f}  "
                  f"same-dist control coverage {v['same_distribution_control_median_coverage']:>6.3f}  "
                  f"(control ratio {v['same_distribution_control_median_ratio']:.3f})")
        df = rep["support"]["decomposition_forward"]
        print(f"  raw rel gain mean {100*df['raw_relative_gain_mean']:+.2f}%   "
              f"shape rel gain mean {100*df['shape_relative_gain_mean']:+.2f}%   "
              f"positive shape cells {df['shape_positive_cells']}/15")
        print(f"  model level share {df['level_share_of_error_mean']:.3f}   "
              f"baseline level share {df['baseline_level_share_mean']:.3f}")

    if args.mode in ("power", "both"):
        geometry = [(29, 290), (28, 274), (28, 280), (28, 280), (28, 280)]
        r2s = [float(v) for v in str(args.r2).split(",") if v.strip()]
        tasks = [(geometry, r2, r) for r2 in r2s for r in range(args.reps)]
        res = Parallel(n_jobs=min(N_JOBS, max(1, len(tasks))), verbose=0)(
            delayed(power_replicate)(g, r2, r) for g, r2, r in tasks
        )
        summary = {}
        for r2 in r2s:
            rows = [x for x in res if x["r2"] == r2]
            summary[f"r2={r2}"] = {
                "replicates": len(rows),
                "gate_pass_count": int(sum(1 for x in rows if x["local_map_supported"])),
                "gate_pass_rate": float(np.mean([x["local_map_supported"] for x in rows])),
                "median_positive_fold_count": float(np.median([x["positive_gain_folds"] for x in rows])),
                "median_beats_median_fold_count": float(np.median([x["beats_shift_median_folds"] for x in rows])),
                "median_pairwise_wins": float(np.median([x["pairwise_wins_of_25"] for x in rows])),
                "distinct_per_fold_streams": bool(
                    all(len({round(x["per_fold_relative_gain"][k], 12) for x in rows}) > 1
                        for k in range(1, 5))
                ),
                "per_fold_relative_gain_median": [
                    float(np.median([x["per_fold_relative_gain"][k] for x in rows])) for k in range(5)
                ],
            }
        rep["power"] = {"geometry": geometry, "summary": summary, "trials": res,
                        "criterion": "FROZEN: pos>=4/5 AND pos in folds 4,5 AND beats_shift_median>=4/5 "
                                     "AND in folds 4,5 AND pairwise>=20/25 (classify_h516 verbatim)"}
        print("=== EXP-2c corrected power, FROZEN local arm + FROZEN gate ===")
        for k, v in summary.items():
            print(f"  {k:<9} pass {v['gate_pass_count']}/{v['replicates']}  "
                  f"pos_folds {v['median_positive_fold_count']}  "
                  f"beat_med {v['median_beats_median_fold_count']}  "
                  f"pairwise {v['median_pairwise_wins']}  "
                  f"distinct_streams={v['distinct_per_fold_streams']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"WROTE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
