#!/usr/bin/env python3
"""
CB16 H5.17-DIAGNOSTIC -- EXP-6S: what the Full102 "state" actually contains.

Motivation
----------
After fixing the self-match defect, the reference nearest-neighbour distance in
the support diagnostic is still ~1e-3 (and ~1e-9 after PCA), and the
SAME-DISTRIBUTION control also reports zero coverage. That pattern has one
obvious candidate explanation: the six rows of a future group are not six
distinct states. They share the market state and differ only in the account
provenance dimensions, so every row's nearest neighbour is its own sibling.

This experiment measures that directly and then redoes the support question in
the correct unit (future GROUP, not row).

Diagnostic only. Already-consumed R10.4 TRAIN cache. No new data, no FINAL,
no authority change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import H516_DIM
from h517_curve2 import materialise_clocks, window

SCEN = 6
MARKET_DIMS = 96          # Operator48 + Medium48
ACCOUNT_DIMS = 6          # Account6
N_JOBS = 6


def group_block(d: dict[str, Any], lo: int, hi: int) -> dict[str, Any]:
    w = window(d, lo, hi)
    x = np.asarray(w["x"])                      # (n_groups, 6, 102)
    return {"x": x, "ts": np.asarray(w["ts"]), "gids": w["gids"], "n_groups": x.shape[0]}


def within_between(block: dict[str, Any]) -> dict[str, float]:
    x = block["x"]
    n = x.shape[0]
    mkt = x[:, :, :MARKET_DIMS]
    acc = x[:, :, MARKET_DIMS:]
    # spread across the six scenarios of the same group
    within_mkt = float(np.mean(np.std(mkt, axis=1)))
    within_acc = float(np.mean(np.std(acc, axis=1)))
    # group representatives
    rep_mkt = mkt.mean(axis=1)
    rep_acc = acc.mean(axis=1)
    rr = np.random.default_rng(1234)
    a = rr.choice(n, size=min(n, 300), replace=False)
    b = rr.choice(n, size=min(n, 300), replace=False)
    cross = np.sqrt(np.maximum(((rep_mkt[a][:, None, :] - rep_mkt[b][None, :, :]) ** 2).sum(-1), 0.0))
    off = cross[~np.eye(len(a), len(b), dtype=bool)]
    return {
        "n_groups": int(n),
        "mean_within_group_std_market96": within_mkt,
        "mean_within_group_std_account6": within_acc,
        "median_between_group_market_distance": float(np.median(off)),
        "ratio_within_to_between": float(within_mkt / max(float(np.median(off)), 1e-300)),
    }


def sibling_diagnostic(block: dict[str, Any]) -> dict[str, Any]:
    """Row-level 1-NN with and without same-group siblings allowed."""
    x = block["x"].reshape(-1, H516_DIM)
    n_groups = block["x"].shape[0]
    gid_of_row = np.repeat(np.arange(n_groups), SCEN)

    nn = NearestNeighbors(n_neighbors=2).fit(x)
    d_all, i_all = nn.kneighbors(x)
    d1_all = d_all[:, 1]
    sib = gid_of_row[i_all[:, 1]] == gid_of_row
    d1_nosib = np.full(len(x), np.nan)
    # exclude same-group rows by brute force over a subsample
    rr = np.random.default_rng(4321)
    sub = rr.choice(len(x), size=min(400, len(x)), replace=False)
    ref = x[gid_of_row != -1]
    for k in sub:
        mask = gid_of_row != gid_of_row[k]
        dd = ((ref[mask] - x[k]) ** 2).sum(-1)
        d1_nosib[k] = float(np.sqrt(dd.min()))
    return {
        "n_rows": int(len(x)),
        "fraction_whose_nearest_neighbour_is_a_sibling": float(sib.mean()),
        "median_1nn_distance_with_siblings": float(np.nanmedian(d1_all)),
        "median_1nn_distance_excluding_siblings": float(np.nanmedian(d1_nosib)),
        "ratio": float(np.nanmedian(d1_nosib) / max(np.nanmedian(d1_all), 1e-300)),
    }


def group_support(tr: dict[str, Any], ev: dict[str, Any]) -> dict[str, Any]:
    """Group-level support in the market-state subspace, siblings resolved by group."""
    def reps(b):
        m = b["x"][:, :, :MARKET_DIMS].mean(axis=1)
        a = b["x"][:, :, MARKET_DIMS:].mean(axis=1)
        return m, a

    tr_m, tr_a = reps(tr)
    ev_m, ev_a = reps(ev)

    out: dict[str, Any] = {}
    for tag, ref, query in (("market96_only", tr_m, ev_m),
                            ("account6_only", tr_a, ev_a)):
        if len(ref) < 3:
            continue
        nn2 = NearestNeighbors(n_neighbors=2).fit(ref)
        d2, _ = nn2.kneighbors(ref)
        d_ref = d2[:, 1]
        thr = float(np.quantile(d_ref, 0.95))
        nn1 = NearestNeighbors(n_neighbors=1).fit(ref)
        d_q, _ = nn1.kneighbors(query)
        d_q = d_q.reshape(-1)
        out[tag] = {
            "n_reference_groups": int(len(ref)), "n_query_groups": int(len(query)),
            "median_reference_group_distance": float(np.median(d_ref)),
            "p95_threshold": thr,
            "median_query_group_distance": float(np.median(d_q)),
            "distance_ratio": float(np.median(d_q) / max(np.median(d_ref), 1e-300)),
            "coverage": float((d_q < thr).mean()),
        }
    # same-distribution control at group level: split the query block's groups in half
    rr = np.random.default_rng(987)
    n_g = len(ev_m)
    perm = rr.permutation(n_g)
    half = max(2, n_g // 2)
    ia, ib = perm[:half], perm[half:]
    if len(ib) >= 3:
        nn2 = NearestNeighbors(n_neighbors=2).fit(ev_m[ia])
        d2, _ = nn2.kneighbors(ev_m[ia])
        d_ref = d2[:, 1]
        thr = float(np.quantile(d_ref, 0.95))
        nn1 = NearestNeighbors(n_neighbors=1).fit(ev_m[ia])
        d_q, _ = nn1.kneighbors(ev_m[ib])
        d_q = d_q.reshape(-1)
        out["same_distribution_control_market96"] = {
            "n_reference_groups": int(len(ia)), "n_query_groups": int(len(ib)),
            "median_reference_group_distance": float(np.median(d_ref)),
            "p95_threshold": thr,
            "median_query_group_distance": float(np.median(d_q)),
            "distance_ratio": float(np.median(d_q) / max(np.median(d_ref), 1e-300)),
            "coverage": float((d_q < thr).mean()),
        }
    return out


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

    gb = [group_block(d, int(b.min()), int(b.max())) for b in blocks]
    rep: dict[str, Any] = {
        "schema": "CB16_H517_DIAGNOSTIC_STATE_STRUCTURE_V1",
        "framing": ("DIAGNOSTIC_ONLY__ALREADY_CONSUMED_R10_4_TRAIN_CACHE__NO_NEW_DATA__"
                    "NO_FINAL__NO_NEW_VERDICT__NO_AUTHORITY_CHANGE"),
        "support_receipt": support_receipt,
        "block_sizes_groups": [b["n_groups"] for b in gb],
        "within_vs_between": [within_between(b) for b in gb],
        "sibling_structure": [sibling_diagnostic(b) for b in gb],
    }

    print("=== does the six-scenario bundle share the market state? ===")
    for i, w in enumerate(rep["within_vs_between"]):
        print(f"  B{i}: groups={w['n_groups']:<5} within-group std(market96)={w['mean_within_group_std_market96']:.3e}  "
              f"std(account6)={w['mean_within_group_std_account6']:.3e}  "
              f"between-group dist={w['median_between_group_market_distance']:.3e}  "
              f"within/between={w['ratio_within_to_between']:.4f}")
    print()
    print("=== is each row's nearest neighbour its own sibling? ===")
    for i, s in enumerate(rep["sibling_structure"]):
        print(f"  B{i}: siblings-are-1NN {100*s['fraction_whose_nearest_neighbour_is_a_sibling']:.1f}%  "
              f"median 1NN with siblings {s['median_1nn_distance_with_siblings']:.3e}  "
              f"without siblings {s['median_1nn_distance_excluding_siblings']:.3e}  "
              f"ratio {s['ratio']:.1f}")
    print()

    tasks = [("forward", i, j) for i in range(6) for j in range(6) if j > i]
    cells = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(group_support)(gb[i], gb[j]) for _, i, j in tasks
    )
    rep["group_level_support_forward"] = [
        {"train_block": i, "eval_block": j, **c} for (_, i, j), c in zip(tasks, cells)
    ]
    print("=== group-level support, 15 forward cells ===")
    for tag in ("market96_only", "account6_only", "same_distribution_control_market96"):
        vals = [c[tag] for c in rep["group_level_support_forward"] if tag in c]
        if not vals:
            continue
        print(f"  {tag:<34} median distance ratio {np.median([v['distance_ratio'] for v in vals]):>10.3f}   "
              f"median coverage {np.median([v['coverage'] for v in vals]):.3f}")
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"WROTE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
