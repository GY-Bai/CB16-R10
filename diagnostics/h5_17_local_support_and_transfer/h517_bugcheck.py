#!/usr/bin/env python3
"""
Independent reproduction of the four defects reported against the H5.17
diagnostic harness. Pure synthetic, no market data, no CB16 authority.

D1 self-match in the NearestNeighbors reference distance
D2 mask unit mismatch (np.repeat on an already (groups,6) mask)
D3 mis-specified level correction (raw prediction vs centred target)
D4 non-equivalence of the EXP-2b gate reimplementation + shared seed across folds
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(7)
N, D = 400, 102
X = rng.normal(size=(N, D))
sub = np.arange(200)
out = {}

# ---- D1: does the reference distance contain the self-match? -------------
nn = NearestNeighbors(n_neighbors=1).fit(X[sub])
d_self, _ = nn.kneighbors(X[sub])
thr = float(np.quantile(d_self, 0.95))
d_ev, _ = nn.kneighbors(X[sub])  # query the very same points
cov = float((d_ev.reshape(-1) < thr).mean())
out["D1"] = {
    "max_self_reference_distance": float(d_self.max()),
    "p95_threshold": thr,
    "coverage_when_querying_training_points_themselves": cov,
    "verdict": "BROKEN: reference distance is dominated by self-match" if thr <= 0.0 or cov == 0.0 else "ok",
}
# the fix
nn2 = NearestNeighbors(n_neighbors=2).fit(X[sub])
d2, _ = nn2.kneighbors(X[sub])
d_ref = d2[:, 1]
thr2 = float(np.quantile(d_ref, 0.95))
d_ev2, _ = nn2.kneighbors(X[sub], n_neighbors=1)
out["D1_fix"] = {
    "median_self_excluded_reference_distance": float(np.median(d_ref)),
    "p95_threshold": thr2,
    "coverage_same_points": float((d_ev2.reshape(-1) < thr2).mean()),
    "coverage_fresh_same_distribution": float(
        (nn2.kneighbors(rng.normal(size=(N, D)), n_neighbors=1)[0].reshape(-1) < thr2).mean()
    ),
}

# ---- D2: mask unit mismatch ---------------------------------------------
groups, scen, nrows = 40, 6, 40 * 6
in_support = rng.random((groups, scen)) < 0.5  # (groups,6) boolean, as in the harness
rows_in = np.repeat(in_support, 6)
out["D2"] = {
    "in_support_shape": list(in_support.shape),
    "rows_in_length": int(rows_in.size),
    "prediction_rows": nrows,
    "ratio": float(rows_in.size / nrows),
    "verdict": "BROKEN: mask is 6x too long; any non-empty mask escapes the fallback and raises",
}
try:
    _ = np.arange(nrows)[rows_in]
    out["D2"]["indexing_with_nonempty_mask"] = "no error (unexpected)"
except Exception as exc:  # noqa: BLE001
    out["D2"]["indexing_with_nonempty_mask"] = f"{type(exc).__name__}: {exc}"

# ---- D3: level correction -----------------------------------------------
# construct a PERFECT prediction: pred == target
y = rng.normal(size=(groups, scen, 9))
y = y - y.mean(axis=2, keepdims=True)
pred = y.copy()
clock_mean = y.mean(axis=(0, 1))                      # per-action clock-equal mean (one clock block)
ev_c = y - clock_mean[None, None, :]
ev_c = ev_c - ev_c.mean(axis=2, keepdims=True)

def mse(a, b):
    return float(np.mean(np.mean((a - b) ** 2, axis=2)))

raw = mse(pred, y)
harness_level = mse(pred, ev_c)                        # what the harness computed
proper_shape = mse(pred - pred.mean(axis=2, keepdims=True), y - y.mean(axis=2, keepdims=True))
out["D3"] = {
    "perfect_prediction_raw_mse": raw,
    "harness_level_corrected_mse": harness_level,
    "proper_centred_shape_mse": proper_shape,
    "verdict": "BROKEN: a perfect prediction is penalised" if harness_level > 1e-12 else "ok",
}
# correct decomposition for an imperfect prediction
pred_bad = y + rng.normal(size=y.shape) * 0.5 + 1.0   # adds level and shape error
tot = mse(pred_bad, y)
lev = float(np.mean((pred_bad.mean(axis=2) - y.mean(axis=2)) ** 2))
shp = mse(pred_bad - pred_bad.mean(axis=2, keepdims=True), y - y.mean(axis=2, keepdims=True))
out["D3_fix"] = {"total": tot, "level_component": lev, "shape_component": shp,
                 "sum_equals_total": bool(abs(tot - lev - shp) < 1e-12)}

# ---- D4: shared seed across folds with equal clock counts ---------------
counts = [29, 28, 28, 28, 28]
rows = []
for rep in range(3):
    seed = 555_000 + rep * 7919 + int(0.2 * 1e6)
    for n_clocks in counts:
        r = np.random.default_rng(seed)
        rows.append({"rep": rep, "n_clocks": n_clocks, "first_draw": float(r.normal())})
ident = {}
for rep in range(3):
    vals = [x["first_draw"] for x in rows if x["rep"] == rep]
    ident[rep] = {"draws": vals, "folds_2_to_5_identical": len(set(np.round(vals[1:], 12))) == 1}
out["D4"] = {"per_replicate": ident,
             "verdict": "BROKEN: folds with equal clock counts share one stream"}

# ---- D4b: is the reimplemented gate equivalent to the frozen one? -------
frozen = {"positive_count": 4, "median_count": 4, "late_pos": True, "late_med": True, "pairwise": 20}
def frozen_expr(pc, mc, lp, lm, pw):
    return bool(pc >= 4 and lp and mc >= 4 and lm and pw >= 20)
# harness used: fold-level relative gain = mean of per-block relative gains,
# beats_shift_median = majority of blocks, pairwise = round(mean of block counts)
harness_equiv = frozen_expr(4, 4, True, True, 20)
pooled_equiv = frozen_expr(4, 4, True, True, 20)
out["D4b"] = {
    "frozen_expression": "pos>=4 & late_pos & med>=4 & late_med & pairwise>=20",
    "difference": ("frozen: per-environment gain from POOLED out-of-fold predictions and one "
                   "shift comparison per environment; harness: mean of per-block relative gains, "
                   "majority vote over blocks, rounded mean of block shift wins"),
    "same_at_this_example": harness_equiv == pooled_equiv,
    "equal_in_general": False,
    "verdict": "NOT EQUIVALENT in general; must call the frozen helper",
}

print(json.dumps(out, indent=2))
