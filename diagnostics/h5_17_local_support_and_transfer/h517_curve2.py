#!/usr/bin/env python3
"""
CB16 H5.17-DIAGNOSTIC (container-only) -- EXP-3: legal-support learning curve.

The frozen H5.16 local arm fits each sub-block using only the other four
sub-blocks of the SAME evaluation environment, i.e. ~23 clocks per fit. The
outer expanding window for fold k legally holds 29/58/86/114/142 clocks - the
same support ladder on which the R6 audit obtained a positive reduced-Teacher
target result.

This experiment holds the evaluation block FIXED and grows the training support
over clocks that strictly precede it, with the frozen RF, the frozen centred-
utility9 target, the frozen clock/dependence equal weights and the frozen
cyclic-shift controls.

  curve rising   -> data-limited   (support was the binding constraint)
  curve flat/neg -> signal-limited (more of the same history does not help)

Diagnostic only. Already-consumed R10.4 TRAIN cache. No new data, no FINAL, no
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
from cb16_local_opt.teacher_temporal_transport_audit_h5 import (
    H5_SCENARIOS,
    _eval_group_scenario_rows_h5,
)
from cb16_local_opt.teacher_vectorized_r11 import build_columnar_teacher_index_r11
from h517_common import fit_predict

R104_ROOT = Path("/cb16/runtime/r104")
N_JOBS = int(os.environ.get("H517_JOBS", "6"))


def materialise_clocks(parents: dict, samples: list, clocks: tuple[int, ...]) -> dict[str, Any]:
    """Rows for every parent whose decision clock is in `clocks`, in H5.16 layout."""
    clock_set = {int(c) for c in clocks}
    sel = {pid: p for pid, p in parents.items() if int(p.decision_time_ms) in clock_set}
    if not sel:
        raise RuntimeError("H517_NO_PARENTS_FOR_CLOCKS")
    ids = set(sel)
    sub_samples = [s for s in samples if s.parent_id in ids]
    index = build_columnar_teacher_index_r11(sub_samples)
    if int(index.feature_dim) != H516_DIM:
        raise RuntimeError(f"H517_FEATURE_DIM:{index.feature_dim}")
    order, rows = _eval_group_scenario_rows_h5(index=index, eval_parents=sel, eval_parent_ids=tuple(sorted(sel)))
    ts_by_gid, scen_count = {}, {}
    for p in sel.values():
        gid, ts = str(p.dependence_group_id), int(p.decision_time_ms)
        if gid in ts_by_gid and ts_by_gid[gid] != ts:
            raise RuntimeError(f"H517_GROUP_CLOCK_DRIFT:{gid}")
        ts_by_gid[gid] = ts
        scen_count[gid] = scen_count.get(gid, 0) + 1
    if not all(scen_count.get(g, 0) == len(H5_SCENARIOS) for g in order):
        raise RuntimeError("H517_SCENARIO_COUNT_DRIFT")
    xs, ys = [], []
    for gid in order:
        xr, yr = [], []
        for sc in H5_SCENARIOS:
            r = int(rows[gid][sc])
            u = np.asarray(index.utilities[r], dtype=np.float64).reshape(-1)
            xs.append(np.asarray(index.features[r], dtype=np.float64).reshape(-1))
            yr.append(u - float(np.mean(u)))
        ys.append(np.stack(yr, axis=0))
    x = np.stack(xs, axis=0).reshape(len(order), 6, H516_DIM)
    y = np.stack(ys, axis=0)
    ts = np.asarray([ts_by_gid[g] for g in order], dtype=np.int64)
    return {
        "future_group_ids": tuple(str(g) for g in order),
        "timestamps_ms": ts,
        "x": np.ascontiguousarray(x),
        "y": np.ascontiguousarray(y),
        "unique_decision_clock_count": int(len(np.unique(ts))),
        "future_group_count": int(len(order)),
    }


def window(d: dict[str, Any], lo: int, hi: int) -> dict[str, Any]:
    ts = np.asarray(d["timestamps_ms"], dtype=np.int64)
    m = (ts >= lo) & (ts <= hi)
    if not m.any():
        raise RuntimeError(f"H517_EMPTY_WINDOW:{lo}:{hi}")
    return {
        "x": np.ascontiguousarray(np.asarray(d["x"])[m]),
        "y": np.ascontiguousarray(np.asarray(d["y"])[m]),
        "ts": np.asarray(ts[m], dtype=np.int64),
        "gids": tuple(str(g) for g, k in zip(d["future_group_ids"], m.tolist()) if k),
        "n_clocks": int(len(np.unique(ts[m]))),
        "n_groups": int(m.sum()),
    }


def task(payload: dict[str, Any]) -> dict[str, Any]:
    d = payload["dataset"]
    tr = window(d, payload["train_lo"], payload["train_hi"])
    ev = window(d, payload["eval_lo"], payload["eval_hi"])
    base = clock_equal_target_mean_h516(tr["y"], tr["ts"])
    baseline_loss = clock_equal_profile_mse_h516(
        np.broadcast_to(base, ev["y"].shape).copy(), ev["y"], ev["ts"]
    )
    model_loss = clock_equal_profile_mse_h516(
        fit_predict(tr["x"], tr["y"], tr["ts"], ev["x"]), ev["y"], ev["ts"]
    )
    gain = float(baseline_loss - model_loss)
    ctrl, recovered = {}, {}
    if tr["n_groups"] > max(H516_SHIFTS):
        for s in H516_SHIFTS:
            sx = shift_group_state_h516(tr["x"], tr["ts"], tr["gids"], int(s))
            ctrl[int(s)] = float(
                baseline_loss
                - clock_equal_profile_mse_h516(fit_predict(sx, tr["y"], tr["ts"], ev["x"]), ev["y"], ev["ts"])
            )
    med = float(statistics.median(ctrl.values())) if ctrl else float("nan")
    return {
        "fold": payload["fold"], "tag": payload["tag"],
        "train_clocks": tr["n_clocks"], "train_groups": tr["n_groups"], "train_rows": tr["n_groups"] * 6,
        "eval_clocks": ev["n_clocks"], "eval_groups": ev["n_groups"],
        "baseline_loss": baseline_loss, "true_gain": gain,
        "relative_gain": float(gain / baseline_loss) if baseline_loss > 0 else float("nan"),
        "positive_gain": bool(gain > 0),
        "median_shift_gain": med,
        "beats_shift_median": bool(ctrl and gain > med),
        "beats_each_shift_count": int(sum(gain > v for v in ctrl.values())) if ctrl else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", default="3,4,5")
    ap.add_argument("--fractions", default="0.25,0.5,0.75,1.0")
    args = ap.parse_args()

    from scripts.r11_science_g0_reduced_teacher_target_information_audit_r6 import load_train_only_support
    from cb16_local_opt.reduced_teacher_target_information_audit_r6 import build_outer_folds_r6

    parents, samples, support_receipt = load_train_only_support(R104_ROOT)
    folds = build_outer_folds_r6(parents)
    want = {int(v) for v in str(args.folds).split(",") if v.strip()}
    fracs = [float(v) for v in str(args.fractions).split(",") if v.strip()]

    rep: dict[str, Any] = {
        "schema": "CB16_H517_DIAGNOSTIC_LEGAL_SUPPORT_LEARNING_CURVE_V2",
        "framing": ("DIAGNOSTIC_ONLY__ALREADY_CONSUMED_R10_4_TRAIN_CACHE__NO_NEW_DATA__"
                    "NO_FINAL__NO_NEW_VERDICT__NO_AUTHORITY_CHANGE"),
        "support_receipt": support_receipt, "curve": {}, "materialisation": [],
    }
    tasks = []
    for f in folds:
        fold = int(f["fold"])
        if fold not in want:
            continue
        eval_clocks = tuple(int(c) for c in f["eval_clocks"])
        past = np.asarray(sorted(int(c) for c in f["train_clocks"]), dtype=np.int64)
        all_clocks = tuple(sorted(set(int(c) for c in f["train_clocks"]) | set(eval_clocks)))
        d = materialise_clocks(parents, samples, all_clocks)
        rep["materialisation"].append({
            "fold": fold, "clocks": d["unique_decision_clock_count"], "groups": d["future_group_count"],
            "legal_past_clocks": len(past), "eval_clocks": len(eval_clocks),
        })
        print(json.dumps({"stage": "built", "fold": fold, "clocks": d["unique_decision_clock_count"],
                          "groups": d["future_group_count"]}), flush=True)
        for fr in fracs:
            k = max(1, int(round(fr * len(past))))
            win = past[-k:]
            tasks.append({
                "dataset": d, "tag": f"frac={fr}", "fold": fold,
                "train_lo": int(win.min()), "train_hi": int(win.max()),
                "eval_lo": int(min(eval_clocks)), "eval_hi": int(max(eval_clocks)),
            })
    out = Parallel(n_jobs=min(N_JOBS, max(1, len(tasks))), verbose=0)(delayed(task)(t) for t in tasks)
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for r in out:
        by_tag.setdefault(r["tag"], []).append(r)
    for tag, rows in sorted(by_tag.items(), key=lambda kv: float(kv[0].split("=")[1])):
        rel = [r["relative_gain"] for r in rows]
        rep["curve"][tag] = {
            "folds": [r["fold"] for r in rows],
            "train_clocks": [r["train_clocks"] for r in rows],
            "train_groups": [r["train_groups"] for r in rows],
            "relative_gain_per_fold": [float(v) for v in rel],
            "mean_relative_gain": float(np.mean(rel)),
            "median_relative_gain": float(np.median(rel)),
            "positive_fold_count": int(sum(1 for r in rows if r["positive_gain"])),
            "beats_shift_median_count": int(sum(1 for r in rows if r["beats_shift_median"])),
            "baseline_loss_per_fold": [r["baseline_loss"] for r in rows],
        }
    rep["raw"] = out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: {"clocks": v["train_clocks"], "rel%": [round(100 * x, 2) for x in v["relative_gain_per_fold"]],
                          "folds": v["folds"]} for k, v in rep["curve"].items()}, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
