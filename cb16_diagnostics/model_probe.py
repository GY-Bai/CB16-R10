from __future__ import annotations

"""Read-only checkpoint and generation weight diagnostics for CB16 R10."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "Operator Brain Stem": ("operator_encoder",),
    "Medium Brain Stem": ("medium_encoder",),
    "Account Brain Stem": ("account_encoder",),
    "Shared Decision Core": ("shared_core",),
    "Direction Head": ("direction_body", "direction_out"),
    "Requested-Risk Head": ("direction_embedding", "sizing_body", "sizing_out"),
}


def _load_state(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    p = Path(path)
    obj = torch.load(p, map_location="cpu", weights_only=True)
    meta: dict[str, Any] = {}
    if isinstance(obj, Mapping):
        for key in ("schema", "generation", "role", "parent_policy_semantic_sha256", "snapshot_hash"):
            if key in obj:
                meta[key] = obj[key]
        state = obj.get("state_dict") or obj.get("model_state") or obj.get("model")
        if state is None and all(torch.is_tensor(v) for v in obj.values()):
            state = obj
    else:
        state = None
    if not isinstance(state, Mapping) or not all(torch.is_tensor(v) for v in state.values()):
        raise RuntimeError(f"CHECKPOINT_STATE_NOT_FOUND:{p}")
    return {str(k): v.detach().cpu().contiguous() for k, v in state.items()}, meta


def semantic_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8") + b"\0")
        h.update(str(t.dtype).encode("ascii") + b"\0")
        h.update(json.dumps(list(t.shape), separators=(",", ":")).encode("ascii") + b"\0")
        h.update(t.numpy().tobytes(order="C"))
    return h.hexdigest()


def _tensor_accumulator(items: list[torch.Tensor]) -> dict[str, Any]:
    count = finite_count = nonfinite = zeros = 0
    sum_v = sum_sq = abs_sum = 0.0
    max_abs = 0.0
    l2_sq = 0.0
    for tensor in items:
        x = tensor.detach().cpu().double().reshape(-1)
        count += x.numel()
        finite = torch.isfinite(x)
        finite_count += int(finite.sum())
        nonfinite += int((~finite).sum())
        if finite.any():
            y = x[finite]
            zeros += int((y == 0).sum())
            sum_v += float(y.sum())
            sum_sq += float((y * y).sum())
            abs_sum += float(y.abs().sum())
            max_abs = max(max_abs, float(y.abs().max()))
            l2_sq += float((y * y).sum())
    mean = sum_v / finite_count if finite_count else None
    variance = max(0.0, sum_sq / finite_count - mean * mean) if finite_count and mean is not None else None
    return {
        "parameter_count": count,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite,
        "zero_count": zeros,
        "zero_fraction": (zeros / finite_count if finite_count else None),
        "mean": mean,
        "std": math.sqrt(variance) if variance is not None else None,
        "mean_abs": (abs_sum / finite_count if finite_count else None),
        "max_abs": max_abs if finite_count else None,
        "l2_norm": math.sqrt(l2_sq),
    }


def _group_for_name(name: str) -> str:
    for group, prefixes in GROUP_PREFIXES.items():
        if name.startswith(prefixes):
            return group
    return "Other"


def checkpoint_stats(state: Mapping[str, torch.Tensor], *, spectral: bool = False) -> dict[str, Any]:
    groups: dict[str, list[torch.Tensor]] = {}
    for name, tensor in state.items():
        groups.setdefault(_group_for_name(name), []).append(tensor)
    out = {
        "global": _tensor_accumulator(list(state.values())),
        "groups": {k: _tensor_accumulator(v) for k, v in sorted(groups.items())},
        "tensor_count": len(state),
    }
    if spectral:
        spectral_rows = []
        for name, tensor in sorted(state.items()):
            if tensor.ndim != 2 or tensor.numel() == 0:
                continue
            x = tensor.detach().cpu().double()
            if not torch.isfinite(x).all():
                spectral_rows.append({"name": name, "status": "NONFINITE"})
                continue
            try:
                s = torch.linalg.svdvals(x)
            except RuntimeError as exc:
                spectral_rows.append({"name": name, "status": "SVD_FAILED", "error": str(exc)[:160]})
                continue
            total = float(s.sum())
            if total <= 0:
                erank = 0.0
            else:
                p = s / total
                entropy = float(-(p[p > 0] * torch.log(p[p > 0])).sum())
                erank = math.exp(entropy)
            spectral_rows.append({
                "name": name,
                "shape": list(tensor.shape),
                "effective_rank": erank,
                "effective_rank_fraction": erank / max(1, min(tensor.shape)),
                "largest_singular_value": float(s.max()) if s.numel() else 0.0,
                "smallest_singular_value": float(s.min()) if s.numel() else 0.0,
                "status": "OK",
            })
        out["spectral"] = spectral_rows
    return out


def compare_states(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor], *, top_n: int = 12) -> dict[str, Any]:
    if set(before) != set(after):
        return {
            "compatible": False,
            "missing_in_before": sorted(set(after) - set(before)),
            "missing_in_after": sorted(set(before) - set(after)),
        }
    total_before_sq = total_after_sq = delta_sq = dot = 0.0
    changed = total = 0
    group_acc: dict[str, dict[str, float | int]] = {}
    tensor_rows: list[dict[str, Any]] = []
    nonfinite = 0
    for name in sorted(before):
        a = before[name].detach().cpu().double()
        b = after[name].detach().cpu().double()
        if a.shape != b.shape:
            return {"compatible": False, "shape_mismatch": name, "before": list(a.shape), "after": list(b.shape)}
        finite = torch.isfinite(a) & torch.isfinite(b)
        nonfinite += int((~finite).sum())
        if not finite.all():
            a = torch.where(finite, a, torch.zeros_like(a))
            b = torch.where(finite, b, torch.zeros_like(b))
        d = b - a
        a_sq = float((a * a).sum())
        b_sq = float((b * b).sum())
        d_sq = float((d * d).sum())
        dp = float((a * b).sum())
        n = a.numel()
        ch = int((a != b).sum())
        total_before_sq += a_sq
        total_after_sq += b_sq
        delta_sq += d_sq
        dot += dp
        total += n
        changed += ch
        group = _group_for_name(name)
        g = group_acc.setdefault(group, {"before_sq": 0.0, "after_sq": 0.0, "delta_sq": 0.0, "dot": 0.0, "count": 0, "changed": 0})
        g["before_sq"] += a_sq
        g["after_sq"] += b_sq
        g["delta_sq"] += d_sq
        g["dot"] += dp
        g["count"] += n
        g["changed"] += ch
        before_norm = math.sqrt(a_sq)
        delta_norm = math.sqrt(d_sq)
        tensor_rows.append({
            "name": name,
            "shape": list(a.shape),
            "before_l2": before_norm,
            "delta_l2": delta_norm,
            "relative_delta_l2": delta_norm / max(before_norm, 1e-30),
            "changed_fraction": ch / max(1, n),
        })

    def summarize(before_sq: float, after_sq: float, d_sq: float, dot_v: float, count_v: int, changed_v: int) -> dict[str, Any]:
        nb = math.sqrt(before_sq)
        na = math.sqrt(after_sq)
        nd = math.sqrt(d_sq)
        cos = dot_v / max(nb * na, 1e-30)
        return {
            "before_l2": nb,
            "after_l2": na,
            "delta_l2": nd,
            "relative_delta_l2": nd / max(nb, 1e-30),
            "cosine_similarity": max(-1.0, min(1.0, cos)),
            "parameter_count": count_v,
            "changed_count": changed_v,
            "changed_fraction": changed_v / max(1, count_v),
        }

    tensor_rows.sort(key=lambda x: x["relative_delta_l2"], reverse=True)
    return {
        "compatible": True,
        "nonfinite_pairs": nonfinite,
        "global": summarize(total_before_sq, total_after_sq, delta_sq, dot, total, changed),
        "groups": {
            k: summarize(float(v["before_sq"]), float(v["after_sq"]), float(v["delta_sq"]), float(v["dot"]), int(v["count"]), int(v["changed"]))
            for k, v in sorted(group_acc.items())
        },
        "top_relative_delta_tensors": tensor_rows[:max(0, int(top_n))],
    }


def probe_checkpoint(path: str | Path, *, spectral: bool = False) -> dict[str, Any]:
    state, meta = _load_state(path)
    return {
        "schema": "CB16_R10_CHECKPOINT_DIAGNOSTIC_R0",
        "path": str(Path(path)),
        "metadata": meta,
        "semantic_sha256": semantic_sha256(state),
        "stats": checkpoint_stats(state, spectral=spectral),
        "final_holdout_2025_09_accessed": False,
    }


def probe_pair(before_path: str | Path, after_path: str | Path, *, spectral_after: bool = False) -> dict[str, Any]:
    before, before_meta = _load_state(before_path)
    after, after_meta = _load_state(after_path)
    return {
        "schema": "CB16_R10_CHECKPOINT_PAIR_DIAGNOSTIC_R0",
        "before_path": str(Path(before_path)),
        "after_path": str(Path(after_path)),
        "before_metadata": before_meta,
        "after_metadata": after_meta,
        "before_semantic_sha256": semantic_sha256(before),
        "after_semantic_sha256": semantic_sha256(after),
        "delta": compare_states(before, after),
        "after_stats": checkpoint_stats(after, spectral=spectral_after),
        "final_holdout_2025_09_accessed": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only CB16 R10 model checkpoint probe")
    ap.add_argument("--checkpoint")
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--spectral", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.checkpoint:
        result = probe_checkpoint(a.checkpoint, spectral=a.spectral)
    elif a.before and a.after:
        result = probe_pair(a.before, a.after, spectral_after=a.spectral)
    else:
        raise SystemExit("provide --checkpoint or both --before and --after")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"schema": result["schema"], "out": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
