from __future__ import annotations

"""M-series M2: frozen-G0 Direction residual research helpers.

The canonical 189,052-parameter G0 policy remains functionally frozen. M2 trains
only a small 256->64->3 residual on detached G0 Shared256 features and detached
G0 Direction logits. No Sizing, Teacher, Physics, Supervisor, or G0 parameter is
a gradient owner.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .canonical_state_alignment_falsification_r41 import R41_SCENARIOS
from .training_runtime_r11 import (
    CANONICAL_BATCH_SIZE_R11,
    CANONICAL_EPOCHS_R11,
    CANONICAL_LR_R11,
    CANONICAL_WEIGHT_DECAY_R11,
    GENERATION_BASE_SEED_R11,
    GRADIENT_CLIP_MAX_NORM_R11,
    PreparedEvidenceR11,
)

M2_RUNTIME = "CB16_R11_M_SERIES_M2_FROZEN_G0_DIRECTION_RESIDUAL_V1"
M2_SHIFTS = (1, 7, 13, 23, 31)
M2_RESIDUAL_PARAMETER_COUNT = 16_643
M2_OBJECTIVES = ("ABS_CE", "SELECTIVE_CHAMPION_PRESERVE_CE")
DIRECTION_VALUES = (-1, 0, 1)


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(obj)).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    x = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("ascii") + b"\0")
    h.update(json.dumps(list(x.shape), separators=(",", ":")).encode("ascii") + b"\0")
    h.update(x.numpy().tobytes(order="C"))
    return h.hexdigest()


class DirectionResidualM2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(256, 64)
        self.output = nn.Linear(64, 3)
        self.activation = nn.SiLU()
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()

    def forward(self, shared256: torch.Tensor) -> torch.Tensor:
        if shared256.ndim != 2 or int(shared256.shape[1]) != 256:
            raise ValueError("M2_SHARED256_SHAPE")
        return self.output(self.activation(self.hidden(shared256)))


def build_direction_residual_m2(
    *, seed: int = GENERATION_BASE_SEED_R11, device: str | torch.device = "cpu"
) -> DirectionResidualM2:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    residual = DirectionResidualM2().to(device=device, dtype=torch.float32)
    count = sum(int(p.numel()) for p in residual.parameters())
    require(count == M2_RESIDUAL_PARAMETER_COUNT, f"M2_RESIDUAL_PARAM_COUNT:{count}")
    require(all(p.dtype == torch.float32 for p in residual.parameters()), "M2_RESIDUAL_NOT_FP32")
    require(
        bool(torch.count_nonzero(residual.output.weight).item()) is False
        and bool(torch.count_nonzero(residual.output.bias).item()) is False,
        "M2_OUTPUT_NOT_EXACT_ZERO_INITIALIZED",
    )
    return residual


def residual_parameter_l2_m2(residual: nn.Module) -> float:
    total = 0.0
    for p in residual.parameters():
        x = p.detach().double()
        total += float(torch.sum(x * x).cpu())
    return math.sqrt(total)


@dataclass(frozen=True)
class FrozenDirectionCacheM2:
    shared256: torch.Tensor
    base_logits: torch.Tensor
    base_probs: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.shared256.shape[0])


def freeze_g0_m2(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None


def cache_frozen_g0_direction_m2(
    model: nn.Module, prepared: PreparedEvidenceR11
) -> FrozenDirectionCacheM2:
    prepared.validate()
    freeze_g0_m2(model)
    device = prepared.packed.device
    require(next(model.parameters()).device == device, "M2_G0_EVIDENCE_DEVICE_DRIFT")
    with torch.inference_mode():
        o = model.operator_encoder(prepared.operator48.detach())
        m = model.medium_encoder(prepared.medium48.detach())
        a = model.account_encoder(prepared.account6.detach())
        shared = model.shared_core(torch.cat([o, m, a], dim=-1))
        logits = model.direction_out(model.direction_body(shared))
        probs = torch.softmax(logits, dim=-1)
    shared = shared.detach()
    logits = logits.detach()
    probs = probs.detach()
    require(shared.shape == (prepared.rows, 256), "M2_SHARED_CACHE_SHAPE")
    require(logits.shape == (prepared.rows, 3), "M2_LOGIT_CACHE_SHAPE")
    require(probs.shape == (prepared.rows, 3), "M2_PROB_CACHE_SHAPE")
    require(not shared.requires_grad and not logits.requires_grad and not probs.requires_grad, "M2_G0_CACHE_AUTOGRAD_TAINT")
    return FrozenDirectionCacheM2(shared256=shared, base_logits=logits, base_probs=probs)


def verify_zero_residual_identity_m2(
    residual: DirectionResidualM2, cache: FrozenDirectionCacheM2
) -> float:
    residual.eval()
    with torch.inference_mode():
        delta = residual(cache.shared256)
        new_probs = torch.softmax(cache.base_logits + delta, dim=-1)
    max_delta = float(torch.max(torch.abs(delta)).detach().cpu())
    max_prob_error = float(torch.max(torch.abs(new_probs - cache.base_probs)).detach().cpu())
    require(max_delta == 0.0, f"M2_INITIAL_DELTA_NOT_ZERO:{max_delta}")
    require(max_prob_error <= 1e-7, f"M2_INITIAL_POLICY_NOT_G0:{max_prob_error}")
    return max_prob_error


def _ordered_admitted_evidence(
    evidence: Sequence[Any], parent_ids: Sequence[str]
) -> list[Any]:
    admitted = [e for e in evidence if bool(e.admission.admitted)]
    by_parent = {str(e.parent_id): e for e in admitted}
    require(len(by_parent) == len(admitted), "M2_DUPLICATE_EVIDENCE_PARENT")
    require(set(by_parent) == set(str(x) for x in parent_ids), "M2_EVIDENCE_PARENT_SET_DRIFT")
    return [by_parent[str(pid)] for pid in parent_ids]


def best_direction_means_from_evidence_m2(
    evidence: Sequence[Any], parent_ids: Sequence[str]
) -> np.ndarray:
    ordered = _ordered_admitted_evidence(evidence, parent_ids)
    means = np.empty((len(ordered), 3), dtype=np.float64)
    for row, e in enumerate(ordered):
        laws = tuple(getattr(e, "action_laws", ()))
        require(bool(laws), f"M2_ACTION_LAWS_MISSING:{row}")
        for cls, direction in enumerate(DIRECTION_VALUES):
            candidates = [law for law in laws if int(law.direction) == int(direction)]
            require(bool(candidates), f"M2_DIRECTION_LAW_MISSING:{row}:{direction}")
            best = max(
                candidates,
                key=lambda law: (float(law.mean_utility), -float(law.requested_risk)),
            )
            means[row, cls] = float(best.mean_utility)
    require(np.isfinite(means).all(), "M2_BEST_DIRECTION_MEANS_NONFINITE")
    return means


def _scenario_from_parent_id_m2(parent_id: str) -> str:
    scenario = str(parent_id).rsplit(":", 1)[-1]
    require(scenario in R41_SCENARIOS, f"M2_UNKNOWN_SCENARIO:{scenario}")
    return scenario


def _group_scenario_rows_m2(
    parent_ids: Sequence[str], dependence_group_ids: Sequence[str]
) -> dict[str, dict[str, int]]:
    require(len(parent_ids) == len(dependence_group_ids), "M2_GROUP_ROW_LENGTH_DRIFT")
    rows: dict[str, dict[str, int]] = {}
    for idx, (pid, gid0) in enumerate(zip(parent_ids, dependence_group_ids)):
        gid = str(gid0)
        scenario = _scenario_from_parent_id_m2(str(pid))
        rows.setdefault(gid, {})
        require(scenario not in rows[gid], f"M2_DUPLICATE_GROUP_SCENARIO:{gid}:{scenario}")
        rows[gid][scenario] = idx
    expected = set(R41_SCENARIOS)
    for gid, mapping in rows.items():
        require(set(mapping) == expected, f"M2_SCENARIO_SET_DRIFT:{gid}:{sorted(mapping)}")
    return rows


def _surface_multiset_sha256_m2(
    teacher_probs: torch.Tensor | np.ndarray, rich_means: np.ndarray
) -> str:
    p = (
        teacher_probs.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        if isinstance(teacher_probs, torch.Tensor)
        else np.asarray(teacher_probs, dtype=np.float32)
    )
    m = np.asarray(rich_means, dtype=np.float64)
    require(p.shape == m.shape and p.ndim == 2 and p.shape[1] == 3, "M2_SURFACE_SHAPE")
    rows = []
    for i in range(len(p)):
        rows.append(
            np.ascontiguousarray(p[i], dtype=np.float32).tobytes(order="C")
            + np.ascontiguousarray(m[i], dtype=np.float64).tobytes(order="C")
        )
    h = hashlib.sha256(b"CB16_R11_M2_TEACHER_SURFACE_MULTISET_V1\0")
    for row in sorted(rows):
        h.update(row)
    return h.hexdigest()


def rotate_rich_direction_means_m2(
    *,
    rich_means: np.ndarray,
    prepared: PreparedEvidenceR11,
    shuffled_prepared: PreparedEvidenceR11,
    shuffle_receipt: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prepared.validate()
    shuffled_prepared.validate()
    require(prepared.parent_ids == shuffled_prepared.parent_ids, "M2_SHUFFLE_PARENT_ORDER_DRIFT")
    require(
        prepared.dependence_group_ids == shuffled_prepared.dependence_group_ids,
        "M2_SHUFFLE_GROUP_ORDER_DRIFT",
    )
    means = np.asarray(rich_means, dtype=np.float64)
    require(means.shape == (prepared.rows, 3), "M2_RICH_MEAN_SHAPE")
    group_rows = _group_scenario_rows_m2(prepared.parent_ids, prepared.dependence_group_ids)
    out = np.empty_like(means)
    seen_dst: set[str] = set()
    mapping = list(shuffle_receipt.get("mapping", ()))
    require(bool(mapping), "M2_SHUFFLE_MAPPING_MISSING")
    for item in mapping:
        dst = str(item["destination_group"])
        src = str(item["source_group"])
        require(dst in group_rows and src in group_rows, f"M2_SHUFFLE_GROUP_UNKNOWN:{dst}:{src}")
        require(dst not in seen_dst, f"M2_SHUFFLE_DEST_DUPLICATE:{dst}")
        seen_dst.add(dst)
        for scenario in R41_SCENARIOS:
            out[group_rows[dst][scenario]] = means[group_rows[src][scenario]]
    require(seen_dst == set(group_rows), "M2_SHUFFLE_MAPPING_INCOMPLETE")
    before = _surface_multiset_sha256_m2(prepared.direction_target_probs, means)
    after = _surface_multiset_sha256_m2(shuffled_prepared.direction_target_probs, out)
    require(before == after, "M2_TEACHER_SURFACE_MULTISET_CHANGED")
    reduced_best = (
        shuffled_prepared.direction_target_probs.detach().cpu().numpy().argmax(axis=1)
    )
    rich_best = np.argmax(out, axis=1)
    require(np.array_equal(reduced_best, rich_best), "M2_SHUFFLED_RICH_REDUCED_BEST_MISMATCH")
    return out, {
        "schema": "CB16_R11_M2_RICH_SURFACE_ROTATION_V1",
        "shift": int(shuffle_receipt["shift"]),
        "teacher_surface_multiset_sha256_before": before,
        "teacher_surface_multiset_sha256_after": after,
        "teacher_surface_multiset_preserved": True,
        "scenario_identity_preserved": True,
        "same_donor_mapping_as_reduced_targets": True,
    }


def build_objective_target_m2(
    *,
    objective: str,
    teacher_probs: torch.Tensor,
    g0_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    require(objective in M2_OBJECTIVES, f"M2_UNKNOWN_OBJECTIVE:{objective}")
    require(teacher_probs.shape == g0_probs.shape, "M2_TARGET_PROB_SHAPE_DRIFT")
    require(teacher_probs.ndim == 2 and teacher_probs.shape[1] == 3, "M2_TARGET_PROB_SHAPE")
    teacher = teacher_probs.detach()
    g0 = g0_probs.detach()
    if objective == "ABS_CE":
        target = teacher.clone()
        preserve_mask = torch.zeros((len(teacher),), dtype=torch.bool, device=teacher.device)
    else:
        teacher_dir = torch.argmax(teacher, dim=-1)
        g0_dir = torch.argmax(g0, dim=-1)
        preserve_mask = teacher_dir == g0_dir
        target = torch.where(preserve_mask[:, None], g0, teacher)
    require(not target.requires_grad, "M2_OBJECTIVE_TARGET_AUTOGRAD_TAINT")
    sums = target.sum(dim=-1)
    require(torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0.0), "M2_OBJECTIVE_TARGET_NOT_DISTRIBUTION")
    return target.detach(), {
        "objective": objective,
        "rows": int(len(target)),
        "champion_preserve_rows": int(preserve_mask.sum().detach().cpu()),
        "teacher_learning_rows": int((~preserve_mask).sum().detach().cpu()) if objective != "ABS_CE" else int(len(target)),
        "target_sha256": _tensor_sha256(target),
    }


def prepare_epoch_permutations_m2(
    rows: int, *, device: str | torch.device
) -> tuple[torch.Tensor, ...]:
    require(rows > 0, "M2_NO_TRAIN_ROWS")
    out = []
    dev = torch.device(device)
    for epoch in range(CANONICAL_EPOCHS_R11):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(GENERATION_BASE_SEED_R11 + epoch))
        ids = torch.randperm(rows, generator=gen, dtype=torch.long)
        out.append(ids.to(dev))
    return tuple(out)


def _weighted_ce_m2(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    row = -(target.detach() * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    denom = weight.detach().sum()
    require(bool(torch.isfinite(denom).item()) and float(denom.detach().cpu()) > 0.0, "M2_WEIGHT_DENOM")
    return (row * weight.detach()).sum() / denom


def train_direction_residual_m2(
    *,
    cache: FrozenDirectionCacheM2,
    target_probs: torch.Tensor,
    group_weight: torch.Tensor,
    permutations: Sequence[torch.Tensor],
    device: str | torch.device,
) -> tuple[DirectionResidualM2, dict[str, Any]]:
    require(cache.rows == len(target_probs) == len(group_weight), "M2_TRAIN_ROW_COUNT_DRIFT")
    require(len(permutations) == CANONICAL_EPOCHS_R11, "M2_EPOCH_PERMUTATION_COUNT")
    residual = build_direction_residual_m2(device=device)
    initial_identity_error = verify_zero_residual_identity_m2(residual, cache)
    optimizer = torch.optim.AdamW(
        residual.parameters(), lr=CANONICAL_LR_R11, weight_decay=CANONICAL_WEIGHT_DECAY_R11
    )
    steps = 0
    last_loss = None
    last_pre_clip = None
    residual.train()
    for epoch, perm in enumerate(permutations):
        require(perm.device == cache.shared256.device, f"M2_PERM_DEVICE:{epoch}")
        for start in range(0, cache.rows, CANONICAL_BATCH_SIZE_R11):
            ids = perm[start : start + CANONICAL_BATCH_SIZE_R11]
            optimizer.zero_grad(set_to_none=True)
            delta = residual(cache.shared256.index_select(0, ids))
            logits = cache.base_logits.index_select(0, ids) + delta
            target = target_probs.index_select(0, ids)
            weight = group_weight.index_select(0, ids)
            loss = _weighted_ce_m2(logits, target, weight)
            require(bool(torch.isfinite(loss).item()), "M2_NONFINITE_TRAIN_LOSS")
            loss.backward()
            pre_clip = torch.nn.utils.clip_grad_norm_(
                residual.parameters(), GRADIENT_CLIP_MAX_NORM_R11
            )
            require(bool(torch.isfinite(pre_clip).item()), "M2_NONFINITE_GRAD_NORM")
            optimizer.step()
            steps += 1
            last_loss = float(loss.detach().cpu())
            last_pre_clip = float(pre_clip.detach().cpu())
    require(steps > 0, "M2_NO_OPTIMIZER_STEPS")
    require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in residual.parameters()), "M2_NONFINITE_RESIDUAL_GRAD")
    return residual, {
        "schema": "CB16_R11_M2_RESIDUAL_TRAINING_RECEIPT_V1",
        "optimizer": "AdamW_FP32",
        "epochs": CANONICAL_EPOCHS_R11,
        "batch_size": CANONICAL_BATCH_SIZE_R11,
        "lr": CANONICAL_LR_R11,
        "weight_decay": CANONICAL_WEIGHT_DECAY_R11,
        "gradient_clip_max_norm": GRADIENT_CLIP_MAX_NORM_R11,
        "base_seed": GENERATION_BASE_SEED_R11,
        "optimizer_steps": int(steps),
        "initial_g0_policy_max_abs_error": initial_identity_error,
        "last_batch_loss": last_loss,
        "last_pre_clip_grad_norm": last_pre_clip,
        "residual_parameter_l2": residual_parameter_l2_m2(residual),
        "residual_state_sha256": _sha256_obj(
            {
                name: _tensor_sha256(param)
                for name, param in residual.state_dict().items()
            }
        ),
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    require(v.shape == w.shape, "M2_WEIGHTED_MEAN_SHAPE")
    denom = float(np.sum(w))
    require(math.isfinite(denom) and denom > 0.0, "M2_WEIGHTED_MEAN_DENOM")
    return float(np.sum(v * w) / denom)


def _weighted_conditional_rate(mask: np.ndarray, event: np.ndarray, weights: np.ndarray) -> float | None:
    m = np.asarray(mask, dtype=bool)
    e = np.asarray(event, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    require(m.shape == e.shape == w.shape, "M2_RATE_SHAPE")
    denom = float(np.sum(w[m]))
    if denom <= 0.0:
        return None
    return float(np.sum(w[m & e]) / denom)


def _jensen_shannon_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64)
    b = np.asarray(q, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    a = np.clip(a, tiny, 1.0)
    b = np.clip(b, tiny, 1.0)
    a = a / a.sum(axis=1, keepdims=True)
    b = b / b.sum(axis=1, keepdims=True)
    m = 0.5 * (a + b)
    return 0.5 * np.sum(a * (np.log(a) - np.log(m)), axis=1) + 0.5 * np.sum(
        b * (np.log(b) - np.log(m)), axis=1
    )


def evaluate_direction_residual_m2(
    *,
    residual: DirectionResidualM2,
    cache: FrozenDirectionCacheM2,
    aligned_teacher_probs: torch.Tensor,
    aligned_best_direction_means: np.ndarray,
    group_weight: torch.Tensor,
) -> dict[str, Any]:
    residual.eval()
    with torch.inference_mode():
        delta = residual(cache.shared256)
        new_logits = cache.base_logits + delta
        new_probs_t = torch.softmax(new_logits, dim=-1)
    new_probs = new_probs_t.detach().cpu().numpy().astype(np.float64, copy=False)
    g0_probs = cache.base_probs.detach().cpu().numpy().astype(np.float64, copy=False)
    teacher_probs = aligned_teacher_probs.detach().cpu().numpy().astype(np.float64, copy=False)
    means = np.asarray(aligned_best_direction_means, dtype=np.float64)
    weight = group_weight.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    require(
        new_probs.shape == g0_probs.shape == teacher_probs.shape == means.shape,
        "M2_EVAL_SHAPE_DRIFT",
    )
    n = len(new_probs)
    row = np.arange(n)
    g0_dir = np.argmax(g0_probs, axis=1).astype(np.int64)
    new_dir = np.argmax(new_probs, axis=1).astype(np.int64)
    teacher_best = np.argmax(means, axis=1).astype(np.int64)
    require(np.array_equal(teacher_best, np.argmax(teacher_probs, axis=1)), "M2_EVAL_RICH_REDUCED_BEST_MISMATCH")

    discrete_gain = means[row, new_dir] - means[row, g0_dir]
    teacher_regret = means[row, teacher_best] - means[row, new_dir]
    soft_gain = np.sum(new_probs * (means - means[row, g0_dir][:, None]), axis=1)
    teacher_agrees = teacher_best == g0_dir
    direction_changed = new_dir != g0_dir
    move_to_teacher = new_dir == teacher_best
    js = _jensen_shannon_rows(g0_probs, new_probs)
    delta_l2 = torch.linalg.vector_norm(delta.detach(), dim=-1).cpu().numpy().astype(np.float64, copy=False)
    ce_rows = -np.sum(teacher_probs * np.log(np.clip(new_probs, np.finfo(np.float64).tiny, 1.0)), axis=1)

    disagreement = ~teacher_agrees
    advantage = means[row, teacher_best] - means[row, g0_dir]
    advantage[np.abs(advantage) < 1e-15] = 0.0
    require(np.all(advantage >= -1e-12), "M2_NEGATIVE_TEACHER_BEST_ADVANTAGE")
    advantage = np.maximum(advantage, 0.0)
    quartiles: list[dict[str, Any]] = []
    if np.any(disagreement):
        a = advantage[disagreement]
        edges = np.quantile(a, [0.25, 0.50, 0.75])
        bins = np.searchsorted(edges, a, side="right")
        mt = move_to_teacher[disagreement]
        ww = weight[disagreement]
        for q in range(4):
            mask = bins == q
            quartiles.append(
                {
                    "quartile": q + 1,
                    "rows": int(np.sum(mask)),
                    "advantage_min": None if not np.any(mask) else float(np.min(a[mask])),
                    "advantage_max": None if not np.any(mask) else float(np.max(a[mask])),
                    "move_to_teacher_rate": None
                    if not np.any(mask)
                    else float(np.sum(ww[mask] * mt[mask]) / np.sum(ww[mask])),
                }
            )

    agreement_change = _weighted_conditional_rate(teacher_agrees, direction_changed, weight)
    disagreement_move = _weighted_conditional_rate(disagreement, move_to_teacher, weight)
    return {
        "schema": "CB16_R11_M2_DIRECTION_RESIDUAL_EVAL_V1",
        "rows": int(n),
        "discrete_champion_relative_gain": _weighted_mean(discrete_gain, weight),
        "discrete_teacher_regret": _weighted_mean(teacher_regret, weight),
        "soft_expected_champion_relative_gain": _weighted_mean(soft_gain, weight),
        "negative_gain_rate": _weighted_mean((discrete_gain < 0.0).astype(np.float64), weight),
        "teacher_g0_agreement_change_rate": agreement_change,
        "teacher_g0_disagreement_move_to_teacher_rate": disagreement_move,
        "direction_change_rate": _weighted_mean(direction_changed.astype(np.float64), weight),
        "direction_ce_to_aligned_eval_teacher": _weighted_mean(ce_rows, weight),
        "jensen_shannon_divergence_vs_g0": _weighted_mean(js, weight),
        "residual_delta_logit_l2_mean": _weighted_mean(delta_l2, weight),
        "residual_delta_logit_l2_p90": float(np.quantile(delta_l2, 0.90)),
        "advantage_quartiles": quartiles,
        "teacher_g0_agreement_rows": int(np.sum(teacher_agrees)),
        "teacher_g0_disagreement_rows": int(np.sum(disagreement)),
        "direction_changed_rows": int(np.sum(direction_changed)),
        "move_to_teacher_rows": int(np.sum(disagreement & move_to_teacher)),
    }


def adjudicate_m2(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(fold_results) == 5, f"M2_FOLD_COUNT:{len(fold_results)}")
    by_fold = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in by_fold) == (1, 2, 3, 4, 5), "M2_FOLD_IDS")

    selective_positive = 0
    selective_gain_vs_shuffle = 0
    selective_move_vs_shuffle = 0
    preservation = 0
    abs_gain_vs_shuffle = 0
    per_fold = []
    for fold in by_fold:
        arms = fold["arms"]
        required = {
            "ABS_CE_ALIGNED",
            "SELECTIVE_ALIGNED",
            *{f"ABS_CE_SHUFFLE_{s}" for s in M2_SHIFTS},
            *{f"SELECTIVE_SHUFFLE_{s}" for s in M2_SHIFTS},
        }
        require(set(arms) == required, f"M2_ARM_SET_DRIFT:FOLD={fold['fold']}")
        sa = arms["SELECTIVE_ALIGNED"]["evaluation"]
        aa = arms["ABS_CE_ALIGNED"]["evaluation"]
        s_sh = [arms[f"SELECTIVE_SHUFFLE_{s}"]["evaluation"] for s in M2_SHIFTS]
        a_sh = [arms[f"ABS_CE_SHUFFLE_{s}"]["evaluation"] for s in M2_SHIFTS]
        med_s_gain = float(statistics.median(float(x["discrete_champion_relative_gain"]) for x in s_sh))
        med_s_move = float(statistics.median(float(x["teacher_g0_disagreement_move_to_teacher_rate"]) for x in s_sh))
        med_a_gain = float(statistics.median(float(x["discrete_champion_relative_gain"]) for x in a_sh))
        s_gain = float(sa["discrete_champion_relative_gain"])
        s_move = float(sa["teacher_g0_disagreement_move_to_teacher_rate"])
        a_gain = float(aa["discrete_champion_relative_gain"])
        s_pres = float(sa["teacher_g0_agreement_change_rate"])
        a_pres = float(aa["teacher_g0_agreement_change_rate"])

        c_positive = s_gain > 0.0
        c_gain = s_gain > med_s_gain
        c_move = s_move > med_s_move
        c_pres = s_pres < a_pres
        c_abs = a_gain > med_a_gain
        selective_positive += int(c_positive)
        selective_gain_vs_shuffle += int(c_gain)
        selective_move_vs_shuffle += int(c_move)
        preservation += int(c_pres)
        abs_gain_vs_shuffle += int(c_abs)
        per_fold.append(
            {
                "fold": int(fold["fold"]),
                "selective_aligned_gain": s_gain,
                "median_selective_shuffle_gain": med_s_gain,
                "selective_aligned_move_to_teacher_rate": s_move,
                "median_selective_shuffle_move_to_teacher_rate": med_s_move,
                "selective_aligned_agreement_change_rate": s_pres,
                "abs_ce_aligned_agreement_change_rate": a_pres,
                "abs_ce_aligned_gain": a_gain,
                "median_abs_ce_shuffle_gain": med_a_gain,
                "selective_gain_positive": c_positive,
                "selective_gain_gt_shuffle_median": c_gain,
                "selective_move_gt_shuffle_median": c_move,
                "selective_preserves_better_than_abs_ce": c_pres,
                "abs_ce_gain_gt_shuffle_median": c_abs,
            }
        )

    selective_pass = (
        selective_positive >= 4
        and selective_gain_vs_shuffle >= 4
        and selective_move_vs_shuffle >= 4
    )
    preservation_pass = preservation >= 4
    abs_secondary_pass = abs_gain_vs_shuffle >= 4
    if selective_pass and preservation_pass:
        legal = (
            "OBJECTIVE_UPDATE_CONTRACT_MISMATCH_SUPPORTED_AS_MATERIAL_MECHANISTIC_BOTTLENECK"
        )
    elif selective_pass:
        legal = (
            "FROZEN_RESIDUAL_EXTRACTS_ALIGNMENT_SPECIFIC_GAIN_BUT_SELECTIVE_PRESERVATION_NOT_ESTABLISHED"
        )
    else:
        legal = "M2_MINIMAL_CHAMPION_RELATIVE_RESIDUAL_MECHANISM_NOT_ESTABLISHED"

    return {
        "schema": "CB16_R11_M_SERIES_M2_ADJUDICATION_SUMMARY_V1",
        "selective_positive_gain_fold_count": int(selective_positive),
        "selective_gain_gt_shuffle_median_fold_count": int(selective_gain_vs_shuffle),
        "selective_move_to_teacher_gt_shuffle_median_fold_count": int(selective_move_vs_shuffle),
        "selective_preservation_better_than_abs_ce_fold_count": int(preservation),
        "abs_ce_gain_gt_shuffle_median_fold_count": int(abs_gain_vs_shuffle),
        "selective_alignment_pass": bool(selective_pass),
        "preservation_pass": bool(preservation_pass),
        "absolute_residual_alignment_secondary_pass": bool(abs_secondary_pass),
        "selective_alignment_conclusion": (
            "FROZEN_G0_SELECTIVE_RESIDUAL_EXTRACTS_ALIGNMENT_SPECIFIC_DIRECTION_GAIN_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
            if selective_pass
            else "FROZEN_G0_SELECTIVE_RESIDUAL_ALIGNMENT_SPECIFIC_GAIN_NOT_ESTABLISHED"
        ),
        "preservation_conclusion": (
            "SELECTIVE_CHAMPION_PRESERVE_OBJECTIVE_REDUCES_UNNECESSARY_DIRECTION_DRIFT_VS_ABSOLUTE_CE"
            if preservation_pass
            else "SELECTIVE_CHAMPION_PRESERVE_OBJECTIVE_DOES_NOT_RELIABLY_REDUCE_AGREEMENT_STATE_DRIFT_VS_ABSOLUTE_CE"
        ),
        "legal_mechanistic_interpretation": legal,
        "per_fold": per_fold,
        "market_information_verdict": False,
        "canonical_promotion_authorized": False,
        "canonical_generation_advance_authorized": False,
    }
