from __future__ import annotations

"""R11 vectorized probabilistic Teacher.

This is a clean-slate execution engine for the already-frozen R6 scientific law.  It is
NOT a new Teacher objective and it does NOT reinterpret realized outcomes as labels.

The old implementation performed most work target-by-target and action-by-action in
Python.  R11 exploits mathematical invariants that are already true under the frozen
CB16 evidence contract:

* all complete parents expose the same nine-action grid;
* nearest-parent geometry inside a future dependence group is action-independent;
* therefore one target selects nearest parents, top-k dependence groups and distance
  weights ONCE, then evaluates all nine utility laws from that shared support;
* BLOCKED_CROSSFIT targets sharing a fold share the exact same train-support set and
  normalization; PREQUENTIAL targets are grouped by their exact support tuple;
* target/support distances are evaluated in float64 columnar blocks with BLAS-friendly
  matrix algebra instead of Python scalar loops.

Legacy R6 remains an oracle during qualification only.  Runtime authority is the R11
semantic freeze and numerical/scientific equivalence gates.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .probabilistic_teacher_r6 import (
    DependenceAwarePredictiveLawR6,
    DependenceAwareTeacherConfigR6,
    DependenceAwareTeacherEvidenceR6,
    EvidenceAdmissionReceiptR6,
    _softmax,
    canonical_hash,
)
from .r102_evidence_cache import ParentContextR102


R11_TEACHER_ENGINE = "CB16_R11_COLUMNAR_VECTORIZED_TEACHER_V1"


@dataclass(frozen=True)
class ColumnarTeacherIndexR11:
    parent_ids: tuple[str, ...]
    student_context_ids: tuple[str, ...]
    parent_timestamps: np.ndarray              # [P] int64
    parent_dep_index: np.ndarray               # [P] int32
    features: np.ndarray                       # [P,F] float64
    utilities: np.ndarray                      # [P,A] float64
    action_directions: np.ndarray              # [A] int8
    action_risks: np.ndarray                   # [A] float64
    dep_ids: tuple[str, ...]
    dep_timestamps: np.ndarray                 # [D] int64
    dep_parent_rows: np.ndarray                # [D,Rmax] int32, -1 padding
    parent_row_by_id: Mapping[str, int]
    dep_row_by_id: Mapping[str, int]

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def action_count(self) -> int:
        return int(self.utilities.shape[1])


@dataclass(frozen=True)
class SupportRegimeR11:
    dep_rows: np.ndarray                       # [G] int32, canonical chronology order
    dep_ids: tuple[str, ...]
    train_dependence_group_hash: str
    mean: np.ndarray                           # [F] float64
    std: np.ndarray                            # [F] float64
    support_parent_rows: np.ndarray            # [G,Rmax] int32, -1 padding
    normalized_support_flat: np.ndarray        # [G*Rmax,F] float64
    normalized_support_norm2: np.ndarray       # [G*Rmax] float64
    valid_support_flat: np.ndarray             # [G*Rmax] bool
    dep_lex_rank: np.ndarray                   # [G] int32

    @property
    def group_count(self) -> int:
        return int(len(self.dep_rows))

    @property
    def max_parents_per_group(self) -> int:
        return int(self.support_parent_rows.shape[1])


@dataclass(frozen=True)
class VectorizedTeacherStatsR11:
    targets: int
    support_regimes: int
    geometry_blocks: int
    max_targets_per_block: int
    feature_dim: int
    actions: int
    engine: str = R11_TEACHER_ENGINE


def _validate_samples_and_build_parent_rows(
    samples: Sequence[CounterfactualBranchSampleR5],
) -> tuple[
    dict[str, list[CounterfactualBranchSampleR5]],
    tuple[tuple[int, float], ...],
]:
    rows_by_parent: dict[str, list[CounterfactualBranchSampleR5]] = {}
    for sample in samples:
        sample.validate()
        rows_by_parent.setdefault(sample.parent_id, []).append(sample)
    if not rows_by_parent:
        raise RuntimeError("R11_TEACHER_NO_SAMPLES")

    grid0: tuple[tuple[int, float], ...] | None = None
    for parent_id, rows in rows_by_parent.items():
        f0 = rows[0].context_features
        c0 = rows[0].student_context_object_id
        t0 = int(rows[0].timestamp)
        d0 = rows[0].dependence_group_id
        seen: dict[tuple[int, float], CounterfactualBranchSampleR5] = {}
        for row in rows:
            if row.context_features != f0:
                raise RuntimeError(f"R11_PARENT_CONTEXT_FEATURES_INCONSISTENT:{parent_id}")
            if row.student_context_object_id != c0:
                raise RuntimeError(f"R11_PARENT_CONTEXT_ID_INCONSISTENT:{parent_id}")
            if int(row.timestamp) != t0:
                raise RuntimeError(f"R11_PARENT_TIMESTAMP_INCONSISTENT:{parent_id}")
            if row.dependence_group_id != d0:
                raise RuntimeError(f"R11_PARENT_DEPENDENCE_GROUP_INCONSISTENT:{parent_id}")
            key = (int(row.direction), float(row.requested_risk))
            if key in seen:
                raise RuntimeError(f"R11_DUPLICATE_ACTION_BRANCH_WITHIN_PARENT:{parent_id}:{key}")
            seen[key] = row
        grid = tuple(sorted(seen, key=lambda x: (x[0], x[1])))
        if grid0 is None:
            grid0 = grid
        elif grid != grid0:
            raise RuntimeError(f"R11_ACTION_GRID_NOT_COMPLETE_OR_UNIFORM:{parent_id}")
    assert grid0 is not None
    if set(d for d, _ in grid0) != {-1, 0, 1}:
        raise RuntimeError(f"R11_ACTION_DIRECTION_GRID_INCOMPLETE:{grid0}")
    if (0, 0.0) not in grid0:
        raise RuntimeError("R11_FLAT_ZERO_ACTION_MISSING")
    return rows_by_parent, grid0


def build_columnar_teacher_index_r11(
    samples: Sequence[CounterfactualBranchSampleR5],
) -> ColumnarTeacherIndexR11:
    """Convert object-heavy branch samples to one compact immutable columnar index."""

    rows_by_parent, grid = _validate_samples_and_build_parent_rows(samples)

    parent_ts: dict[str, int] = {}
    parent_dep: dict[str, str] = {}
    student_id: dict[str, str] = {}
    dep_ts: dict[str, int] = {}
    parents_by_dep: dict[str, list[str]] = {}
    for parent_id, rows in rows_by_parent.items():
        row0 = rows[0]
        parent_ts[parent_id] = int(row0.timestamp)
        parent_dep[parent_id] = str(row0.dependence_group_id)
        student_id[parent_id] = str(row0.student_context_object_id)
        dep = parent_dep[parent_id]
        if dep in dep_ts and dep_ts[dep] != parent_ts[parent_id]:
            raise RuntimeError(f"R11_DEPENDENCE_GROUP_TIMESTAMP_INCONSISTENT:{dep}")
        dep_ts[dep] = parent_ts[parent_id]
        parents_by_dep.setdefault(dep, []).append(parent_id)

    dep_ids = tuple(sorted(parents_by_dep, key=lambda dep: (dep_ts[dep], dep)))
    dep_row_by_id = {dep: i for i, dep in enumerate(dep_ids)}
    for dep in dep_ids:
        parents_by_dep[dep].sort()

    # Physical parent order is dependence-group-major + parent-id lexical order.
    parent_ids = tuple(parent for dep in dep_ids for parent in parents_by_dep[dep])
    parent_row_by_id = {parent: i for i, parent in enumerate(parent_ids)}
    feature_dim = len(rows_by_parent[parent_ids[0]][0].context_features)
    action_count = len(grid)

    features = np.empty((len(parent_ids), feature_dim), dtype=np.float64)
    utilities = np.empty((len(parent_ids), action_count), dtype=np.float64)
    parent_timestamps = np.empty(len(parent_ids), dtype=np.int64)
    parent_dep_index = np.empty(len(parent_ids), dtype=np.int32)
    student_context_ids: list[str] = []

    grid_to_col = {key: i for i, key in enumerate(grid)}
    for row_idx, parent_id in enumerate(parent_ids):
        rows = rows_by_parent[parent_id]
        features[row_idx] = np.asarray(rows[0].context_features, dtype=np.float64)
        parent_timestamps[row_idx] = parent_ts[parent_id]
        parent_dep_index[row_idx] = dep_row_by_id[parent_dep[parent_id]]
        student_context_ids.append(student_id[parent_id])
        seen_cols = set()
        for row in rows:
            col = grid_to_col[(int(row.direction), float(row.requested_risk))]
            utilities[row_idx, col] = float(row.realized_utility)
            seen_cols.add(col)
        if len(seen_cols) != action_count:
            raise RuntimeError(f"R11_ACTION_GRID_COLUMN_MISSING:{parent_id}")

    if not np.all(np.isfinite(features)):
        raise RuntimeError("R11_NONFINITE_CONTEXT_FEATURE")
    if not np.all(np.isfinite(utilities)):
        raise RuntimeError("R11_NONFINITE_REALIZED_UTILITY")

    max_parents = max(len(parents_by_dep[d]) for d in dep_ids)
    dep_parent_rows = np.full((len(dep_ids), max_parents), -1, dtype=np.int32)
    for dep_idx, dep in enumerate(dep_ids):
        rows = [parent_row_by_id[p] for p in parents_by_dep[dep]]
        dep_parent_rows[dep_idx, : len(rows)] = rows

    return ColumnarTeacherIndexR11(
        parent_ids=parent_ids,
        student_context_ids=tuple(student_context_ids),
        parent_timestamps=parent_timestamps,
        parent_dep_index=parent_dep_index,
        features=np.ascontiguousarray(features),
        utilities=np.ascontiguousarray(utilities),
        action_directions=np.asarray([x[0] for x in grid], dtype=np.int8),
        action_risks=np.asarray([x[1] for x in grid], dtype=np.float64),
        dep_ids=dep_ids,
        dep_timestamps=np.asarray([dep_ts[d] for d in dep_ids], dtype=np.int64),
        dep_parent_rows=dep_parent_rows,
        parent_row_by_id=parent_row_by_id,
        dep_row_by_id=dep_row_by_id,
    )


def _fold_by_dep(index: ColumnarTeacherIndexR11, config: DependenceAwareTeacherConfigR6) -> np.ndarray:
    n = len(index.dep_ids)
    return np.asarray(
        [min(config.n_folds - 1, (i * config.n_folds) // max(n, 1)) for i in range(n)],
        dtype=np.int16,
    )


def train_dep_rows_for_target_r11(
    *,
    target_parent_id: str,
    index: ColumnarTeacherIndexR11,
    config: DependenceAwareTeacherConfigR6,
    eligible_train_dependence_groups: set[str] | None,
) -> np.ndarray:
    target_row = index.parent_row_by_id[target_parent_id]
    target_dep_row = int(index.parent_dep_index[target_row])
    target_ts = int(index.dep_timestamps[target_dep_row])

    if config.mode == "PREQUENTIAL":
        keep = index.dep_timestamps < target_ts
    elif config.mode == "BLOCKED_CROSSFIT":
        folds = _fold_by_dep(index, config)
        target_fold = int(folds[target_dep_row])
        excluded = folds == target_fold
        fold_idx = np.flatnonzero(excluded)
        if config.embargo_groups and len(fold_idx):
            lo = max(0, int(fold_idx[0]) - int(config.embargo_groups))
            hi = min(len(index.dep_ids) - 1, int(fold_idx[-1]) + int(config.embargo_groups))
            excluded[lo : hi + 1] = True
        keep = ~excluded
    else:
        raise ValueError(config.mode)

    keep[target_dep_row] = False
    if eligible_train_dependence_groups is not None:
        eligible = np.fromiter(
            (dep in eligible_train_dependence_groups for dep in index.dep_ids),
            dtype=np.bool_,
            count=len(index.dep_ids),
        )
        keep &= eligible
    return np.flatnonzero(keep).astype(np.int32, copy=False)


def group_targets_by_support_r11(
    *,
    target_parent_ids: Sequence[str],
    index: ColumnarTeacherIndexR11,
    config: DependenceAwareTeacherConfigR6,
    eligible_train_dependence_groups: set[str] | None,
) -> list[tuple[np.ndarray, list[str]]]:
    grouped: dict[tuple[int, ...], list[str]] = {}
    for parent_id in target_parent_ids:
        if parent_id not in index.parent_row_by_id:
            continue
        deps = train_dep_rows_for_target_r11(
            target_parent_id=parent_id,
            index=index,
            config=config,
            eligible_train_dependence_groups=eligible_train_dependence_groups,
        )
        grouped.setdefault(tuple(int(x) for x in deps), []).append(parent_id)
    return [
        (np.asarray(key, dtype=np.int32), parents)
        for key, parents in grouped.items()
    ]


def prepare_support_regime_r11(
    *,
    dep_rows: np.ndarray,
    index: ColumnarTeacherIndexR11,
) -> SupportRegimeR11:
    dep_rows = np.asarray(dep_rows, dtype=np.int32)
    dep_ids = tuple(index.dep_ids[int(i)] for i in dep_rows)
    parent_matrix = np.asarray(index.dep_parent_rows[dep_rows], dtype=np.int32)
    valid = parent_matrix >= 0
    parent_rows = parent_matrix[valid]
    if len(parent_rows) == 0:
        feature_dim = index.feature_dim
        return SupportRegimeR11(
            dep_rows=dep_rows,
            dep_ids=dep_ids,
            train_dependence_group_hash=canonical_hash(dep_ids),
            mean=np.zeros(feature_dim, dtype=np.float64),
            std=np.ones(feature_dim, dtype=np.float64),
            support_parent_rows=parent_matrix,
            normalized_support_flat=np.empty((0, feature_dim), dtype=np.float64),
            normalized_support_norm2=np.empty(0, dtype=np.float64),
            valid_support_flat=np.empty(0, dtype=np.bool_),
            dep_lex_rank=np.empty(0, dtype=np.int32),
        )

    x = np.asarray(index.features[parent_rows], dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)

    safe_rows = parent_matrix.copy()
    safe_rows[~valid] = int(parent_rows[0])
    z = (index.features[safe_rows.reshape(-1)] - mean) / std
    z = np.ascontiguousarray(z, dtype=np.float64)
    valid_flat = valid.reshape(-1)
    z[~valid_flat] = 0.0
    norm2 = np.einsum("ij,ij->i", z, z, optimize=True)

    # Candidate sorting in R6 is (distance, dependence_group_id, parent_id).
    # There is one selected parent per dependence group, so the secondary key is dep id.
    lexical = sorted(range(len(dep_ids)), key=lambda i: dep_ids[i])
    dep_lex_rank = np.empty(len(dep_ids), dtype=np.int32)
    for rank, local_idx in enumerate(lexical):
        dep_lex_rank[local_idx] = rank

    return SupportRegimeR11(
        dep_rows=dep_rows,
        dep_ids=dep_ids,
        train_dependence_group_hash=canonical_hash(dep_ids),
        mean=np.ascontiguousarray(mean),
        std=np.ascontiguousarray(std),
        support_parent_rows=parent_matrix,
        normalized_support_flat=z,
        normalized_support_norm2=np.ascontiguousarray(norm2),
        valid_support_flat=np.ascontiguousarray(valid_flat),
        dep_lex_rank=dep_lex_rank,
    )


def _weighted_quantiles_batch_r11(
    values: np.ndarray,          # [B,K,A]
    weights: np.ndarray,         # [B,K]
    quantile_levels: Sequence[float],
) -> np.ndarray:                 # [B,A,Q]
    """Vectorized equivalent of R5 weighted_quantile for positive weights."""

    v = np.transpose(np.asarray(values, dtype=np.float64), (0, 2, 1))  # [B,A,K]
    w0 = np.asarray(weights, dtype=np.float64)[:, None, :]
    order = np.argsort(v, axis=-1, kind="stable")
    sv = np.take_along_axis(v, order, axis=-1)
    sw = np.take_along_axis(np.broadcast_to(w0, v.shape), order, axis=-1)
    c = np.cumsum(sw, axis=-1) - 0.5 * sw
    c /= np.sum(sw, axis=-1, keepdims=True)

    qs = np.asarray(tuple(quantile_levels), dtype=np.float64)
    out = np.empty((*sv.shape[:2], len(qs)), dtype=np.float64)
    k = sv.shape[-1]
    for qi, q in enumerate(qs):
        left = q <= c[..., 0]
        right = q >= c[..., -1]
        hi = np.sum(c < q, axis=-1)
        hi = np.clip(hi, 1, k - 1)
        lo = hi - 1
        clo = np.take_along_axis(c, lo[..., None], axis=-1)[..., 0]
        chi = np.take_along_axis(c, hi[..., None], axis=-1)[..., 0]
        vlo = np.take_along_axis(sv, lo[..., None], axis=-1)[..., 0]
        vhi = np.take_along_axis(sv, hi[..., None], axis=-1)[..., 0]
        frac = (q - clo) / (chi - clo)
        interp = vlo + frac * (vhi - vlo)
        interp = np.where(left, sv[..., 0], interp)
        interp = np.where(right, sv[..., -1], interp)
        out[..., qi] = interp
    return out


def _softmax_batch_r11(x: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / float(temperature)
    z -= np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / np.sum(e, axis=1, keepdims=True)


def _compile_block_r11(
    *,
    target_parent_ids: Sequence[str],
    index: ColumnarTeacherIndexR11,
    regime: SupportRegimeR11,
    config: DependenceAwareTeacherConfigR6,
) -> list[DependenceAwareTeacherEvidenceR6]:
    if not target_parent_ids:
        return []

    target_rows = np.asarray([index.parent_row_by_id[p] for p in target_parent_ids], dtype=np.int32)
    target = (index.features[target_rows] - regime.mean) / regime.std
    b = len(target_rows)
    g = regime.group_count
    rmax = regime.max_parents_per_group
    if g == 0:
        parent_dist = np.empty((b, 0, rmax), dtype=np.float64)
    else:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b.  Float64 keeps ranking error tiny
        # while BLAS turns the dominant geometry into an AVX2-friendly dense operation.
        target_norm2 = np.einsum("ij,ij->i", target, target, optimize=True)
        dot = target @ regime.normalized_support_flat.T
        sq = (
            target_norm2[:, None]
            + regime.normalized_support_norm2[None, :]
            - 2.0 * dot
        ) / float(index.feature_dim)
        np.maximum(sq, 0.0, out=sq)
        flat_dist = np.sqrt(sq, out=sq)
        flat_dist[:, ~regime.valid_support_flat] = np.inf
        parent_dist = flat_dist.reshape(b, g, rmax)

    if g:
        nearest_slot = np.argmin(parent_dist, axis=2)  # first slot = lexical parent tie-break
        nearest_distance = np.take_along_axis(parent_dist, nearest_slot[..., None], axis=2)[..., 0]
        safe_parent_matrix = regime.support_parent_rows.copy()
        first_valid = int(safe_parent_matrix[safe_parent_matrix >= 0][0])
        safe_parent_matrix[safe_parent_matrix < 0] = first_valid
        nearest_parent_row = np.take_along_axis(
            np.broadcast_to(safe_parent_matrix[None, :, :], parent_dist.shape),
            nearest_slot[..., None],
            axis=2,
        )[..., 0]

        # Exact R6 tie rule for dependence groups: distance then dep-id lexical order.
        rank = np.broadcast_to(regime.dep_lex_rank[None, :], nearest_distance.shape)
        order = np.lexsort((rank, nearest_distance), axis=1)
        k = min(int(config.k_dependence_groups), g)
        top_local_dep = order[:, :k]
        top_distance = np.take_along_axis(nearest_distance, top_local_dep, axis=1)
        selected_parent_rows = np.take_along_axis(nearest_parent_row, top_local_dep, axis=1)
        selected_utility = index.utilities[selected_parent_rows]  # [B,K,A]
        weights = np.exp(-0.5 * (top_distance / float(config.distance_temperature)) ** 2) + 1e-12
        weights /= np.sum(weights, axis=1, keepdims=True)
        means = np.einsum("bk,bka->ba", weights, selected_utility, optimize=True)
        centered = selected_utility - means[:, None, :]
        variances = np.einsum("bk,bka->ba", weights, centered * centered, optimize=True)
        stds = np.sqrt(np.maximum(variances, 0.0))
        effective_n = 1.0 / np.sum(weights * weights, axis=1)
        quantiles = _weighted_quantiles_batch_r11(selected_utility, weights, config.quantile_levels)
    else:
        k = 0
        top_local_dep = np.empty((b, 0), dtype=np.int32)
        top_distance = np.empty((b, 0), dtype=np.float64)
        means = np.empty((b, index.action_count), dtype=np.float64)
        stds = np.empty_like(means)
        effective_n = np.zeros(b, dtype=np.float64)
        quantiles = np.empty((b, index.action_count, len(config.quantile_levels)), dtype=np.float64)

    # Best action within each direction. Grid is risk-ascending, therefore np.argmax's
    # first-on-tie behavior exactly implements R6's (mean_utility, -requested_risk).
    action_idx_by_direction = {
        direction: np.flatnonzero(index.action_directions == direction)
        for direction in (-1, 0, 1)
    }
    complete_directions = all(len(v) for v in action_idx_by_direction.values()) and g > 0
    if complete_directions:
        best_action_cols = []
        for direction in (-1, 0, 1):
            cols = action_idx_by_direction[direction]
            local = np.argmax(means[:, cols], axis=1)
            best_action_cols.append(cols[local])
        best_action_cols_arr = np.stack(best_action_cols, axis=1)
        best_means = np.take_along_axis(means, best_action_cols_arr, axis=1)
        probs = _softmax_batch_r11(best_means, config.direction_softmax_temperature)
        best_risks = index.action_risks[best_action_cols_arr]
        best_risks[:, 1] = 0.0
        risk_target = np.sum(probs * best_risks, axis=1)
    else:
        probs = np.full((b, 3), 1.0 / 3.0, dtype=np.float64)
        risk_target = np.zeros(b, dtype=np.float64)

    out: list[DependenceAwareTeacherEvidenceR6] = []
    for i, parent_id in enumerate(target_parent_ids):
        if g:
            selected_dep_ids = [regime.dep_ids[int(x)] for x in top_local_dep[i]]
            support_hash = canonical_hash(selected_dep_ids)
            nearest = float(top_distance[i, 0])
            max_used = float(top_distance[i, -1])
            eff = float(effective_n[i])
            laws = tuple(
                DependenceAwarePredictiveLawR6(
                    direction=int(index.action_directions[a]),
                    requested_risk=float(index.action_risks[a]),
                    mean_utility=float(means[i, a]),
                    std_utility=float(stds[i, a]),
                    quantile_levels=config.quantile_levels,
                    quantiles=tuple(float(x) for x in quantiles[i, a]),
                    effective_dependence_n=eff,
                    unique_dependence_groups=k,
                    nearest_distance=nearest,
                    max_distance_used=max_used,
                    support_dependence_group_hash=support_hash,
                )
                for a in range(index.action_count)
            )
        else:
            laws = ()
            nearest = float("inf")
            eff = 0.0

        reasons: list[str] = []
        if g < int(config.min_train_dependence_groups):
            reasons.append("INSUFFICIENT_TRAIN_DEPENDENCE_GROUPS")
        if not complete_directions:
            reasons.append("ACTION_GRID_SUPPORT_INCOMPLETE")
        if eff < float(config.min_effective_dependence_n):
            reasons.append("INSUFFICIENT_EFFECTIVE_DEPENDENCE_SUPPORT")
        if nearest > float(config.max_nearest_distance):
            reasons.append("TARGET_OUTSIDE_SUPPORTED_CONTEXT")

        admission = EvidenceAdmissionReceiptR6(
            status="ADMITTED" if not reasons else "EVIDENCE_NOT_READY",
            lane=config.lane,
            unique_train_dependence_groups=g,
            minimum_action_effective_dependence_n=eff,
            maximum_action_nearest_distance=nearest,
            reasons=tuple(reasons),
            protocol_hash=config.content_hash,
        )
        parent_row = index.parent_row_by_id[parent_id]
        dep_id = index.dep_ids[int(index.parent_dep_index[parent_row])]
        out.append(DependenceAwareTeacherEvidenceR6(
            evidence_id=f"R6E:{parent_id}:{config.content_hash[:12]}",
            parent_id=parent_id,
            student_context_object_id=index.student_context_ids[parent_row],
            target_dependence_group_id=dep_id,
            timestamp=int(index.parent_timestamps[parent_row]),
            teacher_version=config.teacher_version,
            teacher_protocol_hash=config.content_hash,
            train_dependence_group_hash=regime.train_dependence_group_hash,
            action_laws=laws,
            direction_target_probs=tuple(float(x) for x in probs[i]),
            requested_risk_target=float(risk_target[i]),
            direction_weight=float(config.direction_weight) if admission.admitted else 0.0,
            sizing_weight=float(config.sizing_weight) if admission.admitted else 0.0,
            admission=admission,
        ))
    return out


def compile_teacher_evidence_vectorized_r11(
    *,
    samples: Sequence[CounterfactualBranchSampleR5],
    parents: Mapping[str, ParentContextR102],
    train_config: DependenceAwareTeacherConfigR6,
    val_config: DependenceAwareTeacherConfigR6,
    block_targets: int = 64,
) -> tuple[list[DependenceAwareTeacherEvidenceR6], list[DependenceAwareTeacherEvidenceR6], VectorizedTeacherStatsR11]:
    """Compile the frozen CB16 probabilistic Teacher with a vectorized columnar engine."""

    if int(block_targets) <= 0:
        raise ValueError("block_targets")
    train_config.validate(); val_config.validate()
    index = build_columnar_teacher_index_r11(samples)
    train_parent_ids = sorted(
        p.parent_id for p in parents.values()
        if p.split == "TRAIN" and p.parent_id in index.parent_row_by_id
    )
    val_parent_ids = sorted(
        p.parent_id for p in parents.values()
        if p.split == "VALIDATION" and p.parent_id in index.parent_row_by_id
    )
    eligible_train_groups = {p.dependence_group_id for p in parents.values() if p.split == "TRAIN"}

    compiled: dict[str, DependenceAwareTeacherEvidenceR6] = {}
    regimes = 0
    blocks = 0
    for target_ids, config in ((train_parent_ids, train_config), (val_parent_ids, val_config)):
        support_groups = group_targets_by_support_r11(
            target_parent_ids=target_ids,
            index=index,
            config=config,
            eligible_train_dependence_groups=eligible_train_groups,
        )
        for dep_rows, regime_targets in support_groups:
            regime = prepare_support_regime_r11(dep_rows=dep_rows, index=index)
            regimes += 1
            for start in range(0, len(regime_targets), int(block_targets)):
                chunk = regime_targets[start : start + int(block_targets)]
                for evidence in _compile_block_r11(
                    target_parent_ids=chunk,
                    index=index,
                    regime=regime,
                    config=config,
                ):
                    if evidence.parent_id in compiled:
                        raise RuntimeError(f"R11_DUPLICATE_COMPILED_TARGET:{evidence.parent_id}")
                    compiled[evidence.parent_id] = evidence
                blocks += 1

    train = [compiled[p] for p in train_parent_ids]
    val = [compiled[p] for p in val_parent_ids]
    stats = VectorizedTeacherStatsR11(
        targets=len(train) + len(val),
        support_regimes=regimes,
        geometry_blocks=blocks,
        max_targets_per_block=int(block_targets),
        feature_dim=index.feature_dim,
        actions=index.action_count,
    )
    return train, val, stats
