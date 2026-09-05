from __future__ import annotations

"""
Dependence-group-aware probabilistic Teacher for R6.

R5 correctly prevented counterfactual branches of one parent from becoming independent
samples. R6 extends that rule to multiple AccountStates sharing the same future market path.

Example:
    one BTC future path at timestamp t
    × 1,000 account replicas
    × 9 counterfactual actions

is still ONE independent market-future dependence group.

For a target context and action/risk candidate, each historical dependence group contributes
at most one utility sample: the candidate branch belonging to the nearest Account/Market
context within that group. kNN and effective support therefore operate on independent future
groups rather than account replicas.
"""

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r5 import (
    CounterfactualBranchSampleR5,
    weighted_quantile,
)
from .sharded_experience_lake import ExperienceObject, ExperienceRef, ShardedExperienceLake


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class DependenceAwareTeacherConfigR6:
    teacher_version: str = "CB16_DEPENDENCE_AWARE_KNN_TEACHER_R6"
    mode: Literal["PREQUENTIAL", "BLOCKED_CROSSFIT"] = "PREQUENTIAL"
    n_folds: int = 5
    embargo_groups: int = 1
    k_dependence_groups: int = 64
    min_train_dependence_groups: int = 32
    min_effective_dependence_n: float = 12.0
    max_nearest_distance: float = 8.0
    distance_temperature: float = 2.0
    direction_softmax_temperature: float = 0.002
    quantile_levels: tuple[float, ...] = (
        0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95
    )
    lane: str = "CENTER"
    direction_weight: float = 1.0
    sizing_weight: float = 1.0

    def validate(self):
        if self.mode not in {"PREQUENTIAL", "BLOCKED_CROSSFIT"}:
            raise ValueError("bad mode")
        if self.n_folds < 2:
            raise ValueError("n_folds")
        if self.embargo_groups < 0:
            raise ValueError("embargo_groups")
        if self.k_dependence_groups <= 0:
            raise ValueError("k_dependence_groups")
        if self.min_train_dependence_groups <= 0:
            raise ValueError("min_train_dependence_groups")
        if self.min_effective_dependence_n <= 0:
            raise ValueError("min_effective_dependence_n")
        if self.max_nearest_distance <= 0:
            raise ValueError("max_nearest_distance")
        if self.distance_temperature <= 0 or self.direction_softmax_temperature <= 0:
            raise ValueError("temperatures")
        q = np.asarray(self.quantile_levels)
        if np.any((q <= 0) | (q >= 1)) or np.any(np.diff(q) <= 0):
            raise ValueError("quantiles")

    @property
    def content_hash(self) -> str:
        self.validate()
        return canonical_hash(self)


@dataclass(frozen=True)
class DependenceAwarePredictiveLawR6:
    direction: int
    requested_risk: float
    mean_utility: float
    std_utility: float
    quantile_levels: tuple[float, ...]
    quantiles: tuple[float, ...]
    effective_dependence_n: float
    unique_dependence_groups: int
    nearest_distance: float
    max_distance_used: float
    support_dependence_group_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class EvidenceAdmissionReceiptR6:
    status: str
    lane: str
    unique_train_dependence_groups: int
    minimum_action_effective_dependence_n: float
    maximum_action_nearest_distance: float
    reasons: tuple[str, ...]
    protocol_hash: str

    @property
    def admitted(self) -> bool:
        return self.status == "ADMITTED"

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class DependenceAwareTeacherEvidenceR6:
    evidence_id: str
    parent_id: str
    student_context_object_id: str
    target_dependence_group_id: str
    timestamp: int
    teacher_version: str
    teacher_protocol_hash: str
    train_dependence_group_hash: str
    action_laws: tuple[DependenceAwarePredictiveLawR6, ...]
    direction_target_probs: tuple[float, float, float]
    requested_risk_target: float
    direction_weight: float
    sizing_weight: float
    admission: EvidenceAdmissionReceiptR6

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass
class TeacherIndexR6:
    parents: list[str]
    rows_by_parent: dict[str, list[CounterfactualBranchSampleR5]]
    parent_timestamp: dict[str, int]
    parent_dependence_group: dict[str, str]
    dependence_groups: list[str]
    parents_by_dependence_group: dict[str, list[str]]
    dependence_group_timestamp: dict[str, int]


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / temperature
    z -= np.max(z)
    e = np.exp(np.clip(z, -60, 60))
    return e / e.sum()


class DependenceAwareProbabilisticTeacherR6:
    def __init__(self, config: DependenceAwareTeacherConfigR6 | None = None):
        self.config = config or DependenceAwareTeacherConfigR6()
        self.config.validate()

    @staticmethod
    def index(
        samples: Sequence[CounterfactualBranchSampleR5],
    ) -> TeacherIndexR6:
        rows_by_parent: dict[str, list[CounterfactualBranchSampleR5]] = {}
        parent_ts = {}
        parent_dep = {}
        for s in samples:
            s.validate()
            rows_by_parent.setdefault(s.parent_id, []).append(s)
            parent_ts.setdefault(s.parent_id, s.timestamp)
            if parent_ts[s.parent_id] != s.timestamp:
                raise RuntimeError("PARENT_TIMESTAMP_INCONSISTENT")
            parent_dep.setdefault(s.parent_id, s.dependence_group_id)
            if parent_dep[s.parent_id] != s.dependence_group_id:
                raise RuntimeError("PARENT_DEPENDENCE_GROUP_INCONSISTENT")

        for p, rows in rows_by_parent.items():
            f0 = rows[0].context_features
            c0 = rows[0].student_context_object_id
            for r in rows[1:]:
                if r.context_features != f0:
                    raise RuntimeError("PARENT_CONTEXT_FEATURES_INCONSISTENT")
                if r.student_context_object_id != c0:
                    raise RuntimeError("PARENT_CONTEXT_ID_INCONSISTENT")

        parents_by_dep: dict[str, list[str]] = {}
        dep_ts = {}
        for p, dep in parent_dep.items():
            parents_by_dep.setdefault(dep, []).append(p)
            dep_ts.setdefault(dep, parent_ts[p])
            if dep_ts[dep] != parent_ts[p]:
                raise RuntimeError(
                    "DEPENDENCE_GROUP_MUST_SHARE_DECISION_TIMESTAMP"
                )
        for dep in parents_by_dep:
            parents_by_dep[dep].sort()

        deps = sorted(parents_by_dep, key=lambda d: (dep_ts[d], d))
        parents = sorted(rows_by_parent, key=lambda p: (parent_ts[p], p))
        return TeacherIndexR6(
            parents=parents,
            rows_by_parent=rows_by_parent,
            parent_timestamp=parent_ts,
            parent_dependence_group=parent_dep,
            dependence_groups=deps,
            parents_by_dependence_group=parents_by_dep,
            dependence_group_timestamp=dep_ts,
        )

    def _folds(self, deps: Sequence[str]) -> dict[str, int]:
        n = len(deps)
        return {
            dep: min(
                self.config.n_folds - 1,
                (i * self.config.n_folds) // max(n, 1),
            )
            for i, dep in enumerate(deps)
        }

    def train_dependence_groups_for_target(
        self,
        *,
        target_parent: str,
        index: TeacherIndexR6,
        eligible_train_dependence_groups: set[str] | None = None,
    ) -> list[str]:
        target_dep = index.parent_dependence_group[target_parent]
        target_ts = index.dependence_group_timestamp[target_dep]
        deps = index.dependence_groups

        if self.config.mode == "PREQUENTIAL":
            train = [
                d for d in deps
                if index.dependence_group_timestamp[d] < target_ts
            ]
        else:
            folds = self._folds(deps)
            target_fold = folds[target_dep]
            excluded = {d for d in deps if folds[d] == target_fold}
            fold_idx = [i for i, d in enumerate(deps) if folds[d] == target_fold]
            if self.config.embargo_groups and fold_idx:
                lo = max(0, min(fold_idx) - self.config.embargo_groups)
                hi = min(len(deps) - 1, max(fold_idx) + self.config.embargo_groups)
                excluded.update(deps[lo:hi + 1])
            train = [d for d in deps if d not in excluded]

        # Never allow the target's shared future group into its own Teacher support.
        train = [d for d in train if d != target_dep]
        if eligible_train_dependence_groups is not None:
            train = [d for d in train if d in eligible_train_dependence_groups]
        return train

    def _normalization(
        self,
        *,
        train_deps: Sequence[str],
        index: TeacherIndexR6,
        feature_override: Mapping[str, tuple[float, ...]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        for dep in train_deps:
            for p in index.parents_by_dependence_group[dep]:
                feat = (
                    feature_override[p]
                    if feature_override is not None
                    else index.rows_by_parent[p][0].context_features
                )
                xs.append(feat)
        x = np.asarray(xs, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=0)
        return mean, np.where(std < 1e-8, 1.0, std)

    def predictive_law(
        self,
        *,
        target_features: tuple[float, ...] | np.ndarray,
        train_deps: Sequence[str],
        index: TeacherIndexR6,
        direction: int,
        risk: float,
        feature_override: Mapping[str, tuple[float, ...]] | None = None,
        equal_weight_climatology: bool = False,
    ) -> DependenceAwarePredictiveLawR6 | None:
        if not train_deps:
            return None

        if equal_weight_climatology:
            selected_y = []
            selected_groups = []
            for dep in train_deps:
                vals = []
                for p in index.parents_by_dependence_group[dep]:
                    match = [
                        s for s in index.rows_by_parent[p]
                        if s.direction == direction
                        and abs(s.requested_risk - risk) <= 1e-12
                    ]
                    if match:
                        if len(match) != 1:
                            raise RuntimeError("DUPLICATE_ACTION_BRANCH_WITHIN_PARENT")
                        vals.append(float(match[0].realized_utility))
                if vals:
                    # One independent value per future group.
                    selected_y.append(float(np.mean(vals)))
                    selected_groups.append(dep)
            if not selected_y:
                return None
            y = np.asarray(selected_y, dtype=np.float64)
            w = np.full(len(y), 1.0 / len(y), dtype=np.float64)
            q = weighted_quantile(
                y,
                w,
                np.asarray(self.config.quantile_levels, dtype=np.float64),
            )
            mu = float(np.mean(y))
            return DependenceAwarePredictiveLawR6(
                direction=direction,
                requested_risk=float(risk),
                mean_utility=mu,
                std_utility=float(np.std(y, ddof=0)),
                quantile_levels=self.config.quantile_levels,
                quantiles=tuple(float(x) for x in q),
                effective_dependence_n=float(len(y)),
                unique_dependence_groups=len(y),
                nearest_distance=0.0,
                max_distance_used=0.0,
                support_dependence_group_hash=canonical_hash(selected_groups),
            )

        mean, std = self._normalization(
            train_deps=train_deps,
            index=index,
            feature_override=feature_override,
        )
        t = (np.asarray(target_features, dtype=np.float64) - mean) / std

        # Each dependence group contributes its nearest matching parent only.
        candidates = []
        for dep in train_deps:
            best = None
            for p in index.parents_by_dependence_group[dep]:
                match = [
                    s for s in index.rows_by_parent[p]
                    if s.direction == direction
                    and abs(s.requested_risk - risk) <= 1e-12
                ]
                if not match:
                    continue
                if len(match) != 1:
                    raise RuntimeError("DUPLICATE_ACTION_BRANCH_WITHIN_PARENT")
                feat = (
                    feature_override[p]
                    if feature_override is not None
                    else index.rows_by_parent[p][0].context_features
                )
                z = (np.asarray(feat, dtype=np.float64) - mean) / std
                dist = float(np.sqrt(np.mean((z - t) ** 2)))
                row = (dist, p, float(match[0].realized_utility))
                if best is None or row[:2] < best[:2]:
                    best = row
            if best is not None:
                candidates.append((best[0], dep, best[1], best[2]))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        selected = candidates[: min(
            self.config.k_dependence_groups,
            len(candidates),
        )]
        d = np.asarray([x[0] for x in selected], dtype=np.float64)
        y = np.asarray([x[3] for x in selected], dtype=np.float64)
        deps = [x[1] for x in selected]
        w = np.exp(
            -0.5 * (d / self.config.distance_temperature) ** 2
        ) + 1e-12
        w /= w.sum()
        q = weighted_quantile(
            y,
            w,
            np.asarray(self.config.quantile_levels, dtype=np.float64),
        )
        mu = float(np.sum(w * y))
        var = float(np.sum(w * (y - mu) ** 2))
        eff = float(1.0 / np.sum(w ** 2))
        return DependenceAwarePredictiveLawR6(
            direction=direction,
            requested_risk=float(risk),
            mean_utility=mu,
            std_utility=math.sqrt(max(0.0, var)),
            quantile_levels=self.config.quantile_levels,
            quantiles=tuple(float(x) for x in q),
            effective_dependence_n=eff,
            unique_dependence_groups=len(deps),
            nearest_distance=float(d[0]),
            max_distance_used=float(d[-1]),
            support_dependence_group_hash=canonical_hash(deps),
        )

    def compile_one(
        self,
        *,
        target_parent: str,
        index: TeacherIndexR6,
        eligible_train_dependence_groups: set[str] | None = None,
        feature_override: Mapping[str, tuple[float, ...]] | None = None,
        target_feature_override: tuple[float, ...] | None = None,
    ) -> DependenceAwareTeacherEvidenceR6:
        rows = index.rows_by_parent[target_parent]
        train_deps = self.train_dependence_groups_for_target(
            target_parent=target_parent,
            index=index,
            eligible_train_dependence_groups=eligible_train_dependence_groups,
        )
        target_features = (
            target_feature_override
            if target_feature_override is not None
            else rows[0].context_features
        )
        grid = sorted(
            {(s.direction, float(s.requested_risk)) for s in rows},
            key=lambda x: (x[0], x[1]),
        )
        laws = []
        for d, r in grid:
            law = self.predictive_law(
                target_features=target_features,
                train_deps=train_deps,
                index=index,
                direction=d,
                risk=r,
                feature_override=feature_override,
            )
            if law is not None:
                laws.append(law)

        best_by_direction = {}
        for d in (-1, 0, 1):
            x = [l for l in laws if l.direction == d]
            if x:
                best_by_direction[d] = max(
                    x,
                    key=lambda l: (l.mean_utility, -l.requested_risk),
                )

        reasons = []
        if len(train_deps) < self.config.min_train_dependence_groups:
            reasons.append("INSUFFICIENT_TRAIN_DEPENDENCE_GROUPS")
        if set(best_by_direction) != {-1, 0, 1}:
            reasons.append("ACTION_GRID_SUPPORT_INCOMPLETE")
        min_eff = min(
            (l.effective_dependence_n for l in best_by_direction.values()),
            default=0.0,
        )
        max_nearest = max(
            (l.nearest_distance for l in best_by_direction.values()),
            default=float("inf"),
        )
        if min_eff < self.config.min_effective_dependence_n:
            reasons.append("INSUFFICIENT_EFFECTIVE_DEPENDENCE_SUPPORT")
        if max_nearest > self.config.max_nearest_distance:
            reasons.append("TARGET_OUTSIDE_SUPPORTED_CONTEXT")

        if set(best_by_direction) == {-1, 0, 1}:
            means = np.asarray([
                best_by_direction[-1].mean_utility,
                best_by_direction[0].mean_utility,
                best_by_direction[1].mean_utility,
            ])
            probs = _softmax(
                means,
                self.config.direction_softmax_temperature,
            )
            risks = np.asarray([
                best_by_direction[-1].requested_risk,
                0.0,
                best_by_direction[1].requested_risk,
            ])
            risk_target = float(np.sum(probs * risks))
        else:
            probs = np.asarray([1/3, 1/3, 1/3], dtype=np.float64)
            risk_target = 0.0

        admission = EvidenceAdmissionReceiptR6(
            status="ADMITTED" if not reasons else "EVIDENCE_NOT_READY",
            lane=self.config.lane,
            unique_train_dependence_groups=len(train_deps),
            minimum_action_effective_dependence_n=float(min_eff),
            maximum_action_nearest_distance=float(max_nearest),
            reasons=tuple(reasons),
            protocol_hash=self.config.content_hash,
        )
        dep = index.parent_dependence_group[target_parent]
        return DependenceAwareTeacherEvidenceR6(
            evidence_id=f"R6E:{target_parent}:{self.config.content_hash[:12]}",
            parent_id=target_parent,
            student_context_object_id=rows[0].student_context_object_id,
            target_dependence_group_id=dep,
            timestamp=rows[0].timestamp,
            teacher_version=self.config.teacher_version,
            teacher_protocol_hash=self.config.content_hash,
            train_dependence_group_hash=canonical_hash(train_deps),
            action_laws=tuple(laws),
            direction_target_probs=tuple(float(x) for x in probs),
            requested_risk_target=risk_target,
            direction_weight=self.config.direction_weight if admission.admitted else 0.0,
            sizing_weight=self.config.sizing_weight if admission.admitted else 0.0,
            admission=admission,
        )


    def persist_evidence(
        self,
        *,
        lake: ShardedExperienceLake,
        evidence: DependenceAwareTeacherEvidenceR6,
        policy_generation: int,
        policy_weight_hash: str,
        parent_snapshot_hash: str,
    ) -> ExperienceRef:
        payload = {
            "evidence_id": evidence.evidence_id,
            "parent_id": evidence.parent_id,
            "student_context_object_id": evidence.student_context_object_id,
            "timestamp": evidence.timestamp,
            "target_group_id": evidence.target_dependence_group_id,
            "teacher_identity_hash": evidence.teacher_protocol_hash,
            "generation": int(policy_generation),
            "admitted": evidence.admission.admitted,
            "lane": evidence.admission.lane,
            "payload": {
                "direction_target_probs": list(evidence.direction_target_probs),
                "requested_risk_target": evidence.requested_risk_target,
                "direction_weight": evidence.direction_weight,
                "sizing_weight": evidence.sizing_weight,
                "student_context_object_id": evidence.student_context_object_id,
                "teacher_version": evidence.teacher_version,
                "train_dependence_group_hash": evidence.train_dependence_group_hash,
                "admission_receipt_hash": evidence.admission.content_hash,
            },
            "action_laws": [asdict(x) for x in evidence.action_laws],
            "admission": asdict(evidence.admission),
        }
        obj = ExperienceObject(
            object_id=evidence.evidence_id,
            object_type="EVIDENCE_PACKAGE",
            generation=int(policy_generation),
            policy_weight_hash=policy_weight_hash,
            snapshot_hash=parent_snapshot_hash,
            lineage_hash=evidence.content_hash,
            payload=payload,
        )
        ref, _ = lake.put(obj)
        return ref

    def compile_many(
        self,
        samples: Sequence[CounterfactualBranchSampleR5],
        *,
        target_parent_ids: Sequence[str] | None = None,
        eligible_train_dependence_groups: set[str] | None = None,
    ) -> list[DependenceAwareTeacherEvidenceR6]:
        idx = self.index(samples)
        targets = (
            idx.parents
            if target_parent_ids is None
            else [p for p in idx.parents if p in set(target_parent_ids)]
        )
        return [
            self.compile_one(
                target_parent=p,
                index=idx,
                eligible_train_dependence_groups=eligible_train_dependence_groups,
            )
            for p in targets
        ]
