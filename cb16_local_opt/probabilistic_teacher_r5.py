from __future__ import annotations

"""
Chronological cross-fit probabilistic Teacher / Examiner for historical R&D.

This is not a hindsight BEST_ACTION compiler.

Input:
- pre-outcome Student-visible context features;
- same-future counterfactual economic utilities for a fixed action/risk grid;
- parent chronology/dependence-group identity.

Output:
- an empirical predictive utility law per candidate action;
- probabilistic direction evidence;
- continuous requested-risk target;
- support/admission receipt;
- optional calibration receipts.

Important semantics:
- counterfactual branches from one parent context remain ONE independent support group;
- target parent groups are excluded from their Teacher training support;
- PREQUENTIAL mode uses only earlier groups;
- BLOCKED_CROSSFIT mode excludes the target fold and optional embargo neighbors;
- realized utility samples remain stochastic evidence, not deterministic correct-action labels.
"""

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np

from .sharded_experience_lake import ExperienceObject, ExperienceRef, ShardedExperienceLake


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CounterfactualBranchSampleR5:
    parent_id: str
    student_context_object_id: str
    timestamp: int
    context_features: tuple[float, ...]
    direction: int
    requested_risk: float
    realized_utility: float
    dependence_group_id: str
    market_lineage_hash: str

    def validate(self):
        if self.direction not in {-1, 0, 1}:
            raise ValueError("bad direction")
        if not 0 <= self.requested_risk <= 1:
            raise ValueError("bad risk")
        if self.direction == 0 and self.requested_risk != 0:
            raise ValueError("FLAT_REQUIRES_ZERO_RISK")
        x = np.asarray(self.context_features, dtype=np.float64)
        if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
            raise ValueError("bad context features")
        if not np.isfinite(self.realized_utility):
            raise ValueError("bad utility")

    @property
    def content_hash(self) -> str:
        self.validate()
        return canonical_hash(self)


@dataclass(frozen=True)
class PredictiveUtilityLawR5:
    direction: int
    requested_risk: float
    mean_utility: float
    std_utility: float
    quantile_levels: tuple[float, ...]
    quantiles: tuple[float, ...]
    effective_n: float
    unique_parent_groups: int
    nearest_distance: float
    max_distance_used: float
    support_group_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class EvidenceAdmissionReceiptR5:
    status: str
    lane: str
    unique_train_groups: int
    minimum_action_effective_n: float
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
class HistoricalTeacherEvidenceR5:
    evidence_id: str
    parent_id: str
    student_context_object_id: str
    timestamp: int
    target_group_id: str
    teacher_version: str
    teacher_protocol_hash: str
    train_group_hash: str
    action_laws: tuple[PredictiveUtilityLawR5, ...]
    direction_target_probs: tuple[float, float, float]  # SHORT, FLAT, LONG
    requested_risk_target: float
    direction_weight: float
    sizing_weight: float
    admission: EvidenceAdmissionReceiptR5

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class CrossFitTeacherConfigR5:
    teacher_version: str = "CB16_CROSSFIT_KNN_PROBABILISTIC_TEACHER_R5"
    mode: Literal["PREQUENTIAL", "BLOCKED_CROSSFIT"] = "PREQUENTIAL"
    n_folds: int = 5
    embargo_groups: int = 1
    k_neighbors: int = 64
    min_train_groups: int = 32
    min_effective_n: float = 12.0
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
        if self.k_neighbors <= 0 or self.min_train_groups <= 0:
            raise ValueError("support config")
        if self.min_effective_n <= 0 or self.max_nearest_distance <= 0:
            raise ValueError("admission config")
        if self.distance_temperature <= 0 or self.direction_softmax_temperature <= 0:
            raise ValueError("temperature")
        qs = np.asarray(self.quantile_levels)
        if np.any((qs <= 0) | (qs >= 1)) or np.any(np.diff(qs) <= 0):
            raise ValueError("bad quantiles")

    @property
    def content_hash(self) -> str:
        self.validate()
        return canonical_hash(self)


@dataclass(frozen=True)
class CalibrationReceiptR5:
    samples: int
    qcrps: float
    mean_error: float
    median_abs_error: float
    interval_50_coverage: float
    interval_80_coverage: float
    interval_90_coverage: float
    protocol_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    qs = np.asarray(qs, dtype=np.float64)
    if len(values) == 0 or len(values) != len(weights):
        raise ValueError("weighted_quantile input")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("bad weights")
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    w = weights[order]
    c = np.cumsum(w) - 0.5 * w
    c /= w.sum()
    return np.interp(qs, c, v, left=v[0], right=v[-1])


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / temperature
    z -= np.max(z)
    e = np.exp(np.clip(z, -60, 60))
    return e / e.sum()


class CrossFitProbabilisticTeacherR5:
    def __init__(self, config: CrossFitTeacherConfigR5 | None = None):
        self.config = config or CrossFitTeacherConfigR5()
        self.config.validate()

    @staticmethod
    def _group_samples(
        samples: Sequence[CounterfactualBranchSampleR5],
    ) -> tuple[list[str], dict[str, list[CounterfactualBranchSampleR5]], dict[str, int]]:
        groups: dict[str, list[CounterfactualBranchSampleR5]] = {}
        timestamps: dict[str, int] = {}
        parent_to_dep: dict[str, str] = {}
        for s in samples:
            s.validate()
            groups.setdefault(s.parent_id, []).append(s)
            timestamps.setdefault(s.parent_id, s.timestamp)
            if timestamps[s.parent_id] != s.timestamp:
                raise RuntimeError("PARENT_TIMESTAMP_INCONSISTENT")
            old_dep = parent_to_dep.setdefault(s.parent_id, s.dependence_group_id)
            if old_dep != s.dependence_group_id:
                raise RuntimeError("PARENT_DEPENDENCE_GROUP_INCONSISTENT")

        parents = sorted(groups, key=lambda p: (timestamps[p], p))
        # Require one pre-outcome feature vector per parent, repeated identically across branches.
        for p, rows in groups.items():
            f0 = rows[0].context_features
            c0 = rows[0].student_context_object_id
            for r in rows[1:]:
                if r.context_features != f0:
                    raise RuntimeError("PARENT_CONTEXT_FEATURES_INCONSISTENT")
                if r.student_context_object_id != c0:
                    raise RuntimeError("PARENT_STUDENT_CONTEXT_INCONSISTENT")
        return parents, groups, timestamps

    def _blocked_folds(self, parents: list[str]) -> dict[str, int]:
        n = len(parents)
        # contiguous chronological folds
        fold = {}
        for i, p in enumerate(parents):
            f = min(self.config.n_folds - 1, (i * self.config.n_folds) // max(n, 1))
            fold[p] = int(f)
        return fold

    def train_parents_for_target(
        self,
        target_parent: str,
        parents: list[str],
        timestamps: Mapping[str, int],
    ) -> list[str]:
        idx = parents.index(target_parent)
        if self.config.mode == "PREQUENTIAL":
            # strictly earlier parent contexts only
            candidates = parents[:idx]
            return candidates

        folds = self._blocked_folds(parents)
        tfold = folds[target_parent]
        excluded = {p for p in parents if folds[p] == tfold}
        if self.config.embargo_groups:
            fold_indices = [i for i, p in enumerate(parents) if folds[p] == tfold]
            lo, hi = min(fold_indices), max(fold_indices)
            elo = max(0, lo - self.config.embargo_groups)
            ehi = min(len(parents) - 1, hi + self.config.embargo_groups)
            excluded.update(parents[elo:ehi+1])
        return [p for p in parents if p not in excluded]

    def _normalization(
        self,
        train_parents: Sequence[str],
        groups: Mapping[str, Sequence[CounterfactualBranchSampleR5]],
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray([groups[p][0].context_features for p in train_parents], dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=0)
        std = np.where(std < 1e-8, 1.0, std)
        return mean, std

    def _law(
        self,
        *,
        target_features: np.ndarray,
        train_parents: Sequence[str],
        groups: Mapping[str, Sequence[CounterfactualBranchSampleR5]],
        direction: int,
        risk: float,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> PredictiveUtilityLawR5 | None:
        xs = []
        ys = []
        parent_ids = []
        for p in train_parents:
            # fixed action/risk grid: one branch per parent/candidate
            match = [
                s for s in groups[p]
                if s.direction == direction and abs(s.requested_risk - risk) <= 1e-12
            ]
            if not match:
                continue
            if len(match) != 1:
                raise RuntimeError("DUPLICATE_ACTION_BRANCH_WITHIN_PARENT")
            xs.append(match[0].context_features)
            ys.append(match[0].realized_utility)
            parent_ids.append(p)
        if not xs:
            return None

        x = (np.asarray(xs, dtype=np.float64) - mean) / std
        t = (target_features - mean) / std
        dist = np.sqrt(np.mean((x - t[None, :]) ** 2, axis=1))
        order = np.argsort(dist, kind="mergesort")[: min(self.config.k_neighbors, len(dist))]
        d = dist[order]
        y = np.asarray(ys, dtype=np.float64)[order]
        selected_parents = [parent_ids[i] for i in order]
        # Smooth positive weights. The additive epsilon ensures exact-neighbor stability.
        weights = np.exp(-0.5 * (d / self.config.distance_temperature) ** 2) + 1e-12
        weights /= weights.sum()

        q = weighted_quantile(y, weights, np.asarray(self.config.quantile_levels))
        mu = float(np.sum(weights * y))
        var = float(np.sum(weights * (y - mu) ** 2))
        eff = float(1.0 / np.sum(weights ** 2))
        return PredictiveUtilityLawR5(
            direction=int(direction),
            requested_risk=float(risk),
            mean_utility=mu,
            std_utility=math.sqrt(max(0.0, var)),
            quantile_levels=self.config.quantile_levels,
            quantiles=tuple(float(v) for v in q),
            effective_n=eff,
            unique_parent_groups=len(selected_parents),
            nearest_distance=float(d[0]),
            max_distance_used=float(d[-1]),
            support_group_hash=canonical_hash(selected_parents),
        )

    def compile_one(
        self,
        *,
        target_parent: str,
        parents: list[str],
        groups: Mapping[str, Sequence[CounterfactualBranchSampleR5]],
        timestamps: Mapping[str, int],
    ) -> HistoricalTeacherEvidenceR5:
        target_rows = groups[target_parent]
        target_features = np.asarray(target_rows[0].context_features, dtype=np.float64)
        train_parents = self.train_parents_for_target(target_parent, parents, timestamps)
        train_group_hash = canonical_hash(train_parents)

        # Candidate grid is inherited from the target parent's deterministic same-future
        # counterfactual branch set.
        candidates = sorted(
            {(s.direction, float(s.requested_risk)) for s in target_rows},
            key=lambda x: (x[0], x[1]),
        )

        laws: list[PredictiveUtilityLawR5] = []
        if train_parents:
            mean, std = self._normalization(train_parents, groups)
            for d, r in candidates:
                law = self._law(
                    target_features=target_features,
                    train_parents=train_parents,
                    groups=groups,
                    direction=d,
                    risk=r,
                    mean=mean,
                    std=std,
                )
                if law is not None:
                    laws.append(law)

        # Pick the best expected risk within each direction; this is an estimated expected
        # utility preference, not a realized hindsight winner.
        best_by_direction = {}
        for d in (-1, 0, 1):
            dlaws = [l for l in laws if l.direction == d]
            if dlaws:
                best_by_direction[d] = max(
                    dlaws,
                    key=lambda l: (l.mean_utility, -l.requested_risk),
                )

        reasons = []
        if len(train_parents) < self.config.min_train_groups:
            reasons.append("INSUFFICIENT_TRAIN_GROUPS")
        if set(best_by_direction) != {-1, 0, 1}:
            reasons.append("ACTION_GRID_SUPPORT_INCOMPLETE")
        min_eff = min((l.effective_n for l in best_by_direction.values()), default=0.0)
        max_nearest = max((l.nearest_distance for l in best_by_direction.values()), default=float("inf"))
        if min_eff < self.config.min_effective_n:
            reasons.append("INSUFFICIENT_EFFECTIVE_SUPPORT")
        if max_nearest > self.config.max_nearest_distance:
            reasons.append("TARGET_OUTSIDE_SUPPORTED_CONTEXT")

        if set(best_by_direction) == {-1, 0, 1}:
            means = np.asarray([
                best_by_direction[-1].mean_utility,
                best_by_direction[0].mean_utility,
                best_by_direction[1].mean_utility,
            ])
            probs = _softmax(means, self.config.direction_softmax_temperature)
            best_risks = np.asarray([
                best_by_direction[-1].requested_risk,
                0.0,
                best_by_direction[1].requested_risk,
            ])
            risk_target = float(np.sum(probs * best_risks))
        else:
            probs = np.asarray([1/3, 1/3, 1/3], dtype=np.float64)
            risk_target = 0.0

        admission = EvidenceAdmissionReceiptR5(
            status="ADMITTED" if not reasons else "EVIDENCE_NOT_READY",
            lane=self.config.lane,
            unique_train_groups=len(train_parents),
            minimum_action_effective_n=float(min_eff),
            maximum_action_nearest_distance=float(max_nearest),
            reasons=tuple(reasons),
            protocol_hash=self.config.content_hash,
        )
        return HistoricalTeacherEvidenceR5(
            evidence_id=f"R5E:{target_parent}:{self.config.content_hash[:12]}",
            parent_id=target_parent,
            student_context_object_id=target_rows[0].student_context_object_id,
            timestamp=int(target_rows[0].timestamp),
            target_group_id=target_rows[0].dependence_group_id,
            teacher_version=self.config.teacher_version,
            teacher_protocol_hash=self.config.content_hash,
            train_group_hash=train_group_hash,
            action_laws=tuple(laws),
            direction_target_probs=tuple(float(x) for x in probs),
            requested_risk_target=risk_target,
            direction_weight=self.config.direction_weight if admission.admitted else 0.0,
            sizing_weight=self.config.sizing_weight if admission.admitted else 0.0,
            admission=admission,
        )

    def compile_all(
        self,
        samples: Sequence[CounterfactualBranchSampleR5],
    ) -> list[HistoricalTeacherEvidenceR5]:
        parents, groups, timestamps = self._group_samples(samples)
        return [
            self.compile_one(
                target_parent=p,
                parents=parents,
                groups=groups,
                timestamps=timestamps,
            )
            for p in parents
        ]

    def persist_evidence(
        self,
        *,
        lake: ShardedExperienceLake,
        evidence: HistoricalTeacherEvidenceR5,
        policy_generation: int,
        policy_weight_hash: str,
        parent_snapshot_hash: str,
    ) -> ExperienceRef:
        # The outer Experience object is admitted iff the Evidence Admission Receipt is.
        payload = {
            "evidence_id": evidence.evidence_id,
            "parent_id": evidence.parent_id,
            "student_context_object_id": evidence.student_context_object_id,
            "timestamp": evidence.timestamp,
            "target_group_id": evidence.target_group_id,
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
                "train_group_hash": evidence.train_group_hash,
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


def pinball_loss(y: float, q: float, pred: float) -> float:
    e = y - pred
    return max(q * e, (q - 1.0) * e)


def calibration_receipt(
    observations_and_laws: Sequence[tuple[float, PredictiveUtilityLawR5]],
    *,
    protocol_hash: str,
) -> CalibrationReceiptR5:
    if not observations_and_laws:
        raise ValueError("no calibration samples")
    errors = []
    qloss = []
    cov50 = cov80 = cov90 = 0

    def quant(law, q):
        qs = np.asarray(law.quantile_levels, dtype=np.float64)
        vals = np.asarray(law.quantiles, dtype=np.float64)
        return float(np.interp(q, qs, vals))

    for y, law in observations_and_laws:
        errors.append(float(y - law.mean_utility))
        for q, pred in zip(law.quantile_levels, law.quantiles):
            qloss.append(pinball_loss(float(y), float(q), float(pred)))
        cov50 += quant(law, 0.25) <= y <= quant(law, 0.75)
        cov80 += quant(law, 0.10) <= y <= quant(law, 0.90)
        cov90 += quant(law, 0.05) <= y <= quant(law, 0.95)

    n = len(observations_and_laws)
    e = np.asarray(errors)
    return CalibrationReceiptR5(
        samples=n,
        qcrps=float(2.0 * np.mean(qloss)),
        mean_error=float(np.mean(e)),
        median_abs_error=float(np.median(np.abs(e))),
        interval_50_coverage=float(cov50 / n),
        interval_80_coverage=float(cov80 / n),
        interval_90_coverage=float(cov90 / n),
        protocol_hash=protocol_hash,
    )
