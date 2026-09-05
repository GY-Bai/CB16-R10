from __future__ import annotations

"""
Historical probabilistic-control suite.

Formulations:

F0_CLIMATOLOGY
    No context conditioning. Equal-weight predictive law over eligible chronological
    training parent groups for each action/risk candidate.

F1_ACCOUNT_ONLY
    kNN predictive law using only AccountState6.

F2_TRUE_MARKET_ACCOUNT
    kNN predictive law using the full Market64 + Account6 context.

F3_SHUFFLED_MARKET
    Negative control. AccountState6 stays with its parent. Market64 is deterministically
    permuted *within the eligible Teacher training parent set for each target*.  The target
    receives a market donor from the same eligible historical training set. Therefore the
    shuffle destroys market/account correspondence without importing target/future market
    bytes.

All support is counted in parent chronology/dependence groups, never counterfactual branches.
Lower qCRPS is better.
"""

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r5 import (
    CounterfactualBranchSampleR5,
    CrossFitProbabilisticTeacherR5,
    CrossFitTeacherConfigR5,
    PredictiveUtilityLawR5,
    weighted_quantile,
)


FORMULATIONS = (
    "F0_CLIMATOLOGY",
    "F1_ACCOUNT_ONLY",
    "F2_TRUE_MARKET_ACCOUNT",
    "F3_SHUFFLED_MARKET",
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def pinball_loss(y: float, q: float, pred: float) -> float:
    e = y - pred
    return max(q * e, (q - 1.0) * e)


@dataclass(frozen=True)
class ControlSuiteConfigR6:
    teacher: CrossFitTeacherConfigR5
    market_dim: int = 64
    account_dim: int = 6
    shuffle_seed: int = 20260904
    bootstrap_reps: int = 2000
    bootstrap_alpha: float = 0.05
    minimum_scored_targets: int = 8
    control_version: str = "CB16_HISTORICAL_PROBABILISTIC_CONTROLS_R6"

    def validate(self):
        self.teacher.validate()
        if self.market_dim <= 0 or self.account_dim <= 0:
            raise ValueError("feature dimensions")
        if self.bootstrap_reps <= 0:
            raise ValueError("bootstrap_reps")
        if not 0 < self.bootstrap_alpha < 1:
            raise ValueError("bootstrap_alpha")
        if self.minimum_scored_targets <= 0:
            raise ValueError("minimum_scored_targets")

    @property
    def content_hash(self) -> str:
        self.validate()
        return canonical_hash({
            "teacher_hash": self.teacher.content_hash,
            "market_dim": self.market_dim,
            "account_dim": self.account_dim,
            "shuffle_seed": self.shuffle_seed,
            "bootstrap_reps": self.bootstrap_reps,
            "bootstrap_alpha": self.bootstrap_alpha,
            "minimum_scored_targets": self.minimum_scored_targets,
            "control_version": self.control_version,
        })


@dataclass(frozen=True)
class FormulationScoreR6:
    formulation: str
    scored_targets: int
    scored_branches: int
    qcrps: float
    mean_log_score_proxy: float
    mean_utility_error: float
    median_abs_utility_error: float
    parent_score_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class PairedDeltaReceiptR6:
    left: str
    right: str
    # delta = qCRPS(left) - qCRPS(right); negative means left is better.
    mean_delta: float
    ci_low: float
    ci_high: float
    paired_targets: int
    bootstrap_reps: int

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class HistoricalControlSuiteReceiptR6:
    protocol_hash: str
    formulations: tuple[FormulationScoreR6, ...]
    f2_minus_f0: PairedDeltaReceiptR6
    f3_minus_f2: PairedDeltaReceiptR6
    f1_minus_f0: PairedDeltaReceiptR6
    target_parent_hash: str
    status: str
    reasons: tuple[str, ...]

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


class HistoricalControlSuiteR6:
    def __init__(self, config: ControlSuiteConfigR6):
        config.validate()
        self.config = config
        self.teacher = CrossFitProbabilisticTeacherR5(config.teacher)

    def _candidate_grid(self, rows: Sequence[CounterfactualBranchSampleR5]):
        return sorted(
            {(s.direction, float(s.requested_risk)) for s in rows},
            key=lambda x: (x[0], x[1]),
        )

    def _equal_weight_law(
        self,
        *,
        train_parents: Sequence[str],
        groups: Mapping[str, Sequence[CounterfactualBranchSampleR5]],
        direction: int,
        risk: float,
    ) -> PredictiveUtilityLawR5 | None:
        ys = []
        used = []
        for p in train_parents:
            match = [
                s for s in groups[p]
                if s.direction == direction and abs(s.requested_risk - risk) <= 1e-12
            ]
            if not match:
                continue
            if len(match) != 1:
                raise RuntimeError("DUPLICATE_ACTION_BRANCH_WITHIN_PARENT")
            ys.append(float(match[0].realized_utility))
            used.append(p)
        if not ys:
            return None
        y = np.asarray(ys, dtype=np.float64)
        weights = np.full(len(y), 1.0 / len(y), dtype=np.float64)
        q = weighted_quantile(
            y,
            weights,
            np.asarray(self.config.teacher.quantile_levels, dtype=np.float64),
        )
        mu = float(np.mean(y))
        return PredictiveUtilityLawR5(
            direction=direction,
            requested_risk=float(risk),
            mean_utility=mu,
            std_utility=float(np.std(y, ddof=0)),
            quantile_levels=self.config.teacher.quantile_levels,
            quantiles=tuple(float(x) for x in q),
            effective_n=float(len(y)),
            unique_parent_groups=len(y),
            nearest_distance=0.0,
            max_distance_used=0.0,
            support_group_hash=canonical_hash(used),
        )

    def _features_for(
        self,
        sample: CounterfactualBranchSampleR5,
        formulation: str,
    ) -> tuple[float, ...]:
        x = tuple(sample.context_features)
        expected = self.config.market_dim + self.config.account_dim
        if len(x) != expected:
            raise RuntimeError(
                f"CONTROL_FEATURE_DIM_MISMATCH expected={expected} actual={len(x)}"
            )
        if formulation == "F1_ACCOUNT_ONLY":
            return tuple(x[self.config.market_dim:])
        if formulation == "F2_TRUE_MARKET_ACCOUNT":
            return x
        raise ValueError("features_for formulation")

    def _conditional_laws(
        self,
        *,
        target_parent: str,
        target_features: tuple[float, ...],
        train_parents: Sequence[str],
        groups: Mapping[str, Sequence[CounterfactualBranchSampleR5]],
        formulation: str,
        shuffled_features: Mapping[str, tuple[float, ...]] | None = None,
    ) -> dict[tuple[int, float], PredictiveUtilityLawR5]:
        # Build a temporary groups view with the requested context feature representation.
        temp = {}
        for p in train_parents:
            rows = []
            for s in groups[p]:
                if formulation == "F3_SHUFFLED_MARKET":
                    feat = shuffled_features[p]
                else:
                    feat = self._features_for(s, formulation)
                rows.append(dataclasses.replace(s, context_features=feat))
            temp[p] = rows

        train_x = np.asarray(
            [temp[p][0].context_features for p in train_parents],
            dtype=np.float64,
        )
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0, ddof=0)
        std = np.where(std < 1e-8, 1.0, std)

        laws = {}
        candidates = self._candidate_grid(groups[target_parent])
        target = np.asarray(target_features, dtype=np.float64)
        for d, r in candidates:
            law = self.teacher._law(
                target_features=target,
                train_parents=train_parents,
                groups=temp,
                direction=d,
                risk=r,
                mean=mean,
                std=std,
            )
            if law is not None:
                laws[(d, r)] = law
        return laws

    def _shuffle_features(
        self,
        *,
        target_parent: str,
        train_parents: Sequence[str],
        groups: Mapping[str, Sequence[CounterfactualBranchSampleR5]],
    ) -> tuple[dict[str, tuple[float, ...]], tuple[float, ...]]:
        if not train_parents:
            raise RuntimeError("NO_TRAIN_PARENTS_FOR_SHUFFLE")
        seed_material = (
            f"{self.config.shuffle_seed}|{target_parent}|"
            f"{canonical_hash(list(train_parents))}"
        )
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train_parents))
        market = {}
        accounts = {}
        for p in train_parents:
            x = tuple(groups[p][0].context_features)
            if len(x) != self.config.market_dim + self.config.account_dim:
                raise RuntimeError("SHUFFLE_FEATURE_DIM_MISMATCH")
            market[p] = tuple(x[: self.config.market_dim])
            accounts[p] = tuple(x[self.config.market_dim :])

        shuffled = {}
        for i, p in enumerate(train_parents):
            donor = train_parents[int(perm[i])]
            shuffled[p] = market[donor] + accounts[p]

        # Target market donor is also from eligible train support; target AccountState stays target.
        target_x = tuple(groups[target_parent][0].context_features)
        donor = train_parents[int(rng.integers(0, len(train_parents)))]
        target_features = market[donor] + tuple(target_x[self.config.market_dim :])
        return shuffled, target_features

    def _laws_for_target(
        self,
        *,
        formulation: str,
        target_parent: str,
        parents: list[str],
        groups: Mapping[str, Sequence[CounterfactualBranchSampleR5]],
        timestamps: Mapping[str, int],
        eligible_train_parent_ids: set[str] | None = None,
    ) -> dict[tuple[int, float], PredictiveUtilityLawR5]:
        train = self.teacher.train_parents_for_target(
            target_parent,
            parents,
            timestamps,
        )
        if eligible_train_parent_ids is not None:
            train = [p for p in train if p in eligible_train_parent_ids]
        if len(train) < self.config.teacher.min_train_groups:
            return {}

        candidates = self._candidate_grid(groups[target_parent])
        if formulation == "F0_CLIMATOLOGY":
            return {
                (d, r): law
                for d, r in candidates
                if (law := self._equal_weight_law(
                    train_parents=train,
                    groups=groups,
                    direction=d,
                    risk=r,
                )) is not None
            }

        if formulation == "F1_ACCOUNT_ONLY":
            target_features = self._features_for(
                groups[target_parent][0],
                formulation,
            )
            return self._conditional_laws(
                target_parent=target_parent,
                target_features=target_features,
                train_parents=train,
                groups=groups,
                formulation=formulation,
            )

        if formulation == "F2_TRUE_MARKET_ACCOUNT":
            target_features = self._features_for(
                groups[target_parent][0],
                formulation,
            )
            return self._conditional_laws(
                target_parent=target_parent,
                target_features=target_features,
                train_parents=train,
                groups=groups,
                formulation=formulation,
            )

        if formulation == "F3_SHUFFLED_MARKET":
            shuffled, target_features = self._shuffle_features(
                target_parent=target_parent,
                train_parents=train,
                groups=groups,
            )
            return self._conditional_laws(
                target_parent=target_parent,
                target_features=target_features,
                train_parents=train,
                groups=groups,
                formulation=formulation,
                shuffled_features=shuffled,
            )

        raise ValueError(formulation)

    def _score_target(
        self,
        *,
        target_parent: str,
        laws: Mapping[tuple[int, float], PredictiveUtilityLawR5],
        groups: Mapping[str, Sequence[CounterfactualBranchSampleR5]],
    ) -> tuple[float, int, float, float] | None:
        rows = groups[target_parent]
        losses = []
        errors = []
        log_proxy = []
        for s in rows:
            law = laws.get((s.direction, float(s.requested_risk)))
            if law is None:
                return None
            if law.effective_n < self.config.teacher.min_effective_n:
                return None
            if law.nearest_distance > self.config.teacher.max_nearest_distance:
                return None
            for q, pred in zip(law.quantile_levels, law.quantiles):
                losses.append(
                    pinball_loss(
                        float(s.realized_utility),
                        float(q),
                        float(pred),
                    )
                )
            err = float(s.realized_utility - law.mean_utility)
            errors.append(err)
            # Gaussian-density proxy only as a diagnostic. qCRPS remains status-driving here.
            sigma = max(law.std_utility, 1e-8)
            log_proxy.append(
                0.5 * math.log(2 * math.pi * sigma * sigma)
                + 0.5 * (err / sigma) ** 2
            )
        return (
            float(2.0 * np.mean(losses)),
            len(rows),
            float(np.mean(log_proxy)),
            float(np.mean(errors)),
        )

    def _paired_bootstrap(
        self,
        *,
        left: str,
        right: str,
        parent_scores: Mapping[str, Mapping[str, float]],
    ) -> PairedDeltaReceiptR6:
        shared = sorted(
            set(parent_scores[left]).intersection(parent_scores[right])
        )
        if not shared:
            raise RuntimeError(f"NO_PAIRED_TARGETS:{left}:{right}")
        delta = np.asarray(
            [parent_scores[left][p] - parent_scores[right][p] for p in shared],
            dtype=np.float64,
        )
        seed_material = (
            f"{self.config.content_hash}|{left}|{right}|{canonical_hash(shared)}"
        )
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        n = len(delta)
        boot = np.empty(self.config.bootstrap_reps, dtype=np.float64)
        for i in range(self.config.bootstrap_reps):
            idx = rng.integers(0, n, size=n)
            boot[i] = float(np.mean(delta[idx]))
        a = self.config.bootstrap_alpha
        lo, hi = np.quantile(boot, [a / 2, 1 - a / 2])
        return PairedDeltaReceiptR6(
            left=left,
            right=right,
            mean_delta=float(np.mean(delta)),
            ci_low=float(lo),
            ci_high=float(hi),
            paired_targets=n,
            bootstrap_reps=self.config.bootstrap_reps,
        )

    def evaluate(
        self,
        samples: Sequence[CounterfactualBranchSampleR5],
        *,
        target_parent_ids: Sequence[str] | None = None,
        eligible_train_parent_ids: Sequence[str] | None = None,
    ) -> HistoricalControlSuiteReceiptR6:
        parents, groups, timestamps = self.teacher._group_samples(samples)
        targets = (
            parents
            if target_parent_ids is None
            else [p for p in parents if p in set(target_parent_ids)]
        )
        eligible_train_set = (
            None
            if eligible_train_parent_ids is None
            else set(eligible_train_parent_ids)
        )
        parent_scores: dict[str, dict[str, float]] = {
            f: {} for f in FORMULATIONS
        }
        branch_counts = {f: 0 for f in FORMULATIONS}
        log_proxy = {f: [] for f in FORMULATIONS}
        errors = {f: [] for f in FORMULATIONS}

        for p in targets:
            for f in FORMULATIONS:
                laws = self._laws_for_target(
                    formulation=f,
                    target_parent=p,
                    parents=parents,
                    groups=groups,
                    timestamps=timestamps,
                    eligible_train_parent_ids=eligible_train_set,
                )
                if not laws:
                    continue
                scored = self._score_target(
                    target_parent=p,
                    laws=laws,
                    groups=groups,
                )
                if scored is None:
                    continue
                score, branches, lp, err = scored
                parent_scores[f][p] = score
                branch_counts[f] += branches
                log_proxy[f].append(lp)
                errors[f].append(err)

        formulation_receipts = []
        reasons = []
        for f in FORMULATIONS:
            vals = parent_scores[f]
            if len(vals) < self.config.minimum_scored_targets:
                reasons.append(
                    f"{f}:INSUFFICIENT_SCORED_TARGETS:{len(vals)}"
                )
            err_arr = np.asarray(errors[f], dtype=np.float64)
            formulation_receipts.append(FormulationScoreR6(
                formulation=f,
                scored_targets=len(vals),
                scored_branches=branch_counts[f],
                qcrps=(
                    float(np.mean(list(vals.values())))
                    if vals
                    else float("inf")
                ),
                mean_log_score_proxy=(
                    float(np.mean(log_proxy[f]))
                    if log_proxy[f]
                    else float("inf")
                ),
                mean_utility_error=(
                    float(np.mean(err_arr))
                    if len(err_arr)
                    else float("nan")
                ),
                median_abs_utility_error=(
                    float(np.median(np.abs(err_arr)))
                    if len(err_arr)
                    else float("nan")
                ),
                parent_score_hash=canonical_hash(
                    [(p, vals[p]) for p in sorted(vals)]
                ),
            ))

        # Paired deltas use only targets scored by both formulations.
        f2_f0 = self._paired_bootstrap(
            left="F2_TRUE_MARKET_ACCOUNT",
            right="F0_CLIMATOLOGY",
            parent_scores=parent_scores,
        )
        f3_f2 = self._paired_bootstrap(
            left="F3_SHUFFLED_MARKET",
            right="F2_TRUE_MARKET_ACCOUNT",
            parent_scores=parent_scores,
        )
        f1_f0 = self._paired_bootstrap(
            left="F1_ACCOUNT_ONLY",
            right="F0_CLIMATOLOGY",
            parent_scores=parent_scores,
        )

        return HistoricalControlSuiteReceiptR6(
            protocol_hash=self.config.content_hash,
            formulations=tuple(formulation_receipts),
            f2_minus_f0=f2_f0,
            f3_minus_f2=f3_f2,
            f1_minus_f0=f1_f0,
            target_parent_hash=canonical_hash(targets),
            status="PASS" if not reasons else "INSUFFICIENT_SUPPORT",
            reasons=tuple(reasons),
        )


# ---------------------------------------------------------------------------
# R6 dependence-group-aware control suite (preferred for multi-account campaigns)
# ---------------------------------------------------------------------------

from .probabilistic_teacher_r6 import (
    DependenceAwareProbabilisticTeacherR6,
    DependenceAwareTeacherConfigR6,
    DependenceAwarePredictiveLawR6,
)


@dataclass(frozen=True)
class DependenceAwareControlSuiteConfigR6:
    teacher: DependenceAwareTeacherConfigR6
    market_dim: int = 64
    account_dim: int = 6
    shuffle_seed: int = 20260904
    bootstrap_reps: int = 2000
    bootstrap_alpha: float = 0.05
    minimum_scored_dependence_groups: int = 8
    control_version: str = "CB16_DEPENDENCE_AWARE_CONTROLS_R6"

    def validate(self):
        self.teacher.validate()
        if self.market_dim <= 0 or self.account_dim <= 0:
            raise ValueError("feature dimensions")
        if self.bootstrap_reps <= 0:
            raise ValueError("bootstrap reps")
        if not 0 < self.bootstrap_alpha < 1:
            raise ValueError("bootstrap alpha")
        if self.minimum_scored_dependence_groups <= 0:
            raise ValueError("minimum scored dependence groups")

    @property
    def content_hash(self) -> str:
        self.validate()
        return canonical_hash({
            "teacher_hash": self.teacher.content_hash,
            "market_dim": self.market_dim,
            "account_dim": self.account_dim,
            "shuffle_seed": self.shuffle_seed,
            "bootstrap_reps": self.bootstrap_reps,
            "bootstrap_alpha": self.bootstrap_alpha,
            "minimum_scored_dependence_groups": self.minimum_scored_dependence_groups,
            "control_version": self.control_version,
        })


@dataclass(frozen=True)
class DependenceAwareFormulationScoreR6:
    formulation: str
    scored_parents: int
    scored_dependence_groups: int
    scored_branches: int
    qcrps: float
    dependence_group_score_hash: str


@dataclass(frozen=True)
class DependenceAwareControlSuiteReceiptR6:
    protocol_hash: str
    formulations: tuple[DependenceAwareFormulationScoreR6, ...]
    f2_minus_f0: PairedDeltaReceiptR6
    f3_minus_f2: PairedDeltaReceiptR6
    f1_minus_f0: PairedDeltaReceiptR6
    evaluation_dependence_group_hash: str
    status: str
    reasons: tuple[str, ...]

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


class DependenceAwareHistoricalControlSuiteR6:
    def __init__(self, config: DependenceAwareControlSuiteConfigR6):
        config.validate()
        self.config = config
        self.teacher = DependenceAwareProbabilisticTeacherR6(config.teacher)

    def _feature_override_account_only(self, idx):
        out = {}
        for p in idx.parents:
            x = tuple(idx.rows_by_parent[p][0].context_features)
            if len(x) != self.config.market_dim + self.config.account_dim:
                raise RuntimeError("CONTROL_FEATURE_DIM_MISMATCH")
            out[p] = tuple(x[self.config.market_dim:])
        return out

    def _shuffled_feature_override(
        self,
        *,
        target_parent: str,
        idx,
        train_deps: Sequence[str],
    ):
        if not train_deps:
            raise RuntimeError("NO_TRAIN_DEPENDENCE_GROUPS_FOR_SHUFFLE")
        seed_material = (
            f"{self.config.shuffle_seed}|{target_parent}|"
            f"{canonical_hash(list(train_deps))}"
        )
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train_deps))

        # Market context is common to a dependence group in the intended market-shared
        # multi-account runtime. Validate this before shuffling.
        group_market = {}
        for dep in train_deps:
            markets = set()
            for p in idx.parents_by_dependence_group[dep]:
                x = tuple(idx.rows_by_parent[p][0].context_features)
                if len(x) != self.config.market_dim + self.config.account_dim:
                    raise RuntimeError("CONTROL_FEATURE_DIM_MISMATCH")
                markets.add(tuple(x[:self.config.market_dim]))
            if len(markets) != 1:
                raise RuntimeError(
                    "DEPENDENCE_GROUP_MARKET_FEATURE_NOT_SHARED_ACROSS_ACCOUNTS"
                )
            group_market[dep] = next(iter(markets))

        override = {}
        for i, dep in enumerate(train_deps):
            donor = train_deps[int(perm[i])]
            donor_market = group_market[donor]
            for p in idx.parents_by_dependence_group[dep]:
                x = tuple(idx.rows_by_parent[p][0].context_features)
                account = tuple(x[self.config.market_dim:])
                override[p] = donor_market + account

        target_x = tuple(idx.rows_by_parent[target_parent][0].context_features)
        donor = train_deps[int(rng.integers(0, len(train_deps)))]
        target_feature = group_market[donor] + tuple(
            target_x[self.config.market_dim:]
        )
        return override, target_feature

    def _laws(
        self,
        *,
        formulation: str,
        target_parent: str,
        idx,
        eligible_train_deps: set[str] | None,
    ):
        train_deps = self.teacher.train_dependence_groups_for_target(
            target_parent=target_parent,
            index=idx,
            eligible_train_dependence_groups=eligible_train_deps,
        )
        if len(train_deps) < self.config.teacher.min_train_dependence_groups:
            return {}

        rows = idx.rows_by_parent[target_parent]
        grid = sorted(
            {(s.direction, float(s.requested_risk)) for s in rows},
            key=lambda x: (x[0], x[1]),
        )

        feature_override = None
        target_feature = tuple(rows[0].context_features)
        climatology = False
        if formulation == "F0_CLIMATOLOGY":
            climatology = True
        elif formulation == "F1_ACCOUNT_ONLY":
            feature_override = self._feature_override_account_only(idx)
            target_feature = feature_override[target_parent]
        elif formulation == "F2_TRUE_MARKET_ACCOUNT":
            pass
        elif formulation == "F3_SHUFFLED_MARKET":
            feature_override, target_feature = self._shuffled_feature_override(
                target_parent=target_parent,
                idx=idx,
                train_deps=train_deps,
            )
        else:
            raise ValueError(formulation)

        laws = {}
        for d, r in grid:
            law = self.teacher.predictive_law(
                target_features=target_feature,
                train_deps=train_deps,
                index=idx,
                direction=d,
                risk=r,
                feature_override=feature_override,
                equal_weight_climatology=climatology,
            )
            if law is not None:
                laws[(d, r)] = law
        return laws

    def _score_parent(self, *, parent: str, laws, idx):
        losses = []
        branches = 0
        for s in idx.rows_by_parent[parent]:
            law = laws.get((s.direction, float(s.requested_risk)))
            if law is None:
                return None
            if (
                law.effective_dependence_n
                < self.config.teacher.min_effective_dependence_n
            ):
                return None
            if law.nearest_distance > self.config.teacher.max_nearest_distance:
                return None
            for q, pred in zip(law.quantile_levels, law.quantiles):
                losses.append(
                    pinball_loss(
                        float(s.realized_utility),
                        float(q),
                        float(pred),
                    )
                )
            branches += 1
        return float(2.0 * np.mean(losses)), branches

    def _bootstrap_group_delta(
        self,
        *,
        left: str,
        right: str,
        group_scores,
    ):
        shared = sorted(set(group_scores[left]) & set(group_scores[right]))
        if not shared:
            raise RuntimeError("NO_PAIRED_DEPENDENCE_GROUPS")
        delta = np.asarray(
            [group_scores[left][g] - group_scores[right][g] for g in shared],
            dtype=np.float64,
        )
        seed_material = (
            f"{self.config.content_hash}|{left}|{right}|{canonical_hash(shared)}"
        )
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        boot = np.empty(self.config.bootstrap_reps, dtype=np.float64)
        for i in range(self.config.bootstrap_reps):
            sample = rng.integers(0, len(delta), size=len(delta))
            boot[i] = float(np.mean(delta[sample]))
        a = self.config.bootstrap_alpha
        lo, hi = np.quantile(boot, [a/2, 1-a/2])
        return PairedDeltaReceiptR6(
            left=left,
            right=right,
            mean_delta=float(np.mean(delta)),
            ci_low=float(lo),
            ci_high=float(hi),
            paired_targets=len(shared),
            bootstrap_reps=self.config.bootstrap_reps,
        )

    def evaluate(
        self,
        samples: Sequence[CounterfactualBranchSampleR5],
        *,
        target_parent_ids: Sequence[str],
        eligible_train_dependence_group_ids: Sequence[str],
    ) -> DependenceAwareControlSuiteReceiptR6:
        idx = self.teacher.index(samples)
        target_set = set(target_parent_ids)
        targets = [p for p in idx.parents if p in target_set]
        eligible = set(eligible_train_dependence_group_ids)

        parent_scores = {f: {} for f in FORMULATIONS}
        branch_counts = {f: 0 for f in FORMULATIONS}
        for p in targets:
            for f in FORMULATIONS:
                laws = self._laws(
                    formulation=f,
                    target_parent=p,
                    idx=idx,
                    eligible_train_deps=eligible,
                )
                if not laws:
                    continue
                scored = self._score_parent(parent=p, laws=laws, idx=idx)
                if scored is None:
                    continue
                score, branches = scored
                parent_scores[f][p] = score
                branch_counts[f] += branches

        # Collapse parent/account replicas to one score per independent future group.
        group_scores = {f: {} for f in FORMULATIONS}
        for f in FORMULATIONS:
            by_dep = {}
            for p, score in parent_scores[f].items():
                dep = idx.parent_dependence_group[p]
                by_dep.setdefault(dep, []).append(score)
            group_scores[f] = {
                dep: float(np.mean(vals))
                for dep, vals in by_dep.items()
            }

        reasons = []
        formulations = []
        for f in FORMULATIONS:
            gs = group_scores[f]
            if len(gs) < self.config.minimum_scored_dependence_groups:
                reasons.append(
                    f"{f}:INSUFFICIENT_SCORED_DEPENDENCE_GROUPS:{len(gs)}"
                )
            formulations.append(DependenceAwareFormulationScoreR6(
                formulation=f,
                scored_parents=len(parent_scores[f]),
                scored_dependence_groups=len(gs),
                scored_branches=branch_counts[f],
                qcrps=float(np.mean(list(gs.values()))) if gs else float("inf"),
                dependence_group_score_hash=canonical_hash(
                    [(g, gs[g]) for g in sorted(gs)]
                ),
            ))

        f2_f0 = self._bootstrap_group_delta(
            left="F2_TRUE_MARKET_ACCOUNT",
            right="F0_CLIMATOLOGY",
            group_scores=group_scores,
        )
        f3_f2 = self._bootstrap_group_delta(
            left="F3_SHUFFLED_MARKET",
            right="F2_TRUE_MARKET_ACCOUNT",
            group_scores=group_scores,
        )
        f1_f0 = self._bootstrap_group_delta(
            left="F1_ACCOUNT_ONLY",
            right="F0_CLIMATOLOGY",
            group_scores=group_scores,
        )

        eval_deps = sorted({
            idx.parent_dependence_group[p] for p in targets
        })
        return DependenceAwareControlSuiteReceiptR6(
            protocol_hash=self.config.content_hash,
            formulations=tuple(formulations),
            f2_minus_f0=f2_f0,
            f3_minus_f2=f3_f2,
            f1_minus_f0=f1_f0,
            evaluation_dependence_group_hash=canonical_hash(eval_deps),
            status="PASS" if not reasons else "INSUFFICIENT_SUPPORT",
            reasons=tuple(reasons),
        )
