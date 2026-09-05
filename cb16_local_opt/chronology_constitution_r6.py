from __future__ import annotations

"""
Immutable historical train / validation / tournament chronology constitution.

The split operates on *dependence groups*, not counterfactual branches.

Boundary rule:
For each earlier -> later split boundary, every earlier group whose outcome maturity
timestamp reaches or crosses the first decision timestamp of the later split is PURGED.

Optional embargo:
After each boundary, the first N later dependence groups are excluded entirely.

Therefore:
    max(TRAIN.maturity) < min(VALIDATION.decision)
    max(VALIDATION.maturity) < min(TOURNAMENT.decision)

when all three status-driving splits are non-empty.

This prevents the 72-bar future of an earlier target from crossing into the context period
of a later split.
"""

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ChronologyDependenceGroupR6:
    group_id: str
    parent_ids: tuple[str, ...]
    decision_timestamp: int
    maturity_timestamp: int
    source_hash: str

    def validate(self):
        if not self.group_id or not self.parent_ids:
            raise ValueError("group id/parents required")
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("duplicate parent in group")
        if self.maturity_timestamp <= self.decision_timestamp:
            raise ValueError("maturity must follow decision")

    @property
    def content_hash(self) -> str:
        self.validate()
        return canonical_hash(self)


@dataclass(frozen=True)
class ChronologySplitR6:
    name: str
    group_ids: tuple[str, ...]
    parent_ids: tuple[str, ...]
    first_decision_timestamp: int
    last_decision_timestamp: int
    last_maturity_timestamp: int
    group_count: int
    parent_count: int
    group_hash: str


@dataclass(frozen=True)
class ExcludedChronologyGroupR6:
    group_id: str
    reason: str
    boundary: str


@dataclass(frozen=True)
class ChronologyConstitutionConfigR6:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    tournament_fraction: float = 0.20
    embargo_groups: int = 1
    min_train_groups: int = 24
    min_validation_groups: int = 8
    min_tournament_groups: int = 8
    constitution_version: str = "CB16_HISTORICAL_CHRONOLOGY_CONSTITUTION_R6"

    def validate(self):
        vals = (
            self.train_fraction,
            self.validation_fraction,
            self.tournament_fraction,
        )
        if any(x <= 0 for x in vals):
            raise ValueError("split fractions must be positive")
        if abs(sum(vals) - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to 1")
        if self.embargo_groups < 0:
            raise ValueError("negative embargo")
        if min(
            self.min_train_groups,
            self.min_validation_groups,
            self.min_tournament_groups,
        ) <= 0:
            raise ValueError("minimum split groups must be positive")

    @property
    def content_hash(self) -> str:
        self.validate()
        return canonical_hash(self)


@dataclass(frozen=True)
class HistoricalChronologyConstitutionR6:
    constitution_version: str
    protocol_hash: str
    dataset_hash: str
    horizon_clock_id: str
    source_group_hash: str
    train: ChronologySplitR6
    validation: ChronologySplitR6
    tournament: ChronologySplitR6
    excluded: tuple[ExcludedChronologyGroupR6, ...]
    total_source_groups: int
    total_included_groups: int

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)

    def split_for_group(self, group_id: str) -> str | None:
        if group_id in self.train.group_ids:
            return "TRAIN"
        if group_id in self.validation.group_ids:
            return "VALIDATION"
        if group_id in self.tournament.group_ids:
            return "TOURNAMENT"
        return None


class ChronologyConstitutionBuilderR6:
    def __init__(self, config: ChronologyConstitutionConfigR6 | None = None):
        self.config = config or ChronologyConstitutionConfigR6()
        self.config.validate()

    @staticmethod
    def _validate_source(
        groups: Sequence[ChronologyDependenceGroupR6],
    ) -> list[ChronologyDependenceGroupR6]:
        if not groups:
            raise ValueError("no chronology groups")
        by_id = {}
        parents = set()
        for g in groups:
            g.validate()
            if g.group_id in by_id:
                if by_id[g.group_id].content_hash != g.content_hash:
                    raise RuntimeError("DEPENDENCE_GROUP_ID_CONTENT_CONFLICT")
                raise RuntimeError("DUPLICATE_DEPENDENCE_GROUP_ID")
            by_id[g.group_id] = g
            overlap = parents.intersection(g.parent_ids)
            if overlap:
                raise RuntimeError(f"PARENT_IN_MULTIPLE_DEPENDENCE_GROUPS:{sorted(overlap)}")
            parents.update(g.parent_ids)

        ordered = sorted(
            groups,
            key=lambda g: (g.decision_timestamp, g.group_id),
        )
        # Different groups may share a timestamp (cross-sectional contexts); ordering is
        # deterministic, but split boundaries are moved so equal timestamps do not straddle.
        return ordered

    @staticmethod
    def _move_cut_after_timestamp(
        groups: Sequence[ChronologyDependenceGroupR6],
        cut: int,
    ) -> int:
        if cut <= 0 or cut >= len(groups):
            return cut
        ts = groups[cut - 1].decision_timestamp
        while cut < len(groups) and groups[cut].decision_timestamp == ts:
            cut += 1
        return cut

    @staticmethod
    def _make_split(
        name: str,
        groups: Sequence[ChronologyDependenceGroupR6],
    ) -> ChronologySplitR6:
        if not groups:
            raise RuntimeError(f"EMPTY_STATUS_DRIVING_SPLIT:{name}")
        parents = tuple(p for g in groups for p in g.parent_ids)
        gids = tuple(g.group_id for g in groups)
        return ChronologySplitR6(
            name=name,
            group_ids=gids,
            parent_ids=parents,
            first_decision_timestamp=min(g.decision_timestamp for g in groups),
            last_decision_timestamp=max(g.decision_timestamp for g in groups),
            last_maturity_timestamp=max(g.maturity_timestamp for g in groups),
            group_count=len(groups),
            parent_count=len(parents),
            group_hash=canonical_hash([g.content_hash for g in groups]),
        )

    def build(
        self,
        groups: Sequence[ChronologyDependenceGroupR6],
        *,
        dataset_hash: str,
        horizon_clock_id: str,
    ) -> HistoricalChronologyConstitutionR6:
        ordered = self._validate_source(groups)
        n = len(ordered)

        cut1 = max(1, min(n - 2, round(n * self.config.train_fraction)))
        cut2 = max(cut1 + 1, min(
            n - 1,
            round(n * (self.config.train_fraction + self.config.validation_fraction)),
        ))
        cut1 = self._move_cut_after_timestamp(ordered, cut1)
        cut2 = self._move_cut_after_timestamp(ordered, cut2)
        if not (0 < cut1 < cut2 < n):
            raise RuntimeError("UNABLE_TO_FORM_THREE_CHRONOLOGICAL_SPLITS")

        train = list(ordered[:cut1])
        validation = list(ordered[cut1:cut2])
        tournament = list(ordered[cut2:])
        excluded: list[ExcludedChronologyGroupR6] = []

        def purge_left(
            left: list[ChronologyDependenceGroupR6],
            right: list[ChronologyDependenceGroupR6],
            boundary: str,
        ):
            if not right:
                return
            boundary_ts = right[0].decision_timestamp
            keep = []
            for g in left:
                if g.maturity_timestamp >= boundary_ts:
                    excluded.append(ExcludedChronologyGroupR6(
                        group_id=g.group_id,
                        reason="OUTCOME_MATURITY_CROSSES_NEXT_SPLIT",
                        boundary=boundary,
                    ))
                else:
                    keep.append(g)
            left[:] = keep

        purge_left(train, validation, "TRAIN->VALIDATION")
        purge_left(validation, tournament, "VALIDATION->TOURNAMENT")

        def embargo_right(
            right: list[ChronologyDependenceGroupR6],
            boundary: str,
        ):
            if self.config.embargo_groups <= 0:
                return
            k = min(self.config.embargo_groups, len(right))
            removed = right[:k]
            del right[:k]
            excluded.extend(
                ExcludedChronologyGroupR6(
                    group_id=g.group_id,
                    reason="BOUNDARY_EMBARGO",
                    boundary=boundary,
                )
                for g in removed
            )

        embargo_right(validation, "TRAIN->VALIDATION")
        embargo_right(tournament, "VALIDATION->TOURNAMENT")

        # Purge again because embargo changes the right-side first decision timestamp.
        purge_left(train, validation, "TRAIN->VALIDATION_POST_EMBARGO")
        purge_left(validation, tournament, "VALIDATION->TOURNAMENT_POST_EMBARGO")

        if len(train) < self.config.min_train_groups:
            raise RuntimeError(
                f"INSUFFICIENT_TRAIN_GROUPS_AFTER_PURGE:{len(train)}"
            )
        if len(validation) < self.config.min_validation_groups:
            raise RuntimeError(
                f"INSUFFICIENT_VALIDATION_GROUPS_AFTER_PURGE:{len(validation)}"
            )
        if len(tournament) < self.config.min_tournament_groups:
            raise RuntimeError(
                f"INSUFFICIENT_TOURNAMENT_GROUPS_AFTER_PURGE:{len(tournament)}"
            )

        tr = self._make_split("TRAIN", train)
        va = self._make_split("VALIDATION", validation)
        to = self._make_split("TOURNAMENT", tournament)

        if tr.last_maturity_timestamp >= va.first_decision_timestamp:
            raise RuntimeError("TRAIN_VALIDATION_OUTCOME_LEAKAGE")
        if va.last_maturity_timestamp >= to.first_decision_timestamp:
            raise RuntimeError("VALIDATION_TOURNAMENT_OUTCOME_LEAKAGE")

        source_hash = canonical_hash([g.content_hash for g in ordered])
        return HistoricalChronologyConstitutionR6(
            constitution_version=self.config.constitution_version,
            protocol_hash=self.config.content_hash,
            dataset_hash=dataset_hash,
            horizon_clock_id=horizon_clock_id,
            source_group_hash=source_hash,
            train=tr,
            validation=va,
            tournament=to,
            excluded=tuple(sorted(
                excluded,
                key=lambda x: (x.boundary, x.group_id, x.reason),
            )),
            total_source_groups=n,
            total_included_groups=tr.group_count + va.group_count + to.group_count,
        )


def assert_constitution_disjoint(
    constitution: HistoricalChronologyConstitutionR6,
) -> None:
    a = set(constitution.train.group_ids)
    b = set(constitution.validation.group_ids)
    c = set(constitution.tournament.group_ids)
    if a & b or a & c or b & c:
        raise RuntimeError("SPLIT_GROUP_OVERLAP")
    pa = set(constitution.train.parent_ids)
    pb = set(constitution.validation.parent_ids)
    pc = set(constitution.tournament.parent_ids)
    if pa & pb or pa & pc or pb & pc:
        raise RuntimeError("SPLIT_PARENT_OVERLAP")
    if constitution.train.last_maturity_timestamp >= constitution.validation.first_decision_timestamp:
        raise RuntimeError("TRAIN_VALIDATION_MATURITY_OVERLAP")
    if constitution.validation.last_maturity_timestamp >= constitution.tournament.first_decision_timestamp:
        raise RuntimeError("VALIDATION_TOURNAMENT_MATURITY_OVERLAP")
