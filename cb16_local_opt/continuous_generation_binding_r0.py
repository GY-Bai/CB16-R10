from __future__ import annotations

"""Authority-only binding for pre-league continuous generation.

This module does not alter Physics, Teacher, Student loss, sensory semantics, or
market evidence.  It makes the already-qualified training mechanics auditable
across generations and enforces the hard boundary between contestant production
and frozen league evaluation.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


TRAINING_LEDGER = "CONTESTANT_TRAINING"
LEAGUE_EVALUATION_LEDGER = "LEAGUE_STATUS_DRIVING_EVALUATION"
VALID_LEDGER_KINDS = frozenset({TRAINING_LEDGER, LEAGUE_EVALUATION_LEDGER})


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: str, code: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError(code)


@dataclass(frozen=True)
class EvidenceIntervalR0:
    symbol: str
    start_ms: int
    end_ms: int
    ledger_kind: str
    evidence_id: str

    def validate(self) -> None:
        if not self.symbol:
            raise RuntimeError("CGEN_EMPTY_SYMBOL")
        if self.start_ms >= self.end_ms:
            raise RuntimeError("CGEN_INVALID_INTERVAL")
        if self.ledger_kind not in VALID_LEDGER_KINDS:
            raise RuntimeError("CGEN_INVALID_LEDGER_KIND")
        if not self.evidence_id:
            raise RuntimeError("CGEN_EMPTY_EVIDENCE_ID")

    def overlaps(self, other: "EvidenceIntervalR0") -> bool:
        return max(self.start_ms, other.start_ms) < min(self.end_ms, other.end_ms)


@dataclass(frozen=True)
class GenerationSnapshotR0:
    generation: int
    parent_checkpoint_sha256: str
    parent_policy_hash: str
    child_checkpoint_sha256: str
    child_policy_hash: str
    training_recipe_sha256: str
    training_evidence_ledger_sha256: str
    account_snapshot_sha256: str
    optimizer_snapshot_sha256: str
    authorized_gradient_owners_sha256: str
    final_holdout_touched: bool = False
    fresh_market_data_downloaded: bool = False

    def validate(self) -> None:
        if self.generation < 0:
            raise RuntimeError("CGEN_NEGATIVE_GENERATION")
        for value, code in (
            (self.parent_checkpoint_sha256, "CGEN_PARENT_CHECKPOINT_HASH_INVALID"),
            (self.parent_policy_hash, "CGEN_PARENT_POLICY_HASH_INVALID"),
            (self.child_checkpoint_sha256, "CGEN_CHILD_CHECKPOINT_HASH_INVALID"),
            (self.child_policy_hash, "CGEN_CHILD_POLICY_HASH_INVALID"),
            (self.training_recipe_sha256, "CGEN_RECIPE_HASH_INVALID"),
            (self.training_evidence_ledger_sha256, "CGEN_TRAIN_LEDGER_HASH_INVALID"),
            (self.account_snapshot_sha256, "CGEN_ACCOUNT_HASH_INVALID"),
            (self.optimizer_snapshot_sha256, "CGEN_OPTIMIZER_HASH_INVALID"),
            (self.authorized_gradient_owners_sha256, "CGEN_GRADIENT_OWNER_HASH_INVALID"),
        ):
            _require_sha256(value, code)
        if self.final_holdout_touched:
            raise RuntimeError("CGEN_FINAL_HOLDOUT_FORBIDDEN")
        if self.fresh_market_data_downloaded:
            raise RuntimeError("CGEN_FRESH_DATA_FORBIDDEN")

    @property
    def snapshot_sha256(self) -> str:
        self.validate()
        return canonical_json_sha256({
            "schema": "CB16_R11_CONTINUOUS_GENERATION_SNAPSHOT_R0",
            "generation": self.generation,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "parent_policy_hash": self.parent_policy_hash,
            "child_checkpoint_sha256": self.child_checkpoint_sha256,
            "child_policy_hash": self.child_policy_hash,
            "training_recipe_sha256": self.training_recipe_sha256,
            "training_evidence_ledger_sha256": self.training_evidence_ledger_sha256,
            "account_snapshot_sha256": self.account_snapshot_sha256,
            "optimizer_snapshot_sha256": self.optimizer_snapshot_sha256,
            "authorized_gradient_owners_sha256": self.authorized_gradient_owners_sha256,
            "final_holdout_touched": self.final_holdout_touched,
            "fresh_market_data_downloaded": self.fresh_market_data_downloaded,
        })


def evidence_ledger_sha256(intervals: Sequence[EvidenceIntervalR0], *, expected_kind: str) -> str:
    if expected_kind not in VALID_LEDGER_KINDS:
        raise RuntimeError("CGEN_EXPECTED_LEDGER_KIND_INVALID")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for interval in intervals:
        interval.validate()
        if interval.ledger_kind != expected_kind:
            raise RuntimeError("CGEN_LEDGER_KIND_CROSS_CONTAMINATION")
        if interval.evidence_id in seen:
            raise RuntimeError("CGEN_DUPLICATE_EVIDENCE_ID")
        seen.add(interval.evidence_id)
        rows.append({"symbol": interval.symbol, "start_ms": int(interval.start_ms), "end_ms": int(interval.end_ms), "ledger_kind": interval.ledger_kind, "evidence_id": interval.evidence_id})
    rows.sort(key=lambda x: (x["start_ms"], x["end_ms"], x["symbol"], x["evidence_id"]))
    return canonical_json_sha256({"ledger_kind": expected_kind, "intervals": rows})


def assert_global_quarantine_r0(training: Sequence[EvidenceIntervalR0], league_evaluation: Sequence[EvidenceIntervalR0]) -> None:
    for t in training:
        t.validate()
        if t.ledger_kind != TRAINING_LEDGER:
            raise RuntimeError("CGEN_TRAINING_LEDGER_KIND_DRIFT")
    for e in league_evaluation:
        e.validate()
        if e.ledger_kind != LEAGUE_EVALUATION_LEDGER:
            raise RuntimeError("CGEN_EVALUATION_LEDGER_KIND_DRIFT")
    for t in training:
        for e in league_evaluation:
            if t.overlaps(e):
                raise RuntimeError(f"CGEN_GLOBAL_TRAIN_EVAL_OVERLAP:{t.evidence_id}:{e.evidence_id}")


def validate_generation_transition_r0(previous: GenerationSnapshotR0, current: GenerationSnapshotR0, *, schedule_frozen: bool) -> None:
    previous.validate()
    current.validate()
    if schedule_frozen:
        raise RuntimeError("CGEN_TRAINING_AFTER_LEAGUE_SCHEDULE_FREEZE_FORBIDDEN")
    if current.generation != previous.generation + 1:
        raise RuntimeError("CGEN_GENERATION_NOT_MONOTONIC_PLUS_ONE")
    if current.parent_checkpoint_sha256 != previous.child_checkpoint_sha256:
        raise RuntimeError("CGEN_PARENT_CHECKPOINT_LINEAGE_BREAK")
    if current.parent_policy_hash != previous.child_policy_hash:
        raise RuntimeError("CGEN_PARENT_POLICY_LINEAGE_BREAK")


def assert_account_continuity_r0(post_state_t: Mapping[str, Any], pre_state_t1: Mapping[str, Any]) -> None:
    if canonical_json_sha256(post_state_t) != canonical_json_sha256(pre_state_t1):
        raise RuntimeError("CGEN_ACCOUNT_CONTINUITY_BREAK")


def assert_future_invariance_r0(*, current_snapshot_before: GenerationSnapshotR0, current_snapshot_after_future_poison: GenerationSnapshotR0) -> None:
    if current_snapshot_before.snapshot_sha256 != current_snapshot_after_future_poison.snapshot_sha256:
        raise RuntimeError("CGEN_FUTURE_POISON_CHANGED_CURRENT_GENERATION")


def freeze_roster_entry_r0(snapshot: GenerationSnapshotR0, *, training_recipe_sha256: str, training_evidence_ledger_sha256: str, climatology_hashes_by_symbol: Mapping[str, str], bootstrap_seed: int) -> dict[str, Any]:
    snapshot.validate()
    _require_sha256(training_recipe_sha256, "CGEN_ROSTER_RECIPE_HASH_INVALID")
    _require_sha256(training_evidence_ledger_sha256, "CGEN_ROSTER_LEDGER_HASH_INVALID")
    if training_recipe_sha256 != snapshot.training_recipe_sha256:
        raise RuntimeError("CGEN_ROSTER_RECIPE_NOT_BOUND_TO_SNAPSHOT")
    if training_evidence_ledger_sha256 != snapshot.training_evidence_ledger_sha256:
        raise RuntimeError("CGEN_ROSTER_LEDGER_NOT_BOUND_TO_SNAPSHOT")
    if bootstrap_seed < 0:
        raise RuntimeError("CGEN_ROSTER_BOOTSTRAP_SEED_INVALID")
    if not climatology_hashes_by_symbol:
        raise RuntimeError("CGEN_ROSTER_CLIMATOLOGY_EMPTY")
    for symbol, digest in climatology_hashes_by_symbol.items():
        if not symbol:
            raise RuntimeError("CGEN_ROSTER_CLIMATOLOGY_SYMBOL_EMPTY")
        _require_sha256(digest, "CGEN_ROSTER_CLIMATOLOGY_HASH_INVALID")
    payload = {
        "schema": "CB16_R11_FORMAL_SCIENCE_LEAGUE_ROSTER_ENTRY_R0",
        "generation": snapshot.generation,
        "checkpoint_sha256": snapshot.child_checkpoint_sha256,
        "policy_hash": snapshot.child_policy_hash,
        "generation_snapshot_sha256": snapshot.snapshot_sha256,
        "training_recipe_sha256": training_recipe_sha256,
        "training_evidence_ledger_sha256": training_evidence_ledger_sha256,
        "authorized_gradient_owners_sha256": snapshot.authorized_gradient_owners_sha256,
        "candidate_specific_no_state_climatology_hashes_by_symbol": dict(sorted(climatology_hashes_by_symbol.items())),
        "hierarchical_bootstrap_seed": int(bootstrap_seed),
        "training_locked_after_schedule_freeze": True,
    }
    return {**payload, "roster_entry_sha256": canonical_json_sha256(payload)}
