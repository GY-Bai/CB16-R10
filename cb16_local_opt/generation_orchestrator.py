from __future__ import annotations

"""
Champion / Challenger generation state machine.

Training loss is diagnostic only.  Promotion is based on a frozen TournamentResult
and PromotionRule.  Promotion uses compare-and-swap semantics against the currently
registered Champion, so stale challengers cannot overwrite a newer generation.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def canonical_hash(obj: Any) -> str:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PolicyRecord:
    generation: int
    weight_hash: str
    architecture_hash: str
    tier: str
    checkpoint_path: str
    parent_weight_hash: str | None
    training_snapshot_hash: str | None

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class ChallengerAttempt:
    attempt_id: str
    parent_generation: int
    parent_weight_hash: str
    training_snapshot_hash: str
    experiment_version: str
    architecture_hash: str
    tier: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class TournamentResult:
    attempt_id: str
    champion_weight_hash: str
    challenger_weight_hash: str
    evaluation_dataset_hash: str
    mean_utility_champion: float
    mean_utility_challenger: float
    delta_utility: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    independent_groups: int
    regime_deltas: Mapping[str, float]
    status: str = "COMPLETE"

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class PromotionRule:
    rule_id: str = "CB16_LOCAL_PROMOTION_RULE_R3"
    min_delta_utility: float = 0.0
    min_ci_lower: float = 0.0
    min_independent_groups: int = 1
    max_bad_regime_fraction: float = 0.25
    min_absolute_challenger_utility: float | None = None

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)

    def adjudicate(self, result: TournamentResult) -> tuple[str, tuple[str, ...]]:
        reasons: list[str] = []
        if result.status != "COMPLETE":
            reasons.append("TOURNAMENT_INCOMPLETE")
        if result.delta_utility < self.min_delta_utility:
            reasons.append("DELTA_BELOW_THRESHOLD")
        if result.bootstrap_ci_low < self.min_ci_lower:
            reasons.append("CI_LOWER_BELOW_THRESHOLD")
        if result.independent_groups < self.min_independent_groups:
            reasons.append("INSUFFICIENT_INDEPENDENT_GROUPS")
        if self.min_absolute_challenger_utility is not None:
            if result.mean_utility_challenger < self.min_absolute_challenger_utility:
                reasons.append("CHALLENGER_ABSOLUTE_UTILITY_TOO_LOW")
        if result.regime_deltas:
            bad = sum(float(v) < 0.0 for v in result.regime_deltas.values())
            frac = bad / len(result.regime_deltas)
            if frac > self.max_bad_regime_fraction:
                reasons.append("REGIME_ROBUSTNESS_FAIL")
        return ("PROMOTE" if not reasons else "REJECT", tuple(reasons))


class GenerationStateStore:
    """SQLite WAL state store for one local R&D experiment."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS champion(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                generation INTEGER NOT NULL,
                weight_hash TEXT NOT NULL,
                policy_json BLOB NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attempts(
                attempt_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                parent_generation INTEGER NOT NULL,
                parent_weight_hash TEXT NOT NULL,
                attempt_hash TEXT NOT NULL,
                attempt_json BLOB NOT NULL,
                challenger_hash TEXT,
                challenger_json BLOB,
                tournament_hash TEXT,
                tournament_json BLOB,
                verdict TEXT,
                verdict_reasons_json BLOB,
                promotion_rule_hash TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_attempt_state ON attempts(state, created_at);
            """
        )

    def close(self) -> None:
        self.db.close()

    def current_champion(self) -> PolicyRecord | None:
        row = self.db.execute(
            "SELECT policy_json FROM champion WHERE singleton=1"
        ).fetchone()
        if row is None:
            return None
        return PolicyRecord(**json.loads(row[0]))

    def initialize_champion(self, policy: PolicyRecord) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT weight_hash FROM champion WHERE singleton=1").fetchone()
            if row is not None:
                if row[0] != policy.weight_hash:
                    raise RuntimeError("CHAMPION_ALREADY_INITIALIZED_DIFFERENTLY")
                self.db.execute("COMMIT")
                return
            self.db.execute(
                "INSERT INTO champion(singleton,generation,weight_hash,policy_json,updated_at) VALUES(1,?,?,?,?)",
                (
                    int(policy.generation),
                    policy.weight_hash,
                    json.dumps(asdict(policy), sort_keys=True),
                    time.time(),
                ),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def create_attempt(self, attempt: ChallengerAttempt) -> None:
        champ = self.current_champion()
        if champ is None:
            raise RuntimeError("NO_CHAMPION")
        if champ.generation != attempt.parent_generation or champ.weight_hash != attempt.parent_weight_hash:
            raise RuntimeError("ATTEMPT_PARENT_NOT_CURRENT_CHAMPION")
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT attempt_hash FROM attempts WHERE attempt_id=?",
                (attempt.attempt_id,),
            ).fetchone()
            if row:
                if row[0] != attempt.content_hash:
                    raise RuntimeError("ATTEMPT_ID_CONTENT_CONFLICT")
                self.db.execute("COMMIT")
                return
            self.db.execute(
                """
                INSERT INTO attempts(
                    attempt_id,state,parent_generation,parent_weight_hash,
                    attempt_hash,attempt_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    attempt.attempt_id,
                    "CREATED",
                    attempt.parent_generation,
                    attempt.parent_weight_hash,
                    attempt.content_hash,
                    json.dumps(asdict(attempt), sort_keys=True),
                    now,
                    now,
                ),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def record_challenger(self, attempt_id: str, challenger: PolicyRecord) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT state,parent_weight_hash FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("UNKNOWN_ATTEMPT")
            if row[0] not in {"CREATED", "TRAINED"}:
                raise RuntimeError(f"INVALID_STATE_FOR_CHALLENGER:{row[0]}")
            if challenger.parent_weight_hash != row[1]:
                raise RuntimeError("CHALLENGER_PARENT_HASH_MISMATCH")
            old = self.db.execute(
                "SELECT challenger_hash FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()[0]
            if old and old != challenger.weight_hash:
                raise RuntimeError("CHALLENGER_REWRITE_CONFLICT")
            self.db.execute(
                """
                UPDATE attempts SET state='TRAINED',challenger_hash=?,challenger_json=?,updated_at=?
                WHERE attempt_id=?
                """,
                (
                    challenger.weight_hash,
                    json.dumps(asdict(challenger), sort_keys=True),
                    time.time(),
                    attempt_id,
                ),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def record_tournament(self, result: TournamentResult) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT state,parent_weight_hash,challenger_hash,tournament_hash FROM attempts WHERE attempt_id=?",
                (result.attempt_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("UNKNOWN_ATTEMPT")
            state, parent_hash, challenger_hash, old_hash = row
            if state not in {"TRAINED", "EVALUATED"}:
                raise RuntimeError(f"INVALID_STATE_FOR_TOURNAMENT:{state}")
            if result.champion_weight_hash != parent_hash:
                raise RuntimeError("TOURNAMENT_CHAMPION_PARENT_MISMATCH")
            if result.challenger_weight_hash != challenger_hash:
                raise RuntimeError("TOURNAMENT_CHALLENGER_MISMATCH")
            if old_hash and old_hash != result.content_hash:
                raise RuntimeError("TOURNAMENT_REWRITE_CONFLICT")
            self.db.execute(
                """
                UPDATE attempts SET state='EVALUATED',tournament_hash=?,tournament_json=?,updated_at=?
                WHERE attempt_id=?
                """,
                (
                    result.content_hash,
                    json.dumps(asdict(result), sort_keys=True),
                    time.time(),
                    result.attempt_id,
                ),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def adjudicate(self, attempt_id: str, rule: PromotionRule) -> tuple[str, tuple[str, ...]]:
        row = self.db.execute(
            "SELECT state,tournament_json FROM attempts WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("UNKNOWN_ATTEMPT")
        if row[0] not in {"EVALUATED", "ADJUDICATED"}:
            raise RuntimeError(f"INVALID_STATE_FOR_ADJUDICATION:{row[0]}")
        result = TournamentResult(**json.loads(row[1]))
        verdict, reasons = rule.adjudicate(result)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute(
                "SELECT verdict,promotion_rule_hash FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if old[0] is not None and (old[0] != verdict or old[1] != rule.content_hash):
                raise RuntimeError("ADJUDICATION_REWRITE_CONFLICT")
            self.db.execute(
                """
                UPDATE attempts
                SET state='ADJUDICATED',verdict=?,verdict_reasons_json=?,promotion_rule_hash=?,updated_at=?
                WHERE attempt_id=?
                """,
                (
                    verdict,
                    json.dumps(reasons),
                    rule.content_hash,
                    time.time(),
                    attempt_id,
                ),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return verdict, reasons

    def commit(self, attempt_id: str) -> str:
        """Commit an adjudicated attempt.

        PROMOTE: CAS on current Champion parent hash.
        REJECT: archive attempt and leave Champion untouched.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                """
                SELECT state,parent_generation,parent_weight_hash,challenger_json,verdict
                FROM attempts WHERE attempt_id=?
                """,
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("UNKNOWN_ATTEMPT")
            state, parent_gen, parent_hash, challenger_json, verdict = row
            if state == "COMMITTED":
                self.db.execute("COMMIT")
                return verdict
            if state != "ADJUDICATED":
                raise RuntimeError(f"INVALID_STATE_FOR_COMMIT:{state}")

            champ = self.db.execute(
                "SELECT generation,weight_hash FROM champion WHERE singleton=1"
            ).fetchone()
            if champ is None:
                raise RuntimeError("NO_CHAMPION")

            if verdict == "PROMOTE":
                if int(champ[0]) != int(parent_gen) or champ[1] != parent_hash:
                    raise RuntimeError("STALE_PARENT_PROMOTION_CAS_FAIL")
                challenger = PolicyRecord(**json.loads(challenger_json))
                if challenger.generation <= int(parent_gen):
                    raise RuntimeError("CHALLENGER_GENERATION_NOT_ADVANCING")
                cur = self.db.execute(
                    """
                    UPDATE champion SET generation=?,weight_hash=?,policy_json=?,updated_at=?
                    WHERE singleton=1 AND generation=? AND weight_hash=?
                    """,
                    (
                        challenger.generation,
                        challenger.weight_hash,
                        json.dumps(asdict(challenger), sort_keys=True),
                        time.time(),
                        int(parent_gen),
                        parent_hash,
                    ),
                )
                if cur.rowcount != 1:
                    raise RuntimeError("PROMOTION_COMPARE_AND_SWAP_FAILED")
            elif verdict != "REJECT":
                raise RuntimeError("UNKNOWN_VERDICT")

            self.db.execute(
                "UPDATE attempts SET state='COMMITTED',updated_at=? WHERE attempt_id=?",
                (time.time(), attempt_id),
            )
            self.db.execute("COMMIT")
            return verdict
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def generation_dag(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """
            SELECT attempt_id,state,parent_generation,parent_weight_hash,challenger_hash,verdict
            FROM attempts ORDER BY created_at,attempt_id
            """
        ).fetchall()
        return [
            {
                "attempt_id": r[0],
                "state": r[1],
                "parent_generation": int(r[2]),
                "parent_weight_hash": r[3],
                "challenger_weight_hash": r[4],
                "verdict": r[5],
            }
            for r in rows
        ]
