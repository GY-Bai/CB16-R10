from __future__ import annotations

"""
Single-use FINAL TOURNAMENT F0/F1/F2/F3 examiner.

This is separate from the R6 policy-vs-baseline final economic holdout.

The examiner binds its identity BEFORE reading final-target counterfactual sample bytes.
It then evaluates the frozen probabilistic-law controls on FINAL TOURNAMENT targets using
TRAIN dependence groups only.

Required inputs:
- TRAIN Teacher sample JSONL (already available from historical R&D);
- FINAL TOURNAMENT sample JSONL generated only after the examiner guard is opened;
- immutable chronology constitution;
- dependence-aware control protocol.

Safety properties:
- target final samples are not part of Teacher train support;
- F3 donors come from frozen TRAIN dependence groups only;
- bootstrap unit is dependence group;
- exact identity is idempotent;
- a different protocol/dataset/train sample/final sample identity cannot reuse the guard.
"""

import dataclasses
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .chronology_constitution_r6 import (
    HistoricalChronologyConstitutionR6,
    ChronologySplitR6,
    ExcludedChronologyGroupR6,
)
from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .scientific_controls_r6 import (
    DependenceAwareControlSuiteConfigR6,
    DependenceAwareControlSuiteReceiptR6,
    DependenceAwareHistoricalControlSuiteR6,
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _constitution_from_json(obj) -> HistoricalChronologyConstitutionR6:
    return HistoricalChronologyConstitutionR6(
        constitution_version=obj["constitution_version"],
        protocol_hash=obj["protocol_hash"],
        dataset_hash=obj["dataset_hash"],
        horizon_clock_id=obj["horizon_clock_id"],
        source_group_hash=obj["source_group_hash"],
        train=ChronologySplitR6(**obj["train"]),
        validation=ChronologySplitR6(**obj["validation"]),
        tournament=ChronologySplitR6(**obj["tournament"]),
        excluded=tuple(
            ExcludedChronologyGroupR6(**x) for x in obj["excluded"]
        ),
        total_source_groups=obj["total_source_groups"],
        total_included_groups=obj["total_included_groups"],
    )


def _read_samples(path: str | Path) -> list[CounterfactualBranchSampleR5]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            x["context_features"] = tuple(x["context_features"])
            out.append(CounterfactualBranchSampleR5(**x))
    return out


@dataclass(frozen=True)
class FinalControlExaminerIdentityR7:
    dataset_hash: str
    constitution_hash: str
    tournament_group_hash: str
    train_group_hash: str
    train_samples_sha256: str
    final_sample_protocol_hash: str
    control_protocol_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class FinalControlExaminerReceiptR7:
    identity_hash: str
    train_dependence_groups: int
    final_dependence_groups: int
    final_parent_contexts: int
    final_samples_sha256: str
    controls: DependenceAwareControlSuiteReceiptR6
    status: str
    final_tournament_opened: bool
    result_hash: str


class FinalControlExaminerGuardR7:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            self.path, isolation_level=None, timeout=30
        )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS final_control_examiner(
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            identity_hash TEXT NOT NULL,
            identity_json BLOB NOT NULL,
            state TEXT NOT NULL,
            result_hash TEXT,
            receipt_json BLOB,
            opened_at REAL NOT NULL,
            completed_at REAL
        );
        """)

    def close(self):
        self.db.close()

    def begin(self, identity: FinalControlExaminerIdentityR7) -> str:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT identity_hash,state FROM final_control_examiner WHERE singleton=1"
            ).fetchone()
            if row is None:
                self.db.execute(
                    """
                    INSERT INTO final_control_examiner(
                        singleton,identity_hash,identity_json,state,opened_at
                    ) VALUES(1,?,?,'OPENED',?)
                    """,
                    (
                        identity.content_hash,
                        json.dumps(asdict(identity), sort_keys=True),
                        time.time(),
                    ),
                )
                self.db.execute("COMMIT")
                return "OPENED_NEW"
            if row[0] != identity.content_hash:
                raise RuntimeError(
                    "FINAL_CONTROL_EXAMINER_ALREADY_BOUND_TO_DIFFERENT_IDENTITY"
                )
            self.db.execute("COMMIT")
            return row[1]
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def completed(self, identity: FinalControlExaminerIdentityR7):
        row = self.db.execute(
            """
            SELECT identity_hash,state,receipt_json
            FROM final_control_examiner WHERE singleton=1
            """
        ).fetchone()
        if row is None:
            return None
        if row[0] != identity.content_hash:
            raise RuntimeError("FINAL_CONTROL_EXAMINER_IDENTITY_CONFLICT")
        if row[1] != "COMPLETED":
            return None
        obj = json.loads(row[2])
        obj["controls"]["formulations"] = tuple(
            obj["controls"]["formulations"]
        )
        # Reconstruct nested dataclasses explicitly below in helper.
        return obj

    def complete(
        self,
        identity: FinalControlExaminerIdentityR7,
        receipt: FinalControlExaminerReceiptR7,
    ):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                """
                SELECT identity_hash,state,result_hash
                FROM final_control_examiner WHERE singleton=1
                """
            ).fetchone()
            if row is None or row[0] != identity.content_hash:
                raise RuntimeError("FINAL_CONTROL_EXAMINER_NOT_OPENED")
            if row[1] == "COMPLETED":
                if row[2] != receipt.result_hash:
                    raise RuntimeError(
                        "FINAL_CONTROL_EXAMINER_RESULT_REWRITE_CONFLICT"
                    )
                self.db.execute("COMMIT")
                return
            self.db.execute(
                """
                UPDATE final_control_examiner
                SET state='COMPLETED',result_hash=?,receipt_json=?,completed_at=?
                WHERE singleton=1
                """,
                (
                    receipt.result_hash,
                    json.dumps(asdict(receipt), sort_keys=True),
                    time.time(),
                ),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise


def _control_receipt_from_json(obj):
    from .scientific_controls_r6 import (
        DependenceAwareFormulationScoreR6,
        DependenceAwareControlSuiteReceiptR6,
        PairedDeltaReceiptR6,
    )
    return DependenceAwareControlSuiteReceiptR6(
        protocol_hash=obj["protocol_hash"],
        formulations=tuple(
            DependenceAwareFormulationScoreR6(**x)
            for x in obj["formulations"]
        ),
        f2_minus_f0=PairedDeltaReceiptR6(**obj["f2_minus_f0"]),
        f3_minus_f2=PairedDeltaReceiptR6(**obj["f3_minus_f2"]),
        f1_minus_f0=PairedDeltaReceiptR6(**obj["f1_minus_f0"]),
        evaluation_dependence_group_hash=obj["evaluation_dependence_group_hash"],
        status=obj["status"],
        reasons=tuple(obj["reasons"]),
    )


def evaluate_final_controls_r7(
    *,
    constitution_path: str | Path,
    train_samples_path: str | Path,
    final_samples_path: str | Path,
    controls_config: DependenceAwareControlSuiteConfigR6,
    final_sample_protocol_hash: str,
    guard: FinalControlExaminerGuardR7,
) -> FinalControlExaminerReceiptR7:
    # Administrative metadata and TRAIN identities may be read before OPEN.
    # FINAL sample bytes must not be hashed or parsed until after guard.begin().
    constitution = _constitution_from_json(
        json.loads(Path(constitution_path).read_text())
    )
    train_sha = sha256_file(train_samples_path)
    if not final_sample_protocol_hash:
        raise ValueError("final_sample_protocol_hash required")
    identity = FinalControlExaminerIdentityR7(
        dataset_hash=constitution.dataset_hash,
        constitution_hash=constitution.content_hash,
        tournament_group_hash=constitution.tournament.group_hash,
        train_group_hash=constitution.train.group_hash,
        train_samples_sha256=train_sha,
        final_sample_protocol_hash=final_sample_protocol_hash,
        control_protocol_hash=controls_config.content_hash,
    )
    guard.begin(identity)

    old = guard.completed(identity)
    if old is not None:
        controls = _control_receipt_from_json(old["controls"])
        return FinalControlExaminerReceiptR7(
            identity_hash=old["identity_hash"],
            train_dependence_groups=old["train_dependence_groups"],
            final_dependence_groups=old["final_dependence_groups"],
            final_parent_contexts=old["final_parent_contexts"],
            final_samples_sha256=old["final_samples_sha256"],
            controls=controls,
            status=old["status"],
            final_tournament_opened=old["final_tournament_opened"],
            result_hash=old["result_hash"],
        )

    # From here the exact final-control identity is OPENED.
    # Only now may FINAL sample bytes be touched.
    final_sha = sha256_file(final_samples_path)
    train_samples = _read_samples(train_samples_path)
    final_samples = _read_samples(final_samples_path)
    all_samples = train_samples + final_samples

    final_parent_set = set(constitution.tournament.parent_ids)
    final_samples_parent_set = {s.parent_id for s in final_samples}
    missing = final_parent_set - final_samples_parent_set
    extra = final_samples_parent_set - final_parent_set
    if missing:
        raise RuntimeError(
            f"FINAL_CONTROL_MISSING_TOURNAMENT_PARENTS:{len(missing)}"
        )
    if extra:
        raise RuntimeError(
            f"FINAL_CONTROL_HAS_NON_TOURNAMENT_PARENTS:{len(extra)}"
        )

    train_dep_set = set(constitution.train.group_ids)
    if any(s.dependence_group_id not in train_dep_set for s in train_samples):
        raise RuntimeError("TRAIN_SAMPLE_OUTSIDE_FROZEN_TRAIN_GROUPS")
    final_dep_set = set(constitution.tournament.group_ids)
    if any(s.dependence_group_id not in final_dep_set for s in final_samples):
        raise RuntimeError("FINAL_SAMPLE_OUTSIDE_TOURNAMENT_GROUPS")
    if train_dep_set & final_dep_set:
        raise RuntimeError("TRAIN_FINAL_DEPENDENCE_GROUP_OVERLAP")

    suite = DependenceAwareHistoricalControlSuiteR7Compat(
        controls_config
    )
    controls = suite.evaluate(
        all_samples,
        target_parent_ids=constitution.tournament.parent_ids,
        eligible_train_dependence_group_ids=constitution.train.group_ids,
    )

    material = {
        "identity_hash": identity.content_hash,
        "train_dependence_groups": len(train_dep_set),
        "final_dependence_groups": len(final_dep_set),
        "final_parent_contexts": len(final_parent_set),
        "final_samples_sha256": final_sha,
        "controls_hash": controls.content_hash,
        "controls_status": controls.status,
        "final_tournament_opened": True,
    }
    result_hash = canonical_hash(material)
    receipt = FinalControlExaminerReceiptR7(
        identity_hash=identity.content_hash,
        train_dependence_groups=len(train_dep_set),
        final_dependence_groups=len(final_dep_set),
        final_parent_contexts=len(final_parent_set),
        final_samples_sha256=final_sha,
        controls=controls,
        status=(
            "FINAL_CONTROLS_PASS"
            if controls.status == "PASS"
            else "FINAL_CONTROLS_INSUFFICIENT_SUPPORT"
        ),
        final_tournament_opened=True,
        result_hash=result_hash,
    )
    guard.complete(identity, receipt)
    return receipt


class DependenceAwareHistoricalControlSuiteR7Compat(
    DependenceAwareHistoricalControlSuiteR6
):
    """Named R7 wrapper so final receipts clearly identify the R7 execution surface."""
    pass
