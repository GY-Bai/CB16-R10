from __future__ import annotations

"""Stage-3 E2 multi-generation control-plane / lineage soak for CB16 R11.

This harness exercises real R11Orchestrator authority transitions with deterministic
control receipts and tiny CPU/FP32 content-addressed checkpoint objects.  It does not
perform scientific training, open the final holdout, download market data, or treat
replayed frozen evidence as new scientific evidence.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from .checkpoint_store_r11 import CheckpointStoreR11
from .event_journal_r11 import EventJournalR11
from .integration_adapters_r11 import (
    CheckpointStoreProtocolAdapterR11,
    EvidenceStoreProtocolAdapterR11,
    EventJournalProtocolAdapterR11,
)
from .orchestrator_r11 import IllegalTransition, LineageViolation, R11Orchestrator
from .runtime_events_r11 import (
    AuthorityStamp,
    EvidenceRef,
    FrozenInputReceipt,
    PolicyResult,
    SnapshotSeal,
    TournamentDecision,
    TournamentResult,
    TrainingResult,
    ValidationResult,
    semantic_hash,
)


SCHEMA = "CB16_R11_STAGE3_E2_LINEAGE_SOAK_REPORT_V1"
FAILURE_TAXONOMY = (
    "SEMANTIC_FREEZE_DRIFT",
    "GENERATION_SEQUENCE_VIOLATION",
    "CHAMPION_AUTHORITY_VIOLATION",
    "REJECTED_CHALLENGER_LAUNDERING",
    "SNAPSHOT_OWNERSHIP_VIOLATION",
    "EVIDENCE_LINEAGE_VIOLATION",
    "TEACHER_AUTHORITY_DRIFT",
    "PHYSICS_AUTHORITY_DRIFT",
    "DUPLICATE_COMMIT_VIOLATION",
    "DUPLICATE_RELEASE_VIOLATION",
    "CHECKPOINT_COMMIT_MISMATCH",
    "JOURNAL_AUDIT_FAILURE",
    "RESTART_RECONSTRUCTION_MISMATCH",
    "FUTURE_EVIDENCE_TAINT_ACCEPTED",
    "STALE_GENERATION_EVENT_ACCEPTED",
    "STALE_CHAMPION_EVENT_ACCEPTED",
    "REORDERED_EVENT_ACCEPTED",
    "UNCLASSIFIED_CONTROL_PLANE_FAILURE",
)

FREEZE_REL = Path("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json")
TEACHER_AUTHORITY_ID = "CB16_R11_FROZEN_TEACHER_AUTHORITY"
PHYSICS_AUTHORITY_ID = "CB16_R11_FROZEN_PHYSICS_AUTHORITY"
FROZEN_REPLAY_EVIDENCE_ID = "CB16_R11_STAGE3_E2_FROZEN_REPLAY_CONTROL_RECEIPT_V1"


class Stage3E2InvariantError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _authority_obj(x: AuthorityStamp) -> dict[str, Any]:
    return {
        "generation": int(x.generation),
        "champion_id": x.champion_id,
        "champion_hash": x.champion_hash,
        "teacher_authority_id": x.teacher_authority_id,
        "physics_authority_id": x.physics_authority_id,
    }


def _decision_for_generation(generation: int) -> TournamentDecision:
    """Exercise repeated PROMOTE, repeated REJECT, then alternating decisions."""
    phase = int(generation) % 100
    if phase < 10:
        return TournamentDecision.PROMOTE
    if phase < 20:
        return TournamentDecision.REJECT
    return TournamentDecision.PROMOTE if phase % 2 == 0 else TournamentDecision.REJECT


def _frozen_replay_evidence() -> EvidenceRef:
    payload_hash = semantic_hash(
        {
            "schema": FROZEN_REPLAY_EVIDENCE_ID,
            "role": "ENGINEERING_CONTROL_RECEIPT_ONLY",
            "scientific_evidence_increment": 0,
        }
    )
    return EvidenceRef(
        evidence_id=FROZEN_REPLAY_EVIDENCE_ID,
        payload_hash=payload_hash,
        source_generation=-1,
        producer_champion_id="FROZEN_REPLAY_ORIGIN",
        teacher_authority_id=TEACHER_AUTHORITY_ID,
        physics_authority_id=PHYSICS_AUTHORITY_ID,
        teacher_generation=-1,
        detached_from_autograd=True,
    )


def _snapshot(authority: AuthorityStamp, evidence: EvidenceRef) -> SnapshotSeal:
    body = {
        "schema": "CB16_R11_STAGE3_E2_SNAPSHOT_CONTROL_IDENTITY_V1",
        "generation": int(authority.generation),
        "parent_champion_id": authority.champion_id,
        "evidence_ids": [evidence.evidence_id],
    }
    snapshot_hash = semantic_hash(body)
    return SnapshotSeal(
        snapshot_id=f"stage3-e2-snapshot-g{authority.generation:08d}-{snapshot_hash[:16]}",
        snapshot_hash=snapshot_hash,
        generation=authority.generation,
        parent_champion_id=authority.champion_id,
        evidence_ids=(evidence.evidence_id,),
        immutable=True,
    )


def _expect_rejected(
    name: str,
    fn: Callable[[], Any],
    expected: tuple[type[BaseException], ...] = (LineageViolation, IllegalTransition),
) -> dict[str, Any]:
    try:
        fn()
    except expected as exc:
        return {
            "attack": name,
            "accepted": False,
            "rejected": True,
            "exception": type(exc).__name__,
            "detail": str(exc),
        }
    raise Stage3E2InvariantError(f"E2_EXPECTED_REJECTION_NOT_OBSERVED:{name}")


def _classify_failure(exc: BaseException) -> str:
    text = str(exc)
    mapping = (
        ("FREEZE", "SEMANTIC_FREEZE_DRIFT"),
        ("GENERATION", "GENERATION_SEQUENCE_VIOLATION"),
        ("STALE_POLICY", "CHAMPION_AUTHORITY_VIOLATION"),
        ("SELF_LAUNDER", "REJECTED_CHALLENGER_LAUNDERING"),
        ("SNAPSHOT", "SNAPSHOT_OWNERSHIP_VIOLATION"),
        ("EVIDENCE", "EVIDENCE_LINEAGE_VIOLATION"),
        ("TEACHER", "TEACHER_AUTHORITY_DRIFT"),
        ("PHYSICS", "PHYSICS_AUTHORITY_DRIFT"),
        ("COMMIT", "DUPLICATE_COMMIT_VIOLATION"),
        ("CHECKPOINT", "CHECKPOINT_COMMIT_MISMATCH"),
        ("JOURNAL", "JOURNAL_AUDIT_FAILURE"),
        ("RECOVER", "RESTART_RECONSTRUCTION_MISMATCH"),
    )
    for token, category in mapping:
        if token in text:
            return category
    return "UNCLASSIFIED_CONTROL_PLANE_FAILURE"


def _make_policy_state(generation: int, *, seed: bool = False) -> dict[str, torch.Tensor]:
    # Tiny deterministic FP32 control object. It is a checkpoint identity carrier only,
    # not a trained model and not scientific learning evidence.
    if seed:
        values = [-1.0, 0.0, 1.0, 11.0]
    else:
        values = [float(generation), float(generation % 17), 1.0, 11.0]
    return {"stage3_e2_control_identity": torch.tensor(values, dtype=torch.float32)}


def _verify_recovered(
    recovered: R11Orchestrator,
    authority: AuthorityStamp,
    *,
    commit_id: str,
    checkpoint_id: str,
) -> None:
    r = recovered.record
    if r.authority != authority:
        raise Stage3E2InvariantError("E2_RECOVERED_AUTHORITY_STAMP_MISMATCH")
    if r.commit is None or r.commit.commit_id != commit_id:
        raise Stage3E2InvariantError("E2_RECOVERED_COMMIT_MISMATCH")
    if r.checkpoint is None or r.checkpoint.checkpoint_id != checkpoint_id:
        raise Stage3E2InvariantError("E2_RECOVERED_CHECKPOINT_MISMATCH")
    if not r.next_generation_released:
        raise Stage3E2InvariantError("E2_RECOVERED_RELEASE_FLAG_MISSING")


def run_lineage_soak(
    root: str | Path,
    *,
    generations: int = 100,
    repo_root: str | Path | None = None,
    synchronous: str = "FULL",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a deterministic Stage-3 E2 lineage soak and return a machine-readable report."""
    if int(generations) <= 0:
        raise ValueError("generations must be positive")

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    freeze_path = repo / FREEZE_REL
    freeze_before = _sha256_file(freeze_path) if freeze_path.is_file() else None

    start = time.perf_counter()
    ledger: list[dict[str, Any]] = []
    first_failure: dict[str, Any] | None = None
    journal_audit: dict[str, Any] | None = None
    final_authority: AuthorityStamp | None = None

    journal_store = EventJournalR11(root / "journal", synchronous=synchronous)
    checkpoint_store = CheckpointStoreR11(root / "checkpoints", synchronous=synchronous)
    journal = EventJournalProtocolAdapterR11(journal_store)
    evidence_store = EvidenceStoreProtocolAdapterR11(journal_store)
    checkpoints = CheckpointStoreProtocolAdapterR11(checkpoint_store)

    frozen_evidence = _frozen_replay_evidence()
    rejected_challengers: set[str] = set()
    attack_rejections = 0
    duplicate_idempotent = 0
    restart_checks = 0

    try:
        seed_ref = checkpoint_store.put_state_dict(_make_policy_state(-1, seed=True))
        authority = AuthorityStamp(
            generation=0,
            champion_id=seed_ref.semantic_sha256,
            champion_hash=seed_ref.semantic_sha256,
            teacher_authority_id=TEACHER_AUTHORITY_ID,
            physics_authority_id=PHYSICS_AUTHORITY_ID,
        )

        for generation in range(int(generations)):
            if authority.generation != generation:
                raise Stage3E2InvariantError(
                    f"E2_GENERATION_SEQUENCE_VIOLATION:EXPECTED:{generation}:GOT:{authority.generation}"
                )
            if authority.champion_id != authority.champion_hash:
                raise Stage3E2InvariantError("E2_CHAMPION_ID_HASH_MISMATCH")
            if authority.champion_id in rejected_challengers:
                raise Stage3E2InvariantError(
                    f"E2_REJECTED_CHALLENGER_LAUNDERED_AS_PARENT:{authority.champion_id}"
                )
            if authority.teacher_authority_id != TEACHER_AUTHORITY_ID:
                raise Stage3E2InvariantError("E2_TEACHER_AUTHORITY_DRIFT")
            if authority.physics_authority_id != PHYSICS_AUTHORITY_ID:
                raise Stage3E2InvariantError("E2_PHYSICS_AUTHORITY_DRIFT")

            core = R11Orchestrator(
                authority=authority,
                evidence_store=evidence_store,
                journal=journal,
                checkpoint_store=checkpoints,
            )
            core.verify_frozen_inputs(
                FrozenInputReceipt(
                    input_identity="CB16_R11_STAGE3_E2_FROZEN_REPLAY_ONLY",
                    seal_hash=frozen_evidence.payload_hash,
                    verified=True,
                )
            )

            attacks: list[dict[str, Any]] = []
            attacks.append(
                _expect_rejected(
                    "future_evidence_taint",
                    lambda g=generation: core.accept_teacher_evidence(
                        EvidenceRef(
                            evidence_id=f"forbidden-future-evidence-g{g}",
                            payload_hash=semantic_hash({"forbidden": g}),
                            source_generation=g,
                            producer_champion_id=authority.champion_id,
                            teacher_authority_id=TEACHER_AUTHORITY_ID,
                            physics_authority_id=PHYSICS_AUTHORITY_ID,
                            teacher_generation=g,
                        )
                    ),
                    (LineageViolation,),
                )
            )
            core.accept_teacher_evidence(frozen_evidence)
            snapshot = _snapshot(authority, frozen_evidence)
            core.seal_training_snapshot(snapshot)

            stale_generation = generation - 1 if generation > 0 else generation + 1
            attacks.append(
                _expect_rejected(
                    "stale_generation_event",
                    lambda sg=stale_generation: core.accept_policy_result(
                        PolicyResult(
                            work_id=f"stale-generation-policy-g{generation}",
                            generation=sg,
                            champion_id=authority.champion_id,
                            payload_hash=semantic_hash({"stale_generation": generation}),
                        )
                    ),
                    (LineageViolation,),
                )
            )

            stale_champion = (
                sorted(rejected_challengers)[-1]
                if rejected_challengers
                else semantic_hash({"never_authoritative": generation})
            )
            attacks.append(
                _expect_rejected(
                    "stale_champion_event",
                    lambda sc=stale_champion: core.accept_policy_result(
                        PolicyResult(
                            work_id=f"stale-champion-policy-g{generation}",
                            generation=generation,
                            champion_id=sc,
                            payload_hash=semantic_hash({"stale_champion": generation}),
                        )
                    ),
                    (LineageViolation,),
                )
            )

            attacks.append(
                _expect_rejected(
                    "reordered_validation_before_training",
                    lambda: core.complete_validation(
                        ValidationResult(
                            work_id=f"early-validation-g{generation}",
                            generation=generation,
                            challenger_id="not-yet-created",
                            parent_champion_id=authority.champion_id,
                            snapshot_id=snapshot.snapshot_id,
                            validation_id=f"early-validation-g{generation}",
                            payload_hash=semantic_hash({"early_validation": generation}),
                        )
                    ),
                    (IllegalTransition,),
                )
            )
            attack_rejections += len(attacks)

            policy = PolicyResult(
                work_id=f"policy-g{generation}",
                generation=generation,
                champion_id=authority.champion_id,
                payload_hash=semantic_hash(
                    {"schema": "E2_POLICY_CONTROL_RESULT_V1", "generation": generation, "champion": authority.champion_id}
                ),
            )
            core.accept_policy_result(policy)
            core.start_trace(f"trace-start-g{generation}")
            core.start_challenger_training(work_id=f"train-g{generation}")

            challenger_ref = checkpoint_store.put_state_dict(_make_policy_state(generation))
            challenger_id = challenger_ref.semantic_sha256
            training = TrainingResult(
                work_id=f"train-g{generation}",
                generation=generation,
                challenger_id=challenger_id,
                challenger_hash=challenger_id,
                parent_champion_id=authority.champion_id,
                snapshot_id=snapshot.snapshot_id,
                payload_hash=semantic_hash(
                    {
                        "schema": "E2_TRAINING_CONTROL_RESULT_V1",
                        "generation": generation,
                        "parent": authority.champion_id,
                        "challenger": challenger_id,
                        "snapshot": snapshot.snapshot_hash,
                        "scientific_learning_claimed": False,
                    }
                ),
            )
            core.complete_training(training)
            dup_training = core.complete_training(training)
            if not dup_training.idempotent_duplicate:
                raise Stage3E2InvariantError("E2_DUPLICATE_TRAINING_NOT_IDEMPOTENT")
            duplicate_idempotent += 1

            validation = ValidationResult(
                work_id=f"validation-g{generation}",
                generation=generation,
                challenger_id=challenger_id,
                parent_champion_id=authority.champion_id,
                snapshot_id=snapshot.snapshot_id,
                validation_id=f"stage3-e2-validation-g{generation:08d}",
                payload_hash=semantic_hash(
                    {
                        "schema": "E2_VALIDATION_CONTROL_RESULT_V1",
                        "generation": generation,
                        "challenger": challenger_id,
                        "scientific_verdict": False,
                    }
                ),
            )
            core.complete_validation(validation)
            dup_validation = core.complete_validation(validation)
            if not dup_validation.idempotent_duplicate:
                raise Stage3E2InvariantError("E2_DUPLICATE_VALIDATION_NOT_IDEMPOTENT")
            duplicate_idempotent += 1

            decision = _decision_for_generation(generation)
            tournament = TournamentResult(
                work_id=f"tournament-g{generation}",
                generation=generation,
                challenger_id=challenger_id,
                parent_champion_id=authority.champion_id,
                validation_id=validation.validation_id,
                decision=decision,
                payload_hash=semantic_hash(
                    {
                        "schema": "E2_TOURNAMENT_CONTROL_RECEIPT_V1",
                        "generation": generation,
                        "decision": decision.value,
                        "scientific_verdict": False,
                    }
                ),
            )
            core.decide_tournament(tournament)
            dup_tournament = core.decide_tournament(tournament)
            if not dup_tournament.idempotent_duplicate:
                raise Stage3E2InvariantError("E2_DUPLICATE_TOURNAMENT_NOT_IDEMPOTENT")
            duplicate_idempotent += 1

            commit = core.commit_tournament()
            commit_again = core.commit_tournament()
            if commit_again != commit:
                raise Stage3E2InvariantError("E2_DUPLICATE_COMMIT_CHANGED_IDENTITY")
            duplicate_idempotent += 1

            expected_after = challenger_id if decision is TournamentDecision.PROMOTE else authority.champion_id
            if commit.next_champion_id != expected_after or commit.next_champion_hash != expected_after:
                raise Stage3E2InvariantError("E2_CHAMPION_AUTHORITY_VIOLATION_AT_COMMIT")

            checkpoint = core.seal_checkpoint(commit.next_champion_hash)
            if checkpoint.checkpoint_id != commit.next_champion_hash or checkpoint.commit_id != commit.commit_id:
                raise Stage3E2InvariantError("E2_CHECKPOINT_COMMIT_MISMATCH")

            next_authority = core.release_next_generation()
            attacks.append(
                _expect_rejected(
                    "duplicate_generation_release",
                    core.release_next_generation,
                    (IllegalTransition,),
                )
            )
            attack_rejections += 1

            if next_authority.generation != generation + 1:
                raise Stage3E2InvariantError("E2_NEXT_GENERATION_NUMBER_MISMATCH")
            if next_authority.champion_id != expected_after:
                raise Stage3E2InvariantError("E2_NEXT_GENERATION_CHAMPION_MISMATCH")
            if next_authority.teacher_authority_id != TEACHER_AUTHORITY_ID:
                raise Stage3E2InvariantError("E2_TEACHER_AUTHORITY_DRIFT_ON_RELEASE")
            if next_authority.physics_authority_id != PHYSICS_AUTHORITY_ID:
                raise Stage3E2InvariantError("E2_PHYSICS_AUTHORITY_DRIFT_ON_RELEASE")

            recovered = R11Orchestrator(
                authority=authority,
                evidence_store=evidence_store,
                journal=journal,
                checkpoint_store=checkpoints,
                recover=True,
            )
            _verify_recovered(
                recovered,
                authority,
                commit_id=commit.commit_id,
                checkpoint_id=checkpoint.checkpoint_id,
            )
            restart_checks += 1

            if decision is TournamentDecision.REJECT:
                rejected_challengers.add(challenger_id)
                if next_authority.champion_id == challenger_id:
                    raise Stage3E2InvariantError("E2_REJECTED_CHALLENGER_BECAME_AUTHORITY")

            persisted_evidence = evidence_store.get(frozen_evidence.evidence_id)
            persisted_snapshot = evidence_store.get_snapshot(snapshot.snapshot_id)
            if persisted_evidence != frozen_evidence:
                raise Stage3E2InvariantError("E2_EVIDENCE_LINEAGE_MUTATED")
            if persisted_snapshot != snapshot:
                raise Stage3E2InvariantError("E2_SNAPSHOT_OWNERSHIP_MUTATED")

            ledger.append(
                {
                    "generation": generation,
                    "authoritative_champion_before": authority.champion_id,
                    "authoritative_champion_hash_before": authority.champion_hash,
                    "challenger_id": challenger_id,
                    "challenger_hash": challenger_id,
                    "decision": decision.value,
                    "authoritative_champion_after": next_authority.champion_id,
                    "snapshot": {
                        "snapshot_id": snapshot.snapshot_id,
                        "snapshot_hash": snapshot.snapshot_hash,
                        "generation": snapshot.generation,
                        "parent_champion_id": snapshot.parent_champion_id,
                        "evidence_ids": list(snapshot.evidence_ids),
                        "immutable": snapshot.immutable,
                    },
                    "evidence_identity": {
                        "evidence_id": frozen_evidence.evidence_id,
                        "payload_hash": frozen_evidence.payload_hash,
                        "source_generation": frozen_evidence.source_generation,
                        "teacher_generation": frozen_evidence.teacher_generation,
                        "engineering_replay_only": True,
                        "new_scientific_evidence": False,
                    },
                    "teacher_authority_id": authority.teacher_authority_id,
                    "physics_authority_id": authority.physics_authority_id,
                    "commit_id": commit.commit_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "checkpoint_state_digest": checkpoint.state_digest,
                    "next_generation": next_authority.generation,
                    "duplicate_completion_identity": {
                        "training": dup_training.detail,
                        "validation": dup_validation.detail,
                        "tournament": dup_tournament.detail,
                        "commit_same_receipt": commit_again == commit,
                    },
                    "attack_receipts": attacks,
                    "restart_reconstruction": "PASS",
                }
            )
            authority = next_authority

        final_authority = authority

        journal_audit = journal_store.audit()
        if not journal_audit.get("pass", False):
            raise Stage3E2InvariantError(f"E2_JOURNAL_AUDIT_FAILURE:{journal_audit.get('problems')}")

        commit_count = int(checkpoint_store.conn.execute("SELECT COUNT(*) FROM orchestrator_commits").fetchone()[0])
        checkpoint_count = int(
            checkpoint_store.conn.execute("SELECT COUNT(*) FROM orchestrator_checkpoint_seals").fetchone()[0]
        )
        if commit_count != int(generations):
            raise Stage3E2InvariantError(f"E2_DUPLICATE_OR_MISSING_COMMIT_COUNT:{commit_count}:{generations}")
        if checkpoint_count != int(generations):
            raise Stage3E2InvariantError(
                f"E2_DUPLICATE_OR_MISSING_CHECKPOINT_COUNT:{checkpoint_count}:{generations}"
            )

        previous_after: str | None = None
        all_rejected: set[str] = set()
        for row in ledger:
            if previous_after is not None and row["authoritative_champion_before"] != previous_after:
                raise Stage3E2InvariantError(f"E2_LEDGER_PARENT_DISCONTINUITY:G{row['generation']}")
            if row["authoritative_champion_before"] in all_rejected:
                raise Stage3E2InvariantError(f"E2_REJECTED_CHALLENGER_LAUNDERED_IN_LEDGER:G{row['generation']}")
            if row["snapshot"]["generation"] != row["generation"]:
                raise Stage3E2InvariantError(f"E2_SNAPSHOT_GENERATION_OWNERSHIP_DRIFT:G{row['generation']}")
            if row["snapshot"]["parent_champion_id"] != row["authoritative_champion_before"]:
                raise Stage3E2InvariantError(f"E2_SNAPSHOT_PARENT_OWNERSHIP_DRIFT:G{row['generation']}")
            if row["decision"] == TournamentDecision.REJECT.value:
                all_rejected.add(row["challenger_id"])
                if row["authoritative_champion_after"] != row["authoritative_champion_before"]:
                    raise Stage3E2InvariantError(f"E2_REJECT_DID_NOT_RETAIN_CHAMPION:G{row['generation']}")
            elif row["authoritative_champion_after"] != row["challenger_id"]:
                raise Stage3E2InvariantError(f"E2_PROMOTE_DID_NOT_BIND_CHALLENGER:G{row['generation']}")
            previous_after = row["authoritative_champion_after"]

    except BaseException as exc:
        first_failure = {
            "generation": len(ledger),
            "event": "CONTROL_PLANE_SOAK",
            "exception": type(exc).__name__,
            "detail": str(exc),
            "taxonomy": _classify_failure(exc),
        }
    finally:
        freeze_after = _sha256_file(freeze_path) if freeze_path.is_file() else None
        try:
            if journal_audit is None:
                journal_audit = journal_store.audit()
        except BaseException as audit_exc:
            journal_audit = {
                "schema": "CB16_R11_EVENT_JOURNAL_AUDIT_V1",
                "pass": False,
                "problems": [{"error": type(audit_exc).__name__, "detail": str(audit_exc)}],
            }
            if first_failure is None:
                first_failure = {
                    "generation": len(ledger),
                    "event": "JOURNAL_AUDIT",
                    "exception": type(audit_exc).__name__,
                    "detail": str(audit_exc),
                    "taxonomy": "JOURNAL_AUDIT_FAILURE",
                }
        checkpoint_store.close()
        journal_store.close()

    if freeze_before is not None and freeze_after != freeze_before and first_failure is None:
        first_failure = {
            "generation": len(ledger),
            "event": "SEMANTIC_FREEZE_RECHECK",
            "exception": "Stage3E2InvariantError",
            "detail": f"E2_SEMANTIC_FREEZE_DRIFT:{freeze_before}:{freeze_after}",
            "taxonomy": "SEMANTIC_FREEZE_DRIFT",
        }

    elapsed = max(time.perf_counter() - start, 1e-12)
    passed = (
        first_failure is None
        and len(ledger) == int(generations)
        and bool(journal_audit and journal_audit.get("pass", False))
        and final_authority is not None
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "task": "STAGE3_E2_MULTI_GENERATION_CONTROL_PLANE_LINEAGE_SOAK",
        "verdict": "PASS" if passed else "FAIL",
        "scientific_semantics_changed": False,
        "scientific_verdict_created": False,
        "repeated_frozen_replay_role": "ENGINEERING_CONTROL_WORKLOAD_ONLY",
        "repeated_frozen_replay_counts_as_new_scientific_evidence": False,
        "final_holdout_opened": False,
        "fresh_market_data_downloaded": False,
        "target_generations": int(generations),
        "completed_generations": len(ledger),
        "correctness_identity": {
            "status": "CORRECTNESS_PASS" if passed else "CORRECTNESS_FAIL",
            "real_r11_orchestrator_transitions": True,
            "reset_to_g0_each_iteration": False,
            "semantic_freeze_sha256_before": freeze_before,
            "semantic_freeze_sha256_after": freeze_after,
            "teacher_authority_id": TEACHER_AUTHORITY_ID,
            "physics_authority_id": PHYSICS_AUTHORITY_ID,
            "frozen_replay_evidence_id": frozen_evidence.evidence_id,
            "frozen_replay_evidence_hash": frozen_evidence.payload_hash,
            "checkpoint_identity": "CONTENT_ADDRESSED_FP32_CONTROL_OBJECT",
            "amp": False,
        },
        "runtime_endurance": {
            "status": "CONTROL_PLANE_RUNTIME_PASS" if passed else "CONTROL_PLANE_RUNTIME_FAIL",
            "wall_seconds": elapsed,
            "logical_generations_per_second": len(ledger) / elapsed,
            "restart_reconstruction_checks": restart_checks,
            "attack_rejections": attack_rejections,
            "idempotent_duplicate_checks": duplicate_idempotent,
            "gpu_training_per_generation": False,
            "throughput_optimization_attempted": False,
        },
        "decision_coverage": {
            "promote": sum(1 for row in ledger if row["decision"] == "PROMOTE"),
            "reject": sum(1 for row in ledger if row["decision"] == "REJECT"),
            "repeated_promote_exercised": any(
                a["decision"] == b["decision"] == "PROMOTE" for a, b in zip(ledger, ledger[1:])
            ),
            "repeated_reject_exercised": any(
                a["decision"] == b["decision"] == "REJECT" for a, b in zip(ledger, ledger[1:])
            ),
            "alternating_exercised": any(a["decision"] != b["decision"] for a, b in zip(ledger, ledger[1:])),
        },
        "journal_audit": journal_audit,
        "failure_taxonomy": list(FAILURE_TAXONOMY),
        "first_failure": first_failure,
        "final_authority_identity": None if final_authority is None else _authority_obj(final_authority),
        "generation_ledger": ledger,
    }

    if report_path is not None:
        out = Path(report_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CB16 R11 Stage-3 E2 lineage soak")
    parser.add_argument("--root", required=True, help="Persistent E2 control-plane work directory")
    parser.add_argument("--repo-root", default=".", help="Repository root for freeze hash guard")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--synchronous", choices=("OFF", "NORMAL", "FULL", "EXTRA"), default="FULL")
    parser.add_argument("--report", required=True, help="Machine-readable JSON report path")
    args = parser.parse_args(argv)

    report = run_lineage_soak(
        args.root,
        generations=args.generations,
        repo_root=args.repo_root,
        synchronous=args.synchronous,
        report_path=args.report,
    )
    print(json.dumps({
        "schema": report["schema"],
        "verdict": report["verdict"],
        "completed_generations": report["completed_generations"],
        "target_generations": report["target_generations"],
        "final_authority_identity": report["final_authority_identity"],
        "first_failure": report["first_failure"],
    }, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
