# CB16 R11 Stage-4 INTB — Canonical Persistence & Writer Fencing R0

## Scope

INTB closes the canonical authoritative persistence path for Evidence references/training snapshots, the Event Journal, and checkpoint/Champion commit persistence. It composes the already-qualified S4D live fencing primitive, S4F canonical state-root ownership, and S4G negative legacy-retirement eligibility around the existing R11 protocol adapters.

This integration does not change Teacher, Evidence, Champion/Challenger, gradient, Permission, Physics, requested-risk, replay, or scientific verdict semantics. `DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE` remains unchanged.

## Canonical write admission

Every authoritative mutation is admitted only when all three independent gates hold:

1. The configured S4F control/data roots still verify against the startup receipt.
2. The S4G runtime identity is exactly `R11_CANONICAL_RUNTIME` and the requested canonical capability is eligible. S4G tokens remain `authority_granted=false` and are never positive write authority.
3. The typed `FencingTokenR11` is asserted by the currently live `Stage4AuthorityLeaseR11` immediately before the mutation.

A copied token string is never accepted. PID, TTL, and wall clock are not authority.

## Mutation coverage

The canonical adapters fence the actual Orchestrator protocol calls. INTB uses its exclusive existing-file authorization to add an optional mutation guard to `integration_adapters_r11.py`; the default remains `None`, preserving all pre-Stage-4 adapter behavior. Canonical construction injects the guard and reasserts it at the concrete SQLite/store mutation boundary:

- `EvidenceStoreProtocolAdapterR11.put_once`
- `EvidenceStoreProtocolAdapterR11.seal_snapshot`
- `EventJournalProtocolAdapterR11.append_once` (and `NEXT_GENERATION_RELEASED` additionally requires S4G `RELEASE_GENERATION` eligibility)
- `CheckpointStoreProtocolAdapterR11.atomic_commit`
- `CheckpointStoreProtocolAdapterR11.seal_checkpoint`

The canonical Evidence payload backend additionally fences `put_evidence`, `seal_evidence_set`, segment sealing, and WAL checkpoint mutations. The canonical Journal backend fences trace-batch/outcome seals. The canonical checkpoint backend additionally fences authoritative/reachable `CheckpointStoreR11.put_state_dict`, `seal_generation_checkpoint`, and WAL checkpoint mutations.

## S4F placement

Writable backends are opened only after exact S4F verification.

- Evidence metadata/indexes: `control/hot_indexes/evidence_store`
- Evidence immutable store bytes: `data/evidence_payloads/r11_evidence_store/...`
- Authoritative journal: `control/authoritative_journal_metadata/event_journal`
- Checkpoint SQLite/metadata/generation seals: `control/checkpoint_metadata/checkpoint_store`
- Immutable tensor checkpoint objects: `data/cold_content_artifacts/checkpoint_objects`
- S4D lease/fencing state must live under `control/runtime_lease_fencing_state/...`

The frozen/raw market root is passed only as an external read-only source root. Existing nonempty unclaimed legacy roots are never auto-adopted. S4F storage identity is integrity/placement authority only and does not admit scientific Evidence.

## Failure behavior

Stale fences fail before writer mutation, including after clean handoff and explicit dead-owner recovery. Legacy, replay, diagnostic, and test identities cannot bind canonical writer authority even when their Python methods are reachable. Partial backend-open failure closes every backend already opened and never falls back to legacy.

Orphan payloads, incomplete seals, content conflicts, swapped roots, unclaimed roots, path escape, and symlink traversal remain fail-closed under the S4F contract. Replay remains engineering-only and cannot obtain `MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE` eligibility.

## Ownership

INTB does not modify `evidence_store_r11.py`, `event_journal_r11.py`, or `checkpoint_store_r11.py`. It also does not alter the Orchestrator or scientific engines. INTB uses its sole existing-file exception only for `integration_adapters_r11.py`, adding an optional mutation-guard injection point without changing any serialization, identity, duplicate, commit, snapshot, or checkpoint semantics. Existing callers that omit the guard retain byte-for-byte logical behavior. `evidence_store_r11.py`, `event_journal_r11.py`, and `checkpoint_store_r11.py` remain unchanged; the S4F placement/fencing behavior is additive in `stage4_fenced_persistence_r11.py`.

## Final consolidation responsibilities

INTB PASS is not the final Stage-4 cutover verdict. Final consolidation must bind this persistence bundle to the single canonical runtime spine/recovery/orchestration providers, re-audit the S4A writer registry at the final head with `UNKNOWN_AUTHORITY=0`, execute the production S4H H01-H20 matrix against the consolidated runtime, and prove there is exactly one live canonical writer with stale/legacy writers unable to mutate. Final consolidation also retains responsibility for Permission/Physics closure, recovery identity, worker lifecycle, short canonical-machine lifecycle qualification, and the reserved final cutover adjudication.
