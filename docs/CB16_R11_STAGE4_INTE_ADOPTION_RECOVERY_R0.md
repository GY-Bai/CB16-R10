# CB16 R11 Stage-4 INTE — Adoption / Restart / Recovery Identity R0

## Scope

INTE supplies the concrete `RecoveryProvider` boundary consumed by S4B. It composes only already-qualified primitives present in the immutable Integration Seed: S4C exact authority-adoption verification, S4F canonical target-root verification, a read-only Stage-3 recovery identity inspector, and S4D explicit dead-owner fencing recovery.

INTE does not start workers, open or rewire authoritative writers, train, run tournaments, grant Permission, execute Physics, promote a Champion, advance a scientific generation, create Evidence, or create a scientific verdict.

Scientific status remains `DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`.

## Recovery ordering

`Stage4RecoveryProviderR11.recover_existing_state()` fails closed in this order:

1. Validate the S4B `AuthorityBinding` against the S4C target authority, accepted source generation, and Semantic Freeze identity.
2. Load and contract-bind the published S4C adoption receipt. Missing, partial, conflicting, or drifted adoption is rejected before recovery inspection.
3. Verify exact expected S4F control/data root IDs and frozen-raw identity. S4F remains responsible for rejecting corrupt roots, partial/torn objects, orphan payloads, incomplete seals, symlinks, escapes, and content conflicts.
4. Ask the injected Stage-3 read-only recovery inspector for a fully sealed observation and compare its complete `SourceAuthorityIdentity` to the already accepted S4C source: source SHA, Semantic Freeze, generation, Champion identity/hash, checkpoint identity/hash, Evidence-root identity, journal-head identity, and checkpoint-root identity.
5. Require journal/checkpoint transitions to be complete and their recovery audits to pass. Reject replay-as-new-Evidence, new Evidence, scientific-history rewrite, or a new scientific verdict.
6. Inspect S4D durable lease metadata. An `ACTIVE` residue in ordinary restart mode is rejected. Only explicit dead-owner mode calls `recover_after_dead_owner()`. S4D must prove the kernel lock is free and advance the fencing epoch; PID and wall clock are not authority.
7. Release the temporary recovery lease before S4B's later authority-acquisition phase. S4B must subsequently acquire a fresh live fence; the crashed token remains stale.
8. Return `RecoveredRuntimeState` without inventing identity: `state_id` is the accepted S4C checkpoint identity and `scientific_history_id` is the accepted S4C journal-head identity. Generation remains the accepted source generation.

## Stage-3 inspector boundary

`Stage3RecoveryInspectorR11` is deliberately read-only. INTE has no existing-file ownership for `integration_adapters_r11.py`, journal/checkpoint/evidence stores, or `orchestrator_r11.py`. Final consolidation must bind this protocol to the canonical read side supplied by persistence integration and the already-qualified Stage-3 recovery audit. An inspector PASS is only a negative gate; it never grants writer authority.

## Hostile coverage

The INTE suite covers crash before/after S4C publication; identical/conflicting adoption; source/freeze/generation/Champion/checkpoint/root identity drift; incomplete journal/checkpoint transitions; failed audits; corrupt S4F root markers; orphan S4F objects; crashed durable `ACTIVE` ownership; ordinary-acquire refusal; explicit epoch-advancing recovery; stale tokens; second restart; corrupt S4D metadata; replay/Evidence/history/verdict semantic attacks; and control-namespace placement.

## Final consolidation requirements

INTE PASS is not the Stage-4 cutover verdict. Final consolidation must still bind the Stage-3 inspector to real consolidated canonical persistence readers; compose INTE with the final authority spine; ensure INTB asserts the live S4D fence immediately before every authoritative mutation; ensure workers start only after every authority gate; execute real S4H H01-H20 against the consolidated runtime; rebuild/re-audit the S4A writer registry with `UNKNOWN_AUTHORITY = 0`; prove stale/legacy writers and Permission bypasses fail closed; run the short canonical-machine lifecycle with FP32 and AMP=false; and prove Semantic Freeze/final holdout/frozen market data/scientific status are unchanged.

Only the final consolidated adjudicator may emit `R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED`.
