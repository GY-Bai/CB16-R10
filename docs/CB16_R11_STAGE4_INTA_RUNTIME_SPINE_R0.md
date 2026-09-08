# CB16 R11 Stage-4 INTA — Canonical Runtime Authority Spine R0

## Scope

INTA integrates only the process-level authority spine on top of the immutable
Integration Seed `86a4ac8a9080cd8382600cb998059e9585495a11`.

It composes already-qualified Wave-1 contracts:

1. S4B `CanonicalRuntimeControllerR11` lifecycle.
2. S4C one-time authority adoption receipt verification.
3. S4D single-machine lease and fencing.
4. S4F canonical state-root startup verification.
5. S4G legacy-retirement negative eligibility policy.

Recovery is injected through the S4B `RecoveryProvider` boundary (INTE owns the
real recovery bridge). Orchestration is injected through the S4B
`OrchestrationProvider` boundary (INTF owns worker lifecycle). INTA does not
implement domain writers, workers, Permission/Physics execution, training,
tournaments, or full recovery.

Scientific status remains:

`DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`

No new scientific verdict is created.

## Concrete adapters

`Stage4AuthorityAdoptionProviderR11`

- Loads an already-published S4C receipt using the independently supplied
  `AuthorityAdoptionContract`.
- S4C verifies the complete accepted source identity, including source SHA,
  Semantic Freeze identity, source generation, Champion/checkpoint identities
  and hashes, Evidence root, journal head, and checkpoint-root identity.
- Returns an S4B `AuthorityBinding` only after contract-bound receipt validation.
- `AuthorityBinding.generation` is exactly `source_generation`; adoption never
  advances a generation.

`Stage4StateRootProviderR11`

- Calls S4F `verify_startup()` with exact expected control/data root IDs.
- Requires the S4F frozen raw identity and Semantic Freeze blob identity.
- S4F remains responsible for rejecting swapped/wrong markers, overlapping
  roots, symlinks, unclaimed roots, partial/orphan state, and writable frozen
  raw authority.
- A storage verification receipt is integrity identity, not scientific Evidence.

`Stage4LegacyRetirementProviderR11`

- Requires a signed S4G `R11_CANONICAL_RUNTIME` identity bound to the same S4B
  authority ID.
- Requires S4G eligibility for `ACQUIRE_CANONICAL_RUNTIME_AUTHORITY`.
- Explicitly requires `authority_granted == false` and
  `requires_independent_runtime_authority == true`.
- Therefore a retirement token can deny eligibility but can never grant the
  positive runtime authority supplied by S4D.

`Stage4AuthorityLeaseProviderR11`

- Retains the real `FencingTokenR11` object in process-local state.
- Exposes only an opaque S4B lease handle.
- Every `assert_current()` resolves the handle back to that retained token and
  calls S4D `assert_fencing_token()`. The live local kernel-lock holder and the
  matching durable epoch/nonce must still be current.
- PID, wall clock, TTL, and an opaque copied lease string are not authority.
- Ordinary acquisition is the default and preserves S4D's fail-closed behavior
  when durable metadata is still `ACTIVE` after a crash.
- `EXPLICIT_DEAD_OWNER_RECOVERY` is a distinct construction mode that invokes
  only S4D's already-qualified `recover_after_dead_owner()` primitive. INTA
  does not decide when that mode is legitimate; INTE/final consolidation must
  choose it only after its recovery identity checks.

`CanonicalRuntimeAuthoritySpineR11`

- Is only a provider bundle/factory for `CanonicalRuntimeControllerR11`.
- Recovery and orchestration must be supplied independently.

## Lifecycle proof

The S4B controller plus these providers enforce:

`S4C adoption/verification`
→ `S4F root verification`
→ `S4G legacy-retirement eligibility`
→ injected `recovery`
→ `S4D authority acquisition`
→ `S4D current-fence assertion`
→ injected `worker start`
→ second current-fence assertion
→ `RUNNING`.

No worker starts before all authority gates pass.

On startup failure after a structurally valid lease exists, S4B cleanup stops
any attempted workers and invokes lease release. The INTA lease adapter releases
the retained real S4D token; there is no legacy fallback path.

Shutdown remains the S4B sequence of fence assertion, drain, fence assertion,
runtime seal observation, worker stop, and lease release. S4B rejects a runtime
seal that rewrites recovered generation or scientific-history identity.

## Hostile / fail-closed coverage

The INTA tests exercise:

- duplicate start of one controller;
- a second simultaneous S4D authority owner;
- wrong S4C target identity;
- wrong S4C generation;
- wrong, swapped, unclaimed, and overlapping S4F roots;
- legacy identity requesting canonical runtime authority;
- stale S4D fencing token;
- copied opaque S4B lease with no live provider binding;
- current-fence assertion failure after acquisition and cleanup;
- worker-start failure cleanup;
- ordinary acquisition against crashed durable `ACTIVE`;
- explicit dead-owner recovery with fencing-epoch advance;
- recovery generation manufacture attempt;
- shutdown generation rewrite attempt;
- shutdown scientific-history rewrite attempt.

## Non-goals and final-consolidation requirements

INTA does **not** claim `R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED`.

Final consolidation still must:

- bind the real INTE recovery provider and determine when explicit dead-owner
  recovery is authorized;
- bind the real INTF orchestration provider;
- bind INTB fenced domain writers so every authoritative mutation checks the
  live S4D fence immediately before mutation;
- bind INTC engine wrappers and INTD Permission/Physics execution closure;
- run INTG/S4H H01-H20 against the assembled production runtime;
- rebuild/re-audit the S4A writer registry at final head and require
  `UNKNOWN_AUTHORITY = 0`;
- prove one live canonical writer, stale/legacy writer exclusion, no Permission
  bypass, and replay-not-Evidence;
- run the short canonical-machine startup/steady/drain/shutdown qualification
  with FP32 canonical and `AMP=false`;
- re-prove Semantic Freeze exact, final holdout beginning 2025-09 untouched,
  no fresh market data, and unchanged scientific status.

Long endurance qualification is outside this Integration round.
