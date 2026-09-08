# CB16 R11 Stage-4 INTC — Engine / Challenger / Tournament Authority R0

## Scope

INTC is an additive authority-admission layer around the existing R11 `EngineBundle` contract. It wraps only the computational `execute(WorkItem) -> WorkCompletion` boundary for:

- Teacher acquisition
- H72 / Trace execution
- Champion policy inference
- Challenger training
- Validation
- Tournament computation

INTC does **not** own Evidence/EventJournal/Checkpoint persistence, Champion promotion persistence, Permission/Physics execution, worker lifecycle, recovery, or scientific-engine implementation.

Highest authority remains the semantic contracts. The scientific status remains `DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`.

## Authority gates

Every authoritative engine execution requires all of the following before the inner engine is called:

1. an S4B `RuntimeContext` whose binding/recovery/lease authority IDs agree;
2. runtime generation equality across S4B binding, recovered state, and the supplied R11 `AuthorityStamp`;
3. the current parent Champion ID from the supplied `AuthorityStamp` to equal the `WorkItem.parent_champion_id`;
4. an S4G-signed canonical runtime identity whose subject equals the S4B `authority_id`;
5. an independent S4G canonical identity for the concrete engine;
6. S4G negative eligibility for `ACQUIRE_CANONICAL_RUNTIME_AUTHORITY` for both identities;
7. an injected live-fence assertion against the opaque S4B `AuthorityLease`.

S4G is deliberately used only as a negative eligibility guard. Its admission tokens carry `authority_granted=false` and are never accepted as positive authority. The positive execution gate is the independent live-fence assertion.

INTC uses the S4G runtime-authority capability for all six compute domains because S4G does not define scientific-computation capabilities. INTC intentionally does **not** reinterpret Evidence-admission, snapshot-seal, Champion-transition, Permission, or Physics mutation capabilities as computation semantics.

## Pre/post fence rule

The live fence is asserted immediately before the inner `execute()` call and again immediately after it returns, before the `WorkCompletion` can be accepted. A handoff/stale-fence event during computation therefore causes the computed output to be rejected. The wrapper never falls back to a legacy engine.

The injected fence boundary is structurally compatible with the S4B `AuthorityLeaseProvider.assert_current(AuthorityLease)` contract. Final consolidation must bind it to the concrete live-fence provider that ultimately performs S4D live-owner/fencing validation; an opaque copied token is insufficient.

## Lineage validation

The wrapper never changes engine inputs or outputs. It verifies only authority/lineage identity:

- `WorkItem.kind` must match the wrapped engine domain.
- `WorkItem.generation` must equal the live authority generation.
- `WorkItem.parent_champion_id` must equal the live Champion ID.
- `WorkCompletion` must preserve work ID, work kind, and generation.
- poison-bearing authoritative completions are rejected.
- `PolicyResult`, `TrainingResult`, `ValidationResult`, and `TournamentResult` retain the existing R11 generation/parent-Champion/snapshot identities where those fields exist.

Teacher and Trace payload semantics are intentionally not reinterpreted here. Their result objects pass through unchanged and Evidence admission remains outside INTC.

## Training numeric identity

INTC does not set dtype, AMP, optimizer parameters, loss definitions, gradient owners, model architecture, or training math. Before `TRAIN_CHALLENGER` execution it requires the already-qualified training engine (or a narrow wrapper exposing `training_runtime` / `runtime`) to expose a config proving:

- `amp_enabled is False`
- `str(dtype) == "torch.float32"`

If this numeric identity cannot be observed, execution fails closed. There is no AMP/FP16/BF16/TF32/Triton/TorchInductor fast-path addition and no throughput tuning.

The inner R11 training runtime remains responsible for its frozen loss, gradient owner set, optimizer, clipping, seeds, and architecture checks.

## Tournament boundary

INTC accepts only a `TournamentResult` from the tournament engine. A `TournamentCommitProposal`, `CommitReceipt`, or other promotion object is not a valid tournament-engine output.

The wrapper exposes no promotion/commit method. Existing orchestration keeps tournament computation (`decide_tournament`) separate from the later persistent `checkpoint_store.atomic_commit()` transition. Champion promotion therefore remains a separately fenced control-plane persistence mutation owned outside INTC.

## Replay and noncanonical identities

`LEGACY_REPLAY`, legacy reference/oracle/compatibility, diagnostic, and test identities cannot pass canonical engine execution admission. Replay may remain engineering/recovery material, but INTC contains no Evidence writer and cannot reinterpret replay output as new scientific Evidence.

Python call reachability or implementation of the same `execute()` method is not authority.

## Final consolidation requirements

INTC intentionally leaves these bindings to final consolidation:

- bind `CanonicalEngineAuthorityContextR11.authority_stamp` to the exact Champion/generation identity recovered from the accepted S4C/S4F/Stage-3 state; INTC does not discover or manufacture that identity;
- bind the injected fence assertion to the consolidated INTA/S4D provider and prove the assertion ultimately validates the currently live S4D fence;
- bind each concrete production engine identity to the final re-audited S4A authority registry and S4G key custody;
- expose the already-qualified training runtime config through the concrete training engine adapter so INTC can prove FP32/AMP=false;
- keep all persistent Evidence, Journal, checkpoint, Champion promotion, generation release, Permission, Physics, and worker lifecycle paths outside this wrapper and fenced by their owning Integration tasks;
- execute the real S4H H01-H20 matrix and short canonical-machine lifecycle only after consolidation.

An INTC PASS is only an Integration-fork result. It is **not** `R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED`.
