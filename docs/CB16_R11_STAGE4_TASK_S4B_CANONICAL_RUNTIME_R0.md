# CB16 R11 Stage-4 S4B — Canonical Runtime Entrypoint & Lifecycle R0

## Scope

S4B adds a production-shaped process lifecycle boundary. It does not convert a Stage-2/Stage-3 qualification harness into production authority, and it does not implement authority adoption, lease/fencing, persistent state-root ownership, legacy retirement, recovery semantics, or scientific orchestration semantics.

Semantic contracts remain authority; legacy Python reachability is not authority. The scientific status remains `DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`.

## Canonical lifecycle

`BOOT -> VERIFY_AUTHORITY -> RECOVER_STATE -> ACQUIRE_AUTHORITY -> START_WORKERS -> RUNNING -> DRAIN -> SEAL -> STOPPED`

`FAILED` is a terminal fail-closed sink for runtime failures. Illegal transitions are rejected rather than translated into a legacy fallback.

Startup ordering is deliberate:

1. The authority-adoption boundary binds/verifies an already accepted authority identity.
2. The persistent-root provider verifies canonical state roots.
3. The legacy-retirement guard rejects non-canonical authoritative-write ownership.
4. Recovery reconstructs the existing accepted generation. S4B requires recovered authority ID and generation to equal the adopted binding; recovery may not manufacture a generation.
5. Authority lease/fencing is acquired and asserted current.
6. Only then may workers start and the controller enter `RUNNING`.

Shutdown ordering is `RUNNING -> DRAIN -> SEAL -> STOPPED`. The lease is asserted before drain and again before seal. The seal result is only an identity observation: S4B rejects any generation, authority, or scientific-history identity rewrite. Worker stop and lease release occur after a successful seal. Failure paths perform best-effort local worker stop and lease release and end in `FAILED`.

## Typed integration boundaries

`cb16_local_opt/stage4_canonical_runtime_r11.py` exposes Protocols for:

- `AuthorityAdoptionProvider` — later authority-adoption integration.
- `PersistentStateRootProvider` — later canonical root/storage integration.
- `LegacyRetirementGuard` — later legacy-authority retirement integration.
- `RecoveryProvider` — already-qualified recovery semantics; S4B adds only identity-preservation checks.
- `AuthorityLeaseProvider` — later local singleton/fencing integration; fencing-token mechanics are opaque to S4B.
- `OrchestrationProvider` — qualified R11 worker orchestration; S4B only orders start/drain/seal/stop.

S4B imports no sibling Stage-4 module. `build_fail_closed_runtime()` supplies a deliberately non-authoritative default whose first authority operation raises `IntegrationRequiredError`; there is no automatic fallback to `orchestrator_r11`, `burst_orchestrator_r11`, Stage-2/3 workflows, or legacy runtime.

## Qualification-harness separation

Stage-2/Stage-3 workflows remain qualification evidence and engineering harnesses. Prior burst/endurance/smoke workflows may exercise R11 components and upstream crash/restart contracts, but they are not invoked by the S4B entrypoint and do not receive canonical authority merely because existing Python imports/calls them.

`run_cb16_r11_stage4_runtime.py --describe-lifecycle` is inspection-only. Running it without final integration fails closed because S4B intentionally does not select concrete authority-bearing providers.

## Invariants covered by tests

- no `RUNNING` before all startup gates and authority acquisition pass;
- worker startup cannot precede authority acquisition;
- pre-worker gate failure starts no worker;
- partial worker startup unwinds worker state and releases acquired authority;
- recovery authority/generation must match the adopted binding;
- a mismatched lease cannot start workers;
- double start and illegal shutdown are rejected;
- normal shutdown drains before seal and stop;
- drain failure cannot reach seal;
- shutdown cannot change authority, generation, or scientific-history identity;
- fail-closed defaults have no legacy fallback;
- S4B imports no sibling Stage-4 implementation and contains no scientific verdict decision logic.

## Explicit non-goals / limitations

S4B does not qualify final Stage-4 cutover. Final integration must provide independently qualified concrete providers for adoption, local lease/fencing, canonical state roots, legacy retirement, recovery, and orchestration. S4B does not define process signals, daemon supervision, worker counts, queue depths, performance policy, distributed ownership, evidence semantics, permission semantics, tournament semantics, checkpoint semantics, generation advancement, or scientific truth.

No final holdout is opened, no fresh market data is downloaded, no frozen historical data is mutated, replay is not reinterpreted as evidence, and no new scientific verdict is produced by this task.
