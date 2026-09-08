# CB16 R11 Stage-4 INTF — Worker Lifecycle / Orchestration Provider R0

## Scope

INTF implements the concrete `OrchestrationProvider` boundary consumed by
`CanonicalRuntimeControllerR11`. It owns worker-facility startup, RUNNING
admission, drain, engineering-only runtime seal observation, stop, and
partial-startup cleanup.

It does not own authority adoption, S4D fencing mechanics, authoritative
persistence, scientific engine meaning, Champion promotion, Permission,
Frozen Supervisor, or Frozen Physics.

Authority is injected through `LiveAuthorityAssertionR11`. Final consolidation
must bind that protocol to the current canonical live-fence assertion. INTF
does not infer authority from PID, reachability, TTL, wall clock, or an opaque
string.

## Lifecycle

The provider-local lifecycle is:

`NEW -> STARTING -> RUNNING -> DRAINING -> DRAINED -> SEALED -> STOPPED`

Any startup, drain, seal, or cleanup fault fails closed into `FAILED`.

Before the first worker facility starts, between component starts, and before
RUNNING is accepted, the injected live-authority boundary is asserted. The
provider also asserts live authority around drain and on both sides of the
engineering seal observation. Cleanup itself never depends on authority being
live: stale authority must not prevent workers from being stopped.

Components are registered for cleanup before their `start()` call. Therefore a
component that allocates resources and then raises during start is still
included in reverse-order partial-start cleanup.

## Authoritative work admission and drain

`admit_authoritative_work()` creates an opaque ownership ticket only while the
provider is RUNNING and the current authority assertion succeeds.

`drain_workers()` atomically closes new admission, enters DRAINING, invokes each
component's `begin_drain()`, waits for all already-owned bounded tickets to
complete, invokes `await_drained()`, reasserts live authority, then enters
DRAINED.

This is a lifecycle/admission barrier. It does not change Teacher, Trace,
Training, Validation, Tournament, Evidence, checkpoint, or Champion semantics.

## Existing qualified worker facilities

`ManagedCloseableWorkerComponentR11` adapts already-qualified facilities that
own a `close()` cleanup boundary.

Two additive builders are supplied:

- `build_teacher_worker_component_r11()` creates the existing
  `TeacherWorkerPoolR11` with the canonical 8-worker setting.
- `build_fork_trace_worker_component_r11()` creates the existing
  `ForkProcessTraceRuntimeR11` with the canonical 8-worker setting.

The fork Trace runtime remains lazy exactly as qualified: INTF does not force a
new Trace computation merely to spawn child processes. Final engine assembly
may use the managed component resource only inside an INTF-authorized work
ownership boundary.

## Numeric/runtime identity

INTF fixes the accepted Stage-4 short-smoke defaults:

- Teacher workers: 8
- Trace workers: 8
- Queue depth: 8
- Buffer: 64 MiB
- Nested numeric-library threads: 1
- Numeric mode: FP32
- AMP: false

INTF does not add AMP, FP16, BF16, TF32, Triton, TorchInductor assumptions, or
throughput tuning.

## Runtime seal semantics

`RuntimeSealObservation` is populated from the already-existing authority,
generation, and scientific-history identities in the S4B `RuntimeContext`.
The default seal observer hashes those existing identities plus engineering
topology into an `engineering-runtime-seal:*` identifier.

The seal cannot advance a generation, promote a Champion, create scientific
Evidence, or rewrite scientific history.

Replay remains:

`ENGINEERING_RECOVERY_ONLY__NOT_NEW_SCIENTIFIC_EVIDENCE`

The scientific status remains:

`DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`

## Fail-closed cases covered

The dedicated tests cover:

- no/stale live authority before worker start;
- partial startup cleanup, including the component that fails mid-start;
- double start;
- drain closing new authoritative admission;
- drain waiting for already-owned bounded work;
- seal before drain rejection;
- stop before seal rejection;
- failed seal cleanup and idempotent S4B cleanup re-entry;
- stale authority after startup;
- wrong runtime generation/context;
- authority/generation/scientific-history preservation in the seal;
- restart identity preservation with replay still engineering-only;
- FP32 / AMP=false / canonical topology lock;
- closeable worker cleanup and orphan-worker assertion.

## Known limitations / final consolidation binding

INTF deliberately does not import any INTA-INTH sibling implementation.

Final consolidation still must:

1. Bind `LiveAuthorityAssertionR11` to the actual canonical S4D/authority-spine
   fence validation, not to an opaque string comparison.
2. Assemble real Teacher/Trace/Training/Validation/Tournament engine calls so
   every authoritative submission is bracketed by INTF authoritative-work
   ownership while preserving the engine wrappers' own authority checks.
3. Bind persistence through the separately fenced canonical writer path; INTF
   has no writer authority.
4. Bind recovery through the qualified Stage-4 recovery provider; INTF does not
   reinterpret restart or replay as Evidence.
5. Execute the consolidated hostile H01-H20 matrix and final short canonical
   startup/steady/drain/shutdown qualification.
6. Re-audit the final S4A writer registry and prove UNKNOWN_AUTHORITY=0.

INTF PASS is only fork-level integration evidence. It does not emit or imply
`R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED`.
