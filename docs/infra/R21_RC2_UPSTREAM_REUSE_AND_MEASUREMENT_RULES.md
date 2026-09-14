# R21 RC2 Upstream Reuse and Measurement Rules

> Status: **SOL AUTHORITY SUPPLEMENT — MANDATORY FOR R0-R8 IMPLEMENTATION/REVIEW**  
> Date: 2026-09-13  
> Parent TODO: `docs/infra/R21_RC2_INFRA_UPGRADE_TODO.md`

This supplement closes a gap exposed during R4. The RC2 TODO already defines the sequential program dependency `R0 -> ... -> R8` and global source documents, but task-local upstream/reuse obligations were not explicit enough. Every RC2 task must now identify and inspect relevant accepted code/authority before creating a replacement implementation.

## 1. Mandatory pre-implementation reuse audit

Before coding, each DS task must emit a short `UPSTREAM_REUSE_AUDIT` containing:

- direct upstream task receipts/specs;
- accepted runtime/authority SHAs that constrain the task;
- existing code paths that already implement the needed behavior;
- existing CI/workflow/preflight helpers to reuse;
- historical implementation or architecture references worth consulting;
- explicit `REUSE_AS_IS / REUSE_WITH_ADAPTER / EXTEND / NEW / DO_NOT_TOUCH / LEGACY_REFERENCE_ONLY` classification;
- reason any new implementation is necessary.

A task must not introduce a new helper merely because the existing implementation was not searched first.

## 2. Task-local upstream map

| Task | Direct upstream | Required current references | Historical/reuse references |
|---|---|---|---|
| R0 | Owner/Astra RC2 decisions | `R21_RC2_INFRA_UPGRADE_TODO.md`, current S1 freeze/authorization | qualification framework and prior R11 authority patterns |
| R1 | R0 | `SHANXI_DOCKER_RUNNER_CONTRACT.md`, machine snapshot, shared Shanxi preflight | existing runner/provision workflows and prior Docker handoff artifacts |
| R2 | R0/R1 | live runner/storage contract, R2 capacity authority produced by task | historical R2 evidence-storage architecture and HDD/SSD qualification work |
| R3 | accepted R1 + owner-approved R2 plan | versioned runner definition, exact approved FAST_HOT plan | current R11 runner as rollback/reference; historical R2 hot/cold storage rules |
| R4 | accepted R3 execution surface + accepted S1 runtime identity | exact S1 authorization head, `scripts/run_r11_post_cc_s1_learnability.py`, `cb16_local_opt/post_cc_s1_qualification_v1.py`, existing durable observation/replay/update/checkpoint/provenance writers | historical R2 storage instrumentation/qualification utilities where applicable; existing S1 provenance/audit outputs |
| R5 | accepted R4 measurement + R3 surface | accepted S1 bounded canary and provenance audits | previous CI-B/R4 S1 engineering-smoke evidence |
| R6 | Sol decision based on accepted R4/R5 | exact writer/transaction paths proven by R4 | existing SQLite/durable-store implementation first; no storage redesign without measured need |
| R7 | accepted R3-R6 state | exact implementation identity under qualification, qualification framework primitives | earlier restart/recovery/exactly-once hostile tests and artifact-only provenance patterns |
| R8 | all accepted Recovery evidence | complete task receipts, exact SHA/tree/profile identities | prior Sol qualification-review protocol |

The list is a minimum, not permission to ignore a more relevant existing component discovered during implementation.

## 3. R4-specific measurement integrity rules

R4 is measurement-only. It must not silently invent a measurement model whose errors can decide R5/R6.

### 3.1 Discover first, instrument second

Before adding generic proxies, R4 must inspect the accepted S1 runtime's actual producers for:

- observation/object persistence;
- SQLite indexes/WAL use;
- replay materialization;
- update journal;
- checkpoint store;
- generation-switch receipts;
- provenance/export staging.

The report must bind each measured surface to the producer/consumer code path or state that attribution is unknown.

### 3.2 Instrumentation canary must prove exactness, not only presence

A canary that checks only `count > 0` is insufficient for quantitative metrics.

For known writes, the canary must verify where applicable:

- expected call count exactly matches observed count;
- expected byte count exactly matches observed byte count;
- the same logical write is not counted twice through overlapping wrappers (`builtins.open`, `io.open`, `Path.open`, `os.write`, etc.);
- metrics-output self-writes are excluded;
- an intentionally out-of-root write is either captured as `other` or explicitly declared out of scope;
- atomic temp-write + replace paths are attributed consistently.

If exactness cannot be demonstrated for a metric, that metric must be marked `ESTIMATED`/`LIMITED` and must not be used as a hard selection gate.

### 3.3 Direct measurements and estimates must remain separate

Do not merge a derived SQLite size estimate into the same field as directly intercepted write bytes without preserving separate fields.

At minimum distinguish:

- directly observed Python/application write bytes;
- directly observed write/fsync/commit call counts;
- SQLite logical/page-growth estimate;
- WAL file-size/growth estimate;
- device-level sectors/bytes written.

An estimate may support diagnosis, but it must not be presented as exact physical bytes written.

### 3.4 Root filters cannot define reality

R4 exists to discover the real write path. A predeclared monitored-root allowlist may be used for classification, but it must not cause unknown writes to disappear silently.

The measurement layer should either:

- capture all relevant write-like operations and classify unknown paths as `other`; or
- prove by an independent coverage canary that the configured roots encompass every required producer path.

Only the instrumentation's own metrics directory may be excluded unconditionally.

### 3.5 R4 PASS meaning

R4 `PASS` means the required surfaces were observed with measurement integrity sufficient for the specific metrics used downstream. It does **not** mean every byte metric is exact.

If required surface presence is proven but quantitative integrity is not, use `EVIDENCE_INSUFFICIENT` for any downstream decision that depends on the unproven quantitative metric.

## 4. Review consequence

Sol must review the `UPSTREAM_REUSE_AUDIT` and measurement canary before accepting each task. A green GitHub Actions run is not sufficient when the measurement method itself is under-specified or can double-count/miss writes.

This supplement does not change S1 scientific semantics and does not authorize any new host mutation.