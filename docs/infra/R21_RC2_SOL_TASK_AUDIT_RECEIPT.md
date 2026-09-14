# R21 RC2 Sol Task Audit Receipt

Status: SOL AUTHORITY UPDATED — UPSTREAM/REUSE SUPPLEMENT MANDATORY

Reviewed scope:

- `docs/infra/R21_RC2_INFRA_ARCHITECTURE_RESEARCH.md`
- `docs/infra/R21_RC2_INFRA_UPGRADE_TODO.md`
- `docs/infra/R21_RC2_UPSTREAM_REUSE_AND_MEASUREMENT_RULES.md`

Audit conclusions:

- Recovery and Evolution are separate acceptance stages.
- Recovery is the only direct dependency of frozen S1 CI-C.
- SQLite + SSD is evaluated before storage challengers.
- Storage challengers are evidence-triggered rather than implemented in parallel.
- Host access requirements are explicit per Recovery task.
- R3 is the first Recovery task that requires authorized Shanxi host/container configuration changes.
- R2 is read-only and must produce an owner-approved SSD capacity/provision plan before R3.
- R6 is conditional and creates a new runtime implementation identity.
- R7 distinguishes process-level and container-level recovery; stronger fault classes are not claimed by default.
- S1 scientific identity remains frozen throughout Recovery.
- Every R0-R8 task must now provide a task-local `UPSTREAM_REUSE_AUDIT` before implementation/review.
- Existing code and historical implementations must be classified as `REUSE_AS_IS / REUSE_WITH_ADAPTER / EXTEND / NEW / DO_NOT_TOUCH / LEGACY_REFERENCE_ONLY` before introducing replacement helpers.
- R4 quantitative instrumentation must prove coverage and counting integrity, not merely nonzero observation.
- Directly observed write bytes, SQLite/page/WAL estimates, and device-level write counters must remain distinguishable; estimates must not masquerade as exact physical bytes.

R4 review note:

The R4 episode exposed the previous documentation gap. A monitored-root filter initially caused provenance/output writes to disappear while fsync/SQLite events remained visible. The subsequent broad write interception is directionally appropriate for discovery, but a green run is not sufficient by itself. Sol review must verify that overlapping Python I/O wrappers cannot double-count the same logical write and that any SQLite byte estimate is explicitly separated from exact write-byte metrics before R4 evidence is used to open or skip R6.

Authority consequence:

`docs/infra/R21_RC2_UPSTREAM_REUSE_AND_MEASUREMENT_RULES.md` is a mandatory supplement to the parent RC2 TODO for R0-R8 implementation and Sol review. Where the supplement is stricter about reuse/measurement proof, the stricter requirement applies.
