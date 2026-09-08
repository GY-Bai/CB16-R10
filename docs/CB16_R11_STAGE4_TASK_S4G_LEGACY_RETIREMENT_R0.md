# CB16 R11 Stage-4 S4G — Legacy Runtime Retirement & Authoritative-Write Rejection R0

## Scope

S4G implements a **negative legacy-retirement policy**. It does not delete legacy code and does not make a new implementation scientifically authoritative. Semantic contracts remain authority; legacy Python implementation is not authority.

This task is infrastructure/runtime authority work only. It does not open the final holdout, download market data, mutate frozen history, create Evidence, create a scientific verdict, change Physics/Supervisor/Teacher/Evidence/gradient/Champion-Challenger/requested-risk semantics, or reinterpret replay as Evidence.

Scientific status remains:

`DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`

## Upstream authority inspected read-only

S4G was built from Gatework base `0f18e08ec7250b9b4e45c62803c25be966834390` and inspected the Stage-4 Gatework manifest/document/receipt schema/validator, Semantic Freeze, and the existing R11 authority/runtime persistence surfaces. Relevant existing paths inspected include:

- `cb16_local_opt/rearchitecture_authority_r11.py` — explicitly treats legacy Python as non-authoritative and retains a legacy Brain only as an oracle for frozen functional shape.
- `cb16_local_opt/runtime_protocols_r11.py` — protocol surface for Evidence store, EventJournal and checkpoint/Champion-Challenger commit/seal operations.
- `cb16_local_opt/orchestrator_r11.py` — Evidence admission, training-snapshot sealing, tournament commit, checkpoint and generation lifecycle orchestration.
- `cb16_local_opt/event_journal_r11.py` — authoritative-style durable event/generation outcome mutation surface.
- `cb16_local_opt/evidence_store_r11.py` — immutable Evidence persistence/materialization surface.
- `cb16_local_opt/checkpoint_store_r11.py` — checkpoint objects and generation/Champion seal surface.
- `cb16_local_opt/r102_runtime_authority.py` — earlier runtime authority/performance binding retained as legacy/reference context, not promoted by reachability.
- `cb16_local_opt/r2_legacy_migration.py` — explicit read-only legacy migration/reference path.
- `cb16_local_opt/stage3_e3_crash_recovery_r11.py` — Stage-3 qualification harness using synthetic/frozen engineering replay canaries, not new scientific learning evidence.

No S4A–S4I sibling branch implementation was read, imported, merged, or cherry-picked.

## Policy model

`cb16_local_opt/stage4_legacy_retirement_r11.py` defines seven roles:

| Role | Intended use | Production write authority |
|---|---|---|
| `R11_CANONICAL_RUNTIME` | candidate canonical runtime identity | Not granted by S4G; must pass independent final authority gates |
| `LEGACY_REFERENCE` | read-only reference | denied |
| `LEGACY_ORACLE` | read/reference + oracle comparison | denied |
| `LEGACY_REPLAY` | engineering replay only | denied |
| `LEGACY_COMPATIBILITY_READER` | historical compatibility reads | denied |
| `DIAGNOSTIC` | diagnostics | denied |
| `TEST` | synthetic/unit qualification | denied |

The nine authoritative capabilities covered are:

1. `ACQUIRE_CANONICAL_RUNTIME_AUTHORITY`
2. `MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE`
3. `APPEND_AUTHORITATIVE_EVENT_JOURNAL`
4. `SEAL_TRAINING_SNAPSHOT`
5. `TRANSITION_CHALLENGER_CHAMPION`
6. `SEAL_CHECKPOINT`
7. `RELEASE_GENERATION`
8. `GRANT_PERMISSION`
9. `EXECUTE_ACCOUNT_PHYSICS_TRANSITION`

Every non-canonical role is denied every capability above.

Allowed non-authoritative operations are deliberately narrow: legacy reference reads, legacy oracle comparison, engineering replay, historical compatibility reads, and diagnostics. A role cannot silently use another role's safe capability.

## Admission token semantics

S4G signs `RuntimeIdentity` and `RetirementCapabilityToken` objects with HMAC-SHA256 to make role/capability/subject/issuer tampering fail closed. The signing key is an integration-controlled integrity secret; S4G does not define distributed identity or key distribution.

A valid `RetirementCapabilityToken` always carries:

- `admission_only = true`
- `authority_granted = false`

Therefore an S4G token means only: **this identity/capability pair is not rejected by the legacy-retirement policy**. For an authoritative capability, the returned decision also sets `requires_independent_runtime_authority = true`.

A token is never a lease, fencing token, Permission, checkpoint seal, Evidence receipt, journal receipt, generation-release receipt, or scientific authority.

## Fail-closed cases

The implementation and tests cover:

- all 6 non-canonical roles × all 9 authoritative capabilities denied;
- legacy reference/oracle remain usable for read/comparison;
- legacy replay remains `ENGINEERING_REPLAY` and cannot mint/admit authoritative Evidence;
- failed canonical startup cannot fall back to legacy authority;
- even a successful startup cannot nominate a legacy fallback identity;
- forged role changes invalidate the signed identity;
- forged capability changes invalidate token binding/signature;
- tokens cannot be replayed by another subject identity;
- active legacy authoritative-writer detection fails closed;
- unknown role/capability declarations fail closed;
- active canonical claims require a valid S4G admission token but still receive no positive authority from S4G;
- diagnostic/test roles cannot escalate;
- inactive historical declarations stay historical metadata and do not become live authority.

## Audit CLI

Run:

```bash
python scripts/audit_r11_stage4_legacy_authority.py --self-check
```

The self-check exhaustively attempts the non-canonical-role × authoritative-capability matrix with synthetic identities.

An optional static declaration file can be checked with:

```bash
python scripts/audit_r11_stage4_legacy_authority.py --claims-json /path/to/claims.json
```

Format:

```json
{
  "claims": [
    {
      "subject_id": "legacy-worker-1",
      "role": "LEGACY_REPLAY",
      "capability": "APPEND_AUTHORITATIVE_EVENT_JOURNAL",
      "active": true
    }
  ]
}
```

Static declarations **never establish authority**. They are only a fail-closed detector for unknown or prohibited active claims.

## Final integration contract

Final Stage-4 integration should place S4G before every authoritative mutation consumer discovered by the integrated authority inventory. For an authoritative write to proceed, final integration must require all applicable positive gates **in addition to** S4G, including canonical lifecycle/adoption/lease-fencing/storage/permission controls from their independently qualified interfaces.

Integration must never implement:

```text
if canonical_startup_failed:
    start_legacy_as_authority()
```

Instead, canonical startup failure is terminal/fail-closed for authoritative operation.

Existing legacy modules may remain installed/importable for reference, oracle, engineering replay, diagnostics, and compatibility reading. Import/call reachability is not authority. A legacy path must not be promoted merely because canonical Python can import or call it.

## Known limitations / explicit non-claims

- S4G does not discover the repository-wide writer inventory; final integration supplies discovered identities/claims.
- S4G does not implement or qualify singleton lease/fencing.
- S4G does not implement canonical runtime lifecycle or startup sequencing.
- S4G does not perform one-time authority adoption.
- S4G does not define persistent state roots.
- S4G does not modify permission adapters or Physics/Supervisor semantics.
- S4G does not rewire existing writers; this Wave-1 task is additive only.
- HMAC admission tokens are local integrity bindings. They are not sufficient positive runtime authority and are only as trustworthy as the final integration's key custody and identity provisioning.
- A declared canonical writer in the static CLI is reported as requiring independent runtime proof; the static declaration itself is never accepted as authority.
- Passing S4G qualifies only this retirement guard. It does **not** qualify final Stage-4 canonical authority cutover.
