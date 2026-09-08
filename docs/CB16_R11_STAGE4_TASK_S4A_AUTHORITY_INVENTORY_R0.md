# CB16 R11 Stage-4 S4A — Authority Surface Inventory R0

## Scope

S4A is an administrative/static authority inventory only. It does not retire, rewire, invoke, or grant runtime authority. The governing rule is:

`SEMANTIC_CONTRACTS_ARE_AUTHORITY__LEGACY_PYTHON_IMPLEMENTATION_IS_NOT_AUTHORITY`

Gatework base: `0f18e08ec7250b9b4e45c62803c25be966834390`.

Scientific status remains:

`DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`.

No final holdout is opened, no fresh market data is downloaded, no historical market data is mutated, and no new Evidence or scientific verdict is created by this task.

## Inventory model

`cb16_local_opt/stage4_authority_inventory_r11.py` statically discovers repository-local Python functions/methods and Stage-2/3 workflow entrypoints that directly or indirectly touch one or more of these mutation domains:

- Evidence mint/admission
- authoritative EventJournal
- training snapshot seal
- Challenger creation
- tournament result
- Champion commit/promotion/rejection
- checkpoint seal
- generation advancement/release
- runtime authority ownership
- permission grant
- Physics/account transition

Discovery signals include named mutation APIs, persistent writes in authority-bearing contexts, optimizer parameter mutation, transitive callsites, recovery paths, adapters, CLI/qualification entrypoints, and Stage-2/3 workflows. Discovery is deliberately conservative: false-positive surfaces must be classified, not silently ignored.

The unit of registry identity is `(repository path, symbol/function/class/workflow)`.

## Classification policy

`stage4_authority_registry_builder_r11.py` contains an explicit path review table. It does not infer authority from import or call reachability. A newly discovered path absent from the review table becomes `UNKNOWN_AUTHORITY`, which makes the audit fail.

Classifications:

- `R11_CANONICAL_CANDIDATE`: current contract-governed R11 writer or Frozen Physics/Supervisor implementation surface eligible for later integrated cutover review. **Candidate is not canonical authority.**
- `LEGACY_REFERENCE_ONLY`: legacy/noncanonical engineering, oracle, replay, migration, or compatibility surface. Reachability cannot promote it.
- `QUALIFICATION_ONLY`: Stage-2/3, acceptance, benchmark, or CI harness. It may exercise writers but cannot own production authority.
- `TEST_ONLY`: tests/fakes/hostile fixtures only.
- `NON_AUTHORITATIVE_READ_ONLY`: protocol/declaration surfaces; privileged-looking method names do not grant authority.
- `UNKNOWN_AUTHORITY`: unresolved mutation surface; always fail closed.

The generated machine-readable registry is:

`authority/rearchitecture_r11/CB16_R11_STAGE4_AUTHORITY_WRITER_REGISTRY_V1.json`.

Each entry records path, symbol, authority domains, mutation capability, caller/callsite evidence, discovery evidence, current role, classification, and canonical-eligibility flag.

## Important authority conclusions

1. Existing R11 EventJournal, EvidenceStore, CheckpointStore, orchestrator, I/O path, training runtime, and integration adapters are only **canonical candidates** in S4A. This task does not designate a production writer.
2. Frozen Physics and Supervisor implementation surfaces are inventoried as contract-governed candidates because the Semantic Freeze explicitly fixes their authority boundaries. S4A does not change their behavior.
3. Stage-2/3 workflows and burst/acceptance harnesses are qualification-only and must not become a production runtime by reuse.
4. R2/R9/R10/R10.2 and other legacy/noncanonical writers remain reference-only even when current code imports or calls them.
5. `runtime_protocols_r11.py` is declaration-only; protocol method names are not writer authority.
6. No Stage-4 canonical singleton lease/fence currently exists in S4A. Existing pre-Stage-4 lock/authority surfaces remain noncanonical/legacy inventory entries. Final integration must bind actual writers to the later canonical ownership primitive; S4A does not implement that primitive.

## Fail-closed audit

Run:

```bash
python scripts/build_r11_stage4_authority_registry.py --repo-root .
python scripts/audit_r11_stage4_authority_surface.py --repo-root .
python -m pytest -q tests/test_stage4_gatework_r11.py tests/test_stage4_authority_inventory_r11.py
python -m cb16_local_opt.rearchitecture_authority_r11 static --repo-root .
```

The audit fails for missing/unregistered discoveries, duplicate classifications, `UNKNOWN_AUTHORITY`, legacy/test/qualification escalation, stale registry entries, Semantic Freeze drift, final-holdout/frozen-data changes, any existing-file modification by S4A, or any non-Stage-4-namespaced new path.

## Integration notes

S4A provides inventory evidence only. S4B–S4I and final integration must not import this registry as a capability token. In particular, `R11_CANONICAL_CANDIDATE` must never be treated as proof that a writer possesses runtime authority. Final cutover must independently establish adoption, singleton ownership/fencing, storage ownership, permission closure, legacy retirement, hostile qualification, and integrated receipt adjudication.
