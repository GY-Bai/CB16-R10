# CB16 R11 Stage-4 INTG — Production-Facing Hostile Cutover Harness R0

## Scope

INTG turns the already-qualified S4H hostile-cutover matrix into a production-facing integration contract. It does **not** import INTA–INTF or INTH, does not assemble the final runtime, and does not claim that the integrated runtime has passed H01–H20.

Authority remains the frozen semantic contracts. Scientific status remains:

`DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`

Replay remains engineering/recovery material and is never new scientific Evidence.

## Preserved S4H contract

`cb16_local_opt.stage4_hostile_integration_harness_r11` imports only the qualified Wave-1 S4H contract and checks that the exact H01–H20 IDs, titles, categories and nine common invariants have not drifted.

The common invariants remain:

1. exactly one authoritative generation;
2. exactly one authoritative Champion;
3. no dual writer;
4. no stale-writer mutation;
5. no replay-created scientific Evidence;
6. no Permission bypass;
7. final holdout untouched;
8. Semantic Freeze unchanged;
9. ambiguous authority fails closed.

The INTG runner emits the existing `CB16_R11_STAGE4_HOSTILE_CUTOVER_REPORT_V1` top-level shape. `integrated_runtime_qualified` is permanently `false` in INTG.

## Production adapter contract

Final consolidation must provide `ProductionHostileAdapter`. It is a translation layer over the consolidated runtime and must expose real observations, not synthetic success flags.

Required method families:

- runtime: `start_runtime`, `stop_runtime`, `crash`, `recover`;
- authority: `acquire_authority`, `release_authority`, `fence_strictly_newer`;
- adoption: `adopt_authority`, `begin_adoption`, `seal_pending_adoption`;
- journal/checkpoint transition: `begin_journal_transition`, `commit_pending_journal`, `append_journal`, `commit_transition`, `release_generation`;
- Permission/Physics: `issue_permission`, `physics_transition`;
- Evidence: `admit_evidence`;
- state-root/storage: `place_state_object`, `adopt_state_object`;
- metadata corruption: `corrupt_authority_metadata`;
- observation: `snapshot`.

Native S4D fencing objects may remain opaque. The adapter must not reduce a live native fence to an unverified string; `fence_strictly_newer` must be grounded in the native fencing authority.

`IntegratedAuthoritySnapshot` must be populated from real runtime observations and includes generation/Champion cardinality, writer ownership, stale-writer mutation count, replay Evidence admission count, Permission bypass count, final-holdout access count, Semantic Freeze blob, scientific status, metadata validity, journal cardinality and ambiguity counters.

## Fault hooks

`HostileFaultHooks` wraps each of the twenty case-specific `HostileBoundary` values with `before_boundary` / `after_boundary` hooks. The default hooks are deliberately no-op and are valid only for the deterministic reference self-test.

Final consolidation must bind hooks to real process/filesystem controls where required, including process termination and restart, lease/fence handoff, adoption prepare/seal interruption, journal prepare/commit interruption, orphan/incomplete object construction and durable authority-metadata corruption.

The machine-readable `integration_binding_manifest()` lists every H01–H20 boundary, the target surface and the real-fault requirement.

## H01–H20

The harness keeps the exact qualified S4H meanings:

- H01 duplicate canonical runtime startup
- H02 simultaneous ownership attempt
- H03 stale fencing token
- H04 conflicting authority adoption
- H05 duplicate identical adoption
- H06 legacy authoritative writer attempt
- H07 crash immediately before authority acquisition
- H08 crash immediately after authority acquisition
- H09 crash around adoption seal
- H10 crash around authoritative journal transition
- H11 duplicate commit
- H12 duplicate generation release
- H13 stale Champion writer
- H14 forged Permission
- H15 direct Physics bypass attempt
- H16 replay presented as new Evidence
- H17 orphan/incomplete state object
- H18 restart after successful cutover
- H19 second restart with stale owner
- H20 malformed/corrupt authority metadata

No case may be renamed away, skipped, or converted into a weaker synthetic substitute during final consolidation.

## Reference self-test versus final qualification

`ReferenceProductionHostileAdapter` delegates to the already-qualified S4H deterministic adapter. It exists only to qualify INTG's adapter contract, report compatibility, boundary coverage and fail-closed adjudication.

A PASS using the reference adapter means only that INTG is ready to be bound. It is not evidence that the consolidated runtime passed H01–H20.

The final adjudicator must:

1. bind `ProductionHostileAdapter` to the consolidated canonical runtime;
2. bind real `HostileFaultHooks`;
3. translate native fail-closed failures into the qualified S4H adjudication codes without changing meaning;
4. rerun all H01–H20 against real runtime/process/filesystem boundaries;
5. independently satisfy the final Stage-4 consolidation gates before any reserved cutover verdict is considered.

## Commands

Reference contract qualification:

```bash
python -m pytest -q \
  tests/test_stage4_gatework_r11.py \
  tests/test_stage4_hostile_cutover_r11.py \
  tests/test_stage4_hostile_integration_harness_r11.py

python scripts/run_r11_stage4_integrated_hostile_matrix.py \
  --output stage4_intg_hostile_report.json \
  --manifest-output stage4_intg_binding_manifest.json
```

Final consolidation may pass `--adapter-factory module:function` and `--hooks-factory module:function` after the real consolidated binding exists. INTG itself contains no sibling Integration import.
