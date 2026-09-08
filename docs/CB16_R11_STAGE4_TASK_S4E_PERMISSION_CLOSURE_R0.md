# CB16 R11 Stage-4 S4E — Permission Boundary & Legacy Bypass Closure R0

## Scope and authority

S4E is infrastructure-only. Semantic contracts are authority; legacy Python is not authority. The scientific status remains `DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`.

Frozen separation remains `TRUTH != BELIEF != DECISION != PERMISSION`. The Brain proposes only ActionIntent. The Frozen Supervisor owns Permission. Frozen Physics owns execution/account transition. Requested risk is not confidence.

Authoritative contract references inspected at Gatework base `0f18e08ec7250b9b4e45c62803c25be966834390`:

- `authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json`
- `authority/control_plane_r1/ACTION_INTENT_SCHEMA_V1.json`
- `authority/control_plane_r1/risk_supervisor_r1.py`
- `cb16_local_opt/r102_physics.py` (legacy/reference implementation, not semantic authority)
- `cb16_local_opt/typed_central_brain_r10.py` (legacy/reference implementation, not semantic authority)
- `cb16_local_opt/integration_adapters_r11.py`

## Adapter audit

`integration_adapters_r11.py` exposes persistence/control-plane adapters for Evidence references/snapshot seals, EventJournal events, and checkpoint/tournament persistence. It contains no ActionIntent conversion, Permission grant, `allowed_action`, `execute_physics`, or direct account-transition method. Therefore S4E deliberately does **not** modify that existing file: there is no permission writer in it to repair, and changing persistence behavior would exceed S4E.

The new Stage-4 boundary is additive and must be used by final cutover integration for any Brain/legacy/recovery action ingress.

## Closure contract

`stage4_permission_boundary_r11.py` enforces:

1. `ActionIntentV1` is validated against the frozen semantic shape: schema, direction, requested-risk multiplier, and complete lineage only.
2. Compatibility input may translate only `direction` + `requested_risk` representation into ActionIntent. It cannot supply Permission.
3. `confidence` is rejected rather than aliased to requested risk.
4. Privileged-looking fields (`allowed_action`, Permission/token, Supervisor decision, executable action, direct Physics/account transition, approval) fail closed even if nested.
5. Recovery may reconstruct only ActionIntent representation. Any stale permission/token causes rejection.
6. The execution bridge accepts only ActionIntent plus state inputs. It calls the already-frozen chain in order: `supervise -> executable_action -> execute_physics`.
7. The bridge has no API accepting caller-supplied Supervisor decisions, executable actions, Permission, or account transitions.
8. Missing Supervisor, Physics, contract, or malformed authority API fails closed.
9. Stage-4's returned engineering receipt explicitly states `permission_minted_by_stage4=false` and `scientific_semantics_changed=false`; that receipt itself is not Permission authority.

## Hostile cases

Tests reject all required hostile cases: forged direct executable action, forged legacy `allowed_action`, direct account transition, requested-risk/confidence/permission conflation, stale recovery Permission, compatibility authority upgrade, missing Supervisor/Physics, malformed ActionIntent, and legacy objects with privileged-looking fields.

## What S4E does not do

S4E does not modify Frozen Supervisor or Physics, does not create evidence, does not advance generation, does not alter Teacher/gradient/Champion semantics, does not open the final holdout, and does not download or mutate market data. It does not claim final Stage-4 cutover qualification.

## Final integration requirement

Final integration must route every production Brain/compatibility/recovery action ingress through `FrozenPermissionBoundaryR11` (or prove an equivalent boundary) and must not expose the vendored Supervisor `execute_physics` function as a caller-facing authoritative-write API. Legacy/reference paths may remain readable but cannot inject `RiskSupervisorDecisionV1`, `ExecutableActionV1`, Permission fields, or account transitions into this boundary.
