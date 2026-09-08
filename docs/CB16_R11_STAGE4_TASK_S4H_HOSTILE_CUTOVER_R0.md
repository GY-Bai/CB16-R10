# CB16 R11 Stage-4 Task S4H — Hostile Cutover / Split-Brain Harness R0

## Scope

S4H qualifies the **adversarial harness**, not the production implementations owned by S4B–S4G and not the final integrated Stage-4 cutover. The harness is intentionally independent: it imports no sibling Stage-4 implementation and ships a deterministic in-memory reference adapter solely so the protocol can be tested on its own branch.

The scientific status remains `DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`. S4H does not open the final holdout, download market data, mutate frozen data, mint a scientific verdict, reinterpret replay as evidence, or change Physics/Permission/Teacher/Evidence/gradient/Champion-Challenger/requested-risk semantics.

## Integration boundary

`HostileCutoverAdapter` is the only production-facing contract. A later integrator can wrap the concrete Stage-4 runtime, lease, adoption, state-root, permission and legacy-retirement implementations behind that interface. S4H itself does not know those module names.

The adapter surface covers runtime lifecycle, local authority acquisition/fencing, adoption, journal transitions, commit/release, Supervisor-issued Permission, Physics execution, Evidence admission, state-object sealing and corrupt-metadata recovery. Failures are represented as deterministic `FailClosed` codes in the reference model; a production adapter may translate native exceptions into those adjudication classes.

## Required hostile matrix

The matrix contains exactly the twenty Wave-1 required classes: duplicate runtime startup; simultaneous ownership; stale fence; conflicting and duplicate adoption; legacy writer; crash before/after acquisition; crash around adoption and journal seals; duplicate commit/release; stale Champion writer; forged Permission; direct Physics bypass; replay laundering; orphan/incomplete object; first and second restart fencing; and corrupt metadata.

Every case checks the same nine invariants:

1. exactly one authoritative generation;
2. exactly one authoritative Champion;
3. no dual writer;
4. no stale writer mutation;
5. no new scientific Evidence from replay;
6. no Permission bypass;
7. no final-holdout access;
8. Semantic Freeze identity unchanged;
9. ambiguous authority fails closed.

## Crash model

The fake uses explicit prepare/seal boundaries for adoption and journal writes. A crash discards unsealed pending control state. Restart never manufactures a new generation, Champion, adoption or Evidence. A newly acquired fence is monotonically newer than all previous owner tokens; old tokens cannot mutate authority.

## Idempotency model

Byte/semantic-identical adoption, commit, journal append and generation release are explicit no-ops. Conflicting reuse of the same authority slot fails closed. A generation release is the sole operation in the fake that advances the synthetic generation and binds the committed next Champion; duplicate release cannot advance it twice.

## Permission and replay model

Only a token marked as issued by `FROZEN_SUPERVISOR` can authorize the reference Physics transition. A forged or missing Permission is rejected. `source=REPLAY` is engineering-only and is rejected by the Evidence-admission path without incrementing scientific Evidence count.

## Machine-readable output

`scripts/run_r11_stage4_hostile_cutover_matrix.py` emits `CB16_R11_STAGE4_HOSTILE_CUTOVER_REPORT_V1`. The JSON schema is `authority/rearchitecture_r11/CB16_R11_STAGE4_HOSTILE_CUTOVER_REPORT_SCHEMA_V1.json`. The report permanently carries `integrated_runtime_qualified=false`; S4H cannot emit `R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED`.

## Integration qualification requirement

Final Stage-4 integration must supply a production adapter over the real S4B–S4G components and rerun the same matrix. A PASS on this branch only means the hostile qualification protocol and reference model are internally ready. It is not evidence that real cutover ownership, persistence or recovery has passed.
