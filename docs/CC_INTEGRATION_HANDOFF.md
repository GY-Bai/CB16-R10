# R11 CC Integration Handoff

Qualification date: 2026-09-12 UTC.  
Canonical merge: PR #99 -> `main` at `daa889d758ce80c7d1ce73ea37110937e1b146c0`.

Final machine-readable authority:

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

Receipt-qualified runtime code head:

`fc7adaae66da5b7735a3ab6aee7a8c7bd3721ed2`

Final PR handoff head:

`5d236e9d2368a79e830bec1971b949de88c032df`

The final PR head re-ran the complete Shanxi integration qualification after the authority/docs commits and also PASSed. Later documentation-maintenance commits on `main` do not change the frozen science identity or runtime qualification.

## Canonical ownership

- Thread A: runtime, account continuity, signed economics, permission/execution correctness oracle.
- Thread B: stochastic policy, true behavior likelihood, Critic, V-trace learner, checkpoint/retention.
- Thread C: immutable experience, replay, arithmetic economic evaluation contracts.
- Thread D: performance implementation only, constrained by Thread A semantics.
- Integration: W-01 through W-05 binding, closed-loop wiring, equivalence gate, performance selection and canonical handoff.

Selected topology:

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

## Frozen thread lineage

All four implementation threads started from:

`89d62bf966f476e598f0e2f5c5e8e03c15a8db51`

Frozen heads:

- A: `d904fa67f66026fc2bc9c320fe5a888ff5f98db6`
- B: `f06babb485daee0a2f7e978c51d6263b03fc2fea`
- C: `626241fc1043e10326e538f93cf08d9cfac75b67`
- D: `26742447af209d52943085e9d37996eae522b93c`

Do not reconstruct current authority from moving thread branches; use the integration receipt.

## Qualification

### Frozen qualification

Workflow run `34710702090`, job `103598868269` passed all mandatory gates.

Joined tests: `155 passed, 1 deselected`.

The single deselection was Thread A's independent-branch sibling-isolation assertion, intentionally inapplicable after authorized A/B/C/D integration; no scientific behavior assertion was deselected.

PASS:

- integrated closed loop and provenance;
- hostile/recovery cases;
- exactly-once learner update;
- reference-fast semantic equivalence;
- legacy-import/fallback firewall;
- FINAL/fresh-data firewall.

Reference and fast produced the same:

- semantic checksum `c864052eab1c107ab73a88d96ddb527bfc60819a41165b1495ecc74d56644988`;
- final-account checksum `29d33e7639029f40c6edfb1c7fe9fff26e4af3f418cf5109a066a55de13e282e`.

Frozen qualification artifact: ID `10303497599`, SHA256 `9b0be77738e680de071e92d622f9fc25539373ec4c49a54714f471cafde4b903`.

### Final-head confirmation

Final PR head `5d236e9d2368a79e830bec1971b949de88c032df` later ran workflow `34712758444` successfully.

It again produced:

- `155 passed, 1 deselected`;
- all mandatory integration gates PASS;
- the same semantic checksum;
- the same final-account checksum;
- the same selected topology.

This confirms that the final authority/docs commits did not alter or break the runtime. The repeated benchmark is confirmation only; it does not replace the preregistered frozen benchmark in the receipt.

## Shanxi selection

Frozen workload: 16 accounts, 64 market steps, 1024 transitions per run, 7 alternating repetitions per topology.

Receipt-authoritative first qualification:

- Reference median: 7.535366 transitions/s; 135.892535 s median wall time.
- Fast median: 638.765768 transitions/s; 1.603092 s median wall time.
- Median speedup: 84.769x.

The selection rule was frozen before measurement:

`PERFORMANCE FIRST AMONG SEMANTICALLY QUALIFIED IMPLEMENTATIONS`

Thread D's integrated fast spine therefore wins the hard cutover without acquiring account-science authority.

## Hard cutover

Canonical CC has no compatibility or runtime fallback to:

- `gpu_inference_broker.py`
- `multiprocess_trajectory_farm.py`
- `vectorized_physics.py`

Failure is fail-closed. Those files may remain as historical/reference material only.

## Evidence boundary

Strongest justified evidence:

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

This is not ECONOMIC or TRANSFER evidence. FINAL remained sealed and fresh market data was not used.

No statement in this handoff authorizes real-market deployment, FINAL access, fresh-data consumption, or economic promotion beyond the evidence actually obtained.

## Remaining owner-open decision

The only current owner-open scientific decision is the master precedence rule when Buy-and-Hold and FLAT component outcomes conflict.

Until explicitly resolved:

- both component outcomes are reported;
- no unified master winner is fabricated;
- promotion must fail closed as `UNRESOLVED_OWNER_DECISION` or equivalent.

See `docs/OPEN_QUESTIONS.md`.

## Next work

Future historical science, capacity experiments or further training should begin from the canonical CC authority above, not from the old A/B/C/D branches or AC/BC task plans.

`docs/BRAIN_CAPACITY_ROADMAP_3700X_1060.md` is a future capacity-planning document, not a model-expansion authority.
