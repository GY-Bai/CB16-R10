# R11 CC Integration Handoff

Qualification date: 2026-09-12 UTC.

The final machine-readable authority is:

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

Qualified integration code head: `fc7adaae66da5b7735a3ab6aee7a8c7bd3721ed2`.

## Canonical ownership

- Thread A: runtime, account continuity, signed economics, permission/execution correctness oracle.
- Thread B: stochastic policy, true behavior likelihood, Critic, V-trace learner, checkpoint/retention.
- Thread C: immutable experience, replay, arithmetic economic evaluation contracts.
- Thread D: performance implementation only, constrained by Thread A semantics.
- Integration: W-01 through W-05 binding, closed-loop wiring, equivalence gate, performance selection and canonical handoff.

The selected topology is `CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`.

## Qualification

Workflow run `34710702090`, job `103598868269` passed all mandatory gates. Joined tests were `155 passed, 1 deselected`; the single deselection was the Thread-A independent-branch sibling-isolation assertion, which is intentionally inapplicable after the authorized A/B/C/D join.

Closed-loop, provenance, hostile/recovery, exactly-once learner update, reference-fast equivalence, legacy-import firewall and FINAL/fresh-data firewall all passed.

Reference and fast produced the same semantic checksum `c864052eab1c107ab73a88d96ddb527bfc60819a41165b1495ecc74d56644988` and final-account checksum `29d33e7639029f40c6edfb1c7fe9fff26e4af3f418cf5109a066a55de13e282e`.

## Shanxi selection

Workload: 16 accounts, 64 market steps, 1024 transitions per run, 7 alternating repetitions per topology.

- Reference median: 7.535366 transitions/s; 135.892535 s median wall time.
- Fast median: 638.765768 transitions/s; 1.603092 s median wall time.
- Median speedup: 84.769x.

The performance rule was frozen before the run: choose the highest median end-to-end compliant throughput among semantic-PASS implementations. Thread D's integrated fast spine therefore wins the hard cutover.

No compatibility or runtime fallback to the historical `gpu_inference_broker.py`, `multiprocess_trajectory_farm.py`, or `vectorized_physics.py` is part of the canonical CC path.

## Evidence boundary

Strongest justified evidence: `INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`.

This is not ECONOMIC or TRANSFER evidence. FINAL remained sealed and fresh market data was not used.

The remaining owner-open decision is Thread C's unresolved precedence question when buy-and-hold and FLAT component outcomes conflict. Integration did not invent a master winner rule.
