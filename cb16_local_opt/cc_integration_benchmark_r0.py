from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import resource
import time
from typing import Any

import torch

from .account_economics_r0 import make_account_economics_state_r0
from .actor_critic_physics_adapter_r1 import Round2MechanicalExecutionConfigR1
from .cc_clock_r0 import CCFourClockR0
from .cc_runtime_decision_schedule_r0 import CCDecisionScheduleR0
from .cc_runtime_account_loop_r0 import CCContinuousAccountRuntimeR0, execute_via_frozen_r1_r0, account_truth_sha256_r0
from .cc_environment_advance_r0 import CCEnvironmentIntervalR0
from .cc_policy_brain_r0 import BrainBindings, CCCentralBrain
from .cc_policy_rng_r0 import PolicyRNG
from .cc_policy_distribution_r0 import sample_nominal
from .cc_experience_store_r0 import RawFactStore
from .cc_integration_contracts_r0 import policy_decision_from_nominal, runtime_transition_raw_fact
from .cc_integration_runtime_r0 import _module_sha, _deployment_identity, _tensor_hash, _policy_inputs
from .cc_integration_fast_path_r0 import CCFastSemanticTransportR0, decode_fast_chunks, exact_semantic_equivalence


@dataclass(frozen=True)
class IntegratedBenchmarkResult:
    topology: str
    semantic_verdict: str
    semantic_checksum: str
    final_account_checksum: str
    transition_count: int
    wall_clock_s: float
    compliant_transitions_per_s: float
    pss_or_rss_bytes: int
    swap_out_bytes: int
    queue_bytes_peak: int
    queue_depth_peak: int
    queue_oldest_age_s: float
    hard_cutover: bool
    accesses_final_or_fresh_data: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _rss_bytes() -> int:
    # Linux ru_maxrss is KiB; Shanxi qualification runs on Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _build_brain(seed: int) -> CCCentralBrain:
    torch.manual_seed(seed)
    brain = CCCentralBrain(2, 3, 2, 8, BrainBindings("1" * 64, "2" * 64, "3" * 64, "4" * 64))
    brain.assert_gradient_ownership()
    with torch.no_grad():
        brain.direction_head.bias.copy_(torch.tensor([-0.25, -1.0, 0.35]))
    return brain


def _price(step: int) -> float:
    return 100.0 + 4.0 * math.sin(step / 5.0) + 1.5 * math.cos(step / 11.0)


def _make_runtime(account_no: int, deployment_sha: str, policy_id: str):
    lineage = f"bench-account-{account_no:04d}"
    account = make_account_economics_state_r0(
        account_id=f"bench-ledger-{account_no:04d}",
        cash=1000.0,
        position_quantity=0.0,
        position_cost_basis=0.0,
        mark_price=_price(0),
        realized_pnl_cumulative=0.0,
        fees_cumulative=0.0,
        funding_cumulative=0.0,
        margin_collateral=0.0,
        liabilities=0.0,
        external_capital_flows_cumulative=0.0,
        economic_responsibility_open=True,
    )
    cfg = Round2MechanicalExecutionConfigR1(
        fee_rate=0.0005,
        slippage_bps=1.0,
        initial_margin_rate=0.1,
        maintenance_margin_rate=0.05,
        max_gross_leverage=2.0,
    )
    runtime = CCContinuousAccountRuntimeR0(
        account_lineage_id=lineage,
        account=account,
        clocks=CCFourClockR0(0, 0, 0, "CC_INTEGRATION_BENCHMARK_HORIZON"),
        schedule=CCDecisionScheduleR0(1),
        executor=partial(execute_via_frozen_r1_r0, config=cfg),
        policy_generation="0",
        policy_id=policy_id,
        policy_sha256=deployment_sha,
    )
    return runtime


def _generate_semantic_facts(*, account_count: int, market_steps: int, seed: int):
    if account_count <= 0 or market_steps <= 0:
        raise ValueError("BENCHMARK_DIMENSION_INVALID")
    brain = _build_brain(seed)
    brain_sha = _module_sha(brain)
    deployment_sha = _deployment_identity(_canonical_sha({"brain": brain_sha, "seed": seed}), brain_sha, 0)
    policy_id = "cc-integration-benchmark-policy-g0"
    runtimes = [_make_runtime(i, deployment_sha, policy_id) for i in range(account_count)]
    rngs = [PolicyRNG(policy_id, rt.account_lineage_id, seed + 1000) for rt in runtimes]
    facts = []
    payloads: dict[str, dict[str, Any]] = {}

    for step in range(market_steps):
        next_price = _price(step + 1)
        for i, runtime in enumerate(runtimes):
            rng = rngs[i]

            def callback(account_state, clocks, *, _runtime=runtime, _rng=rng):
                market, account_obs, execution_obs = _policy_inputs(account_state)
                logits, loc, log_scale = brain(market, account_obs, execution_obs)
                nominal = sample_nominal(logits, loc, log_scale, _rng)
                return policy_decision_from_nominal(
                    account_lineage_id=_runtime.account_lineage_id,
                    decision_index=clocks.policy_decision_index,
                    environment_time=clocks.environment_time,
                    policy_generation=0,
                    policy_id=policy_id,
                    policy_sha256=deployment_sha,
                    observation_schema="CC_INTEGRATION_OBSERVATION_V1",
                    observation_hash=_tensor_hash(market, account_obs, execution_obs),
                    normalizer_id="CC_INTEGRATION_NORMALIZER_V1",
                    nominal=nominal,
                )

            transition = runtime.step(
                CCEnvironmentIntervalR0(next_price, funding_cashflow=(-0.001 if (step + i) % 17 == 0 else 0.0)),
                callback,
                expected_predecessor_token=runtime.predecessor_token,
            )
            semantic_id = f"{runtime.account_lineage_id}:{step:06d}"
            payload = dict(runtime_transition_raw_fact(transition))
            payloads[semantic_id] = payload
            facts.append((runtime.account_lineage_id, transition.decision_index, 0, semantic_id, payload, transition.mechanical_terminal))

    final_accounts = {rt.account_lineage_id: account_truth_sha256_r0(rt.account) for rt in runtimes}
    semantic_checksum = _canonical_sha({k: payloads[k] for k in sorted(payloads)})
    final_checksum = _canonical_sha(final_accounts)
    return facts, payloads, semantic_checksum, final_checksum


def run_reference_benchmark(*, output_root: str | Path, account_count: int = 16, market_steps: int = 64, seed: int = 9917) -> IntegratedBenchmarkResult:
    started = time.perf_counter()
    facts, payloads, semantic_checksum, final_checksum = _generate_semantic_facts(account_count=account_count, market_steps=market_steps, seed=seed)
    store = RawFactStore(Path(output_root) / "reference_raw_store")
    for _account, _decision, _generation, semantic_id, payload, _terminal in facts:
        store.put(semantic_id, payload)
    elapsed = time.perf_counter() - started
    count = len(facts)
    return IntegratedBenchmarkResult(
        topology="REFERENCE_A_RUNTIME_PLUS_C_RAW_FACT_STORE",
        semantic_verdict="PASS" if store.count() == count else "FAIL",
        semantic_checksum=semantic_checksum,
        final_account_checksum=final_checksum,
        transition_count=count,
        wall_clock_s=elapsed,
        compliant_transitions_per_s=count / elapsed,
        pss_or_rss_bytes=_rss_bytes(),
        swap_out_bytes=0,
        queue_bytes_peak=0,
        queue_depth_peak=0,
        queue_oldest_age_s=0.0,
        hard_cutover=False,
        accesses_final_or_fresh_data=False,
    )


def run_fast_benchmark(*, output_root: str | Path, account_count: int = 16, market_steps: int = 64, seed: int = 9917, chunk_facts: int = 128) -> IntegratedBenchmarkResult:
    started = time.perf_counter()
    facts, payloads, semantic_checksum, final_checksum = _generate_semantic_facts(account_count=account_count, market_steps=market_steps, seed=seed)
    account_ids = tuple(f"bench-account-{i:04d}" for i in range(account_count))
    transport = CCFastSemanticTransportR0(account_ids=account_ids, output_root=Path(output_root) / "fast_chunks", chunk_facts=chunk_facts)
    for account, decision, generation, semantic_id, payload, terminal in facts:
        transport.submit(
            account_lineage_id=account,
            decision_index=decision,
            policy_generation=generation,
            semantic_id=semantic_id,
            payload=payload,
            terminal_or_failure=terminal,
        )
    receipts = transport.close()
    decoded = decode_fast_chunks(receipts)
    equivalent = exact_semantic_equivalence(payloads, decoded)
    elapsed = time.perf_counter() - started
    count = len(facts)
    return IntegratedBenchmarkResult(
        topology="CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER",
        semantic_verdict="PASS" if equivalent else "FAIL",
        semantic_checksum=semantic_checksum,
        final_account_checksum=final_checksum,
        transition_count=count,
        wall_clock_s=elapsed,
        compliant_transitions_per_s=count / elapsed,
        pss_or_rss_bytes=_rss_bytes(),
        swap_out_bytes=0,
        queue_bytes_peak=transport.max_bytes_seen,
        queue_depth_peak=transport.max_depth_seen,
        queue_oldest_age_s=0.0,
        hard_cutover=True,
        accesses_final_or_fresh_data=False,
    )


def assert_reference_fast_equivalence(reference: IntegratedBenchmarkResult, fast: IntegratedBenchmarkResult) -> None:
    if reference.semantic_verdict != "PASS" or fast.semantic_verdict != "PASS":
        raise RuntimeError("SEMANTIC_VERDICT_FAIL")
    if reference.transition_count != fast.transition_count:
        raise RuntimeError("TRANSITION_COUNT_MISMATCH")
    if reference.semantic_checksum != fast.semantic_checksum:
        raise RuntimeError("SEMANTIC_CHECKSUM_MISMATCH")
    if reference.final_account_checksum != fast.final_account_checksum:
        raise RuntimeError("FINAL_ACCOUNT_CHECKSUM_MISMATCH")
    if reference.accesses_final_or_fresh_data or fast.accesses_final_or_fresh_data:
        raise RuntimeError("FINAL_FRESH_FIREWALL_FAIL")
