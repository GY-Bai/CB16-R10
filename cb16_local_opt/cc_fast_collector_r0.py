
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
from typing import Sequence

import numpy as np

from .cc_fast_account_kernel_r0 import AccountKernelConfig
from .cc_fast_account_workers_r0 import AccountWorkerCommand, PersistentAccountWorkerPool
from .cc_fast_fact_queue_r0 import FactOutputQueue, BackpressureRequired, encode_fact
from .cc_fast_market_cache_r0 import MarketCacheKey, SharedMarketOwner
from .cc_fast_metrics_r0 import FastMetrics
from .cc_fast_policy_broker_r0 import BatchedPolicyBroker, PolicyRequest, PolicySpec
from .cc_fast_scheduler_r0 import AccountSerialScheduler, ScheduledAccountTask
from .cc_fast_wire_r0 import semantic_sha256
from .cc_fast_workload_r0 import SyntheticWorkload
from .cc_fast_writer_r0 import ContiguousChunkWriter, ChunkReceipt

COLLECTOR_SCHEMA = "CB16_R11_CC_FAST_COLLECTOR_V1"


@dataclass(frozen=True)
class CollectorConfig:
    worker_count: int = 2
    queue_max_bytes: int = 4 * 1024 * 1024
    queue_max_age_s: float = 30.0
    writer_chunk_facts: int = 128
    max_steps: int | None = None

    def validate(self) -> "CollectorConfig":
        if self.worker_count not in {2, 4, 6}:
            raise ValueError("COLLECTOR_WORKER_COUNT_INVALID")
        if self.queue_max_bytes <= 0 or self.queue_max_age_s <= 0 or self.writer_chunk_facts <= 0:
            raise ValueError("COLLECTOR_CONFIG_INVALID")
        if self.max_steps is not None and self.max_steps <= 1:
            raise ValueError("COLLECTOR_MAX_STEPS_INVALID")
        return self


@dataclass(frozen=True)
class CollectorResult:
    workload_id: str
    transitions: int
    policy_decisions: int
    terminal_transitions: int
    chunk_receipts: tuple[ChunkReceipt, ...]
    semantic_checksum: str
    final_account_checksum: str
    queue_peak_bytes: int
    queue_peak_depth: int
    batch_sizes: tuple[int, ...]
    worker_pids: tuple[int, ...]
    worker_cuda_initialized: tuple[bool, ...]
    metrics: dict[str, object]


def _initial_snapshot(initial_equity: float) -> dict[str, float | int | bool]:
    return {"decision_index":0,"quantity":0.0,"avg_cost":0.0,"cash":float(initial_equity),"liability":0.0,"equity":float(initial_equity),"realized_pnl":0.0,"fees_paid":0.0,"funding_paid":0.0,"terminal":False}


def _observation(snapshot: dict[str, float | int | bool], mark: float) -> tuple[float, ...]:
    return (float(mark),float(snapshot["quantity"]),float(snapshot["avg_cost"]),float(snapshot["cash"]),float(snapshot["liability"]),float(snapshot["equity"]),float(snapshot["realized_pnl"]),float(snapshot["fees_paid"]),float(snapshot["funding_paid"]))


def _flush_queue(queue: FactOutputQueue, writer: ContiguousChunkWriter, *, chunk_id: str, count: int) -> ChunkReceipt | None:
    if queue.depth == 0:
        return None
    n = min(count, queue.depth)
    return writer.write_chunk([queue.get() for _ in range(n)], chunk_id=chunk_id)


def run_collector(workload: SyntheticWorkload, policy: PolicySpec, *, output_root: str, config: CollectorConfig = CollectorConfig(), account_kernel_config: AccountKernelConfig = AccountKernelConfig()) -> CollectorResult:
    config.validate(); policy.validate()
    if workload.accesses_final_or_fresh_data:
        raise RuntimeError("FINAL_OR_FRESH_DATA_FORBIDDEN")
    snapshots = {account_id:_initial_snapshot(workload.config.initial_equity) for account_id in workload.lineage_ids}
    scheduler = AccountSerialScheduler(workload.lineage_ids, max_starvation_ticks=workload.config.decision_interval_max * 8)
    broker = BatchedPolicyBroker([policy], execution_mode="CPU")
    queue = FactOutputQueue(max_bytes=config.queue_max_bytes, max_age_s=config.queue_max_age_s)
    writer = ContiguousChunkWriter(output_root)
    metrics = FastMetrics(); receipts=[]; fact_hashes=[]; terminal_transitions=0; queue_peak_bytes=queue_peak_depth=0
    max_steps = min(workload.config.market_steps, config.max_steps or workload.config.market_steps)
    market_matrix = np.ascontiguousarray(np.column_stack([workload.market, workload.funding]), dtype=np.float64)
    cache_key = MarketCacheKey(market_source_lineage=workload.market_lineage_id, visible_window_identity=f"synthetic:0:{max_steps}", preprocessing_version="SYNTHETIC_WORKLOAD_V1", frozen_organ_identity="NO_FROZEN_ORGAN_IN_D_SYNTHETIC_FIXTURE", normalizer_identity="CC_FAST_SYNTHETIC_NOOP_NORMALIZER_V1")
    with SharedMarketOwner(cache_key, market_matrix) as market_owner:
        with PersistentAccountWorkerPool(worker_count=config.worker_count, account_ids=workload.lineage_ids, market_descriptor=market_owner.descriptor, initial_equity=workload.config.initial_equity, kernel_config=account_kernel_config) as workers:
            worker_pids = tuple(sorted(status.pid for status in workers.statuses.values()))
            worker_cuda = tuple(status.cuda_initialized for _,status in sorted(workers.statuses.items()))
            for step in range(max_steps - 1):
                ready_idx = workload.ready_accounts(step)
                requests=[]
                for i in ready_idx.tolist():
                    account_id=workload.lineage_ids[i]; snapshot=snapshots[account_id]
                    if bool(snapshot["terminal"]): continue
                    dindex=int(snapshot["decision_index"])
                    scheduler.submit(ScheduledAccountTask(account_id,dindex,policy.policy_generation,step), now_tick=step)
                    obs=_observation(snapshot,float(workload.market[step])); obs_hash=semantic_sha256({"schema":"CC_FAST_OBS_V1","values":obs})
                    requests.append(PolicyRequest(account_lineage_id=account_id,decision_index=dindex,environment_time_ns=step,policy_generation=policy.policy_generation,policy_sha256=policy.policy_sha256,observation=obs,observation_hash=obs_hash,normalizer_id="CC_FAST_SYNTHETIC_NOOP_NORMALIZER_V1",rng_stream_id=f"{account_id}:gen:{policy.policy_generation}",rng_counter=dindex,stochastic=True).validate())
                if not requests: continue
                with metrics.phase("policy_inference"):
                    responses=broker.infer(requests)
                metrics.policy_decisions += len(responses)
                if len(responses)!=len(requests): raise RuntimeError("BROKER_RESPONSE_COUNT_MISMATCH")
                commands=[]
                for ordinal,(response,request) in enumerate(zip(responses,requests)):
                    if response.account_lineage_id!=request.account_lineage_id or response.decision_index!=request.decision_index or response.policy_generation!=request.policy_generation or response.policy_sha256!=request.policy_sha256:
                        raise RuntimeError("OUT_OF_ORDER_OR_CONTAMINATED_POLICY_RESPONSE")
                    commands.append(AccountWorkerCommand(ordinal=ordinal,account_lineage_id=response.account_lineage_id,decision_index=response.decision_index,step=step,nominal_direction=response.nominal_direction,nominal_target_risk=response.nominal_target_risk,policy_decision_ref=semantic_sha256({"account":response.account_lineage_id,"decision":response.decision_index,"policy":response.policy_sha256,"rng":[response.rng_stream_id,response.rng_counter]})))
                with metrics.phase("account_kernel"):
                    worker_results=workers.execute(commands)
                if len(worker_results)!=len(commands): raise RuntimeError("ACCOUNT_WORKER_RESULT_COUNT_MISMATCH")
                for response,request,worker_result in zip(responses,requests,worker_results):
                    if worker_result.account_lineage_id!=request.account_lineage_id or worker_result.decision_index!=request.decision_index:
                        raise RuntimeError("ACCOUNT_WORKER_RESULT_ORDER_VIOLATION")
                    snapshots[worker_result.account_lineage_id]=dict(worker_result.snapshot)
                    scheduler.commit(worker_result.account_lineage_id,worker_result.decision_index,now_tick=step)
                    transition=worker_result.transition
                    if transition.mechanical_terminal: terminal_transitions+=1
                    payload={"wire":"CCEnvironmentTransitionV1","transition_sha256":transition.content_sha256,"account_lineage_id":transition.account_lineage_id,"decision_index":transition.decision_index,"environment_time_before_ns":transition.environment_time_before_ns,"environment_time_after_ns":transition.environment_time_after_ns,"policy_decision":asdict(response),"execution_legs":[asdict(x) for x in transition.execution_legs],"fees":transition.fees,"funding":transition.funding,"realized_pnl":transition.realized_pnl,"unrealized_pnl_delta":transition.unrealized_pnl_delta,"liability_delta":transition.liability_delta,"post_equity":transition.post_equity,"boundary_type":transition.boundary_type,"mechanical_terminal":transition.mechanical_terminal}
                    env=encode_fact(payload,semantic_id=transition.content_sha256,terminal_or_failure=transition.mechanical_terminal)
                    while True:
                        try:
                            queue.put(env,block=False); break
                        except BackpressureRequired:
                            with metrics.phase("serialization_write"):
                                receipt=_flush_queue(queue,writer,chunk_id=f"chunk-{len(receipts):08d}",count=config.writer_chunk_facts)
                            if receipt is None: raise
                            receipts.append(receipt)
                    fact_hashes.append(transition.content_sha256); metrics.transitions+=1
                    queue_peak_bytes=max(queue_peak_bytes,queue.bytes_depth); queue_peak_depth=max(queue_peak_depth,queue.depth)
                if queue.depth>=config.writer_chunk_facts:
                    with metrics.phase("serialization_write"):
                        receipt=_flush_queue(queue,writer,chunk_id=f"chunk-{len(receipts):08d}",count=config.writer_chunk_facts)
                    if receipt: receipts.append(receipt)
            while queue.depth:
                with metrics.phase("serialization_write"):
                    receipt=_flush_queue(queue,writer,chunk_id=f"chunk-{len(receipts):08d}",count=config.writer_chunk_facts)
                if receipt: receipts.append(receipt)
    queue.close()
    semantic_checksum=hashlib.sha256("".join(fact_hashes).encode("ascii")).hexdigest()
    final_account_checksum=semantic_sha256({account_id:snapshots[account_id] for account_id in sorted(snapshots)})
    return CollectorResult(workload_id=workload.workload_id,transitions=metrics.transitions,policy_decisions=metrics.policy_decisions,terminal_transitions=terminal_transitions,chunk_receipts=tuple(receipts),semantic_checksum=semantic_checksum,final_account_checksum=final_account_checksum,queue_peak_bytes=queue_peak_bytes,queue_peak_depth=queue_peak_depth,batch_sizes=tuple(broker.batch_sizes),worker_pids=worker_pids,worker_cuda_initialized=worker_cuda,metrics=metrics.snapshot_system())
