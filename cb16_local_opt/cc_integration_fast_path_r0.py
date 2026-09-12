from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Iterable, Mapping

from .cc_fast_scheduler_r0 import AccountSerialScheduler, ScheduledAccountTask
from .cc_fast_fact_queue_r0 import FactOutputQueue, encode_fact
from .cc_fast_writer_r0 import ContiguousChunkWriter, ChunkReceipt


@dataclass(frozen=True)
class FastTransportReport:
    semantic_verdict: str
    fact_count: int
    chunk_count: int
    wall_clock_s: float
    compliant_transitions_per_s: float
    queue_max_bytes_seen: int
    queue_max_depth_seen: int
    receipts: tuple[ChunkReceipt, ...]


class CCFastSemanticTransportR0:
    """Thread-D scheduling/transport around Thread-A authoritative semantic facts.

    It never recomputes account economics and therefore cannot become a second
    scientific authority. A fast-path failure fails closed; there is no legacy
    performance fallback.
    """

    def __init__(
        self,
        *,
        account_ids: Iterable[str],
        output_root: str | Path,
        chunk_facts: int = 128,
        max_queue_bytes: int = 8 * 1024 * 1024,
        max_queue_age_s: float = 30.0,
    ) -> None:
        ids = tuple(account_ids)
        if chunk_facts <= 0:
            raise ValueError("CHUNK_FACTS_INVALID")
        self.scheduler = AccountSerialScheduler(ids, max_starvation_ticks=max(64, len(ids) * 8))
        self.queue = FactOutputQueue(max_bytes=max_queue_bytes, max_age_s=max_queue_age_s)
        self.writer = ContiguousChunkWriter(output_root)
        self.chunk_facts = int(chunk_facts)
        self.receipts: list[ChunkReceipt] = []
        self._tick = 0
        self._chunk_index = 0
        self.max_bytes_seen = 0
        self.max_depth_seen = 0

    def _flush(self) -> None:
        if self.queue.depth == 0:
            return
        batch = []
        while self.queue.depth and len(batch) < self.chunk_facts:
            batch.append(self.queue.get())
        receipt = self.writer.write_chunk(batch, chunk_id=f"cc-integration-{self._chunk_index:08d}")
        if not self.writer.verify(receipt):
            raise RuntimeError("FAST_TRANSPORT_DURABILITY_VERIFY_FAILED")
        self._chunk_index += 1
        self.receipts.append(receipt)

    def submit(
        self,
        *,
        account_lineage_id: str,
        decision_index: int,
        policy_generation: int,
        semantic_id: str,
        payload: Mapping[str, Any],
        terminal_or_failure: bool,
    ) -> None:
        task = ScheduledAccountTask(
            account_lineage_id=account_lineage_id,
            decision_index=int(decision_index),
            policy_generation=int(policy_generation),
            enqueue_tick=self._tick,
        )
        self.scheduler.submit(task, now_tick=self._tick)
        envelope = encode_fact(payload, semantic_id=semantic_id, terminal_or_failure=terminal_or_failure)
        try:
            self.queue.put(envelope, block=False)
        except Exception:
            self.scheduler.cancel_fail_closed(account_lineage_id)
            raise
        self.max_bytes_seen = max(self.max_bytes_seen, self.queue.bytes_depth)
        self.max_depth_seen = max(self.max_depth_seen, self.queue.depth)
        self.scheduler.commit(account_lineage_id, int(decision_index), now_tick=self._tick)
        self._tick += 1
        if self.queue.depth >= self.chunk_facts:
            self._flush()

    def close(self) -> tuple[ChunkReceipt, ...]:
        while self.queue.depth:
            self._flush()
        self.queue.close()
        return tuple(self.receipts)


def decode_fast_chunks(receipts: Iterable[ChunkReceipt]) -> dict[str, Mapping[str, Any]]:
    decoded: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        receipt.validate()
        for line in Path(receipt.path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            semantic_id = row["semantic_id"]
            if semantic_id in decoded:
                raise RuntimeError("DUPLICATE_FAST_FACT_ID")
            decoded[semantic_id] = json.loads(row["payload"])
    return decoded


def exact_semantic_equivalence(
    reference: Mapping[str, Mapping[str, Any]],
    fast: Mapping[str, Mapping[str, Any]],
) -> bool:
    if tuple(sorted(reference)) != tuple(sorted(fast)):
        return False
    return all(reference[key] == fast[key] for key in reference)


def transport_facts_fast(
    *,
    facts: Iterable[tuple[str, int, int, str, Mapping[str, Any], bool]],
    account_ids: Iterable[str],
    output_root: str | Path,
    chunk_facts: int = 128,
) -> FastTransportReport:
    started = time.perf_counter()
    transport = CCFastSemanticTransportR0(account_ids=account_ids, output_root=output_root, chunk_facts=chunk_facts)
    reference: dict[str, Mapping[str, Any]] = {}
    count = 0
    for account_id, decision_index, generation, semantic_id, payload, terminal_or_failure in facts:
        if semantic_id in reference:
            raise RuntimeError("DUPLICATE_REFERENCE_FACT_ID")
        reference[semantic_id] = dict(payload)
        transport.submit(
            account_lineage_id=account_id,
            decision_index=decision_index,
            policy_generation=generation,
            semantic_id=semantic_id,
            payload=payload,
            terminal_or_failure=terminal_or_failure,
        )
        count += 1
    receipts = transport.close()
    decoded = decode_fast_chunks(receipts)
    equivalent = exact_semantic_equivalence(reference, decoded)
    elapsed = time.perf_counter() - started
    if not equivalent:
        raise RuntimeError("REFERENCE_FAST_SEMANTIC_MISMATCH")
    return FastTransportReport(
        semantic_verdict="PASS",
        fact_count=count,
        chunk_count=len(receipts),
        wall_clock_s=elapsed,
        compliant_transitions_per_s=(count / elapsed if elapsed > 0 else float("inf")),
        queue_max_bytes_seen=transport.max_bytes_seen,
        queue_max_depth_seen=transport.max_depth_seen,
        receipts=receipts,
    )


def summarize_repetitions(values: Iterable[float]) -> dict[str, float]:
    rows = [float(x) for x in values]
    if len(rows) < 2:
        raise ValueError("AT_LEAST_TWO_REPETITIONS_REQUIRED")
    return {
        "n": float(len(rows)),
        "median": float(statistics.median(rows)),
        "min": float(min(rows)),
        "max": float(max(rows)),
        "pstdev": float(statistics.pstdev(rows)),
    }
