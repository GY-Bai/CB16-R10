from __future__ import annotations

"""R11 bounded SSD/HDD throughput runtime.

This module changes storage orchestration only.  Evidence/event/checkpoint scientific
identity remains owned by the existing R11 stores.  The writer owns one sequential HDD
payload lane, batches durability work, and exposes an explicit barrier before a
scientific generation can be sealed.
"""

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .checkpoint_store_r11 import CheckpointObjectRefR11, CheckpointStoreR11
from .evidence_store_r11 import (
    EvidenceItemR11,
    EvidenceMaterializeReceiptR11,
    EvidenceStoreR11,
    FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX,
    PayloadRefR11,
    sha256_bytes,
    sha256_obj,
)
from .event_journal_r11 import EventItemR11, EventJournalR11, TraceBatchSealR11

FAIL_DURING_BATCH_APPEND = "DURING_BATCH_APPEND"
FAIL_AFTER_METADATA_COMMIT_BEFORE_EVENT_SEAL = "AFTER_METADATA_COMMIT_BEFORE_EVENT_SEAL"
FailHook = Callable[[str], None]

_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_OBJECT_OVERHEAD = 512
_INDEX_BYTES_PER_TIMESTAMP = 96


class IORuntimeErrorR11(RuntimeError):
    pass


class IORuntimeBudgetExceededR11(IORuntimeErrorR11):
    pass


class StorageWriterFailedR11(IORuntimeErrorR11):
    pass


@dataclass(frozen=True)
class IORuntimeBudgetR11:
    """Explicit 16 GiB-class host RAM partition for Task C runtime state."""

    total_ram_bytes: int = 16 * _GIB
    market_cache_bytes: int = 4 * _GIB
    evidence_queue_bytes: int = 512 * _MIB
    checkpoint_staging_bytes: int = 2 * _GIB
    process_reserve_bytes: int = 4 * _GIB

    def __post_init__(self) -> None:
        values = (
            self.total_ram_bytes,
            self.market_cache_bytes,
            self.evidence_queue_bytes,
            self.checkpoint_staging_bytes,
            self.process_reserve_bytes,
        )
        if any(int(x) <= 0 for x in values):
            raise ValueError("R11_IO_BUDGETS_MUST_BE_POSITIVE")
        allocated = sum(int(x) for x in values[1:])
        if allocated > int(self.total_ram_bytes):
            raise IORuntimeBudgetExceededR11(
                f"R11_IO_RAM_BUDGET_OVERCOMMITTED:{allocated}:{self.total_ram_bytes}"
            )


@dataclass(frozen=True)
class IOThroughputConfigR11:
    evidence_metadata_root: Path
    evidence_payload_roots: tuple[Path, ...]
    journal_root: Path
    checkpoint_root: Path | None = None
    segment_target_bytes: int = 256 * _MIB
    evidence_codec: str = "zstd"
    sqlite_synchronous: str = "FULL"
    queue_max_items: int = 256
    writer_batch_max_objects: int = 128
    writer_batch_max_bytes: int = 32 * _MIB
    budgets: IORuntimeBudgetR11 = field(default_factory=IORuntimeBudgetR11)

    def __post_init__(self) -> None:
        roots = tuple(Path(x).expanduser().resolve() for x in self.evidence_payload_roots)
        if len(roots) != 1:
            raise ValueError("R11_IO_SINGLE_SEQUENTIAL_HDD_LANE_REQUIRED")
        object.__setattr__(
            self, "evidence_metadata_root", Path(self.evidence_metadata_root).expanduser().resolve()
        )
        object.__setattr__(self, "evidence_payload_roots", roots)
        object.__setattr__(self, "journal_root", Path(self.journal_root).expanduser().resolve())
        if self.checkpoint_root is not None:
            object.__setattr__(
                self, "checkpoint_root", Path(self.checkpoint_root).expanduser().resolve()
            )
        if self.segment_target_bytes <= 0:
            raise ValueError("R11_IO_SEGMENT_TARGET_MUST_BE_POSITIVE")
        if self.queue_max_items <= 0 or self.writer_batch_max_objects <= 0:
            raise ValueError("R11_IO_ITEM_LIMITS_MUST_BE_POSITIVE")
        if self.writer_batch_max_bytes <= 0:
            raise ValueError("R11_IO_BATCH_BYTES_MUST_BE_POSITIVE")
        if self.writer_batch_max_bytes > self.budgets.evidence_queue_bytes:
            raise IORuntimeBudgetExceededR11("R11_IO_BATCH_EXCEEDS_EVIDENCE_QUEUE_BUDGET")


@dataclass(frozen=True)
class ImmutableEvidenceObjectR11:
    evidence_id: str
    parent_snapshot_hash: str
    lineage_hash: str
    teacher_protocol_hash: str
    payload_bytes: bytes

    @classmethod
    def from_item(cls, item: EvidenceItemR11) -> "ImmutableEvidenceObjectR11":
        return cls(
            evidence_id=str(item.evidence_id),
            parent_snapshot_hash=str(item.parent_snapshot_hash),
            lineage_hash=str(item.lineage_hash),
            teacher_protocol_hash=str(item.teacher_protocol_hash),
            payload_bytes=bytes(item.payload_bytes),
        )

    @property
    def content_hash(self) -> str:
        return sha256_bytes(self.payload_bytes)

    @property
    def identity_hash(self) -> str:
        return sha256_obj(
            {
                "evidence_id": self.evidence_id,
                "content_hash": self.content_hash,
                "parent_snapshot_hash": self.parent_snapshot_hash,
                "lineage_hash": self.lineage_hash,
                "teacher_protocol_hash": self.teacher_protocol_hash,
            }
        )


@dataclass(frozen=True)
class ImmutableEventObjectR11:
    event_id: str
    event_type: str
    generation: int
    policy_weight_hash: str
    snapshot_hash: str
    lineage_hash: str
    payload_bytes: bytes

    @classmethod
    def from_item(cls, item: EventItemR11) -> "ImmutableEventObjectR11":
        return cls(
            event_id=str(item.event_id),
            event_type=str(item.event_type),
            generation=int(item.generation),
            policy_weight_hash=str(item.policy_weight_hash),
            snapshot_hash=str(item.snapshot_hash),
            lineage_hash=str(item.lineage_hash),
            payload_bytes=bytes(item.payload_bytes),
        )

    def thaw(self) -> EventItemR11:
        return EventItemR11(
            event_id=self.event_id,
            event_type=self.event_type,
            generation=self.generation,
            policy_weight_hash=self.policy_weight_hash,
            snapshot_hash=self.snapshot_hash,
            lineage_hash=self.lineage_hash,
            payload=json.loads(self.payload_bytes),
        )


@dataclass(frozen=True)
class EvidenceBatchResultR11:
    refs: tuple[PayloadRefR11, ...]
    receipt: EvidenceMaterializeReceiptR11
    payload_fsync_count: int
    metadata_transaction_count: int


@dataclass(frozen=True)
class TraceCommitResultR11:
    evidence: EvidenceBatchResultR11
    trace_batch: TraceBatchSealR11


@dataclass(frozen=True)
class IORuntimeStatsR11:
    submitted_requests: int
    durable_requests: int
    queue_peak_items: int
    queue_peak_bytes: int
    payload_fsyncs: int
    evidence_metadata_transactions: int
    writer_batches: int
    created_payloads: int
    reused_payloads: int
    checkpoint_created: int
    checkpoint_reused: int


class MarketCacheBudgetGuardR11:
    """Admission-only budget for campaign-lifetime market cache.

    R11 market cache entries remain resident for the campaign; this guard therefore
    fails closed instead of evicting an entry and changing reuse/lifetime semantics.
    """

    def __init__(self, max_bytes: int):
        if int(max_bytes) <= 0:
            raise ValueError("R11_MARKET_CACHE_BUDGET_MUST_BE_POSITIVE")
        self.max_bytes = int(max_bytes)
        self._used_bytes = 0
        self._symbols: dict[str, int] = {}

    @staticmethod
    def accounting_bytes(entry: Any) -> int:
        arrays = int(
            entry.open_time_ms.nbytes + entry.ohlcv.nbytes + entry.funding_rate.nbytes
        )
        return arrays + len(entry.index_by_time_ms) * _INDEX_BYTES_PER_TIMESTAMP + _OBJECT_OVERHEAD

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def admit(self, entry: Any) -> int:
        symbol = str(entry.symbol)
        old = self._symbols.get(symbol)
        if old is not None:
            return old
        charge = self.accounting_bytes(entry)
        proposed = self._used_bytes + charge
        if proposed > self.max_bytes:
            raise IORuntimeBudgetExceededR11(
                f"R11_MARKET_CACHE_BUDGET_EXCEEDED:{symbol}:{proposed}:{self.max_bytes}"
            )
        self._symbols[symbol] = charge
        self._used_bytes = proposed
        return charge

    def reset(self) -> None:
        self._symbols.clear()
        self._used_bytes = 0


@dataclass(frozen=True)
class _PendingPayload:
    content_hash: str
    raw: bytes
    record: bytes
    crc32: int
    codec: str
    stored_bytes: int


def _catalog_plan(
    store: EvidenceStoreR11, rows: Sequence[ImmutableEvidenceObjectR11]
) -> tuple[dict[str, ImmutableEvidenceObjectR11], int]:
    by_id: dict[str, ImmutableEvidenceObjectR11] = {}
    for item in rows:
        old = by_id.get(item.evidence_id)
        if old is not None and old.identity_hash != item.identity_hash:
            raise RuntimeError(f"R11_EVIDENCE_ID_CONTENT_CONFLICT:{item.evidence_id}")
        by_id[item.evidence_id] = item
    created = 0
    for evidence_id, item in by_id.items():
        row = store.conn.execute(
            "SELECT identity_hash FROM evidence_catalog WHERE evidence_id=?", (evidence_id,)
        ).fetchone()
        if row is None:
            created += 1
        elif str(row[0]) != item.identity_hash:
            raise RuntimeError(f"R11_EVIDENCE_ID_CONTENT_CONFLICT:{evidence_id}")
    return by_id, created


def _insert_catalog(
    store: EvidenceStoreR11,
    catalog: Mapping[str, ImmutableEvidenceObjectR11],
    now: float,
) -> None:
    for evidence_id, item in catalog.items():
        row = store.conn.execute(
            "SELECT identity_hash FROM evidence_catalog WHERE evidence_id=?", (evidence_id,)
        ).fetchone()
        if row is not None:
            if str(row[0]) != item.identity_hash:
                raise RuntimeError(f"R11_EVIDENCE_ID_CONTENT_CONFLICT:{evidence_id}")
            continue
        store.conn.execute(
            "INSERT INTO evidence_catalog VALUES(?,?,?,?,?,?,?)",
            (
                evidence_id,
                item.content_hash,
                item.identity_hash,
                item.parent_snapshot_hash,
                item.lineage_hash,
                item.teacher_protocol_hash,
                now,
            ),
        )


def _write_payload_group(
    store: EvidenceStoreR11,
    group: Sequence[_PendingPayload],
    *,
    catalog: Mapping[str, ImmutableEvidenceObjectR11] | None,
    fail_hook: FailHook | None,
) -> tuple[dict[str, PayloadRefR11], int, int]:
    segment_id, path, committed_end, payload_count = store._active_row(0)
    size = path.stat().st_size if path.exists() else 0
    if size != committed_end:
        raise RuntimeError(
            f"R11_ACTIVE_TAIL_NOT_RECOVERED:{segment_id}:{size}:{committed_end}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    refs: dict[str, PayloadRefR11] = {}
    offset = committed_end
    with path.open("ab", buffering=0) as handle:
        for ordinal, pending in enumerate(group):
            handle.write(pending.record)
            end = offset + len(pending.record)
            refs[pending.content_hash] = PayloadRefR11(
                content_hash=pending.content_hash,
                lane=0,
                segment_id=segment_id,
                segment_path=str(path),
                offset=offset,
                end_offset=end,
                raw_bytes=len(pending.raw),
                stored_bytes=pending.stored_bytes,
                codec=pending.codec,
                crc32=pending.crc32,
            )
            offset = end
            if fail_hook is not None and len(group) > 1 and ordinal == 0:
                fail_hook(FAIL_DURING_BATCH_APPEND)
        handle.flush()
        os.fsync(handle.fileno())
    if fail_hook is not None:
        fail_hook(FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX)

    now = time.time()
    store.conn.execute("BEGIN IMMEDIATE")
    try:
        for pending in group:
            ref = refs[pending.content_hash]
            old = store._payload_ref(pending.content_hash)
            if old is not None:
                if old != ref:
                    raise RuntimeError(
                        f"R11_CONTENT_ADDRESS_COLLISION_OR_RACE:{pending.content_hash}"
                    )
                continue
            store.conn.execute(
                "INSERT INTO payloads VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ref.content_hash,
                    0,
                    ref.segment_id,
                    ref.segment_path,
                    ref.offset,
                    ref.end_offset,
                    ref.raw_bytes,
                    ref.stored_bytes,
                    ref.codec,
                    ref.crc32,
                    now,
                ),
            )
        store.conn.execute(
            "UPDATE segments SET committed_end=?,payload_count=? WHERE segment_id=?",
            (offset, payload_count + len(group), segment_id),
        )
        if catalog is not None:
            _insert_catalog(store, catalog, now)
        store.conn.execute("COMMIT")
    except BaseException:
        if store.conn.in_transaction:
            store.conn.execute("ROLLBACK")
        raise
    return refs, 1, 1


def put_evidence_batched_r11(
    store: EvidenceStoreR11,
    items: Sequence[ImmutableEvidenceObjectR11],
    *,
    fail_hook: FailHook | None = None,
) -> EvidenceBatchResultR11:
    """Append sequentially and amortize fsync/metadata transaction per touched segment."""

    if store.lane_count != 1:
        raise RuntimeError("R11_IO_SINGLE_SEQUENTIAL_HDD_LANE_REQUIRED")
    rows = list(items)
    if not rows:
        empty = EvidenceMaterializeReceiptR11(0, 0, 0, 0, 0, 0)
        return EvidenceBatchResultR11((), empty, 0, 0)

    catalog, created_evidence = _catalog_plan(store, rows)
    unique_payloads = {item.content_hash: item.payload_bytes for item in rows}
    refs: dict[str, PayloadRefR11] = {}
    pending: list[_PendingPayload] = []
    for content_hash, raw in sorted(unique_payloads.items()):
        old = store._payload_ref(content_hash)
        if old is not None:
            refs[content_hash] = old
            continue
        h, record, crc, codec, stored_len = store._encode(raw)
        if h != content_hash:
            raise RuntimeError("R11_CONTENT_HASH_DRIFT")
        pending.append(
            _PendingPayload(h, raw, record, int(crc), str(codec), int(stored_len))
        )

    fsyncs = 0
    metadata_transactions = 0
    todo = deque(pending)
    while todo:
        _segment_id, _path, committed_end, _count = store._active_row(0)
        if committed_end and committed_end + len(todo[0].record) > store.segment_target_bytes:
            store._seal_active(0, fail_hook=fail_hook)
            committed_end = 0

        group: list[_PendingPayload] = []
        projected = committed_end
        while todo:
            candidate = todo[0]
            if group and projected + len(candidate.record) > store.segment_target_bytes:
                break
            group.append(todo.popleft())
            projected += len(candidate.record)
            if projected >= store.segment_target_bytes:
                break

        final_group = not todo
        group_refs, group_fsyncs, group_txns = _write_payload_group(
            store,
            group,
            catalog=catalog if final_group else None,
            fail_hook=fail_hook,
        )
        refs.update(group_refs)
        fsyncs += group_fsyncs
        metadata_transactions += group_txns
        if not final_group:
            store._seal_active(0, fail_hook=fail_hook)

    if not pending:
        store.conn.execute("BEGIN IMMEDIATE")
        try:
            _insert_catalog(store, catalog, time.time())
            store.conn.execute("COMMIT")
            metadata_transactions += 1
        except BaseException:
            if store.conn.in_transaction:
                store.conn.execute("ROLLBACK")
            raise

    created_payloads = len(pending)
    receipt = EvidenceMaterializeReceiptR11(
        input_count=len(rows),
        unique_payload_count=len(unique_payloads),
        created_payload_count=created_payloads,
        reused_payload_count=len(unique_payloads) - created_payloads,
        created_evidence_count=created_evidence,
        reused_evidence_count=len(rows) - created_evidence,
    )
    return EvidenceBatchResultR11(
        refs=tuple(refs[item.content_hash] for item in rows),
        receipt=receipt,
        payload_fsync_count=fsyncs,
        metadata_transaction_count=metadata_transactions,
    )


@dataclass
class _Completion:
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


@dataclass(frozen=True)
class DurabilityTicketR11:
    sequence: int
    _completion: _Completion = field(repr=False, compare=False)


@dataclass
class _Request:
    sequence: int
    kind: str
    weight: int
    completion: _Completion
    evidence: tuple[ImmutableEvidenceObjectR11, ...] = ()
    events: tuple[ImmutableEventObjectR11, ...] = ()
    trace_batch_id: str | None = None
    fail_hook: FailHook | None = None
    evidence_set_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    checkpoint_state: Mapping[str, Any] | None = None


class _BoundedQueue:
    """Queued plus in-flight requests remain charged until durable completion."""

    def __init__(self, max_items: int, max_bytes: int):
        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self.ready: deque[_Request] = deque()
        self.items = 0
        self.bytes = 0
        self.peak_items = 0
        self.peak_bytes = 0
        self.closed = False
        self.failure: BaseException | None = None
        self.cv = threading.Condition()

    def put(
        self,
        req: _Request,
        timeout: float | None,
        assign_sequence: Callable[[], int],
    ) -> int:
        if req.weight > self.max_bytes:
            raise IORuntimeBudgetExceededR11(
                f"R11_IO_REQUEST_EXCEEDS_QUEUE_BUDGET:{req.weight}:{self.max_bytes}"
            )
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.cv:
            while self.items >= self.max_items or self.bytes + req.weight > self.max_bytes:
                self._assert_available()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("R11_IO_QUEUE_BACKPRESSURE_TIMEOUT")
                self.cv.wait(remaining)
            self._assert_available()
            req.sequence = int(assign_sequence())
            self.ready.append(req)
            self.items += 1
            self.bytes += req.weight
            self.peak_items = max(self.peak_items, self.items)
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.cv.notify_all()
            return req.sequence

    def get(self) -> _Request | None:
        with self.cv:
            while not self.ready and not self.closed and self.failure is None:
                self.cv.wait()
            if self.failure is not None:
                return None
            return self.ready.popleft() if self.ready else None

    def done(self, req: _Request) -> None:
        with self.cv:
            self.items -= 1
            self.bytes -= req.weight
            if self.items < 0 or self.bytes < 0:
                raise RuntimeError("R11_IO_QUEUE_ACCOUNTING_UNDERFLOW")
            self.cv.notify_all()

    def fail(self, exc: BaseException) -> list[_Request]:
        with self.cv:
            self.failure = self.failure or exc
            pending = list(self.ready)
            self.ready.clear()
            for req in pending:
                self.items -= 1
                self.bytes -= req.weight
            self.cv.notify_all()
            return pending

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def _assert_available(self) -> None:
        if self.failure is not None:
            raise StorageWriterFailedR11("R11_STORAGE_WRITER_FAILED") from self.failure
        if self.closed:
            raise IORuntimeErrorR11("R11_STORAGE_WRITER_QUEUE_CLOSED")


class IOThroughputRuntimeR11:
    """Single bounded storage writer for evidence, journal and checkpoint IO."""

    def __init__(self, config: IOThroughputConfigR11):
        self.config = config
        self.market_budget = MarketCacheBudgetGuardR11(config.budgets.market_cache_bytes)
        self._queue = _BoundedQueue(
            config.queue_max_items, config.budgets.evidence_queue_bytes
        )
        self._seq = 0
        self._durable = 0
        self._stats = {name: 0 for name in IORuntimeStatsR11.__dataclass_fields__}
        self._state = threading.Condition()
        self._checkpoint_cv = threading.Condition()
        self._checkpoint_bytes = 0
        self._failure: BaseException | None = None
        self._closed = False
        self._startup = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._writer_main, name="cb16-r11-storage-writer", daemon=True
        )
        self._thread.start()
        self._startup.wait()
        if self._startup_error is not None:
            raise StorageWriterFailedR11("R11_STORAGE_WRITER_STARTUP_FAILED") from self._startup_error

    @staticmethod
    def _weight(
        evidence: Sequence[ImmutableEvidenceObjectR11],
        events: Sequence[ImmutableEventObjectR11] = (),
    ) -> int:
        return sum(len(x.payload_bytes) + _OBJECT_OVERHEAD for x in evidence) + sum(
            len(x.payload_bytes) + _OBJECT_OVERHEAD for x in events
        )

    def _next_sequence(self) -> int:
        with self._state:
            self._seq += 1
            return self._seq

    def _submit(
        self,
        kind: str,
        weight: int,
        timeout: float | None,
        **kwargs: Any,
    ) -> DurabilityTicketR11:
        with self._state:
            if self._closed:
                raise IORuntimeErrorR11("R11_IO_RUNTIME_CLOSED")
            if self._failure is not None:
                raise StorageWriterFailedR11("R11_STORAGE_WRITER_FAILED") from self._failure
        completion = _Completion()
        req = _Request(0, kind, int(weight), completion, **kwargs)
        sequence = self._queue.put(req, timeout, self._next_sequence)
        with self._state:
            self._stats["submitted_requests"] += 1
            self._stats["queue_peak_items"] = max(
                self._stats["queue_peak_items"], self._queue.peak_items
            )
            self._stats["queue_peak_bytes"] = max(
                self._stats["queue_peak_bytes"], self._queue.peak_bytes
            )
        return DurabilityTicketR11(sequence, completion)

    def submit_evidence(
        self,
        items: Iterable[EvidenceItemR11],
        *,
        fail_hook: FailHook | None = None,
        timeout: float | None = None,
    ) -> DurabilityTicketR11:
        rows = tuple(ImmutableEvidenceObjectR11.from_item(item) for item in items)
        if not rows:
            raise ValueError("R11_IO_EMPTY_EVIDENCE_SUBMISSION")
        weight = self._weight(rows)
        if len(rows) > self.config.writer_batch_max_objects:
            raise IORuntimeBudgetExceededR11("R11_IO_EVIDENCE_OBJECT_LIMIT_EXCEEDED")
        if weight > self.config.writer_batch_max_bytes:
            raise IORuntimeBudgetExceededR11("R11_IO_EVIDENCE_BATCH_BYTES_EXCEEDED")
        return self._submit(
            "EVIDENCE", weight, timeout, evidence=rows, fail_hook=fail_hook
        )

    def submit_trace_commit(
        self,
        evidence_items: Iterable[EvidenceItemR11],
        events: Iterable[EventItemR11],
        *,
        trace_batch_id: str,
        fail_hook: FailHook | None = None,
        timeout: float | None = None,
    ) -> DurabilityTicketR11:
        evidence = tuple(
            ImmutableEvidenceObjectR11.from_item(item) for item in evidence_items
        )
        frozen_events = tuple(ImmutableEventObjectR11.from_item(item) for item in events)
        if not evidence or not frozen_events:
            raise ValueError("R11_IO_TRACE_COMMIT_REQUIRES_EVIDENCE_AND_EVENTS")
        weight = self._weight(evidence, frozen_events)
        if len(evidence) > self.config.writer_batch_max_objects:
            raise IORuntimeBudgetExceededR11("R11_IO_TRACE_OBJECT_LIMIT_EXCEEDED")
        if weight > self.config.writer_batch_max_bytes:
            raise IORuntimeBudgetExceededR11("R11_IO_TRACE_BATCH_BYTES_EXCEEDED")
        return self._submit(
            "TRACE",
            weight,
            timeout,
            evidence=evidence,
            events=frozen_events,
            trace_batch_id=str(trace_batch_id),
            fail_hook=fail_hook,
        )

    def seal_evidence_set(
        self,
        evidence_set_id: str,
        evidence_ids: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        ticket = self._submit(
            "EVIDENCE_SET",
            max(_OBJECT_OVERHEAD, 64 * len(evidence_ids)),
            timeout,
            evidence_set_id=str(evidence_set_id),
            evidence_ids=tuple(str(x) for x in evidence_ids),
        )
        return self.wait(ticket, timeout=timeout)

    def put_checkpoint_durable(
        self,
        state: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> CheckpointObjectRefR11:
        if self.config.checkpoint_root is None:
            raise IORuntimeErrorR11("R11_IO_CHECKPOINT_STORE_NOT_CONFIGURED")
        tensor_bytes = 0
        for value in state.values():
            if not callable(getattr(value, "numel", None)) or not callable(
                getattr(value, "element_size", None)
            ):
                raise TypeError("R11_CHECKPOINT_NON_TENSOR_VALUE")
            tensor_bytes += int(value.numel()) * int(value.element_size())
        # CheckpointStore canonicalizes to a CPU clone before serialization; charge both
        # the submitted state reference and that staging clone against the explicit budget.
        charge = max(_OBJECT_OVERHEAD, tensor_bytes * 2)
        limit = self.config.budgets.checkpoint_staging_bytes
        if charge > limit:
            raise IORuntimeBudgetExceededR11(
                f"R11_CHECKPOINT_STAGING_BUDGET_EXCEEDED:{charge}:{limit}"
            )
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._checkpoint_cv:
            while self._checkpoint_bytes + charge > limit:
                if self._failure is not None:
                    raise StorageWriterFailedR11("R11_STORAGE_WRITER_FAILED") from self._failure
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("R11_CHECKPOINT_STAGING_BACKPRESSURE_TIMEOUT")
                self._checkpoint_cv.wait(remaining)
            self._checkpoint_bytes += charge
        try:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            ticket = self._submit(
                "CHECKPOINT", _OBJECT_OVERHEAD, remaining, checkpoint_state=state
            )
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            return self.wait(ticket, timeout=remaining)
        finally:
            with self._checkpoint_cv:
                self._checkpoint_bytes -= charge
                self._checkpoint_cv.notify_all()

    def wait(self, ticket: DurabilityTicketR11, *, timeout: float | None = None) -> Any:
        if not ticket._completion.event.wait(timeout):
            raise TimeoutError(f"R11_IO_DURABILITY_WAIT_TIMEOUT:{ticket.sequence}")
        if ticket._completion.error is not None:
            raise StorageWriterFailedR11(
                f"R11_STORAGE_WRITER_REQUEST_FAILED:{ticket.sequence}"
            ) from ticket._completion.error
        return ticket._completion.result

    def durable_barrier(
        self,
        ticket: DurabilityTicketR11 | None = None,
        *,
        timeout: float | None = None,
    ) -> int:
        with self._state:
            target = self._seq if ticket is None else int(ticket.sequence)
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._durable < target:
                if self._failure is not None:
                    raise StorageWriterFailedR11(
                        "R11_STORAGE_WRITER_FAILED_AT_BARRIER"
                    ) from self._failure
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"R11_IO_DURABILITY_BARRIER_TIMEOUT:{target}")
                self._state.wait(remaining)
            return self._durable

    def stats(self) -> IORuntimeStatsR11:
        with self._state:
            self._stats["queue_peak_items"] = max(
                self._stats["queue_peak_items"], self._queue.peak_items
            )
            self._stats["queue_peak_bytes"] = max(
                self._stats["queue_peak_bytes"], self._queue.peak_bytes
            )
            return IORuntimeStatsR11(**self._stats)

    def close(self, *, drain: bool = True, raise_on_failure: bool = True) -> None:
        with self._state:
            if self._closed:
                return
        error: BaseException | None = None
        if drain and self._failure is None:
            try:
                self.durable_barrier()
            except BaseException as exc:  # surface durable-barrier failure after cleanup
                error = exc
        with self._state:
            self._closed = True
        self._queue.close()
        self._thread.join()
        if error is None and self._failure is not None:
            error = StorageWriterFailedR11("R11_STORAGE_WRITER_FAILED")
            error.__cause__ = self._failure
        if error is not None and raise_on_failure:
            raise error

    def __enter__(self) -> "IOThroughputRuntimeR11":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(raise_on_failure=exc is None)

    def _record_batch_stats(self, result: EvidenceBatchResultR11) -> None:
        with self._state:
            self._stats["payload_fsyncs"] += result.payload_fsync_count
            self._stats["evidence_metadata_transactions"] += (
                result.metadata_transaction_count
            )
            self._stats["writer_batches"] += 1
            self._stats["created_payloads"] += result.receipt.created_payload_count
            self._stats["reused_payloads"] += result.receipt.reused_payload_count

    def _mark_success(self, req: _Request, result: Any) -> None:
        self._queue.done(req)
        with self._state:
            if req.sequence != self._durable + 1:
                raise RuntimeError(
                    f"R11_IO_DURABLE_SEQUENCE_GAP:{self._durable}:{req.sequence}"
                )
            self._durable = req.sequence
            self._stats["durable_requests"] += 1
            self._state.notify_all()
        req.completion.result = result
        req.completion.event.set()

    def _writer_main(self) -> None:
        evidence: EvidenceStoreR11 | None = None
        journal: EventJournalR11 | None = None
        checkpoints: CheckpointStoreR11 | None = None
        active: _Request | None = None
        try:
            evidence = EvidenceStoreR11(
                metadata_root=self.config.evidence_metadata_root,
                payload_roots=self.config.evidence_payload_roots,
                segment_target_bytes=self.config.segment_target_bytes,
                codec=self.config.evidence_codec,
                sqlite_synchronous=self.config.sqlite_synchronous,
            )
            journal = EventJournalR11(
                self.config.journal_root, synchronous=self.config.sqlite_synchronous
            )
            if self.config.checkpoint_root is not None:
                checkpoints = CheckpointStoreR11(
                    self.config.checkpoint_root,
                    synchronous=self.config.sqlite_synchronous,
                )
            self._startup.set()

            while True:
                req = self._queue.get()
                if req is None:
                    return
                active = req
                if req.kind == "EVIDENCE":
                    result = put_evidence_batched_r11(
                        evidence, req.evidence, fail_hook=req.fail_hook
                    )
                    self._record_batch_stats(result)
                elif req.kind == "TRACE":
                    evidence_result = put_evidence_batched_r11(
                        evidence, req.evidence, fail_hook=req.fail_hook
                    )
                    self._record_batch_stats(evidence_result)
                    if req.fail_hook is not None:
                        req.fail_hook(FAIL_AFTER_METADATA_COMMIT_BEFORE_EVENT_SEAL)
                    trace_batch = journal.seal_trace_batch(
                        (event.thaw() for event in req.events),
                        trace_batch_id=req.trace_batch_id,
                        fail_hook=req.fail_hook,
                    )
                    result = TraceCommitResultR11(evidence_result, trace_batch)
                elif req.kind == "EVIDENCE_SET":
                    result = evidence.seal_evidence_set(
                        str(req.evidence_set_id), req.evidence_ids
                    )
                elif req.kind == "CHECKPOINT":
                    if checkpoints is None or req.checkpoint_state is None:
                        raise IORuntimeErrorR11(
                            "R11_IO_CHECKPOINT_STORE_NOT_CONFIGURED"
                        )
                    result = checkpoints.put_state_dict(req.checkpoint_state)
                    with self._state:
                        key = "checkpoint_created" if result.created else "checkpoint_reused"
                        self._stats[key] += 1
                else:
                    raise RuntimeError(f"R11_IO_UNKNOWN_REQUEST_KIND:{req.kind}")
                self._mark_success(req, result)
                active = None
        except BaseException as exc:
            if not self._startup.is_set():
                self._startup_error = exc
                self._startup.set()
            self._failure = exc
            if active is not None:
                active.completion.error = exc
                active.completion.event.set()
                try:
                    self._queue.done(active)
                except Exception:
                    pass
            for req in self._queue.fail(exc):
                req.completion.error = exc
                req.completion.event.set()
            with self._state:
                self._state.notify_all()
            with self._checkpoint_cv:
                self._checkpoint_cv.notify_all()
        finally:
            for store in (checkpoints, journal, evidence):
                if store is not None:
                    try:
                        store.close()
                    except Exception:
                        pass
            self._startup.set()
