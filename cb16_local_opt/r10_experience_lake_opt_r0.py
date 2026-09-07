from __future__ import annotations

"""Experimental Experience Lake persistence candidates for R10 qualification.

This module is deliberately not wired into the canonical R10 campaign. It provides
bulk persistence candidates that must pass semantic, exactly-once, crash/recovery,
and host benchmark gates before any later runtime adoption.

Candidate levels:
- SHARD_PARALLEL_OBJECTWISE: uses the existing per-object put() unchanged, but runs
  independent shards concurrently. This preserves per-object durability semantics.
- SHARD_PARALLEL_BATCHED: preserves payload fsync and immutable identity, but batches
  metadata transactions within each shard. Fault-prefix behavior can differ, so this
  candidate requires recovery-equivalence qualification before adoption.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .sharded_experience_lake import ExperienceObject, ExperienceRef, ShardedExperienceLake

MODE_OBJECTWISE = "SHARD_PARALLEL_OBJECTWISE"
MODE_BATCHED = "SHARD_PARALLEL_BATCHED"


@dataclass(frozen=True)
class PutManyReceiptR0:
    schema: str
    mode: str
    object_count: int
    created_count: int
    reused_count: int
    shards_used: int
    max_workers: int
    batch_size: int | None
    wall_seconds: float
    objects_per_second: float
    semantic_note: str


@dataclass(frozen=True)
class _Prepared:
    input_index: int
    obj: ExperienceObject
    raw: bytes
    payload_hash: str
    identity_hash: str
    payload_path: str
    stored_bytes: int
    created_at: float


FailHook = Callable[[str, int, int], None]


def _chunks(rows: Sequence[tuple[int, ExperienceObject]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _normalized_workers(lake: ShardedExperienceLake, max_workers: int | None) -> int:
    if max_workers is None:
        return max(1, int(lake.shard_count))
    return max(1, min(int(max_workers), int(lake.shard_count)))


def _bucket_by_shard(lake: ShardedExperienceLake, objects: Sequence[ExperienceObject]):
    buckets: dict[int, list[tuple[int, ExperienceObject]]] = {}
    for i, obj in enumerate(objects):
        idx = int(lake._index(obj.object_id))
        buckets.setdefault(idx, []).append((i, obj))
    return buckets


def _with_shard(ref: ExperienceRef, shard: int) -> ExperienceRef:
    return ExperienceRef(
        object_id=ref.object_id,
        object_type=ref.object_type,
        generation=ref.generation,
        shard=int(shard),
        identity_hash=ref.identity_hash,
        payload_hash=ref.payload_hash,
        payload_path=ref.payload_path,
        bytes_raw=ref.bytes_raw,
        bytes_stored=ref.bytes_stored,
    )


def _put_shard_objectwise(
    lake: ShardedExperienceLake,
    shard_idx: int,
    rows: Sequence[tuple[int, ExperienceObject]],
):
    out: list[tuple[int, ExperienceRef, bool]] = []
    for input_index, obj in rows:
        ref, created = lake.put(obj)
        if int(ref.shard) != int(shard_idx):
            raise RuntimeError(f"R10_LAKE_OPT_SHARD_ROUTE_DRIFT:{obj.object_id}")
        out.append((input_index, ref, bool(created)))
    return out


def _old_ref_from_row(obj: ExperienceObject, old: Sequence[Any]) -> ExperienceRef:
    return ExperienceRef(
        object_id=obj.object_id,
        object_type=str(old[0]),
        generation=int(old[1]),
        shard=-1,
        identity_hash=str(old[5]),
        payload_hash=str(old[6]),
        payload_path=str(old[7]),
        bytes_raw=int(old[8]),
        bytes_stored=int(old[9]),
    )


def _put_one_batch(
    lake: ShardedExperienceLake,
    shard_idx: int,
    rows: Sequence[tuple[int, ExperienceObject]],
    *,
    batch_ordinal: int,
    fail_hook: FailHook | None,
):
    shard = lake._shards[int(shard_idx)]
    prepared: list[_Prepared] = []

    for input_index, obj in rows:
        if not obj.object_id or not obj.object_type:
            raise ValueError("object id/type required")
        raw = obj.payload_bytes
        payload_hash = obj.payload_hash
        identity_hash = obj.identity_hash
        path, stored_bytes = shard._atomic_write_payload(payload_hash, raw)
        prepared.append(_Prepared(
            input_index=input_index,
            obj=obj,
            raw=raw,
            payload_hash=payload_hash,
            identity_hash=identity_hash,
            payload_path=str(path),
            stored_bytes=int(stored_bytes),
            created_at=time.time(),
        ))

    if fail_hook is not None:
        fail_hook("AFTER_PAYLOADS_BEFORE_METADATA", int(shard_idx), int(batch_ordinal))

    conn = shard.conn
    conn.execute("BEGIN IMMEDIATE")
    out: list[tuple[int, ExperienceRef, bool]] = []
    try:
        for p in prepared:
            old = conn.execute(
                """
                SELECT object_type,generation,policy_weight_hash,snapshot_hash,lineage_hash,
                       identity_hash,payload_hash,payload_path,bytes_raw,bytes_stored
                FROM objects WHERE object_id=?
                """,
                (p.obj.object_id,),
            ).fetchone()
            if old is not None:
                if old[5] != p.identity_hash:
                    raise RuntimeError(f"EXPERIENCE_ID_CONTENT_CONFLICT:{p.obj.object_id}")
                out.append((p.input_index, _old_ref_from_row(p.obj, old), False))
                continue

            conn.execute(
                """
                INSERT INTO objects(
                    object_id,object_type,generation,policy_weight_hash,snapshot_hash,
                    lineage_hash,identity_hash,payload_hash,payload_path,
                    bytes_raw,bytes_stored,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    p.obj.object_id,
                    p.obj.object_type,
                    int(p.obj.generation),
                    p.obj.policy_weight_hash,
                    p.obj.snapshot_hash,
                    p.obj.lineage_hash,
                    p.identity_hash,
                    p.payload_hash,
                    p.payload_path,
                    len(p.raw),
                    p.stored_bytes,
                    p.created_at,
                ),
            )
            out.append((
                p.input_index,
                ExperienceRef(
                    object_id=p.obj.object_id,
                    object_type=p.obj.object_type,
                    generation=int(p.obj.generation),
                    shard=-1,
                    identity_hash=p.identity_hash,
                    payload_hash=p.payload_hash,
                    payload_path=p.payload_path,
                    bytes_raw=len(p.raw),
                    bytes_stored=p.stored_bytes,
                ),
                True,
            ))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise

    if fail_hook is not None:
        fail_hook("AFTER_METADATA_COMMIT", int(shard_idx), int(batch_ordinal))
    return out


def _put_shard_batched(
    lake: ShardedExperienceLake,
    shard_idx: int,
    rows: Sequence[tuple[int, ExperienceObject]],
    *,
    batch_size: int,
    fail_hook: FailHook | None,
):
    out: list[tuple[int, ExperienceRef, bool]] = []
    for batch_ordinal, batch in enumerate(_chunks(rows, int(batch_size))):
        for input_index, ref, created in _put_one_batch(
            lake,
            shard_idx,
            batch,
            batch_ordinal=batch_ordinal,
            fail_hook=fail_hook,
        ):
            out.append((input_index, _with_shard(ref, shard_idx), created))
    return out


def put_many_r10_opt_r0(
    lake: ShardedExperienceLake,
    objects: Iterable[ExperienceObject],
    *,
    mode: str = MODE_OBJECTWISE,
    max_workers: int | None = None,
    batch_size: int = 128,
    fail_hook: FailHook | None = None,
) -> tuple[list[ExperienceRef], PutManyReceiptR0]:
    objs = list(objects)
    if mode not in {MODE_OBJECTWISE, MODE_BATCHED}:
        raise ValueError(f"UNKNOWN_R10_LAKE_OPT_MODE:{mode}")
    if mode == MODE_BATCHED and int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if not objs:
        receipt = PutManyReceiptR0(
            schema="CB16_R10_EXPERIENCE_LAKE_PUT_MANY_R0",
            mode=mode,
            object_count=0,
            created_count=0,
            reused_count=0,
            shards_used=0,
            max_workers=0,
            batch_size=(int(batch_size) if mode == MODE_BATCHED else None),
            wall_seconds=0.0,
            objects_per_second=0.0,
            semantic_note="QUALIFICATION_ONLY_NOT_CANONICAL_RUNTIME",
        )
        return [], receipt

    buckets = _bucket_by_shard(lake, objs)
    workers = min(_normalized_workers(lake, max_workers), len(buckets))
    started = time.perf_counter()
    rows: list[tuple[int, ExperienceRef, bool]] = []

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cb16-lake-r0") as pool:
        futs = []
        for shard_idx, shard_rows in sorted(buckets.items()):
            if mode == MODE_OBJECTWISE:
                futs.append(pool.submit(_put_shard_objectwise, lake, shard_idx, shard_rows))
            else:
                futs.append(pool.submit(
                    _put_shard_batched,
                    lake,
                    shard_idx,
                    shard_rows,
                    batch_size=int(batch_size),
                    fail_hook=fail_hook,
                ))
        for fut in as_completed(futs):
            rows.extend(fut.result())

    rows.sort(key=lambda x: x[0])
    if [i for i, _, _ in rows] != list(range(len(objs))):
        raise RuntimeError("R10_LAKE_OPT_INPUT_ORDER_COVERAGE_FAIL")
    refs = [ref for _, ref, _ in rows]
    created_count = sum(1 for _, _, created in rows if created)
    wall = max(0.0, time.perf_counter() - started)
    rate = (len(objs) / wall) if wall > 0 else math.inf
    receipt = PutManyReceiptR0(
        schema="CB16_R10_EXPERIENCE_LAKE_PUT_MANY_R0",
        mode=mode,
        object_count=len(objs),
        created_count=created_count,
        reused_count=len(objs) - created_count,
        shards_used=len(buckets),
        max_workers=workers,
        batch_size=(int(batch_size) if mode == MODE_BATCHED else None),
        wall_seconds=wall,
        objects_per_second=rate,
        semantic_note=(
            "PER_OBJECT_DURABILITY_PATH_PRESERVED__CROSS_SHARD_ORDER_NONAUTHORITATIVE"
            if mode == MODE_OBJECTWISE
            else "PAYLOAD_FSYNC_PRESERVED__METADATA_BATCHED__RECOVERY_EQUIVALENCE_REQUIRED"
        ),
    )
    return refs, receipt


def semantic_ref_tuple(ref: ExperienceRef) -> tuple[Any, ...]:
    return (
        ref.object_id,
        ref.object_type,
        int(ref.generation),
        int(ref.shard),
        ref.identity_hash,
        ref.payload_hash,
        int(ref.bytes_raw),
        int(ref.bytes_stored),
    )
