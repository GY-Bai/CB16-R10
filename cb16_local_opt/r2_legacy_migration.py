from __future__ import annotations

"""Read-only legacy Experience Lake -> R2 semantic migration qualification.

The legacy R10 Experience Lake makes generation and current Champion authority part of
EVIDENCE_PACKAGE identity.  R2 deliberately removes those facts from immutable evidence
content.  This module therefore compares a semantic projection rather than legacy object
identity.

This module is stdlib-only and does not import torch or the legacy runtime package.
"""

import json
import sqlite3
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .r2_evidence_storage import (
    R2EvidenceItem,
    R2EvidenceSetRef,
    R2EvidenceStore,
    R2MaterializeReceipt,
    canonical_json_bytes,
    sha256_bytes,
    sha256_obj,
)

LEGACY_EVIDENCE_TYPE = "EVIDENCE_PACKAGE"
LEGACY_SCHEMA = "CB16_R10_2_EVIDENCE_PACKAGE_V1"
R2_SCHEMA = "CB16_R2_EVIDENCE_PACKAGE_V1"
PROJECTION_SCHEMA = "CB16_EVIDENCE_SEMANTIC_PROJECTION_V1"


@dataclass(frozen=True)
class LegacyEvidenceRow:
    object_id: str
    generation: int
    policy_weight_hash: str
    parent_snapshot_hash: str
    lineage_hash: str
    identity_hash: str
    payload_hash: str
    payload_path: str
    bytes_raw: int
    bytes_stored: int
    shard_db: str


@dataclass(frozen=True)
class LegacyGenerationQualification:
    generation: int
    legacy_object_count: int
    unique_legacy_payload_count: int
    projection_mismatch_count: int
    metadata_mismatch_count: int
    duplicate_evidence_id_count: int
    evidence_set_hash: str | None
    created_payload_count: int | None
    reused_payload_count: int | None
    semantic_projection_aggregate_hash: str


def semantic_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project either legacy or R2 payload into generation-independent evidence semantics."""
    obj = dict(payload)
    obj.pop("generation", None)
    obj["schema"] = PROJECTION_SCHEMA
    return obj


def semantic_projection_hash(payload: Mapping[str, Any]) -> str:
    return sha256_obj(semantic_projection(payload))


def _evidence_id_from_legacy_object_id(object_id: str, generation: int) -> str:
    prefix = f"R102:G{int(generation)}:"
    if not object_id.startswith(prefix):
        raise RuntimeError(f"R2_LEGACY_OBJECT_ID_FORMAT_MISMATCH:{object_id}:{prefix}")
    evidence_id = object_id[len(prefix):]
    if not evidence_id:
        raise RuntimeError(f"R2_LEGACY_EMPTY_EVIDENCE_ID:{object_id}")
    return evidence_id


def _open_legacy_ro(db_path: Path) -> sqlite3.Connection:
    # mode=ro prevents accidental mutation of canonical metadata.  Do not use immutable=1:
    # active WAL content must remain visible when the source campaign is still open.
    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def iter_legacy_generation_rows(
    legacy_lake_root: str | Path,
    *,
    generation: int,
) -> Iterator[LegacyEvidenceRow]:
    root = Path(legacy_lake_root)
    dbs = sorted((root / "metadata").glob("experience_*.sqlite"))
    if not dbs:
        raise FileNotFoundError(f"R2_LEGACY_METADATA_SHARDS_NOT_FOUND:{root}")
    rows: list[LegacyEvidenceRow] = []
    for db in dbs:
        conn = _open_legacy_ro(db)
        try:
            query = """
                SELECT object_id,generation,policy_weight_hash,snapshot_hash,lineage_hash,
                       identity_hash,payload_hash,payload_path,bytes_raw,bytes_stored
                FROM objects
                WHERE generation=? AND object_type=?
                ORDER BY object_id
            """
            for r in conn.execute(query, (int(generation), LEGACY_EVIDENCE_TYPE)):
                rows.append(LegacyEvidenceRow(
                    object_id=str(r[0]), generation=int(r[1]), policy_weight_hash=str(r[2]),
                    parent_snapshot_hash=str(r[3]), lineage_hash=str(r[4]), identity_hash=str(r[5]),
                    payload_hash=str(r[6]), payload_path=str(r[7]), bytes_raw=int(r[8]),
                    bytes_stored=int(r[9]), shard_db=str(db),
                ))
        finally:
            conn.close()
    rows.sort(key=lambda x: x.object_id)
    yield from rows


def read_legacy_payload(row: LegacyEvidenceRow) -> dict[str, Any]:
    stored = Path(row.payload_path).read_bytes()
    try:
        raw = zlib.decompress(stored)
    except zlib.error as exc:
        raise RuntimeError(f"R2_LEGACY_PAYLOAD_DECOMPRESS_FAIL:{row.object_id}") from exc
    if sha256_bytes(raw) != row.payload_hash:
        raise RuntimeError(f"R2_LEGACY_PAYLOAD_HASH_MISMATCH:{row.object_id}")
    if len(raw) != row.bytes_raw:
        raise RuntimeError(f"R2_LEGACY_PAYLOAD_SIZE_MISMATCH:{row.object_id}")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError(f"R2_LEGACY_PAYLOAD_NOT_OBJECT:{row.object_id}")
    return obj


def legacy_row_to_r2_item(
    row: LegacyEvidenceRow,
    payload: Mapping[str, Any],
) -> R2EvidenceItem:
    if payload.get("schema") != LEGACY_SCHEMA:
        raise RuntimeError(f"R2_LEGACY_SCHEMA_MISMATCH:{row.object_id}:{payload.get('schema')}")
    if int(payload.get("generation", -1)) != int(row.generation):
        raise RuntimeError(f"R2_LEGACY_GENERATION_MISMATCH:{row.object_id}")
    teacher_protocol_hash = str(payload.get("teacher_protocol_hash", ""))
    if not teacher_protocol_hash:
        raise RuntimeError(f"R2_LEGACY_TEACHER_PROTOCOL_MISSING:{row.object_id}")

    r2_payload = dict(payload)
    r2_payload.pop("generation", None)
    r2_payload["schema"] = R2_SCHEMA
    item = R2EvidenceItem(
        evidence_id=_evidence_id_from_legacy_object_id(row.object_id, row.generation),
        parent_snapshot_hash=row.parent_snapshot_hash,
        lineage_hash=row.lineage_hash,
        teacher_protocol_hash=teacher_protocol_hash,
        payload=r2_payload,
    )
    if semantic_projection_hash(payload) != semantic_projection_hash(item.payload):
        raise RuntimeError(f"R2_LEGACY_SEMANTIC_PROJECTION_MISMATCH:{row.object_id}")
    return item


def qualify_legacy_generation(
    *,
    legacy_lake_root: str | Path,
    generation: int,
    store: R2EvidenceStore | None = None,
    evidence_set_id: str = "R2_TRAINING_EVIDENCE_SET_V1",
    limit: int | None = None,
) -> tuple[LegacyGenerationQualification, R2EvidenceSetRef | None, R2MaterializeReceipt | None]:
    rows = list(iter_legacy_generation_rows(legacy_lake_root, generation=generation))
    if limit is not None and int(limit) > 0:
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError(f"R2_LEGACY_NO_EVIDENCE_ROWS:G{generation}")

    items: list[R2EvidenceItem] = []
    projection_rows: list[tuple[str, str]] = []
    projection_mismatches = 0
    metadata_mismatches = 0
    evidence_ids: list[str] = []

    for row in rows:
        payload = read_legacy_payload(row)
        try:
            item = legacy_row_to_r2_item(row, payload)
        except RuntimeError as exc:
            if "SEMANTIC_PROJECTION_MISMATCH" in str(exc):
                projection_mismatches += 1
                continue
            raise
        # These are legacy metadata fields that remain immutable evidence authority in R2.
        if item.parent_snapshot_hash != row.parent_snapshot_hash or item.lineage_hash != row.lineage_hash:
            metadata_mismatches += 1
        evidence_ids.append(item.evidence_id)
        projection_rows.append((item.evidence_id, semantic_projection_hash(item.payload)))
        items.append(item)

    duplicate_ids = len(evidence_ids) - len(set(evidence_ids))
    aggregate_hash = sha256_obj({
        "schema":"CB16_R2_LEGACY_PROJECTION_AGGREGATE_V1",
        "entries":[{"evidence_id":eid,"projection_hash":h} for eid,h in sorted(projection_rows)],
    })

    evidence_set = None
    receipt = None
    if store is not None:
        evidence_set, receipt = store.materialize_evidence_set(
            evidence_set_id=evidence_set_id,
            items=items,
        )

    q = LegacyGenerationQualification(
        generation=int(generation),
        legacy_object_count=len(rows),
        unique_legacy_payload_count=len({r.payload_hash for r in rows}),
        projection_mismatch_count=projection_mismatches,
        metadata_mismatch_count=metadata_mismatches,
        duplicate_evidence_id_count=duplicate_ids,
        evidence_set_hash=(evidence_set.content_hash if evidence_set else None),
        created_payload_count=(receipt.created_payload_count if receipt else None),
        reused_payload_count=(receipt.reused_payload_count if receipt else None),
        semantic_projection_aggregate_hash=aggregate_hash,
    )
    return q, evidence_set, receipt


def qualify_legacy_generations(
    *,
    legacy_lake_root: str | Path,
    generations: Sequence[int],
    store: R2EvidenceStore | None = None,
    evidence_set_id: str = "R2_TRAINING_EVIDENCE_SET_V1",
    limit: int | None = None,
) -> dict[str, Any]:
    out = []
    for generation in generations:
        q, _set_ref, _receipt = qualify_legacy_generation(
            legacy_lake_root=legacy_lake_root,
            generation=int(generation),
            store=store,
            evidence_set_id=evidence_set_id,
            limit=limit,
        )
        out.append(asdict(q))
    hashes = [x["semantic_projection_aggregate_hash"] for x in out]
    set_hashes = [x["evidence_set_hash"] for x in out if x["evidence_set_hash"] is not None]
    return {
        "schema":"CB16_R2_LEGACY_MIGRATION_QUALIFICATION_V1",
        "generations":out,
        "all_projection_hashes_equal":len(set(hashes)) == 1,
        "all_materialized_evidence_set_hashes_equal":(len(set(set_hashes)) == 1 if set_hashes else None),
        "all_projection_mismatches_zero":all(x["projection_mismatch_count"] == 0 for x in out),
        "all_metadata_mismatches_zero":all(x["metadata_mismatch_count"] == 0 for x in out),
        "all_duplicate_evidence_ids_zero":all(x["duplicate_evidence_id_count"] == 0 for x in out),
        "writes_to_legacy_lake":False,
        "legacy_generation_and_policy_authority_removed_from_r2_content":True,
    }
