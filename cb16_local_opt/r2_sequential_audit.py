from __future__ import annotations

"""Physical-order R2 pack audit.

Production HDD verification must scan segment files sequentially.  Hash-order locator
reads would turn an append-only pack back into random seeks and defeat the storage layout.
"""

import os
import zlib
from pathlib import Path
from typing import Any

from .r2_evidence_storage import PACK_HEADER, PACK_MAGIC, _decompress, sha256_bytes, sha256_obj


def _advise_sequential(fd: int) -> None:
    try:
        advice = getattr(os, "POSIX_FADV_SEQUENTIAL")
        os.posix_fadvise(fd, 0, 0, advice)
    except (AttributeError, OSError):
        pass


def audit_r2_store_sequential(store, *, verify_payloads: bool = True) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    indexed = {
        (str(path), int(offset)): (str(h), int(lane))
        for h, lane, path, offset in store.conn.execute(
            "SELECT content_hash,lane,segment_path,offset FROM payloads"
        )
    }
    seen: set[tuple[str, int]] = set()
    segment_count = 0
    payload_count = 0
    bytes_scanned = 0

    for lane in range(store.lane_count):
        for path in store._segments(lane):
            segment_count += 1
            offset = 0
            with path.open("rb", buffering=1024 * 1024) as f:
                _advise_sequential(f.fileno())
                while True:
                    record_offset = offset
                    one = f.read(1)
                    if not one:
                        break
                    tag_len = one[0]
                    tag = f.read(tag_len)
                    header = f.read(PACK_HEADER.size)
                    if len(tag) != tag_len or len(header) != PACK_HEADER.size:
                        errors.append({"path":str(path),"offset":record_offset,"error":"INCOMPLETE_RECORD_HEADER"})
                        break
                    magic, hash_b, raw_len, stored_len, crc = PACK_HEADER.unpack(header)
                    if magic != PACK_MAGIC:
                        errors.append({"path":str(path),"offset":record_offset,"error":"PACK_MAGIC_MISMATCH"})
                        break
                    stored = f.read(stored_len)
                    if len(stored) != stored_len:
                        errors.append({"path":str(path),"offset":record_offset,"error":"INCOMPLETE_RECORD_PAYLOAD"})
                        break
                    end = f.tell()
                    h = hash_b.hex()
                    key = (str(path), record_offset)
                    seen.add(key)
                    payload_count += 1
                    bytes_scanned += end - record_offset
                    idx = indexed.get(key)
                    if idx != (h, lane):
                        errors.append({"path":str(path),"offset":record_offset,"error":"LOCATOR_INDEX_MISMATCH","pack_hash":h,"index":idx})
                    if verify_payloads:
                        if zlib.crc32(stored) & 0xFFFFFFFF != crc:
                            errors.append({"path":str(path),"offset":record_offset,"error":"CRC_MISMATCH"})
                        else:
                            try:
                                raw = _decompress(stored, tag.decode("ascii"))
                                if len(raw) != raw_len or sha256_bytes(raw) != h:
                                    errors.append({"path":str(path),"offset":record_offset,"error":"CONTENT_HASH_MISMATCH"})
                            except Exception as exc:
                                errors.append({"path":str(path),"offset":record_offset,"error":repr(exc)})
                    offset = end

    missing = sorted(set(indexed) - seen)
    for path, offset in missing:
        errors.append({"path":path,"offset":offset,"error":"INDEX_POINTS_TO_MISSING_PACK_RECORD"})

    sets = store.conn.execute(
        "SELECT evidence_set_hash,manifest_path,object_count FROM evidence_sets"
    ).fetchall()
    for h, path, n in sets:
        try:
            import json
            obj = json.loads(Path(path).read_text())
            if sha256_obj(obj) != h or int(obj["object_count"]) != int(n):
                raise RuntimeError("evidence-set manifest mismatch")
        except Exception as exc:
            errors.append({"evidence_set_hash":h,"error":repr(exc)})

    snaps = store.conn.execute(
        "SELECT snapshot_id,content_hash,manifest_path,evidence_set_hash FROM generation_snapshots"
    ).fetchall()
    for sid, h, path, set_h in snaps:
        try:
            import json
            obj = json.loads(Path(path).read_text())
            if sha256_obj(obj) != h or obj["evidence_set_hash"] != set_h:
                raise RuntimeError("generation snapshot mismatch")
        except Exception as exc:
            errors.append({"snapshot_id":sid,"error":repr(exc)})

    return {
        "schema":"CB16_R2_SEQUENTIAL_PACK_AUDIT_V1",
        "physical_order":"LANE_SEGMENT_OFFSET_ASCENDING",
        "sequential_readahead_hint":True,
        "payload_lanes":store.lane_count,
        "segments":segment_count,
        "payload_objects":payload_count,
        "bytes_scanned":bytes_scanned,
        "evidence_sets":len(sets),
        "generation_snapshots":len(snaps),
        "errors":errors,
        "pass":not errors,
    }
