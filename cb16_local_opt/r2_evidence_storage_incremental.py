from __future__ import annotations

"""R2 incremental payload-index recovery.

The original R2EvidenceStore recovery scans every pack from byte zero on each open.
For an append-only pack this is unnecessary once SQLite already contains durable
locators.  This R2-only subclass validates the last indexed record of each segment,
then scans only bytes after that verified boundary to recover fsync'd orphan records.

Full sequential payload/hash audit remains unchanged and still runs at qualification
boundaries; this optimization changes startup recovery cost, not audit authority.
"""

import os
import time
from pathlib import Path
from typing import Any

from .r2_evidence_storage import R2EvidenceStore


class R2IncrementalEvidenceStore(R2EvidenceStore):
    last_recovery_receipt: dict[str, Any] | None = None

    def _verified_indexed_end(self, path: Path, lane: int) -> int:
        row = self.conn.execute(
            "SELECT content_hash,offset FROM payloads "
            "WHERE segment_path=? AND lane=? ORDER BY offset DESC LIMIT 1",
            (str(path), int(lane)),
        ).fetchone()
        if row is None:
            return 0
        expected_hash, offset = str(row[0]), int(row[1])
        size = path.stat().st_size
        if offset < 0 or offset >= size:
            raise RuntimeError(
                f"R2_INDEXED_PACK_BOUNDARY_OUT_OF_RANGE:{path}:{offset}:{size}"
            )
        decoded = self._decode(path, offset)
        if decoded is None:
            raise RuntimeError(f"R2_INDEXED_PACK_BOUNDARY_INCOMPLETE:{path}:{offset}")
        ref, _raw, end = decoded
        if ref.content_hash != expected_hash:
            raise RuntimeError(
                f"R2_INDEXED_PACK_BOUNDARY_HASH_MISMATCH:{path}:{offset}"
            )
        if end > size:
            raise RuntimeError(f"R2_INDEXED_PACK_BOUNDARY_PAST_EOF:{path}:{end}:{size}")
        return int(end)

    def recover_payload_index(self) -> dict[str, Any]:
        """Recover only unindexed append tails after one verified boundary canary."""
        discovered = 0
        truncated: list[dict[str, Any]] = []
        segment_receipts: list[dict[str, Any]] = []
        total_bytes = 0
        tail_bytes_scanned = 0
        indexed_prefix_bytes_skipped = 0

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for lane in range(self.lane_count):
                for path in self._segments(lane):
                    size = int(path.stat().st_size)
                    total_bytes += size
                    offset = self._verified_indexed_end(path, lane)
                    if offset > size:
                        raise RuntimeError(
                            f"R2_INDEXED_PACK_END_PAST_FILE:{path}:{offset}:{size}"
                        )
                    indexed_prefix_bytes_skipped += offset
                    start_offset = offset
                    recovered_here = 0
                    while offset < size:
                        record_offset = offset
                        decoded = self._decode(path, offset)
                        if decoded is None:
                            with path.open("r+b") as handle:
                                handle.truncate(offset)
                                handle.flush()
                                os.fsync(handle.fileno())
                            truncated.append(
                                {"path": str(path), "from": size, "to": offset}
                            )
                            size = offset
                            break
                        ref, _raw, end = decoded
                        old = self.conn.execute(
                            "SELECT lane,segment_path,offset FROM payloads WHERE content_hash=?",
                            (ref.content_hash,),
                        ).fetchone()
                        if old is None:
                            self.conn.execute(
                                "INSERT INTO payloads VALUES(?,?,?,?,?,?,?,?,?)",
                                (
                                    ref.content_hash,
                                    int(lane),
                                    str(path),
                                    int(record_offset),
                                    int(ref.raw_bytes),
                                    int(ref.stored_bytes),
                                    ref.codec,
                                    int(ref.crc32),
                                    time.time(),
                                ),
                            )
                            discovered += 1
                            recovered_here += 1
                        offset = int(end)
                    scanned = max(0, int(size) - int(start_offset))
                    tail_bytes_scanned += scanned
                    segment_receipts.append(
                        {
                            "lane": int(lane),
                            "path": str(path),
                            "file_bytes": int(size),
                            "verified_indexed_end": int(start_offset),
                            "tail_bytes_scanned": int(scanned),
                            "recovered_payloads": int(recovered_here),
                        }
                    )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

        receipt = {
            "schema": "CB16_R2_INCREMENTAL_PAYLOAD_RECOVERY_V1",
            "mode": "VERIFY_LAST_INDEXED_RECORD_THEN_SCAN_APPEND_TAIL",
            "discovered_payloads": int(discovered),
            "truncated_tails": truncated,
            "segments": segment_receipts,
            "total_pack_bytes": int(total_bytes),
            "indexed_prefix_bytes_skipped": int(indexed_prefix_bytes_skipped),
            "tail_bytes_scanned": int(tail_bytes_scanned),
            "full_payload_hash_audit_still_required_at_qualification_boundary": True,
        }
        self.last_recovery_receipt = receipt
        return receipt
