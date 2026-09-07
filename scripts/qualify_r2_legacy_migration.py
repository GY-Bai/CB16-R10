from __future__ import annotations

import argparse
import json
from pathlib import Path

from cb16_local_opt.r2_evidence_storage import R2EvidenceStore
from cb16_local_opt.r2_legacy_migration import qualify_legacy_generations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-lake-root", required=True)
    ap.add_argument("--generations", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metadata-root")
    ap.add_argument("--payload-root", action="append", default=[])
    ap.add_argument("--codec", default="zstd")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    legacy = Path(args.legacy_lake_root).resolve()
    out = Path(args.out).resolve()
    if legacy == out or legacy in out.parents:
        raise RuntimeError("R2_MIGRATION_OUTPUT_MUST_BE_OUTSIDE_LEGACY_LAKE")

    store = None
    if not args.verify_only:
        if not args.metadata_root or not args.payload_root:
            raise RuntimeError("R2_MIGRATION_MATERIALIZE_REQUIRES_METADATA_AND_PAYLOAD_ROOTS")
        meta = Path(args.metadata_root).resolve()
        payloads = [Path(x).resolve() for x in args.payload_root]
        if legacy == meta or legacy in meta.parents:
            raise RuntimeError("R2_METADATA_MUST_BE_OUTSIDE_LEGACY_LAKE")
        for p in payloads:
            if legacy == p or legacy in p.parents:
                raise RuntimeError("R2_PAYLOAD_MUST_BE_OUTSIDE_LEGACY_LAKE")
        store = R2EvidenceStore(
            metadata_root=meta,
            payload_roots=payloads,
            codec=args.codec,
            sqlite_synchronous="FULL",
            recover_on_open=True,
        )

    try:
        result = qualify_legacy_generations(
            legacy_lake_root=legacy,
            generations=args.generations,
            store=store,
            limit=(args.limit if args.limit > 0 else None),
        )
        if store is not None:
            result["r2_store_stats"] = store.stats()
            result["r2_store_audit"] = store.audit(verify_payloads=True)
        result.update({
            "status":"PASS" if (
                result["all_projection_mismatches_zero"]
                and result["all_metadata_mismatches_zero"]
                and result["all_duplicate_evidence_ids_zero"]
            ) else "FAIL",
            "legacy_lake_root":str(legacy),
            "writes_to_legacy_lake":False,
            "scientific_semantics_changed":False,
            "final_holdout_2025_09_accessed":False,
        })
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result,sort_keys=True,indent=2),encoding="utf-8")
        print(json.dumps(result,sort_keys=True,indent=2))
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    main()
