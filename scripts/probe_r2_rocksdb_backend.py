from __future__ import annotations

import json


def main():
    result = {
        "schema": "CB16_R2_ROCKSDB_BACKEND_CAPABILITY_PROBE_R0",
        "rocksdict_importable": False,
        "capabilities": {},
        "error": None,
    }
    try:
        import rocksdict
        from rocksdict import DBPath, Options, Rdict, WriteBatch, WriteOptions

        result["rocksdict_importable"] = True
        result["rocksdict_version"] = getattr(rocksdict, "__version__", None)
        result["capabilities"] = {
            "Rdict": Rdict is not None,
            "Options": Options is not None,
            "DBPath": DBPath is not None,
            "WriteBatch": WriteBatch is not None,
            "WriteOptions": WriteOptions is not None,
            "set_db_paths": hasattr(Options(), "set_db_paths"),
            "set_wal_dir": hasattr(Options(), "set_wal_dir"),
            "set_bytes_per_sync": hasattr(Options(), "set_bytes_per_sync"),
            "set_wal_bytes_per_sync": hasattr(Options(), "set_wal_bytes_per_sync"),
            "enable_pipelined_write": hasattr(Options(), "set_enable_pipelined_write"),
        }
    except Exception as exc:
        result["error"] = repr(exc)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
