from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def run(cmd):
    p = subprocess.run(cmd, text=True, capture_output=True)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()

    result = {
        "schema": "CB16_R2_STORAGE_TOPOLOGY_PROBE_R0",
        "lsblk": run([
            "lsblk", "-J", "-b",
            "-o", "NAME,KNAME,TYPE,ROTA,SIZE,FSTYPE,MOUNTPOINTS,MODEL,TRAN"
        ]),
        "findmnt": run(["findmnt", "-J", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]),
        "mounts_of_interest": {},
        "writes_to_canonical_run_root": False,
        "final_holdout_2025_09_accessed": False,
    }
    for candidate in ["/", "/data", "/data/cb16_hdd", "/tmp"]:
        p = Path(candidate)
        if p.exists():
            st = os.statvfs(p)
            result["mounts_of_interest"][candidate] = {
                "free_bytes": st.f_bavail * st.f_frsize,
                "total_bytes": st.f_blocks * st.f_frsize,
            }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
