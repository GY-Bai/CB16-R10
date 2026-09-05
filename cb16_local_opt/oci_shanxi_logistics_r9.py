from __future__ import annotations

"""Immutable OCI -> Shanxi transport publisher for R9.

R8 already provides deterministic portable bundles and a settle+SHA receiver. R9 adds
an explicit publisher protocol so a partially transferred file never appears under its
final `.tar.gz` name on Shanxi.

Remote protocol (rsync + ssh):
1. upload bundle to `<name>.partial.<sha>`;
2. upload SHA sidecar to `<name>.sha256.partial.<sha>`;
3. remote fsync best-effort via `sync -f` when available;
4. atomic remote rename bundle to final name;
5. atomic remote rename sidecar LAST;
6. R8 watcher sees the bundle only after final sidecar exists and settlement passes.

The transport receipt binds local bytes and remote names, not scientific market claims.
"""

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class TransportPublicationReceiptR9:
    transport_version: str
    status: str
    source_bundle: str
    bundle_sha256: str
    bundle_bytes: int
    destination_kind: str
    destination: str
    remote_final_bundle_name: str
    remote_final_sidecar_name: str
    published_at_unix: float

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def _validate_sidecar(bundle: Path) -> str:
    side = Path(str(bundle) + ".sha256")
    if not side.is_file():
        raise RuntimeError("R9_TRANSPORT_SHA256_SIDECAR_MISSING")
    expected = side.read_text().strip().split()[0].lower()
    actual = sha256_file(bundle)
    if expected != actual:
        raise RuntimeError("R9_TRANSPORT_LOCAL_BUNDLE_SHA_MISMATCH")
    return actual


def publish_bundle_local_r9(
    *,
    bundle: str | Path,
    destination_inbox: str | Path,
) -> TransportPublicationReceiptR9:
    src = Path(bundle)
    digest = _validate_sidecar(src)
    dst = Path(destination_inbox)
    dst.mkdir(parents=True, exist_ok=True)
    final_bundle = dst / src.name
    final_side = Path(str(final_bundle) + ".sha256")
    partial_bundle = dst / (src.name + f".partial.{digest[:16]}")
    partial_side = dst / (src.name + f".sha256.partial.{digest[:16]}")

    shutil.copyfile(src, partial_bundle)
    if sha256_file(partial_bundle) != digest:
        partial_bundle.unlink(missing_ok=True)
        raise RuntimeError("R9_TRANSPORT_LOCAL_COPY_SHA_MISMATCH")
    partial_side.write_text(f"{digest}  {src.name}\n")
    os.replace(partial_bundle, final_bundle)
    os.replace(partial_side, final_side)  # sidecar published last
    return TransportPublicationReceiptR9(
        transport_version="CB16_OCI_SHANXI_IMMUTABLE_TRANSPORT_R9",
        status="PUBLISHED",
        source_bundle=str(src.resolve()),
        bundle_sha256=digest,
        bundle_bytes=src.stat().st_size,
        destination_kind="LOCAL_OR_MOUNTED_INBOX",
        destination=str(dst.resolve()),
        remote_final_bundle_name=final_bundle.name,
        remote_final_sidecar_name=final_side.name,
        published_at_unix=time.time(),
    )


def publish_bundle_ssh_r9(
    *,
    bundle: str | Path,
    remote: str,
    remote_inbox: str,
    rsync_bin: str = "rsync",
    ssh_bin: str = "ssh",
) -> TransportPublicationReceiptR9:
    src = Path(bundle)
    digest = _validate_sidecar(src)
    if not remote or not remote_inbox.startswith("/"):
        raise ValueError("remote and absolute remote_inbox required")
    part_bundle = src.name + f".partial.{digest[:16]}"
    part_side = src.name + f".sha256.partial.{digest[:16]}"
    final_bundle = src.name
    final_side = src.name + ".sha256"

    # Ensure destination exists before transfer.
    subprocess.run(
        [ssh_bin, remote, "mkdir", "-p", remote_inbox],
        check=True,
    )
    subprocess.run(
        [rsync_bin, "-a", "--partial", str(src), f"{remote}:{remote_inbox}/{part_bundle}"],
        check=True,
    )
    local_side_tmp = src.with_name(src.name + ".r9_publish_sidecar.tmp")
    try:
        local_side_tmp.write_text(f"{digest}  {src.name}\n")
        subprocess.run(
            [rsync_bin, "-a", str(local_side_tmp), f"{remote}:{remote_inbox}/{part_side}"],
            check=True,
        )
    finally:
        local_side_tmp.unlink(missing_ok=True)

    # Verify remote bytes before atomic publication. sha256sum exits nonzero on mismatch.
    remote_script = (
        "set -euo pipefail; "
        f"cd {json.dumps(remote_inbox)}; "
        f"test \"$(sha256sum {json.dumps(part_bundle)} | awk '{{print $1}}')\" = {json.dumps(digest)}; "
        f"mv -f {json.dumps(part_bundle)} {json.dumps(final_bundle)}; "
        f"mv -f {json.dumps(part_side)} {json.dumps(final_side)}; "
        "sync"
    )
    subprocess.run([ssh_bin, remote, "bash", "-lc", remote_script], check=True)
    return TransportPublicationReceiptR9(
        transport_version="CB16_OCI_SHANXI_IMMUTABLE_TRANSPORT_R9",
        status="PUBLISHED",
        source_bundle=str(src.resolve()),
        bundle_sha256=digest,
        bundle_bytes=src.stat().st_size,
        destination_kind="SSH_RSYNC",
        destination=f"{remote}:{remote_inbox}",
        remote_final_bundle_name=final_bundle,
        remote_final_sidecar_name=final_side,
        published_at_unix=time.time(),
    )
