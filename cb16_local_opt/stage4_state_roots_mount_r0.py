from __future__ import annotations

"""Mount-aware, semantics-preserving S4F deployment adapter.

Docker bind mounts can retain source inode mode bits while the mounted filesystem
itself is read-only. Accept OS-reported ST_RDONLY as read-only; on a writable
filesystem preserve the original write-bit fail-closed rule.
"""

import os
import stat
from pathlib import Path

from .stage4_state_roots_r11 import (
    Stage4FrozenRootWritableError,
    Stage4RootSetR11,
    Stage4StateRootsR11,
)


class Stage4MountAwareStateRootsR11(Stage4StateRootsR11):
    """S4F root verifier that recognizes filesystem-level read-only mounts."""

    def __init__(self, roots: Stage4RootSetR11):
        super().__init__(roots)

    def _assert_frozen_read_only(self) -> None:
        root: Path = self.frozen_raw_root
        try:
            flags = os.statvfs(root).f_flag
        except OSError as exc:
            raise Stage4FrozenRootWritableError(
                f"STAGE4_FROZEN_RAW_READONLY_STATUS_UNAVAILABLE:{root}"
            ) from exc

        if flags & os.ST_RDONLY:
            return

        mode = root.lstat().st_mode
        write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        if mode & write_bits:
            raise Stage4FrozenRootWritableError(f"STAGE4_FROZEN_RAW_ROOT_WRITABLE:{root}")


__all__ = ["Stage4MountAwareStateRootsR11"]
