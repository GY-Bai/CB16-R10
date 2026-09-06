from pathlib import Path

import pytest

from cb16_local_opt.r10_run_root_lock import RunRootExclusiveLock


def test_r104_run_root_lock_is_exclusive_and_releasable(tmp_path: Path):
    first = RunRootExclusiveLock(tmp_path).acquire()
    try:
        with pytest.raises(RuntimeError, match="R10_4_RUN_ROOT_ALREADY_ACTIVE"):
            RunRootExclusiveLock(tmp_path).acquire()
    finally:
        first.release()

    second = RunRootExclusiveLock(tmp_path).acquire()
    second.release()
