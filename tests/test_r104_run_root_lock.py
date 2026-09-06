from pathlib import Path

import pytest

from cb16_local_opt.r10_run_root_lock import RunRootExclusiveLock
from cb16_local_opt.r104_writer_contract import (
    CANONICAL_R104_ROOT,
    enforce_r104_writer_contract,
    inspect_r104_writer_contract,
)


def test_r104_run_root_lock_is_exclusive_and_releasable(tmp_path: Path):
    first = RunRootExclusiveLock(tmp_path).acquire()
    try:
        with pytest.raises(RuntimeError, match="R10_4_RUN_ROOT_ALREADY_ACTIVE"):
            RunRootExclusiveLock(tmp_path).acquire()
    finally:
        first.release()

    second = RunRootExclusiveLock(tmp_path).acquire()
    second.release()


def test_canonical_r104_root_rejects_non_ci_writer():
    with pytest.raises(RuntimeError, match="R10_4_CANONICAL_WRITER_REQUIRED"):
        enforce_r104_writer_contract(
            CANONICAL_R104_ROOT,
            effective_user="bgy",
        )


def test_canonical_r104_root_allows_cb16_ci_writer():
    contract = enforce_r104_writer_contract(
        CANONICAL_R104_ROOT,
        effective_user="cb16-ci",
    )
    assert contract.canonical is True
    assert contract.authorized is True


def test_manual_debug_root_remains_available_to_non_ci_user(tmp_path: Path):
    contract = inspect_r104_writer_contract(
        tmp_path / "manual-debug-r104",
        effective_user="bgy",
    )
    assert contract.canonical is False
    assert contract.authorized is True
