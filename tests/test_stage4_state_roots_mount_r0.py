from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import cb16_local_opt.stage4_state_roots_mount_r0 as mount_adapter
from cb16_local_opt.stage4_state_roots_mount_r0 import Stage4MountAwareStateRootsR11
from cb16_local_opt.stage4_state_roots_r11 import (
    Stage4FrozenRootWritableError,
    Stage4RootSetR11,
)


def _store(tmp_path: Path, *, frozen_mode: int) -> Stage4MountAwareStateRootsR11:
    frozen = tmp_path / "raw"
    frozen.mkdir()
    (frozen / "sentinel").write_text("raw\n", encoding="utf-8")
    frozen.chmod(frozen_mode)
    return Stage4MountAwareStateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=tmp_path / "control",
            data_root=tmp_path / "data",
            frozen_raw_root=frozen,
            frozen_raw_identity="fixture-raw-v1",
        )
    )


def test_mount_readonly_flag_accepts_source_inode_0755(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _store(tmp_path, frozen_mode=0o755)
    real_statvfs = mount_adapter.os.statvfs

    def fake_statvfs(path: Path):
        if Path(path) == store.frozen_raw_root:
            return SimpleNamespace(f_flag=os.ST_RDONLY)
        return real_statvfs(path)

    monkeypatch.setattr(mount_adapter.os, "statvfs", fake_statvfs)
    receipt = store.initialize()
    assert receipt.control_root_id
    assert receipt.data_root_id


def test_writable_filesystem_with_0755_still_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path, frozen_mode=0o755)
    with pytest.raises(Stage4FrozenRootWritableError, match="FROZEN_RAW_ROOT_WRITABLE"):
        store.initialize()


def test_mode_readonly_fixture_remains_accepted(tmp_path: Path) -> None:
    store = _store(tmp_path, frozen_mode=0o555)
    receipt = store.initialize()
    assert receipt.sealed_object_count == 0
