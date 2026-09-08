from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import pytest

from cb16_local_opt.stage4_authority_lease_r11 import (
    AuthorityBusyError,
    AuthorityLeaseNotInitializedError,
    AuthorityRecoveryRequiredError,
    AlreadyAcquiredError,
    CorruptAuthorityLeaseError,
    FencingTokenR11,
    STATE_FILE,
    StaleFencingTokenError,
    Stage4AuthorityLeaseR11,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="S4D uses POSIX flock")
CTX = mp.get_context("spawn")


def _contender(root: str, owner: str, gate, hold, out) -> None:
    gate.wait(10)
    lease = Stage4AuthorityLeaseR11(root, owner_id=owner)
    try:
        token = lease.acquire_authority()
    except AuthorityBusyError:
        out.put((owner, "BUSY", None))
        return
    out.put((owner, "ACQUIRED", (token.epoch, token.owner_nonce)))
    hold.wait(10)
    lease.release_authority(token)


def _crash_owner(root: str, out) -> None:
    lease = Stage4AuthorityLeaseR11(root, owner_id="crash-owner")
    token = lease.acquire_authority()
    out.put((token.epoch, token.owner_nonce))
    while True:
        time.sleep(1)


def _try_acquire(root: str, owner: str, out) -> None:
    lease = Stage4AuthorityLeaseR11(root, owner_id=owner)
    try:
        token = lease.acquire_authority()
    except Exception as exc:
        out.put(type(exc).__name__)
        return
    out.put("ACQUIRED")
    lease.release_authority(token)


def _try_recover(root: str, owner: str, out) -> None:
    lease = Stage4AuthorityLeaseR11(root, owner_id=owner)
    try:
        token = lease.recover_after_dead_owner()
    except Exception as exc:
        out.put(type(exc).__name__)
        return
    out.put("RECOVERED")
    lease.release_authority(token)


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "stage4-lease"
    Stage4AuthorityLeaseR11.initialize(root)
    return root


def test_two_processes_concurrent_acquire_exactly_one_owner(tmp_path: Path) -> None:
    root = _root(tmp_path)
    gate, hold, out = CTX.Event(), CTX.Event(), CTX.Queue()
    procs = [CTX.Process(target=_contender, args=(str(root), f"owner-{i}", gate, hold, out)) for i in range(2)]
    for p in procs:
        p.start()
    gate.set()
    first = out.get(timeout=10)
    second = out.get(timeout=10)
    assert sorted([first[1], second[1]]) == ["ACQUIRED", "BUSY"]
    hold.set()
    for p in procs:
        p.join(10)
        assert p.exitcode == 0


def test_second_owner_denied_and_same_process_double_acquire_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path)
    lease = Stage4AuthorityLeaseR11(root, owner_id="owner-a")
    token = lease.acquire_authority()
    with pytest.raises(AlreadyAcquiredError):
        lease.acquire_authority()
    peer = Stage4AuthorityLeaseR11(root, owner_id="same-pid-peer")
    with pytest.raises(AuthorityBusyError):
        peer.acquire_authority()
    out = CTX.Queue()
    p = CTX.Process(target=_try_acquire, args=(str(root), "owner-b", out))
    p.start(); p.join(10)
    assert p.exitcode == 0
    assert out.get(timeout=2) == "AuthorityBusyError"
    lease.release_authority(token)


def test_stale_token_rejected_after_clean_handoff(tmp_path: Path) -> None:
    root = _root(tmp_path)
    first = Stage4AuthorityLeaseR11(root, owner_id="first")
    old = first.acquire_authority(); first.release_authority(old)
    second = Stage4AuthorityLeaseR11(root, owner_id="second")
    current = second.acquire_authority()
    assert current.epoch == old.epoch + 1
    assert not second.validate_fencing_token(old)
    with pytest.raises(StaleFencingTokenError):
        second.assert_fencing_token(old)
    second.assert_fencing_token(current)
    second.release_authority(current)


def test_sigkill_requires_explicit_recovery_and_fences_old_owner(tmp_path: Path) -> None:
    root = _root(tmp_path)
    out = CTX.Queue()
    p = CTX.Process(target=_crash_owner, args=(str(root), out))
    p.start()
    epoch, nonce = out.get(timeout=10)
    old = FencingTokenR11(epoch, nonce)
    os.kill(p.pid, signal.SIGKILL); p.join(10)
    assert p.exitcode == -signal.SIGKILL
    recovering = Stage4AuthorityLeaseR11(root, owner_id="recovery")
    with pytest.raises(AuthorityRecoveryRequiredError):
        recovering.acquire_authority()
    current = recovering.recover_after_dead_owner()
    assert current.epoch == old.epoch + 1
    assert not recovering.validate_fencing_token(old)
    with pytest.raises(StaleFencingTokenError):
        recovering.assert_fencing_token(old)
    recovering.release_authority(current)


def test_pid_reuse_does_not_establish_authority(tmp_path: Path) -> None:
    root = _root(tmp_path)
    out = CTX.Queue(); p = CTX.Process(target=_crash_owner, args=(str(root), out))
    p.start(); old_epoch, _ = out.get(timeout=10)
    os.kill(p.pid, signal.SIGKILL); p.join(10)
    state_path = root / STATE_FILE
    raw = json.loads(state_path.read_text())
    raw["owner_pid"] = os.getpid()
    state_path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")
    lease = Stage4AuthorityLeaseR11(root, owner_id="recovery")
    with pytest.raises(AuthorityRecoveryRequiredError):
        lease.acquire_authority()
    token = lease.recover_after_dead_owner()
    assert token.epoch == old_epoch + 1
    lease.release_authority(token)


def test_corrupt_and_missing_metadata_fail_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    state = root / STATE_FILE
    state.write_text("{not-json")
    lease = Stage4AuthorityLeaseR11(root, owner_id="owner")
    with pytest.raises(CorruptAuthorityLeaseError):
        lease.inspect_current_authority()
    state.unlink()
    with pytest.raises(AuthorityLeaseNotInitializedError):
        lease.acquire_authority()


def test_wrong_fence_cannot_release_or_authorize_old_writer(tmp_path: Path) -> None:
    root = _root(tmp_path)
    lease = Stage4AuthorityLeaseR11(root, owner_id="owner")
    token = lease.acquire_authority()
    forged = FencingTokenR11(token.epoch, "0" * 32 if token.owner_nonce != "0" * 32 else "1" * 32)
    with pytest.raises(StaleFencingTokenError):
        lease.release_authority(forged)
    lease.assert_fencing_token(token)
    out = CTX.Queue(); p = CTX.Process(target=_try_acquire, args=(str(root), "attacker", out))
    p.start(); p.join(10)
    assert out.get(timeout=2) == "AuthorityBusyError"
    lease.release_authority(token)
    next_owner = Stage4AuthorityLeaseR11(root, owner_id="next")
    next_token = next_owner.acquire_authority()
    assert next_token.epoch > token.epoch
    assert not next_owner.validate_fencing_token(token)
    next_owner.release_authority(next_token)


def test_recovery_is_denied_while_live_owner_still_holds_lock(tmp_path: Path) -> None:
    root = _root(tmp_path)
    owner = Stage4AuthorityLeaseR11(root, owner_id="owner")
    token = owner.acquire_authority()
    out = CTX.Queue()
    p = CTX.Process(target=_try_recover, args=(str(root), "pretend-recovery", out))
    p.start(); p.join(10)
    assert out.get(timeout=2) == "AuthorityBusyError"
    owner.release_authority(token)


def test_symlink_metadata_is_never_trusted(tmp_path: Path) -> None:
    root = _root(tmp_path)
    state = root / STATE_FILE
    target = tmp_path / "outside.json"
    target.write_text(state.read_text())
    state.unlink(); state.symlink_to(target)
    lease = Stage4AuthorityLeaseR11(root, owner_id="owner")
    with pytest.raises(CorruptAuthorityLeaseError):
        lease.inspect_current_authority()
