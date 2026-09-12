import hashlib
from pathlib import Path
import time

import numpy as np
import pytest

from cb16_local_opt.cc_fast_account_kernel_r0 import AccountKernelConfig
from cb16_local_opt.cc_fast_account_workers_r0 import AccountWorkerCommand, PersistentAccountWorkerPool
from cb16_local_opt.cc_fast_fact_queue_r0 import BackpressureRequired, FactOutputQueue, encode_fact
from cb16_local_opt.cc_fast_market_cache_r0 import MarketCacheKey, SharedMarketOwner, SharedMarketView
from cb16_local_opt.cc_fast_policy_broker_r0 import BatchedPolicyBroker, PolicyRequest, PolicySpec
from cb16_local_opt.cc_fast_runtime_guard_r0 import run_with_deadline, validate_policy_response_alignment
from cb16_local_opt.cc_fast_scheduler_r0 import AccountSerialScheduler, ScheduledAccountTask
from cb16_local_opt.cc_fast_storage_tier_r0 import StorageTierManager, TierConfig


def _req(account="a", decision=0, gen=1, sha="1" * 64):
    return PolicyRequest(account, decision, decision, gen, sha, (1.0,), hashlib.sha256(b"x").hexdigest(), "n", f"stream-{account}", decision)


def test_stale_policy_and_duplicate_account_fail_closed():
    broker = BatchedPolicyBroker([PolicySpec(1, "p", "1" * 64, (0, 0, 0))])
    with pytest.raises(RuntimeError):
        broker.infer([_req(gen=2, sha="2" * 64)])
    sched = AccountSerialScheduler(["a"])
    sched.submit(ScheduledAccountTask("a", 0, 1, 0), now_tick=0)
    with pytest.raises(RuntimeError):
        sched.submit(ScheduledAccountTask("a", 0, 1, 0), now_tick=0)


def test_queue_saturation_backpressures_instead_of_dropping_failure():
    f = encode_fact({"failure": "x" * 20}, semantic_id="failure", terminal_or_failure=True)
    other = encode_fact({"x": "y" * 20}, semantic_id="other")
    q = FactOutputQueue(max_bytes=f.nbytes + 1, max_age_s=10)
    q.put(f)
    with pytest.raises(BackpressureRequired):
        q.put(other)
    assert q.get().semantic_id == "failure"


def test_writer_slowdown_trips_age_backpressure_without_losing_terminal_fact():
    terminal = encode_fact({"terminal": True}, semantic_id="terminal", terminal_or_failure=True)
    later = encode_fact({"later": True}, semantic_id="later")
    q = FactOutputQueue(max_bytes=4096, max_age_s=0.002)
    q.put(terminal)
    time.sleep(0.01)
    with pytest.raises(BackpressureRequired, match="OLDEST_AGE"):
        q.put(later)
    retained = q.get()
    assert retained.semantic_id == "terminal"
    assert retained.terminal_or_failure is True


def test_policy_broker_delay_hits_declared_deadline_fail_closed():
    broker = BatchedPolicyBroker([PolicySpec(1, "p", "1" * 64, (0.0, -1.0, 1.0))])
    request = _req()

    def delayed_call():
        time.sleep(0.05)
        return broker.infer([request])

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="POLICY_BROKER_DEADLINE_EXCEEDED"):
        run_with_deadline(delayed_call, timeout_s=0.005, failure_code="POLICY_BROKER_DEADLINE_EXCEEDED")
    assert time.monotonic() - started < 0.04


def test_out_of_order_policy_response_is_rejected():
    broker = BatchedPolicyBroker([PolicySpec(1, "p", "1" * 64, (0.0, -1.0, 1.0))])
    requests = [_req("a", 0), _req("b", 0)]
    responses = broker.infer(requests)
    with pytest.raises(RuntimeError, match="OUT_OF_ORDER_OR_CONTAMINATED_POLICY_RESPONSE"):
        validate_policy_response_alignment(requests, list(reversed(responses)))


def test_shared_memory_lifecycle_error_is_visible():
    arr = np.arange(8, dtype=np.float64)
    owner = SharedMarketOwner(MarketCacheKey("m", "w", "p", "o", "n"), arr)
    desc = owner.descriptor
    owner.close(unlink=True)
    with pytest.raises(FileNotFoundError):
        SharedMarketView(desc)


def test_worker_crash_is_not_silently_recovered_with_legacy_fallback():
    market = np.column_stack([np.linspace(100, 101, 8), np.zeros(8)]).astype(np.float64)
    with SharedMarketOwner(MarketCacheKey("m", "w", "p", "o", "n"), market) as owner:
        pool = PersistentAccountWorkerPool(
            worker_count=2,
            account_ids=["a", "b"],
            market_descriptor=owner.descriptor,
            initial_equity=1000,
            kernel_config=AccountKernelConfig(fee_rate=0, maintenance_margin_fraction=0),
        )
        try:
            wid = pool.owner_for("a")
            pool.terminate_worker(wid)
            with pytest.raises(RuntimeError, match="CRASHED"):
                pool.execute([AccountWorkerCommand(0, "a", 0, 0, "FLAT", 0, "f" * 64)], timeout_s=0.5)
        finally:
            pool.close(force=True)


def test_ssd_hdd_capacity_and_partial_archive_leave_hot_fact_intact(tmp_path, monkeypatch):
    ssd = tmp_path / "ssd"
    hdd = tmp_path / "hdd"
    ssd.mkdir()
    src = ssd / "fact"
    src.write_bytes(b"abcdef")
    mgr = StorageTierManager(ssd_root=str(ssd), hdd_root=str(hdd), config=TierConfig(10, 10, 100))
    import cb16_local_opt.cc_fast_storage_tier_r0 as mod
    real = mod.shutil.copyfile

    def partial_then_fail(a, b):
        Path(b).write_bytes(b"ab")
        raise OSError("simulated partial write")

    monkeypatch.setattr(mod.shutil, "copyfile", partial_then_fail)
    with pytest.raises(OSError):
        mgr.archive(src, reclaim_hot=True)
    assert src.exists()
    assert not (hdd / "fact").exists()
    monkeypatch.setattr(mod.shutil, "copyfile", real)
