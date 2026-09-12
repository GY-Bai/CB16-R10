import hashlib
from pathlib import Path
import numpy as np
import pytest
from cb16_local_opt.cc_fast_account_kernel_r0 import AccountKernelConfig
from cb16_local_opt.cc_fast_account_workers_r0 import *
from cb16_local_opt.cc_fast_fact_queue_r0 import *
from cb16_local_opt.cc_fast_market_cache_r0 import *
from cb16_local_opt.cc_fast_policy_broker_r0 import *
from cb16_local_opt.cc_fast_scheduler_r0 import *
from cb16_local_opt.cc_fast_storage_tier_r0 import *
def _req(gen=1,sha="1"*64): return PolicyRequest("a",0,0,gen,sha,(1.0,),hashlib.sha256(b"x").hexdigest(),"n","s",0)
def test_stale_policy_and_duplicate_account_fail_closed():
    broker=BatchedPolicyBroker([PolicySpec(1,"p","1"*64,(0,0,0))])
    with pytest.raises(RuntimeError): broker.infer([_req(2,"2"*64)])
    sched=AccountSerialScheduler(["a"]); sched.submit(ScheduledAccountTask("a",0,1,0),now_tick=0)
    with pytest.raises(RuntimeError): sched.submit(ScheduledAccountTask("a",0,1,0),now_tick=0)
def test_queue_saturation_backpressures_instead_of_dropping_failure():
    f=encode_fact({"failure":"x"*20},semantic_id="failure",terminal_or_failure=True); other=encode_fact({"x":"y"*20},semantic_id="other"); q=FactOutputQueue(max_bytes=f.nbytes+1,max_age_s=10); q.put(f)
    with pytest.raises(BackpressureRequired): q.put(other)
    assert q.get().semantic_id=="failure"
def test_shared_memory_lifecycle_error_is_visible():
    arr=np.arange(8,dtype=np.float64); owner=SharedMarketOwner(MarketCacheKey("m","w","p","o","n"),arr); desc=owner.descriptor; owner.close(unlink=True)
    with pytest.raises(FileNotFoundError): SharedMarketView(desc)
def test_worker_crash_is_not_silently_recovered_with_legacy_fallback():
    market=np.column_stack([np.linspace(100,101,8),np.zeros(8)]).astype(np.float64)
    with SharedMarketOwner(MarketCacheKey("m","w","p","o","n"),market) as owner:
        pool=PersistentAccountWorkerPool(worker_count=2,account_ids=["a","b"],market_descriptor=owner.descriptor,initial_equity=1000,kernel_config=AccountKernelConfig(fee_rate=0,maintenance_margin_fraction=0))
        try:
            wid=pool.owner_for("a"); pool.terminate_worker(wid)
            with pytest.raises(RuntimeError,match="CRASHED"): pool.execute([AccountWorkerCommand(0,"a",0,0,"FLAT",0,"f"*64)],timeout_s=.5)
        finally: pool.close(force=True)
def test_ssd_hdd_capacity_and_partial_archive_leave_hot_fact_intact(tmp_path,monkeypatch):
    ssd=tmp_path/"ssd"; hdd=tmp_path/"hdd"; ssd.mkdir(); src=ssd/"fact"; src.write_bytes(b"abcdef"); mgr=StorageTierManager(ssd_root=str(ssd),hdd_root=str(hdd),config=TierConfig(10,10,100)); import cb16_local_opt.cc_fast_storage_tier_r0 as mod; real=mod.shutil.copyfile
    def partial_then_fail(a,b): Path(b).write_bytes(b"ab"); raise OSError("simulated partial write")
    monkeypatch.setattr(mod.shutil,"copyfile",partial_then_fail)
    with pytest.raises(OSError): mgr.archive(src,reclaim_hot=True)
    assert src.exists(); assert not (hdd/"fact").exists(); monkeypatch.setattr(mod.shutil,"copyfile",real)
