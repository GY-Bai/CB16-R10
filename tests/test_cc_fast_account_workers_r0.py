import numpy as np
from cb16_local_opt.cc_fast_account_workers_r0 import *
from cb16_local_opt.cc_fast_market_cache_r0 import *
from cb16_local_opt.cc_fast_account_kernel_r0 import AccountKernelConfig
def test_persistent_spawn_workers_single_cuda_owner_and_account_ownership():
    market=np.column_stack([np.linspace(100,101,8),np.zeros(8)]).astype(np.float64); key=MarketCacheKey("m","w","p","o","n")
    with SharedMarketOwner(key,market) as owner:
        with PersistentAccountWorkerPool(worker_count=2,account_ids=["a","b","c","d"],market_descriptor=owner.descriptor,initial_equity=1000,kernel_config=AccountKernelConfig(fee_rate=0,maintenance_margin_fraction=0)) as pool:
            assert len(pool.statuses)==2; assert all(not s.cuda_initialized for s in pool.statuses.values()); assert all(s.numeric_threads["OMP_NUM_THREADS"]=="1" for s in pool.statuses.values()); assert pool.owner_for("a")!=pool.owner_for("b"); out1=pool.execute([AccountWorkerCommand(0,"a",0,0,"LONG",.2,"1"*64),AccountWorkerCommand(1,"b",0,0,"SHORT",.2,"2"*64)]); pids1={x.worker_pid for x in out1}; out2=pool.execute([AccountWorkerCommand(0,"a",1,1,"FLAT",0.0,"3"*64),AccountWorkerCommand(1,"b",1,1,"FLAT",0.0,"4"*64)]); assert {x.worker_pid for x in out2}==pids1; assert [x.decision_index for x in out2]==[1,1]
