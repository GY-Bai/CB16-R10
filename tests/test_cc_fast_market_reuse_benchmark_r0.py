import numpy as np
from cb16_local_opt.cc_fast_market_cache_r0 import *
def test_reuse_avoids_n_materialized_market_copies():
    n=16; market=np.arange(20000,dtype=np.float64).reshape(10000,2); repeated_bytes=n*market.nbytes
    with SharedMarketOwner(MarketCacheKey("m","w","p","o","n"),market) as owner:
        descriptors=[owner.descriptor for _ in range(n)]; shared_allocated_bytes=owner.descriptor.nbytes; assert len({d.shm_name for d in descriptors})==1; assert shared_allocated_bytes==market.nbytes; assert repeated_bytes==n*shared_allocated_bytes; assert shared_allocated_bytes<repeated_bytes
