import numpy as np
import pytest
from cb16_local_opt.cc_fast_market_cache_r0 import *
def key(): return MarketCacheKey("market","window","prep","organ","norm")
def test_cache_key_binds_all_semantic_axes():
    assert key().key_sha256!=MarketCacheKey("market","window","prep2","organ","norm").key_sha256
def test_shared_market_one_allocation_many_views():
    arr=np.arange(100,dtype=np.float64).reshape(50,2)
    with SharedMarketOwner(key(),arr) as owner:
        with SharedMarketView(owner.descriptor) as v1, SharedMarketView(owner.descriptor) as v2:
            assert v1.descriptor.shm_name==v2.descriptor.shm_name==owner.descriptor.shm_name; assert v1.verify_content() and v2.verify_content(); assert np.array_equal(v1.array,arr); assert not v1.array.flags.writeable
            with pytest.raises(ValueError): v1.array[0,0]=9
