from cb16_local_opt.cc_fast_topology_search_r0 import *
def test_candidate_matrix_is_bounded_and_semantics_fixed():
    m=candidate_matrix(writer_chunk_facts=(64,),prefetch_depths=(0,)); assert len(m)==3*3*2; assert {x.worker_count for x in m}=={2,4,6}; assert {x.active_accounts for x in m}=={16,32,64}; assert all(x.market_reuse for x in m)
