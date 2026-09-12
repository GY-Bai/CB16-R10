from dataclasses import asdict
from cb16_local_opt.cc_runtime_generation_switch_r0 import switch_generation_r0
from tests.cc_thread_a_support_r0 import runtime
def test_generation_switch_preserves_account_clocks_lineage_and_token():
    r=runtime(); before=(asdict(r.account),asdict(r.clocks),r.account_lineage_id,r.predecessor_token); rec=switch_generation_r0(r,new_policy_generation='g2',new_policy_id='p2',new_policy_sha256='c'*64); assert before==(asdict(r.account),asdict(r.clocks),r.account_lineage_id,r.predecessor_token); assert rec.old_policy_id=='p1' and r.policy_id=='p2'
