from cb16_local_opt.cc_continual_retention_r0 import run_retention
def test_a_b_a_replay_retention_without_manual_switch():
 r=run_retention(); assert r.a_initial>=.99 and r.b_after_adapt>=.99 and r.a_after_b_with_replay>r.a_after_b_no_replay and r.a_after_revisit>=.99
