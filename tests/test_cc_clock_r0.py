from cb16_local_opt.cc_clock_r0 import CCFourClockR0
def test_four_clocks_are_independent():
    c=CCFourClockR0(1,2,3,'h'); assert c.advance_environment().policy_decision_index==2; assert c.advance_decision().environment_time==1; assert c.next_chunk().objective_horizon_id=='h'; assert c.with_horizon('x').compute_chunk_index==3
