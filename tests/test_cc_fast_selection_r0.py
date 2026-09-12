import pytest
from cb16_local_opt.cc_fast_selection_r0 import *
def test_winner_only_among_measured_semantically_valid_candidates():
    rows=[CandidateResult("fast-invalid",False,True,True,1000,1,True),CandidateResult("safe-a",True,True,True,100,10,True),CandidateResult("safe-b",True,True,True,120,9,True),CandidateResult("unmeasured",True,True,True,999,1,False)]; assert select_fast_candidate(rows).candidate_id=="safe-b"
    with pytest.raises(RuntimeError): select_fast_candidate([rows[-1]])
