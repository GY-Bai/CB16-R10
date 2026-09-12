from cb16_local_opt.cc_economic_toys_r0 import run_known_answer_toys

def test_all_five_known_answer_toys_pass():
    r=run_known_answer_toys(); assert len(r)==5 and all(r.values()),r
