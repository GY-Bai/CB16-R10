from cb16_local_opt.cc_fast_io_benchmark_r0 import *
def test_physical_reorder_does_not_change_logical_selection(tmp_path):
    p=tmp_path/"x"; p.write_bytes(b"0123456789abcdef"); sel=[SampleLocation("b",str(p),8,4),SampleLocation("a",str(p),0,4),SampleLocation("c",str(p),4,4)]; ref=read_preselected(sel,physical_strategy="naive")
    for mode in ("grouped_sequential","ssd_active","bounded_prefetch"): assert read_preselected(sel,physical_strategy=mode)==ref
    assert ref==[b"89ab",b"0123",b"4567"]
