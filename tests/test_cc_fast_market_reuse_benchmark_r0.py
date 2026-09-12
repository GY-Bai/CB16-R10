from cb16_local_opt.cc_fast_market_reuse_benchmark_r0 import benchmark_market_reuse


def test_reuse_reports_speed_and_memory_deltas_without_semantic_change():
    report = benchmark_market_reuse(account_count=8, rows=4000, repeats=2)
    assert report.account_count == 8
    assert report.repeated_materialized_bytes == 8 * report.shared_materialized_bytes
    assert report.memory_reduction_ratio == 8.0
    assert report.repeated_compute_s > 0.0
    assert report.shared_compute_s > 0.0
    assert report.speedup_ratio > 0.0
