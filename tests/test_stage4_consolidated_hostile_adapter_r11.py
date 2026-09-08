from cb16_local_opt.stage4_consolidated_hostile_adapter_r11 import (
    ConsolidatedProductionHostileAdapterR11,
)
from cb16_local_opt.stage4_hostile_integration_harness_r11 import (
    run_integrated_hostile_matrix,
)


def test_real_consolidated_adapter_passes_h01_h20() -> None:
    report = run_integrated_hostile_matrix(ConsolidatedProductionHostileAdapterR11)
    failed = [row for row in report["cases"] if not row["passed"]]
    assert report["case_count"] == 20
    assert report["passed_cases"] == 20, failed
    assert report["failed_cases"] == 0
    assert report["status"] == "PASS"
    # INTG deliberately withholds this field even when the actual adapter passes;
    # the final cutover report, not INTG, owns the infrastructure verdict.
    assert report["integrated_runtime_qualified"] is False
