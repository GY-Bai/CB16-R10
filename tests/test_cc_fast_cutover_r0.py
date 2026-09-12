import json
from pathlib import Path
def test_cutover_manifest_is_cc_only_and_does_not_fake_shanxi_numbers():
    root=Path(__file__).resolve().parents[1]; d=json.loads((root/"authority/rearchitecture_r11/CB16_R11_CC_FAST_CUTOVER_V1.json").read_text()); assert d["active_route"]=="CC_FAST_R0"; assert d["compatibility_layer"] is False; assert d["fallback"] is False; assert d["legacy_runtime_status"]["gpu_inference_broker.py"]=="REFERENCE_ONLY_FOR_CC"
    if d["shanxi_measurement"]["status"]!="PERFORMANCE_MEASURED": assert d["performance_winner"] is None; assert d["shanxi_measurement"]["performance_metrics"] is None
