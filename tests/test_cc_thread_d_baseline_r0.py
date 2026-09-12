import json
from pathlib import Path
BASE="89d62bf966f476e598f0e2f5c5e8e03c15a8db51"
def _load(name):
    root=Path(__file__).resolve().parents[1]; return json.loads((root/"authority/rearchitecture_r11"/name).read_text())
def test_baseline_is_exact_and_sibling_free():
    doc=_load("CB16_R11_CC_THREAD_D_BASELINE_V1.json"); assert doc["frozen_implementation_base"]==BASE; assert doc["implementation_branch"]=="ai/r11-cc-thread-d-fast-cutover-r0"; assert doc["sibling_branch_dependencies"]==[]; assert doc["sibling_code_consumption_forbidden"] is True; assert doc["final_data_access"]=="FORBIDDEN"; assert doc["fresh_market_download"]=="FORBIDDEN"; assert doc["planning_docs_read_from_main"]["docs/cc/CC_THREAD_D_PERFORMANCE_HARD_CUTOVER.md"]=="748290ac8eb4342d7d8ae6136e0e536aa0b0bece"
def test_legacy_retirement_registry_is_hard_cutover():
    doc=_load("CB16_R11_CC_LEGACY_PERF_RETIREMENT_V1.json"); assert doc["hard_cutover"] is True; assert doc["compatibility_layer"] is False; assert doc["fallback"] is False
    for name in ("cb16_local_opt/gpu_inference_broker.py","cb16_local_opt/multiprocess_trajectory_farm.py","cb16_local_opt/vectorized_physics.py"):
        assert doc["modules"][name]["status"]=="REFERENCE_ONLY_FOR_CC"; assert doc["modules"][name]["active_runtime_import_forbidden"] is True
