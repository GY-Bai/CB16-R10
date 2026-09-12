import ast
from pathlib import Path
FORBIDDEN={"cb16_local_opt.gpu_inference_broker","cb16_local_opt.multiprocess_trajectory_farm","cb16_local_opt.vectorized_physics"}
def test_active_cc_fast_modules_have_no_legacy_imports_or_fallback_tokens():
    root=Path(__file__).resolve().parents[1]/"cb16_local_opt"; fast=list(root.glob("cc_fast_*.py")); assert fast
    for path in fast:
        tree=ast.parse(path.read_text(),filename=str(path)); imported=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imported.update(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom): imported.add(node.module or "")
        assert not (FORBIDDEN&imported),(path,FORBIDDEN&imported)
    router=(root/"cc_fast_router_r0.py").read_text().lower(); assert 'cc_fast_route="cc_fast_r0"' in router.replace(" ",""); assert "gpu_inference_broker" not in router
