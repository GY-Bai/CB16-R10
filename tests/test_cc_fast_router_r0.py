import pytest
from cb16_local_opt.cc_fast_router_r0 import *
def test_only_cc_fast_route_exists():
    assert resolve_fast_route("CC_FAST_R0")=="CC_FAST_R0"
    for bad in ("legacy","compat","auto","fallback","CC_FAST_R1"):
        with pytest.raises(RuntimeError): resolve_fast_route(bad)
