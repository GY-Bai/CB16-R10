from __future__ import annotations
CC_FAST_ROUTE="CC_FAST_R0"
FORBIDDEN_ROUTE_TOKENS=("legacy","compat","fallback")
def resolve_fast_route(route:str)->str:
    if route!=CC_FAST_ROUTE: raise RuntimeError("UNKNOWN_OR_LEGACY_PERFORMANCE_ROUTE")
    return CC_FAST_ROUTE
def route_has_fallback(route_config:dict[str,object])->bool:
    text=" ".join(f"{k}={v}" for k,v in route_config.items()).lower(); return any(token in text for token in FORBIDDEN_ROUTE_TOKENS)
