from __future__ import annotations
import time
import numpy as np
try:
    import numba as nb
except Exception:
    nb=None
NUMBA_AVAILABLE=nb is not None
if NUMBA_AVAILABLE:
    @nb.njit(cache=True)
    def _mark_equity_numba(cash,quantity,price,liability,out):
        for i in range(cash.shape[0]):
            out[i]=cash[i]+quantity[i]*price-liability[i]
else:
    _mark_equity_numba=None

def mark_equity_numpy(cash,quantity,price,liability):
    return np.asarray(cash)+np.asarray(quantity)*float(price)-np.asarray(liability)

def mark_equity_numba(cash,quantity,price,liability):
    if not NUMBA_AVAILABLE: raise RuntimeError("NUMBA_CANDIDATE_UNAVAILABLE")
    out=np.empty_like(np.asarray(cash,dtype=np.float64))
    _mark_equity_numba(np.asarray(cash,dtype=np.float64),np.asarray(quantity,dtype=np.float64),float(price),np.asarray(liability,dtype=np.float64),out)
    return out

def benchmark_numba_candidate(cash,quantity,price,liability,*,repeats=20):
    cash=np.asarray(cash,dtype=np.float64); quantity=np.asarray(quantity,dtype=np.float64); liability=np.asarray(liability,dtype=np.float64)
    t0=time.perf_counter(); ref=None
    for _ in range(repeats): ref=mark_equity_numpy(cash,quantity,price,liability)
    numpy_s=time.perf_counter()-t0
    if not NUMBA_AVAILABLE:
        return {"available":False,"numpy_s":numpy_s,"cold_compile_s":0.0,"warm_s":0.0,"semantic_equal":False}
    t0=time.perf_counter(); candidate=mark_equity_numba(cash,quantity,price,liability); cold=time.perf_counter()-t0
    t0=time.perf_counter()
    for _ in range(repeats): candidate=mark_equity_numba(cash,quantity,price,liability)
    warm=time.perf_counter()-t0
    return {"available":True,"numpy_s":numpy_s,"cold_compile_s":cold,"warm_s":warm,"semantic_equal":bool(np.allclose(ref,candidate,rtol=0.0,atol=0.0))}
