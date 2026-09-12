import numpy as np
import pytest
from cb16_local_opt.cc_fast_numba_kernel_r0 import *
def test_numba_candidate_semantics_when_available():
    c=np.arange(32,dtype=float)+100; q=np.linspace(-2,2,32); l=np.linspace(-3,3,32); ref=mark_equity_numpy(c,q,10,l)
    if NUMBA_AVAILABLE:
        got=mark_equity_numba(c,q,10,l); assert np.array_equal(ref,got); b=benchmark_numba_candidate(c,q,10,l,repeats=2); assert b["semantic_equal"] is True
    else:
        with pytest.raises(RuntimeError): mark_equity_numba(c,q,10,l)
