from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import threading
import time
from typing import Any, Deque

@dataclass(frozen=True)
class FactEnvelope:
    payload: bytes
    created_monotonic: float
    terminal_or_failure: bool
    semantic_id: str
    @property
    def nbytes(self) -> int:
        return len(self.payload)

def encode_fact(payload: Any, *, semantic_id: str, terminal_or_failure: bool = False) -> FactEnvelope:
    if not semantic_id:
        raise ValueError("FACT_SEMANTIC_ID_EMPTY")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return FactEnvelope(raw, time.monotonic(), bool(terminal_or_failure), semantic_id)

class BackpressureRequired(RuntimeError):
    pass

class FactOutputQueue:
    def __init__(self, *, max_bytes: int, max_age_s: float):
        if max_bytes <= 0 or max_age_s <= 0:
            raise ValueError("QUEUE_LIMIT_INVALID")
        self.max_bytes=int(max_bytes); self.max_age_s=float(max_age_s); self._q:Deque[FactEnvelope]=deque(); self._bytes=0; self._cv=threading.Condition(); self._closed=False
    @property
    def bytes_depth(self):
        with self._cv: return self._bytes
    @property
    def depth(self):
        with self._cv: return len(self._q)
    @property
    def oldest_age_s(self):
        with self._cv: return 0.0 if not self._q else max(0.0,time.monotonic()-self._q[0].created_monotonic)
    def _age_guard(self):
        if self._q and time.monotonic()-self._q[0].created_monotonic>self.max_age_s: raise BackpressureRequired("FACT_QUEUE_OLDEST_AGE_EXCEEDED")
    def put(self,item:FactEnvelope,*,block:bool=False,timeout_s:float|None=None):
        if item.nbytes>self.max_bytes: raise BackpressureRequired("FACT_LARGER_THAN_QUEUE_BUDGET")
        deadline=None if timeout_s is None else time.monotonic()+timeout_s
        with self._cv:
            while True:
                if self._closed: raise RuntimeError("FACT_QUEUE_CLOSED")
                self._age_guard()
                if self._bytes+item.nbytes<=self.max_bytes:
                    self._q.append(item); self._bytes+=item.nbytes; self._cv.notify_all(); return
                if not block: raise BackpressureRequired("FACT_QUEUE_BYTE_BUDGET_EXCEEDED")
                remaining=None if deadline is None else deadline-time.monotonic()
                if remaining is not None and remaining<=0: raise BackpressureRequired("FACT_QUEUE_BACKPRESSURE_TIMEOUT")
                self._cv.wait(remaining)
    def get(self,*,block:bool=False,timeout_s:float|None=None):
        deadline=None if timeout_s is None else time.monotonic()+timeout_s
        with self._cv:
            while not self._q:
                if self._closed: raise RuntimeError("FACT_QUEUE_CLOSED")
                if not block: raise IndexError("FACT_QUEUE_EMPTY")
                remaining=None if deadline is None else deadline-time.monotonic()
                if remaining is not None and remaining<=0: raise TimeoutError("FACT_QUEUE_GET_TIMEOUT")
                self._cv.wait(remaining)
            item=self._q.popleft(); self._bytes-=item.nbytes; self._cv.notify_all(); return item
    def close(self):
        with self._cv: self._closed=True; self._cv.notify_all()
