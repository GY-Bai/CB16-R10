from __future__ import annotations

"""
Single-owner GPU inference broker with shared-memory request/response buffers.

Why this exists:
- GTX1060 should have one process owning CUDA to avoid duplicate model/VRAM state.
- CPU trajectory workers should not initialize CUDA.
- Many small account batches can be micro-batched into one large GPU forward.
- Shared-memory arrays prevent repeatedly pickling/copying large NumPy payloads.

The broker is CPU-testable. On the Shanxi machine, the broker subprocess is the CUDA owner.
"""

import atexit
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SharedNDArrayRef:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def open(self) -> tuple[shared_memory.SharedMemory, np.ndarray]:
        shm = shared_memory.SharedMemory(name=self.name)
        arr = np.ndarray(self.shape, dtype=np.dtype(self.dtype), buffer=shm.buf)
        return shm, arr


class OwnedSharedArray:
    def __init__(self, shape: tuple[int, ...], dtype: str | np.dtype):
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.dtype(dtype)
        nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self._closed = False

    @property
    def ref(self) -> SharedNDArrayRef:
        return SharedNDArrayRef(self.shm.name, self.shape, str(self.dtype))

    def close(self, unlink: bool = True):
        if self._closed:
            return
        self.shm.close()
        if unlink:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
        self._closed = True


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    policy_weight_hash: str
    market: SharedNDArrayRef   # [N,64]
    account: SharedNDArrayRef  # [N,6]
    direction_out: SharedNDArrayRef # [N] int8
    risk_out: SharedNDArrayRef      # [N] float32
    rows: int
    submitted_monotonic: float

    @property
    def identity_hash(self) -> str:
        return canonical_hash({
            "request_id": self.request_id,
            "policy_weight_hash": self.policy_weight_hash,
            "market_shape": self.market.shape,
            "account_shape": self.account.shape,
            "rows": self.rows,
        })


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    status: str
    rows: int
    policy_weight_hash: str
    latency_ms: float
    broker_batch_rows: int
    error: str | None = None


@dataclass(frozen=True)
class BrokerConfig:
    tier: str = "TIER_1"
    device: str = "cuda"
    max_batch_rows: int = 16384
    max_wait_ms: float = 2.0
    start_method: str = "spawn"
    cpu_threads: int = 1

    def validate(self):
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be cuda/cpu")
        if self.max_batch_rows <= 0 or self.max_wait_ms < 0:
            raise ValueError("bad batch/wait setting")
        if self.start_method not in {"spawn", "forkserver"}:
            raise ValueError("fork is intentionally unsupported")


def _weight_hash(model) -> str:
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        t = tensor.detach().cpu().contiguous()
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _load_model(tier: str, state_dict_path: str | None, device: str):
    import torch
    from .trader_capacity_ladder import build_trader

    model = build_trader(tier)
    if state_dict_path:
        try:
            state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_dict_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    return model, dev


def _broker_main(
    config: BrokerConfig,
    request_q,
    response_q,
    state_dict_path: str | None,
    ready_q,
):
    os.environ["OMP_NUM_THREADS"] = str(config.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(config.cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(config.cpu_threads)

    import torch
    torch.set_num_threads(max(1, config.cpu_threads))
    model, device = _load_model(config.tier, state_dict_path, config.device)
    policy_hash = _weight_hash(model)
    ready_q.put({
        "status": "READY",
        "device": str(device),
        "policy_weight_hash": policy_hash,
        "tier": config.tier,
    })

    STOP = "__CB16_BROKER_STOP__"
    carry = None
    while True:
        if carry is None:
            item = request_q.get()
        else:
            item = carry
            carry = None
        if item == STOP:
            break

        pending = [item]
        rows = item.rows
        deadline = time.monotonic() + config.max_wait_ms / 1000.0

        # Micro-batch requests until row budget or wait deadline.
        while rows < config.max_batch_rows:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                nxt = request_q.get(timeout=remaining)
            except queue.Empty:
                break
            if nxt == STOP:
                carry = STOP
                break
            if rows + nxt.rows > config.max_batch_rows:
                carry = nxt
                break
            pending.append(nxt)
            rows += nxt.rows

        opened = []
        try:
            market_parts = []
            account_parts = []
            for req in pending:
                if req.policy_weight_hash != policy_hash:
                    raise RuntimeError(
                        f"POLICY_HASH_MISMATCH request={req.policy_weight_hash} broker={policy_hash}"
                    )
                m_shm, m = req.market.open()
                a_shm, a = req.account.open()
                opened.extend([m_shm, a_shm])
                if m.shape != (req.rows, model.spec.market_dim):
                    raise RuntimeError("MARKET_SHAPE_MISMATCH")
                if a.shape != (req.rows, model.spec.account_dim):
                    raise RuntimeError("ACCOUNT_SHAPE_MISMATCH")
                market_parts.append(np.array(m, copy=True))
                account_parts.append(np.array(a, copy=True))

            market_np = np.concatenate(market_parts, axis=0).astype(np.float32, copy=False)
            account_np = np.concatenate(account_parts, axis=0).astype(np.float32, copy=False)
            market_t = torch.from_numpy(market_np)
            account_t = torch.from_numpy(account_np)
            if device.type == "cuda":
                market_t = market_t.pin_memory().to(device, non_blocking=True)
                account_t = account_t.pin_memory().to(device, non_blocking=True)
            else:
                market_t = market_t.to(device)
                account_t = account_t.to(device)

            with torch.inference_mode():
                out = model(market_t, account_t)
                action = model.compose_action(out)
            d_all = action["direction"].detach().cpu().numpy().astype(np.int8, copy=False)
            r_all = action["requested_risk"].detach().cpu().numpy().astype(np.float32, copy=False)

            offset = 0
            broker_rows = len(d_all)
            for req in pending:
                d_shm, d = req.direction_out.open()
                r_shm, r = req.risk_out.open()
                opened.extend([d_shm, r_shm])
                d[:] = d_all[offset:offset+req.rows]
                r[:] = r_all[offset:offset+req.rows]
                latency = (time.monotonic() - req.submitted_monotonic) * 1000.0
                response_q.put(InferenceResponse(
                    request_id=req.request_id,
                    status="PASS",
                    rows=req.rows,
                    policy_weight_hash=policy_hash,
                    latency_ms=latency,
                    broker_batch_rows=broker_rows,
                ))
                offset += req.rows
        except Exception as exc:
            for req in pending:
                response_q.put(InferenceResponse(
                    request_id=req.request_id,
                    status="FAIL",
                    rows=req.rows,
                    policy_weight_hash=policy_hash,
                    latency_ms=(time.monotonic()-req.submitted_monotonic)*1000,
                    broker_batch_rows=rows,
                    error=repr(exc),
                ))
        finally:
            for shm in opened:
                try:
                    shm.close()
                except Exception:
                    pass

        if carry == STOP:
            break


class GPUInferenceBroker:
    def __init__(
        self,
        config: BrokerConfig,
        *,
        state_dict_path: str | None = None,
        expected_policy_hash: str | None = None,
    ):
        config.validate()
        self.config = config
        self.state_dict_path = state_dict_path
        self.expected_policy_hash = expected_policy_hash
        self.ctx = mp.get_context(config.start_method)
        self.request_q = self.ctx.Queue(maxsize=128)
        self.response_q = self.ctx.Queue(maxsize=128)
        self.ready_q = self.ctx.Queue(maxsize=1)
        self.proc = None
        self.policy_weight_hash = None
        self.device = None
        self._responses: dict[str, InferenceResponse] = {}

    def start(self, timeout: float = 30.0) -> dict[str, Any]:
        if self.proc is not None and self.proc.is_alive():
            return {
                "status":"READY",
                "policy_weight_hash":self.policy_weight_hash,
                "device":self.device,
            }
        self.proc = self.ctx.Process(
            target=_broker_main,
            args=(
                self.config,
                self.request_q,
                self.response_q,
                self.state_dict_path,
                self.ready_q,
            ),
            daemon=True,
            name="cb16-gpu-inference-broker",
        )
        self.proc.start()
        ready = self.ready_q.get(timeout=timeout)
        self.policy_weight_hash = ready["policy_weight_hash"]
        self.device = ready["device"]
        if self.expected_policy_hash and self.policy_weight_hash != self.expected_policy_hash:
            self.stop()
            raise RuntimeError("BROKER_LOADED_UNEXPECTED_POLICY_HASH")
        return ready

    def stop(self, timeout: float = 10.0):
        if self.proc is None:
            return
        if self.proc.is_alive():
            self.request_q.put("__CB16_BROKER_STOP__")
            self.proc.join(timeout=timeout)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2)
        self.proc = None

    def submit(
        self,
        market: np.ndarray,
        account: np.ndarray,
    ) -> tuple[str, OwnedSharedArray, OwnedSharedArray, OwnedSharedArray, OwnedSharedArray]:
        if self.proc is None or not self.proc.is_alive():
            raise RuntimeError("BROKER_NOT_RUNNING")
        market = np.ascontiguousarray(market, dtype=np.float32)
        account = np.ascontiguousarray(account, dtype=np.float32)
        if market.ndim != 2 or account.ndim != 2 or len(market) != len(account):
            raise ValueError("market/account must be aligned 2D arrays")
        n = len(market)
        m = OwnedSharedArray(market.shape, np.float32)
        a = OwnedSharedArray(account.shape, np.float32)
        d = OwnedSharedArray((n,), np.int8)
        r = OwnedSharedArray((n,), np.float32)
        m.array[:] = market
        a.array[:] = account
        req_id = uuid.uuid4().hex
        req = InferenceRequest(
            request_id=req_id,
            policy_weight_hash=str(self.policy_weight_hash),
            market=m.ref,
            account=a.ref,
            direction_out=d.ref,
            risk_out=r.ref,
            rows=n,
            submitted_monotonic=time.monotonic(),
        )
        self.request_q.put(req)
        return req_id, m, a, d, r

    def wait(self, request_id: str, timeout: float = 30.0) -> InferenceResponse:
        if request_id in self._responses:
            return self._responses.pop(request_id)
        deadline = time.monotonic() + timeout
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"INFERENCE_TIMEOUT:{request_id}")
            resp = self.response_q.get(timeout=rem)
            if resp.request_id == request_id:
                return resp
            self._responses[resp.request_id] = resp

    def infer(self, market: np.ndarray, account: np.ndarray, timeout: float = 30.0):
        req_id, m, a, d, r = self.submit(market, account)
        try:
            resp = self.wait(req_id, timeout=timeout)
            if resp.status != "PASS":
                raise RuntimeError(f"BROKER_INFERENCE_FAIL:{resp.error}")
            return d.array.copy(), r.array.copy(), resp
        finally:
            for x in (m,a,d,r):
                x.close(unlink=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
