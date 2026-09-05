from __future__ import annotations

"""
Integrated Market Encoder + Trader inference broker.

One subprocess owns CUDA:
    unique [M,32,5] market windows
        -> Frozen Market Encoder
        -> [M,64] latent
        -> account_to_market_index gather
        -> [N,64] market latent + [N,6] AccountState
        -> Trader
        -> direction + requested risk

This avoids:
- a separate CUDA process for the ~66K Market Encoder;
- recomputing the same market window for every account;
- sending 64D latent buffers back and forth between CPU workers.

The actual historical encoder architecture can be supplied by dotted factory path.  The
built-in reference factory remains `REFERENCE_CONFORMANCE_ONLY`.
"""

import dataclasses
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import queue
import time
import uuid
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .gpu_inference_broker import OwnedSharedArray, SharedNDArrayRef


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class EncoderLoadSpec:
    model_factory: str
    architecture_id: str
    artifact_path: str
    expected_artifact_sha256: str | None
    expected_parameter_count: int | None
    authority: str

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class IntegratedBrokerConfig:
    trader_tier: str = "TIER_1"
    trader_state_dict_path: str | None = None
    device: str = "cuda"
    max_account_rows: int = 16384
    max_market_windows: int = 256
    start_method: str = "spawn"
    cpu_threads: int = 1

    def validate(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device")
        if self.start_method not in {"spawn", "forkserver"}:
            raise ValueError("plain fork intentionally unsupported")
        if self.max_account_rows <= 0 or self.max_market_windows <= 0:
            raise ValueError("bad limits")


@dataclass(frozen=True)
class IntegratedRequest:
    request_id: str
    market_windows: SharedNDArrayRef          # [M,32,5] float32/64
    account_state6: SharedNDArrayRef          # [N,6]
    account_to_market: SharedNDArrayRef       # [N] int32/int64
    direction_out: SharedNDArrayRef           # [N] int8
    risk_out: SharedNDArrayRef                # [N] float32
    market_latent_out: SharedNDArrayRef | None # optional [M,64]
    markets: int
    accounts: int
    submitted_monotonic: float


@dataclass(frozen=True)
class IntegratedResponse:
    request_id: str
    status: str
    markets: int
    accounts: int
    trader_weight_hash: str
    encoder_artifact_sha256: str
    encoder_weight_hash: str
    latency_ms: float
    error: str | None = None


def reference_encoder_factory():
    from .market_encoder_r5 import ReferenceGrammarEncoderR5
    return ReferenceGrammarEncoderR5()


def _resolve_factory(path: str):
    if path == "REFERENCE_R5":
        return reference_encoder_factory
    if ":" not in path:
        raise ValueError("model_factory must be module:function or REFERENCE_R5")
    module, name = path.split(":", 1)
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError("encoder factory is not callable")
    return fn


def _trader_weight_hash(model) -> str:
    from .market_encoder_r5 import state_dict_hash
    return state_dict_hash(model.state_dict())


def _broker_main(config, encoder_spec, request_q, response_q, ready_q):
    os.environ["OMP_NUM_THREADS"] = str(config.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(config.cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(config.cpu_threads)

    import torch
    from .market_encoder_r5 import (
        FrozenMarketEncoderArtifact,
        WindowNormalizer,
    )
    from .trader_capacity_ladder import build_trader

    torch.set_num_threads(max(1, config.cpu_threads))
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")

    factory = _resolve_factory(encoder_spec.model_factory)
    encoder_model = factory()
    artifact = FrozenMarketEncoderArtifact(
        model=encoder_model,
        architecture_id=encoder_spec.architecture_id,
        artifact_path=encoder_spec.artifact_path,
        normalizer=WindowNormalizer(),
        expected_artifact_sha256=encoder_spec.expected_artifact_sha256,
        expected_parameter_count=encoder_spec.expected_parameter_count,
        authority=encoder_spec.authority,
    )
    artifact.model.to(device).eval()

    trader = build_trader(config.trader_tier)
    if config.trader_state_dict_path:
        try:
            tstate = torch.load(config.trader_state_dict_path, map_location="cpu", weights_only=True)
        except TypeError:
            tstate = torch.load(config.trader_state_dict_path, map_location="cpu")
        if isinstance(tstate, Mapping) and "model_state_dict" in tstate:
            tstate = tstate["model_state_dict"]
        trader.load_state_dict(tstate, strict=True)
    trader.to(device).eval()
    trader_hash = _trader_weight_hash(trader)

    ready_q.put({
        "status": "READY",
        "device": str(device),
        "trader_weight_hash": trader_hash,
        "encoder_receipt": asdict(artifact.receipt),
        "encoder_receipt_hash": artifact.receipt.content_hash,
    })

    STOP = "__CB16_INTEGRATED_BROKER_STOP__"
    while True:
        req = request_q.get()
        if req == STOP:
            break
        opened = []
        try:
            if req.markets > config.max_market_windows:
                raise RuntimeError("TOO_MANY_MARKET_WINDOWS")
            if req.accounts > config.max_account_rows:
                raise RuntimeError("TOO_MANY_ACCOUNTS")

            w_shm, windows = req.market_windows.open()
            a_shm, accounts = req.account_state6.open()
            i_shm, mapping = req.account_to_market.open()
            d_shm, dout = req.direction_out.open()
            r_shm, rout = req.risk_out.open()
            opened.extend([w_shm, a_shm, i_shm, d_shm, r_shm])
            latent_arr = None
            if req.market_latent_out is not None:
                z_shm, latent_arr = req.market_latent_out.open()
                opened.append(z_shm)

            if windows.shape != (req.markets, 32, 5):
                raise RuntimeError("WINDOW_SHAPE_MISMATCH")
            if accounts.shape != (req.accounts, 6):
                raise RuntimeError("ACCOUNT_SHAPE_MISMATCH")
            if mapping.shape != (req.accounts,):
                raise RuntimeError("ACCOUNT_MARKET_MAP_SHAPE_MISMATCH")
            idx = np.asarray(mapping, dtype=np.int64)
            if np.any((idx < 0) | (idx >= req.markets)):
                raise RuntimeError("ACCOUNT_MARKET_INDEX_OUT_OF_RANGE")

            normalized = artifact.normalizer.transform_numpy(np.asarray(windows))
            wt = torch.from_numpy(np.ascontiguousarray(normalized))
            at = torch.from_numpy(np.ascontiguousarray(accounts, dtype=np.float32))
            it = torch.from_numpy(np.ascontiguousarray(idx, dtype=np.int64))

            if device.type == "cuda":
                wt = wt.pin_memory().to(device, non_blocking=True)
                at = at.pin_memory().to(device, non_blocking=True)
                it = it.pin_memory().to(device, non_blocking=True)
            else:
                wt, at, it = wt.to(device), at.to(device), it.to(device)

            with torch.inference_mode():
                z = artifact.model(wt)
                z_accounts = z.index_select(0, it)
                out = trader(z_accounts, at)
                action = trader.compose_action(out)

            d = action["direction"].detach().cpu().numpy().astype(np.int8, copy=False)
            r = action["requested_risk"].detach().cpu().numpy().astype(np.float32, copy=False)
            dout[:] = d
            rout[:] = r
            if latent_arr is not None:
                latent_arr[:] = z.detach().cpu().numpy().astype(np.float32, copy=False)

            response_q.put(IntegratedResponse(
                request_id=req.request_id,
                status="PASS",
                markets=req.markets,
                accounts=req.accounts,
                trader_weight_hash=trader_hash,
                encoder_artifact_sha256=artifact.receipt.artifact_sha256,
                encoder_weight_hash=artifact.receipt.state_dict_weight_hash,
                latency_ms=(time.monotonic() - req.submitted_monotonic) * 1000.0,
            ))
        except Exception as exc:
            response_q.put(IntegratedResponse(
                request_id=req.request_id,
                status="FAIL",
                markets=req.markets,
                accounts=req.accounts,
                trader_weight_hash=trader_hash,
                encoder_artifact_sha256=artifact.receipt.artifact_sha256,
                encoder_weight_hash=artifact.receipt.state_dict_weight_hash,
                latency_ms=(time.monotonic() - req.submitted_monotonic) * 1000.0,
                error=repr(exc),
            ))
        finally:
            for shm in opened:
                try:
                    shm.close()
                except Exception:
                    pass


class IntegratedMarketTraderBroker:
    def __init__(
        self,
        config: IntegratedBrokerConfig,
        encoder_spec: EncoderLoadSpec,
    ):
        config.validate()
        self.config = config
        self.encoder_spec = encoder_spec
        self.ctx = mp.get_context(config.start_method)
        self.request_q = self.ctx.Queue(maxsize=64)
        self.response_q = self.ctx.Queue(maxsize=64)
        self.ready_q = self.ctx.Queue(maxsize=1)
        self.proc = None
        self.ready = None
        self._responses = {}

    def start(self, timeout: float = 30.0):
        if self.proc is not None and self.proc.is_alive():
            return self.ready
        self.proc = self.ctx.Process(
            target=_broker_main,
            args=(self.config, self.encoder_spec, self.request_q, self.response_q, self.ready_q),
            daemon=True,
            name="cb16-integrated-market-trader-broker",
        )
        self.proc.start()
        self.ready = self.ready_q.get(timeout=timeout)
        return self.ready

    def stop(self):
        if self.proc is None:
            return
        if self.proc.is_alive():
            self.request_q.put("__CB16_INTEGRATED_BROKER_STOP__")
            self.proc.join(timeout=10)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2)
        self.proc = None

    def infer(
        self,
        market_windows: np.ndarray,
        account_state6: np.ndarray,
        account_to_market: np.ndarray,
        *,
        return_market_latent: bool = False,
        timeout: float = 30.0,
    ):
        if self.proc is None or not self.proc.is_alive():
            raise RuntimeError("BROKER_NOT_RUNNING")
        windows = np.ascontiguousarray(market_windows, dtype=np.float32)
        accounts = np.ascontiguousarray(account_state6, dtype=np.float32)
        mapping = np.ascontiguousarray(account_to_market, dtype=np.int64)
        m, n = len(windows), len(accounts)

        wbuf = OwnedSharedArray(windows.shape, np.float32)
        abuf = OwnedSharedArray(accounts.shape, np.float32)
        ibuf = OwnedSharedArray(mapping.shape, np.int64)
        dbuf = OwnedSharedArray((n,), np.int8)
        rbuf = OwnedSharedArray((n,), np.float32)
        zbuf = OwnedSharedArray((m, 64), np.float32) if return_market_latent else None
        wbuf.array[:] = windows
        abuf.array[:] = accounts
        ibuf.array[:] = mapping

        req_id = uuid.uuid4().hex
        req = IntegratedRequest(
            request_id=req_id,
            market_windows=wbuf.ref,
            account_state6=abuf.ref,
            account_to_market=ibuf.ref,
            direction_out=dbuf.ref,
            risk_out=rbuf.ref,
            market_latent_out=None if zbuf is None else zbuf.ref,
            markets=m,
            accounts=n,
            submitted_monotonic=time.monotonic(),
        )
        self.request_q.put(req)
        try:
            deadline = time.monotonic() + timeout
            while True:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError("INTEGRATED_INFERENCE_TIMEOUT")
                resp = self.response_q.get(timeout=rem)
                if resp.request_id == req_id:
                    break
                self._responses[resp.request_id] = resp
            if resp.status != "PASS":
                raise RuntimeError(f"INTEGRATED_BROKER_FAIL:{resp.error}")
            z = None if zbuf is None else zbuf.array.copy()
            return dbuf.array.copy(), rbuf.array.copy(), z, resp
        finally:
            for buf in (wbuf, abuf, ibuf, dbuf, rbuf):
                buf.close(unlink=True)
            if zbuf is not None:
                zbuf.close(unlink=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
