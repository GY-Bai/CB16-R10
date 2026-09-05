from __future__ import annotations

"""Strict installation contract for the real frozen Market Grammar Encoder.

R9 can test the contract in the sandbox with a reference fixture, but a real
historical campaign requires authority == USER_FROZEN_ENCODER and a local model
factory that reconstructs the exact historical architecture.

The installation receipt binds:
- model factory import path;
- architecture id;
- artifact SHA256;
- state_dict hash;
- parameter count;
- window/channel/latent dimensions;
- normalization hash;
- CPU deterministic canary output hash;
- optional CUDA output hash and CPU/CUDA max absolute error.

No campaign may substitute REFERENCE_CONFORMANCE_ONLY for the real encoder.
"""

import dataclasses
import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .market_encoder_r5 import (
    FrozenMarketEncoderArtifact,
    WindowNormalizer,
    MarketEncoderArtifactReceipt,
    REFERENCE_ARCHITECTURE_ID,
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tensor_sha256(x: np.ndarray) -> str:
    a = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def resolve_model_factory_r9(spec: str) -> Callable[[], nn.Module]:
    if spec == "REFERENCE_R5":
        from .market_encoder_r5 import ReferenceGrammarEncoderR5
        return ReferenceGrammarEncoderR5
    if ":" not in spec:
        raise ValueError("factory must be module:callable")
    module, name = spec.split(":", 1)
    obj = getattr(importlib.import_module(module), name)
    if not callable(obj):
        raise TypeError("encoder factory target is not callable")
    return obj


def deterministic_encoder_windows_r9(rows: int = 16) -> np.ndarray:
    if rows <= 0:
        raise ValueError("rows")
    out = np.empty((rows, 32, 5), dtype=np.float64)
    for i in range(rows):
        t = np.arange(32, dtype=np.float64)
        base = 100.0 + i * 0.7
        close = base * np.exp(0.0008 * t + 0.0015 * np.sin((t + i) / 5.0))
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) * (1.0015 + i * 1e-6)
        low = np.minimum(open_, close) * (0.9985 - i * 1e-6)
        vol = 1000.0 + 2.0 * t + 10.0 * np.cos((t + i) / 7.0)
        out[i] = np.stack([open_, high, low, close, vol], axis=1)
    return out


@dataclass(frozen=True)
class EncoderInstallPolicyR9:
    require_authority: str = "USER_FROZEN_ENCODER"
    require_window_bars: int = 32
    require_raw_channels: int = 5
    require_latent_dim: int = 64
    cpu_cuda_max_abs_error: float = 1e-4
    require_cuda_canary_on_real_install: bool = True

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class EncoderInstallationReceiptR9:
    installation_version: str
    status: str
    factory_spec: str
    architecture_id: str
    authority: str
    artifact_path: str
    artifact_sha256: str
    state_dict_weight_hash: str
    parameter_count: int
    window_bars: int
    raw_channels: int
    latent_dim: int
    normalization_hash: str
    cpu_canary_sha256: str
    cuda_canary_sha256: str | None
    cpu_cuda_max_abs_error: float | None
    cuda_device_name: str | None
    policy_hash: str
    artifact_receipt_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def install_encoder_authority_r9(
    *,
    factory_spec: str,
    architecture_id: str,
    artifact_path: str | Path,
    expected_artifact_sha256: str,
    expected_parameter_count: int,
    expected_state_dict_hash: str | None = None,
    expected_normalization_hash: str | None = None,
    authority: str = "USER_FROZEN_ENCODER",
    policy: EncoderInstallPolicyR9 | None = None,
    allow_reference_fixture: bool = False,
    cuda_device: str = "cuda",
) -> EncoderInstallationReceiptR9:
    policy = policy or EncoderInstallPolicyR9()
    if authority != policy.require_authority:
        if not (allow_reference_fixture and authority == "REFERENCE_CONFORMANCE_ONLY"):
            raise RuntimeError("R9_REAL_ENCODER_AUTHORITY_REQUIRED")
    if factory_spec == "REFERENCE_R5" and authority == "USER_FROZEN_ENCODER":
        # Prevent accidentally relabeling the R5 compatibility fixture as historical truth.
        raise RuntimeError("REFERENCE_ENCODER_CANNOT_BE_RELABELED_USER_FROZEN")

    factory = resolve_model_factory_r9(factory_spec)
    model = factory()
    if not isinstance(model, nn.Module):
        raise TypeError("encoder factory must return torch.nn.Module")
    normalizer = WindowNormalizer()
    artifact = FrozenMarketEncoderArtifact(
        model=model,
        architecture_id=architecture_id,
        artifact_path=artifact_path,
        normalizer=normalizer,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_parameter_count=expected_parameter_count,
        authority=authority,
    )
    ar = artifact.receipt
    if ar.window_bars != policy.require_window_bars:
        raise RuntimeError("ENCODER_WINDOW_BARS_MISMATCH")
    if ar.raw_channels != policy.require_raw_channels:
        raise RuntimeError("ENCODER_RAW_CHANNELS_MISMATCH")
    if ar.latent_dim != policy.require_latent_dim:
        raise RuntimeError("ENCODER_LATENT_DIM_MISMATCH")
    if expected_state_dict_hash and ar.state_dict_weight_hash != expected_state_dict_hash:
        raise RuntimeError("ENCODER_STATE_DICT_HASH_MISMATCH")
    if expected_normalization_hash and ar.normalization_hash != expected_normalization_hash:
        raise RuntimeError("ENCODER_NORMALIZATION_HASH_MISMATCH")

    windows = deterministic_encoder_windows_r9()
    cpu = artifact.encode_numpy(windows, device="cpu", batch_size=8)
    if cpu.shape != (len(windows), 64) or not np.all(np.isfinite(cpu)):
        raise RuntimeError("ENCODER_CPU_CANARY_BAD_OUTPUT")
    cpu_hash = tensor_sha256(cpu)

    cuda_hash = None
    max_err = None
    cuda_name = None
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu = artifact.encode_numpy(windows, device=cuda_device, batch_size=8)
        if gpu.shape != cpu.shape or not np.all(np.isfinite(gpu)):
            raise RuntimeError("ENCODER_CUDA_CANARY_BAD_OUTPUT")
        max_err = float(np.max(np.abs(cpu.astype(np.float64) - gpu.astype(np.float64))))
        if max_err > policy.cpu_cuda_max_abs_error:
            raise RuntimeError(
                f"ENCODER_CPU_CUDA_NUMERIC_MISMATCH:{max_err}>{policy.cpu_cuda_max_abs_error}"
            )
        cuda_hash = tensor_sha256(gpu)
        cuda_name = torch.cuda.get_device_name(torch.device(cuda_device))
    elif policy.require_cuda_canary_on_real_install and authority == "USER_FROZEN_ENCODER":
        raise RuntimeError("ENCODER_REAL_INSTALL_REQUIRES_CUDA_CANARY")

    receipt = EncoderInstallationReceiptR9(
        installation_version="CB16_REAL_FROZEN_ENCODER_INSTALL_R9",
        status=(
            "USER_FROZEN_ENCODER_INSTALLED"
            if authority == "USER_FROZEN_ENCODER"
            else "REFERENCE_FIXTURE_INSTALLED"
        ),
        factory_spec=factory_spec,
        architecture_id=architecture_id,
        authority=authority,
        artifact_path=str(Path(artifact_path).resolve()),
        artifact_sha256=ar.artifact_sha256,
        state_dict_weight_hash=ar.state_dict_weight_hash,
        parameter_count=ar.parameter_count,
        window_bars=ar.window_bars,
        raw_channels=ar.raw_channels,
        latent_dim=ar.latent_dim,
        normalization_hash=ar.normalization_hash,
        cpu_canary_sha256=cpu_hash,
        cuda_canary_sha256=cuda_hash,
        cpu_cuda_max_abs_error=max_err,
        cuda_device_name=cuda_name,
        policy_hash=policy.content_hash,
        artifact_receipt_hash=ar.content_hash,
    )
    return receipt


def save_encoder_install_receipt_r9(
    receipt: EncoderInstallationReceiptR9,
    path: str | Path,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = asdict(receipt)
    obj["content_hash"] = receipt.content_hash
    p.write_text(json.dumps(obj, indent=2) + "\n")
    return p


def load_encoder_install_receipt_r9(path: str | Path) -> EncoderInstallationReceiptR9:
    obj = json.loads(Path(path).read_text())
    claimed = obj.pop("content_hash", None)
    r = EncoderInstallationReceiptR9(**obj)
    if claimed and claimed != r.content_hash:
        raise RuntimeError("ENCODER_INSTALL_RECEIPT_HASH_MISMATCH")
    return r
