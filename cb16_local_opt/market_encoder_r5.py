from __future__ import annotations

"""
32-bar OHLCV -> 64D Market Encoder integration.

Two distinct concepts are deliberately separated:

1. `ReferenceGrammarEncoderR5`
   A ~66K parameter compatibility/conformance model supplied with this runtime.
   It is NOT claimed to be the user's historical frozen Grammar Encoder.

2. `FrozenMarketEncoderArtifact`
   A strict artifact loader for the actual versioned encoder weights.  Real experiments
   must bind file SHA256, architecture id, parameter count and state-dict weight hash.

The rolling service accepts only MarketEvents up to the decision timestamp.  No future bar
is available to the encoder.
"""

import dataclasses
import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .event_source_contracts import MarketEvent


WINDOW_BARS = 32
RAW_CHANNELS = 5
LATENT_DIM = 64
REFERENCE_ARCHITECTURE_ID = "CB16_REFERENCE_GRAMMAR_ENCODER_R5_32x5_TO_64"


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        h.update(name.encode())
        t = tensor.detach().cpu().contiguous()
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class ReferenceGrammarEncoderR5(nn.Module):
    """Reference-compatible ~66K encoder.

    It is intentionally simple and fast on a GTX1060/3700X.  The historical frozen encoder
    can be loaded through `FrozenMarketEncoderArtifact` without requiring this architecture
    if an importable model factory is supplied by the local project.
    """

    def __init__(self):
        super().__init__()
        # 160 -> 288 -> 64.  With LayerNorms this is ~65.6K parameters.
        self.input_norm = nn.LayerNorm(WINDOW_BARS * RAW_CHANNELS)
        self.fc1 = nn.Linear(WINDOW_BARS * RAW_CHANNELS, 288)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(288, LATENT_DIM)
        self.output_norm = nn.LayerNorm(LATENT_DIM)

    def forward(self, normalized_window: torch.Tensor) -> torch.Tensor:
        if normalized_window.ndim != 3 or tuple(normalized_window.shape[-2:]) != (WINDOW_BARS, RAW_CHANNELS):
            raise ValueError("encoder input must be [B,32,5]")
        x = normalized_window.reshape(normalized_window.shape[0], -1)
        x = self.input_norm(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.output_norm(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass(frozen=True)
class WindowNormalizationConfig:
    """Window-local normalization; uses no data later than the current window."""

    price_mode: str = "LOG_REL_FIRST_CLOSE"
    volume_mode: str = "LOG_REL_MEDIAN"
    epsilon: float = 1e-8
    clip: float = 12.0

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


class WindowNormalizer:
    def __init__(self, config: WindowNormalizationConfig | None = None):
        self.config = config or WindowNormalizationConfig()

    def transform_numpy(self, windows: np.ndarray) -> np.ndarray:
        x = np.asarray(windows, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != (WINDOW_BARS, RAW_CHANNELS):
            raise ValueError("windows must have shape [B,32,5]")
        if not np.all(np.isfinite(x)):
            raise ValueError("nonfinite OHLCV")
        prices = x[:, :, :4]
        volume = x[:, :, 4]
        if np.any(prices <= 0) or np.any(volume < 0):
            raise ValueError("invalid OHLCV")

        if self.config.price_mode != "LOG_REL_FIRST_CLOSE":
            raise ValueError("unsupported price normalization")
        ref = x[:, 0, 3:4]  # first close in the already-observed 32-bar window
        p = np.log(np.maximum(prices, self.config.epsilon) / np.maximum(ref[:, None, :], self.config.epsilon))

        if self.config.volume_mode == "LOG_REL_MEDIAN":
            med = np.median(volume, axis=1, keepdims=True)
            v = np.log1p(volume / np.maximum(med, self.config.epsilon))
        elif self.config.volume_mode == "LOG1P":
            v = np.log1p(volume)
        else:
            raise ValueError("unsupported volume normalization")

        out = np.concatenate([p, v[:, :, None]], axis=2)
        out = np.clip(out, -self.config.clip, self.config.clip)
        return out.astype(np.float32, copy=False)


@dataclass(frozen=True)
class MarketEncoderArtifactReceipt:
    architecture_id: str
    artifact_path: str
    artifact_sha256: str
    state_dict_weight_hash: str
    parameter_count: int
    window_bars: int
    raw_channels: int
    latent_dim: int
    normalization_hash: str
    authority: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


class FrozenMarketEncoderArtifact:
    """Strict loader for a versioned encoder artifact."""

    def __init__(
        self,
        *,
        model: nn.Module,
        architecture_id: str,
        artifact_path: str | Path,
        normalizer: WindowNormalizer,
        expected_artifact_sha256: str | None = None,
        expected_parameter_count: int | None = None,
        authority: str = "USER_FROZEN_ENCODER",
    ):
        self.model = model
        self.architecture_id = architecture_id
        self.artifact_path = Path(artifact_path)
        self.normalizer = normalizer
        if not self.artifact_path.is_file():
            raise FileNotFoundError(self.artifact_path)
        actual_sha = sha256_file(self.artifact_path)
        if expected_artifact_sha256 is not None and actual_sha != expected_artifact_sha256:
            raise RuntimeError("MARKET_ENCODER_ARTIFACT_SHA256_MISMATCH")

        try:
            payload = torch.load(self.artifact_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(self.artifact_path, map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            payload = payload["model_state_dict"]
        if not isinstance(payload, Mapping):
            raise RuntimeError("MARKET_ENCODER_ARTIFACT_NOT_STATE_DICT")
        self.model.load_state_dict(payload, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        pc = parameter_count(self.model)
        if expected_parameter_count is not None and pc != expected_parameter_count:
            raise RuntimeError(
                f"MARKET_ENCODER_PARAMETER_COUNT_MISMATCH expected={expected_parameter_count} actual={pc}"
            )
        self.receipt = MarketEncoderArtifactReceipt(
            architecture_id=architecture_id,
            artifact_path=str(self.artifact_path),
            artifact_sha256=actual_sha,
            state_dict_weight_hash=state_dict_hash(self.model.state_dict()),
            parameter_count=pc,
            window_bars=WINDOW_BARS,
            raw_channels=RAW_CHANNELS,
            latent_dim=LATENT_DIM,
            normalization_hash=self.normalizer.config.content_hash,
            authority=authority,
        )

    def to(self, device: str | torch.device):
        self.model.to(device)
        return self

    @torch.inference_mode()
    def encode_numpy(
        self,
        windows: np.ndarray,
        *,
        device: str | torch.device = "cpu",
        batch_size: int = 8192,
    ) -> np.ndarray:
        z = []
        dev = torch.device(device)
        self.model.to(dev).eval()
        normalized = self.normalizer.transform_numpy(windows)
        for i in range(0, len(normalized), batch_size):
            t = torch.from_numpy(np.ascontiguousarray(normalized[i:i+batch_size]))
            if dev.type == "cuda":
                t = t.pin_memory().to(dev, non_blocking=True)
            else:
                t = t.to(dev)
            out = self.model(t)
            z.append(out.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(z, axis=0) if z else np.empty((0, LATENT_DIM), np.float32)


def create_reference_encoder_artifact(path: str | Path, *, seed: int = 12345) -> MarketEncoderArtifactReceipt:
    """Create a deterministic REFERENCE-only artifact for CI/conformance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = ReferenceGrammarEncoderR5()
        torch.save(model.state_dict(), path)
    finally:
        torch.random.set_rng_state(cpu_state)
    loader = FrozenMarketEncoderArtifact(
        model=ReferenceGrammarEncoderR5(),
        architecture_id=REFERENCE_ARCHITECTURE_ID,
        artifact_path=path,
        normalizer=WindowNormalizer(),
        expected_parameter_count=parameter_count(ReferenceGrammarEncoderR5()),
        authority="REFERENCE_CONFORMANCE_ONLY",
    )
    return loader.receipt


@dataclass(frozen=True)
class MarketWindow:
    symbol: str
    timeframe: str
    end_timestamp: int
    events_hash: str
    ohlcv: np.ndarray

    @property
    def content_hash(self) -> str:
        return canonical_hash({
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "end_timestamp": self.end_timestamp,
            "events_hash": self.events_hash,
            "shape": tuple(self.ohlcv.shape),
            "data_sha256": hashlib.sha256(np.ascontiguousarray(self.ohlcv).tobytes()).hexdigest(),
        })


class RollingMarketWindow:
    def __init__(self, *, symbol: str, timeframe: str, interval_ms: int, window_bars: int = WINDOW_BARS):
        self.symbol = symbol
        self.timeframe = timeframe
        self.interval_ms = int(interval_ms)
        self.window_bars = int(window_bars)
        self._events: Deque[MarketEvent] = deque(maxlen=self.window_bars)
        self._last_ts: int | None = None

    def push(self, event: MarketEvent) -> MarketWindow | None:
        event.validate()
        if event.symbol != self.symbol or event.timeframe != self.timeframe:
            raise RuntimeError("MARKET_WINDOW_STREAM_ID_MISMATCH")
        if self._last_ts is not None:
            expected = self._last_ts + self.interval_ms
            if event.timestamp != expected:
                raise RuntimeError(
                    f"MARKET_WINDOW_GAP expected={expected} got={event.timestamp}"
                )
        self._last_ts = event.timestamp
        self._events.append(event)
        if len(self._events) < self.window_bars:
            return None
        arr = np.asarray(
            [[e.open, e.high, e.low, e.close, e.volume] for e in self._events],
            dtype=np.float64,
        )
        events_hash = canonical_hash([e.content_hash for e in self._events])
        return MarketWindow(
            symbol=self.symbol,
            timeframe=self.timeframe,
            end_timestamp=event.timestamp,
            events_hash=events_hash,
            ohlcv=arr,
        )


@dataclass(frozen=True)
class MarketLatentReceipt:
    symbol: str
    timeframe: str
    end_timestamp: int
    window_hash: str
    encoder_artifact_hash: str
    encoder_weight_hash: str
    latent_sha256: str
    latent_dim: int

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


class MarketEncoderService:
    """Small batched service; CPU by default, can share CUDA owner in an integrated broker."""

    def __init__(self, artifact: FrozenMarketEncoderArtifact, *, device: str = "cpu"):
        self.artifact = artifact
        self.device = device

    def encode_windows(self, windows: Sequence[MarketWindow]) -> tuple[np.ndarray, tuple[MarketLatentReceipt, ...]]:
        if not windows:
            return np.empty((0, LATENT_DIM), np.float32), ()
        x = np.stack([w.ohlcv for w in windows], axis=0)
        z = self.artifact.encode_numpy(x, device=self.device)
        receipts = []
        for w, latent in zip(windows, z):
            receipts.append(MarketLatentReceipt(
                symbol=w.symbol,
                timeframe=w.timeframe,
                end_timestamp=w.end_timestamp,
                window_hash=w.content_hash,
                encoder_artifact_hash=self.artifact.receipt.artifact_sha256,
                encoder_weight_hash=self.artifact.receipt.state_dict_weight_hash,
                latent_sha256=hashlib.sha256(np.ascontiguousarray(latent).tobytes()).hexdigest(),
                latent_dim=int(latent.shape[0]),
            ))
        return z, tuple(receipts)
