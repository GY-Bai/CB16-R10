from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

G0_FILE_SHA256 = "b123a89ff3a7838861ccad592e71305b4bc09c4b9e3a05f9fd22564762f4304f"
G0_TENSOR_SEMANTIC_SHA256 = "ab4acc5962f7285efad5c1b7a3e60c3d53fb3102891364a46dc8250768bf9185"
PHYSICS_BUNDLE_SHA256 = "19f89018ef9b7c7301fe13c57e6b2abb512ffc770b29a7b3fb9df9a1be9f47de"
PHYSICS_CONTRACT_SHA256 = "d1da4242141dc2e5aa257eddc50a2573b8d41471faa3e6270acb596e6c459321"
V55_KERNEL_AGGREGATE_SHA256 = "518f90f3f0db790b0e6e49ec05def3b98344d92984cf2956d9dd9a2cc5701d87"
RISK_SUPERVISOR_SHA256 = "1fee663d23d400dca7900df5b839fdebe126b9190bac02069f4ea4a70451e9f6"

CANONICAL_SYMBOLS_R102 = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
ALL_SUPPORTED_SYMBOLS_R102 = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT",
)
FORBIDDEN_FINAL_MONTH = "2025-09"
FORBIDDEN_FINAL_START_MS = int(datetime(2025, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
TRAIN_END_MS = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
VALIDATION_END_MS = FORBIDDEN_FINAL_START_MS
HOUR_MS = 3_600_000
H72 = 72
TRAIN_VALIDATION_PURGE_HOURS = 128
# Binance funding calc_time in official archives can carry tiny millisecond settlement jitter
# around the scheduled UTC hour (e.g. +2 ms). R10.2.1 canonicalizes only bounded jitter.
FUNDING_CANONICAL_JITTER_TOLERANCE_MS = 1_000
FUNDING_CANONICALIZATION_POLICY = "NEAREST_UTC_HOUR_IF_ABS_JITTER_LE_1000MS__RAW_TIME_AUDITED__NO_FORWARD_FILL"


def canonical_json_bytes(obj: Any) -> bytes:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def atomic_write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(canonical_json_bytes(obj) + b"\n")
    os.replace(tmp, p)


def atomic_write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def hardlink_or_copy(src: str | Path, dst: str | Path) -> str:
    s, d = Path(src), Path(dst)
    if not s.is_file():
        raise FileNotFoundError(s)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.unlink(missing_ok=True)
    try:
        os.link(s, d)
        return "HARDLINK"
    except OSError:
        shutil.copy2(s, d)
        return "COPY"


def utc_iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()


def month_key_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m")


def assert_not_forbidden_timestamp(ms: int, *, what: str) -> None:
    if month_key_from_ms(ms) == FORBIDDEN_FINAL_MONTH:
        raise RuntimeError(f"FINAL_HOLDOUT_ACCESS_FAIL_CLOSED:{what}:{utc_iso_from_ms(ms)}")


def torch_tensor_semantic_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Serialization-independent checkpoint identity over sorted tensor keys/dtypes/shapes/bytes."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8") + b"\0")
        h.update(str(t.dtype).encode("ascii") + b"\0")
        h.update(json.dumps(list(t.shape), separators=(",", ":")).encode("ascii") + b"\0")
        h.update(t.numpy().tobytes(order="C"))
    return h.hexdigest()


def load_brain_checkpoint(path: str | Path, model: torch.nn.Module, *, device: str | torch.device) -> dict[str, Any]:
    p = Path(path)
    obj = torch.load(p, map_location="cpu", weights_only=True)
    state = obj.get("state_dict") or obj.get("model_state") or obj.get("model") or obj
    if not isinstance(state, Mapping):
        raise RuntimeError("G0_CHECKPOINT_STATE_NOT_FOUND")
    model.load_state_dict(state, strict=True)
    model.to(device)
    return dict(obj) if isinstance(obj, dict) else {"state_dict": state}


def model_state_semantic_sha256(model: torch.nn.Module) -> str:
    return torch_tensor_semantic_sha256(model.state_dict())


def model_parameter_l2_delta(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    s = 0.0
    for k in a:
        x = a[k].detach().cpu().double() - b[k].detach().cpu().double()
        s += float(torch.sum(x * x))
    return math.sqrt(s)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
