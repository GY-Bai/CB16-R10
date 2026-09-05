from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


@dataclass(frozen=True)
class SourceAssetSpecR10:
    asset_id: str
    repo_id: str
    revision: str
    filename: str
    source_sha256: str
    config_filename: str
    config_sha256: str


SOURCE_ASSETS_R10 = {
    "kronos_small": SourceAssetSpecR10(
        asset_id="kronos_small",
        repo_id="NeoQuasar/Kronos-small",
        revision="901c26c1332695a2a8f243eb2f37243a37bea320",
        filename="model.safetensors",
        source_sha256="b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020",
        config_filename="config.json",
        config_sha256="5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396",
    ),
    "kronos_tokenizer_base": SourceAssetSpecR10(
        asset_id="kronos_tokenizer_base",
        repo_id="NeoQuasar/Kronos-Tokenizer-base",
        revision="0e0117387f39004a9016484a186a908917e22426",
        filename="model.safetensors",
        source_sha256="59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
        config_filename="config.json",
        config_sha256="2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2",
    ),
    "timesfm_2p5_200m": SourceAssetSpecR10(
        asset_id="timesfm_2p5_200m",
        repo_id="google/timesfm-2.5-200m-transformers",
        revision="5a9806b9b291fad9233b5249d88263f1846304d3",
        filename="model.safetensors",
        source_sha256="b53f6d52114e2ad786890f3c4637ce05f580b7800d6e24401f88b398b76035ef",
        config_filename="config.json",
        config_sha256="452ecae918f67b2e7d0f2892ab424d1876939e70d077db3708a4fe8ca03a7de5",
    ),
}


PREFIX_SPECS_R10 = {
    "kronos_model_l5": {
        "source": "kronos_small",
        "output_relpath": "assets/operator/runtime/kronos_model_l5.safetensors",
        "role": "Kronos-small embedding + temporal embedding + transformer layers 1..5 only; D45=L5-L4",
        "key_count": 79,
        "tensor_bytes": 59062912,
        "semantic_sha256": "b17dd3cb0f8fdb97e2327debb004a9c791f615d21b77432c2d2b3c4efb149cdc",
    },
    "kronos_tokenizer_encode": {
        "source": "kronos_tokenizer_base",
        "output_relpath": "assets/operator/runtime/kronos_tokenizer_encode.safetensors",
        "role": "Kronos tokenizer encode path only: embed + encoder + quant_embed (+ quantizer state if any); decoder/reconstruction path excluded",
        "key_count": 48,
        "tensor_bytes": 7911056,
        "semantic_sha256": "b315317b2ea619c13f7b9b3a87da085750ced2965e85e97fe17a5f32eb8cf089",
    },
    "timesfm_layer3": {
        "source": "timesfm_2p5_200m",
        "output_relpath": "assets/medium/runtime/timesfm_layer3.safetensors",
        "role": "TimesFM input patch tokenizer + transformer layers 1..3 only; Medium48 taps last token after Layer3",
        "key_count": 45,
        "tensor_bytes": 125253440,
        "semantic_sha256": "23eab66683b2581927d2d78ffcdb7c15b6d40ff7da2cd13e2277451c9bdebb2c",
    },
}


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tensor_semantic_sha256_from_dict(tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(tensors):
        t = tensors[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8")); h.update(b"\0")
        h.update(str(t.dtype).encode("ascii")); h.update(b"\0")
        h.update(json.dumps(list(t.shape), separators=(",", ":")).encode("ascii")); h.update(b"\0")
        h.update(t.numpy().tobytes(order="C"))
    return h.hexdigest()


def tensor_semantic_sha256_file(path: str | Path) -> tuple[str, int, int]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return (
        tensor_semantic_sha256_from_dict(tensors),
        len(tensors),
        sum(t.numel() * t.element_size() for t in tensors.values()),
    )


def _selector(prefix_id: str) -> Callable[[str], bool]:
    if prefix_id == "kronos_model_l5":
        return lambda k: (
            k.startswith("embedding.")
            or k.startswith("time_emb.")
            or any(k.startswith(f"transformer.{i}.") for i in range(5))
        )
    if prefix_id == "kronos_tokenizer_encode":
        return lambda k: (
            k.startswith("embed.")
            or k.startswith("encoder.")
            or k.startswith("quant_embed.")
            or k.startswith("tokenizer.")
        )
    if prefix_id == "timesfm_layer3":
        return lambda k: (
            k.startswith("model.input_ff_layer.")
            or any(k.startswith(f"model.layers.{i}.") for i in range(3))
        )
    raise KeyError(prefix_id)


def extract_prefix(prefix_id: str, source_weight: str | Path, output_path: str | Path) -> dict:
    spec = PREFIX_SPECS_R10[prefix_id]
    pred = _selector(prefix_id)
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(source_weight), framework="pt", device="cpu") as f:
        for key in f.keys():
            if pred(key):
                tensors[key] = f.get_tensor(key)
    tensors = dict(sorted(tensors.items()))
    semantic = tensor_semantic_sha256_from_dict(tensors)
    tensor_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    if len(tensors) != int(spec["key_count"]):
        raise RuntimeError(f"PREFIX_KEY_COUNT_MISMATCH:{prefix_id}:{len(tensors)}!={spec['key_count']}")
    if tensor_bytes != int(spec["tensor_bytes"]):
        raise RuntimeError(f"PREFIX_TENSOR_BYTES_MISMATCH:{prefix_id}:{tensor_bytes}!={spec['tensor_bytes']}")
    if semantic != spec["semantic_sha256"]:
        raise RuntimeError(f"PREFIX_SEMANTIC_HASH_MISMATCH:{prefix_id}:{semantic}!={spec['semantic_sha256']}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(out),
        metadata={
            "cb16_prefix_id": prefix_id,
            "cb16_role": str(spec["role"]),
            "source_sha256": SOURCE_ASSETS_R10[spec["source"]].source_sha256,
        },
    )
    post_semantic, post_count, post_bytes = tensor_semantic_sha256_file(out)
    if (post_semantic, post_count, post_bytes) != (semantic, len(tensors), tensor_bytes):
        raise RuntimeError(f"PREFIX_ROUNDTRIP_MISMATCH:{prefix_id}")
    return {
        "prefix_id": prefix_id,
        "source_asset": spec["source"],
        "output_path": str(out),
        "key_count": len(tensors),
        "tensor_bytes": tensor_bytes,
        "semantic_sha256": semantic,
        "serialized_sha256": sha256_file(out),
        "role": spec["role"],
    }


def verify_installed_prefix(package_root: str | Path, prefix_id: str) -> dict:
    spec = PREFIX_SPECS_R10[prefix_id]
    path = Path(package_root) / spec["output_relpath"]
    if not path.is_file():
        raise FileNotFoundError(path)
    semantic, count, nbytes = tensor_semantic_sha256_file(path)
    if semantic != spec["semantic_sha256"]:
        raise RuntimeError(f"PREFIX_SEMANTIC_HASH_MISMATCH:{prefix_id}")
    if count != spec["key_count"] or nbytes != spec["tensor_bytes"]:
        raise RuntimeError(f"PREFIX_SHAPE_CONTRACT_MISMATCH:{prefix_id}")
    return {
        "prefix_id": prefix_id,
        "path": str(path),
        "semantic_sha256": semantic,
        "key_count": count,
        "tensor_bytes": nbytes,
        "serialized_sha256": sha256_file(path),
    }
