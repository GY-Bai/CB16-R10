from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .binance_archive_input_r10 import SensoryDecisionFrameR10
from .frozen_layer_contract_r10 import PREFIX_SPECS_R10, verify_installed_prefix

OP_REDUCER_SHA = "c61b20341ea7d859842821bfd401baad17b210d02317da6324be7de9d3d54423"
MEDIUM_PORTABLE_SHA = "897ba29937817b225df73c6ad2b5bbf183d2a18a0488dcdcff5d7b43ade436c2"


def sha256_file(p: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class SensoryAssetPathsR10:
    package_root: Path

    @property
    def kronos_model_prefix(self): return self.package_root / PREFIX_SPECS_R10["kronos_model_l5"]["output_relpath"]
    @property
    def kronos_tokenizer_prefix(self): return self.package_root / PREFIX_SPECS_R10["kronos_tokenizer_encode"]["output_relpath"]
    @property
    def operator_reducers(self): return self.package_root / "assets/operator/operator_reducers_v1.npz"
    @property
    def kronos_source(self): return self.package_root / "third_party/kronos"
    @property
    def kronos_model_config(self): return self.package_root / "assets/operator/kronos_small/config.json"
    @property
    def kronos_tokenizer_config(self): return self.package_root / "assets/operator/tokenizer_base/config.json"
    @property
    def timesfm_prefix(self): return self.package_root / PREFIX_SPECS_R10["timesfm_layer3"]["output_relpath"]
    @property
    def timesfm_source(self): return self.package_root / "third_party/timesfm/src"
    @property
    def medium_adapter(self): return self.package_root / "assets/medium/CANONICAL_NONLINEAR48_SEED24680_PORTABLE.npz"

    def verify(self) -> dict:
        out = {
            "kronos_model_l5": verify_installed_prefix(self.package_root, "kronos_model_l5"),
            "kronos_tokenizer_encode": verify_installed_prefix(self.package_root, "kronos_tokenizer_encode"),
            "timesfm_layer3": verify_installed_prefix(self.package_root, "timesfm_layer3"),
        }
        for key, path, expected in [
            ("operator_reducers", self.operator_reducers, OP_REDUCER_SHA),
            ("medium_adapter", self.medium_adapter, MEDIUM_PORTABLE_SHA),
        ]:
            if not path.is_file(): raise FileNotFoundError(path)
            actual = sha256_file(path)
            if actual != expected: raise RuntimeError(f"ASSET_HASH_MISMATCH:{key}:{actual}!={expected}")
            out[key] = {"path": str(path), "sha256": actual}
        return out


class FrozenOperator48R10:
    """Exact deployed Operator48 path, but only the historically used Kronos layers are resident.

    Historical authority: D45=L5-L4 for both 60x1m Micro and 32x1h Macro.
    Therefore the active Kronos model contains embedding/time-embedding and transformer layers 1..5 only.
    The tokenizer contains only its encode path; decoder/reconstruction modules are not active runtime dependencies.
    """
    def __init__(self, paths: SensoryAssetPathsR10, *, device: str = "cpu"):
        sys.path.insert(0, str(paths.kronos_source))
        from model.kronos import KronosTokenizer, Kronos
        self.device = torch.device(device)
        tcfg = json.loads(paths.kronos_tokenizer_config.read_text())
        mcfg = json.loads(paths.kronos_model_config.read_text())

        tok = KronosTokenizer(**tcfg)
        # Encode-only tokenizer. These reconstruction modules never participate in the frozen Operator D45 path.
        del tok.decoder, tok.head, tok.post_quant_embed_pre, tok.post_quant_embed
        tok.load_state_dict(load_file(str(paths.kronos_tokenizer_prefix), device="cpu"), strict=True)
        self.tok = tok.eval().to(self.device)

        model = Kronos(**mcfg)
        # 1-based L5 tap requires exactly transformer blocks 0..4. L6-L8, final norm, dependency layer and heads are unused.
        model.transformer = torch.nn.ModuleList(list(model.transformer[:5]))
        del model.norm, model.dep_layer, model.head
        model.load_state_dict(load_file(str(paths.kronos_model_prefix), device="cpu"), strict=True)
        self.model = model.eval().to(self.device)

        for p in self.tok.parameters(): p.requires_grad_(False)
        for p in self.model.parameters(): p.requires_grad_(False)
        self.r = np.load(paths.operator_reducers, allow_pickle=False)

    @staticmethod
    def _prep(W):
        W = np.asarray(W, dtype=np.float32)
        if W.ndim == 2: W = W[None]
        if W.shape[-1] != 5: raise ValueError("expected OHLCV")
        o = W[:, :, :4]; v = W[:, :, 4:5]; amt = v * o.mean(2, keepdims=True)
        x = np.concatenate([o, v, amt], 2).astype(np.float32)
        m = x.mean(1, keepdims=True); s = x.std(1, keepdims=True)
        return np.clip((x - m) / (s + 1e-5), -5, 5)

    def _d45(self, W, stamp):
        X = self._prep(W); T = np.asarray(stamp, dtype=np.float32)
        if T.ndim == 2: T = T[None]
        if X.shape[:2] != T.shape[:2]: raise ValueError("window/stamp shape mismatch")
        with torch.inference_mode():
            x = torch.from_numpy(X).to(self.device)
            st = torch.from_numpy(T).to(self.device)
            z = self.tok.embed(x)
            for L in self.tok.encoder: z = L(z)
            z = self.tok.quant_embed(z)
            _, _, ids = self.tok.tokenizer(z, half=True, collect_metrics=False)
            h = self.model.embedding([ids[0], ids[1]]) + self.model.time_emb(st)
            h = self.model.token_drop(h); h4 = None
            for li, L in enumerate(self.model.transformer, 1):
                h = L(h)
                if li == 4: h4 = h[:, -1].clone()
                if li == 5: return (h[:, -1] - h4).detach().cpu().numpy().astype(np.float32)
        raise RuntimeError("KRONOS_LAYER5_NOT_REACHED")

    def _reduce(self, Z, scale):
        Z = np.asarray(Z, dtype=np.float64)
        S = (Z - self.r[f"{scale}_sc_mean"]) / self.r[f"{scale}_sc_scale"]
        return ((S - self.r[f"{scale}_pca_mean"]) @ self.r[f"{scale}_pca_components"].T).astype(np.float32)

    def encode_batch(self, micro_60x5, micro_stamp, macro_32x5, macro_stamp) -> np.ndarray:
        mi = self._reduce(self._d45(micro_60x5, micro_stamp), "micro")
        ma = self._reduce(self._d45(macro_32x5, macro_stamp), "macro")
        out = np.concatenate([mi, ma], axis=1)
        if out.shape[1] != 48: raise RuntimeError(f"OPERATOR48_SHAPE:{out.shape}")
        return out


class FrozenMedium48R10:
    """Exact Medium48 path with TimesFM truncated after the frozen Layer3 tap.

    Active input: current 64 complete hourly closes.
    Active foundation tensors: input patch tokenizer + TimesFM transformer layers 1..3 only.
    Layers 4..20 and forecast output heads are intentionally absent from the runtime.
    """
    def __init__(self, paths: SensoryAssetPathsR10, *, device: str = "cpu"):
        sys.path.insert(0, str(paths.timesfm_source))
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
        self._TimesFM = TimesFM_2p5_200M_torch_module
        self.device = torch.device(device)
        self.model = self._load_model(paths.timesfm_prefix).eval().to(self.device)
        for p in self.model.parameters(): p.requires_grad_(False)
        z = np.load(paths.medium_adapter, allow_pickle=False)
        self.scaler_mean = z["scaler_mean"].astype(np.float32)
        self.scaler_scale = z["scaler_scale"].astype(np.float32)
        self.w0 = torch.from_numpy(z["state_net__0__weight"].astype(np.float32)).to(self.device)
        self.b0 = torch.from_numpy(z["state_net__0__bias"].astype(np.float32)).to(self.device)
        self.w2 = torch.from_numpy(z["state_net__2__weight"].astype(np.float32)).to(self.device)
        self.b2 = torch.from_numpy(z["state_net__2__bias"].astype(np.float32)).to(self.device)
        self.sm = torch.from_numpy(self.scaler_mean).to(self.device)
        self.ss = torch.from_numpy(self.scaler_scale).to(self.device)

    @staticmethod
    def _convert_hf_layer3(hf):
        out = {}; p = "model.input_ff_layer."
        out["tokenizer.hidden_layer.weight"] = hf[p+"input_layer.weight"]
        out["tokenizer.hidden_layer.bias"] = hf[p+"input_layer.bias"]
        out["tokenizer.output_layer.weight"] = hf[p+"output_layer.weight"]
        out["tokenizer.output_layer.bias"] = hf[p+"output_layer.bias"]
        out["tokenizer.residual_layer.weight"] = hf[p+"residual_layer.weight"]
        out["tokenizer.residual_layer.bias"] = hf[p+"residual_layer.bias"]
        for i in range(3):
            a=f"model.layers.{i}."; b=f"stacked_xf.{i}."
            out[b+"pre_attn_ln.scale"] = hf[a+"input_layernorm.weight"]
            out[b+"post_attn_ln.scale"] = hf[a+"post_attention_layernorm.weight"]
            out[b+"pre_ff_ln.scale"] = hf[a+"pre_feedforward_layernorm.weight"]
            out[b+"post_ff_ln.scale"] = hf[a+"post_feedforward_layernorm.weight"]
            out[b+"attn.qkv_proj.weight"] = torch.cat([hf[a+"self_attn.q_proj.weight"], hf[a+"self_attn.k_proj.weight"], hf[a+"self_attn.v_proj.weight"]], 0)
            out[b+"attn.out.weight"] = hf[a+"self_attn.o_proj.weight"]
            out[b+"attn.query_ln.scale"] = hf[a+"self_attn.q_norm.weight"]
            out[b+"attn.key_ln.scale"] = hf[a+"self_attn.k_norm.weight"]
            out[b+"attn.per_dim_scale.per_dim_scale"] = hf[a+"self_attn.scaling"]
            out[b+"ff0.weight"] = hf[a+"mlp.ff0.weight"]
            out[b+"ff1.weight"] = hf[a+"mlp.ff1.weight"]
        return out

    def _load_model(self, weight_path: Path):
        hf = load_file(str(weight_path), device="cpu")
        m = self._TimesFM()
        m.stacked_xf = torch.nn.ModuleList(list(m.stacked_xf[:3]))
        del m.output_projection_point, m.output_projection_quantiles
        native = self._convert_hf_layer3(hf)
        m.load_state_dict(native, strict=True)
        del hf, native
        return m

    def _preprocess(self, x: torch.Tensor):
        from timesfm.torch import util as t5_util
        B, T = x.shape; p = self.model.p
        if T % p != 0: raise ValueError(f"TimesFM input length {T} not divisible by patch {p}")
        masks = torch.zeros_like(x, dtype=torch.bool)
        patched = x.reshape(B, -1, p); pm = masks.reshape(B, -1, p)
        n=torch.zeros(B, device=x.device); mu=torch.zeros(B, device=x.device); sigma=torch.zeros(B, device=x.device)
        mus=[]; sigs=[]
        for i in range(patched.shape[1]):
            (n,mu,sigma),_ = t5_util.update_running_stats(n,mu,sigma,patched[:,i],pm[:,i]); mus.append(mu); sigs.append(sigma)
        mu=torch.stack(mus,1); sigma=torch.stack(sigs,1)
        norm=t5_util.revin(patched,mu,sigma,reverse=False)
        return torch.where(pm,0.,norm),pm

    def layer3_batch(self, close_64: np.ndarray) -> torch.Tensor:
        x = np.asarray(close_64, dtype=np.float32)
        if x.ndim == 1: x = x[None]
        if x.shape[1] != 64: raise ValueError(f"Medium48 requires 64 hourly closes; got {x.shape}")
        with torch.inference_mode():
            xb = torch.from_numpy(x).to(self.device)
            norm, pm = self._preprocess(xb)
            h = self.model.tokenizer(torch.cat([norm, pm.to(norm.dtype)], -1))
            for li, layer in enumerate(self.model.stacked_xf, 1):
                h, _ = layer(h, pm[..., -1], None)
                if li == 3: return h[:, -1, :]
        raise RuntimeError("TIMESFM_LAYER3_NOT_REACHED")

    def encode_batch(self, close_64: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            l3 = self.layer3_batch(close_64)
            z = (l3 - self.sm) / self.ss
            z = F.linear(z, self.w0, self.b0)
            z = F.gelu(z)
            z = F.linear(z, self.w2, self.b2)
        out = z.detach().cpu().numpy().astype(np.float32)
        if out.shape[1] != 48: raise RuntimeError(f"MEDIUM48_SHAPE:{out.shape}")
        return out


@dataclass(frozen=True)
class TypedSensoryBatchR10:
    operator48: np.ndarray
    medium48: np.ndarray
    ordered4h30: np.ndarray
    symbols: tuple[str, ...]
    decision_time_ms: np.ndarray


class FrozenSensoryStackR10:
    """Frozen production sensory body. No gradients are owned here."""
    def __init__(self, package_root: str | Path, *, device: str = "cpu", verify_hashes: bool = True):
        self.paths = SensoryAssetPathsR10(Path(package_root))
        self.verified_hashes = self.paths.verify() if verify_hashes else {}
        self.operator = FrozenOperator48R10(self.paths, device=device)
        self.medium = FrozenMedium48R10(self.paths, device=device)

    def encode_frames(self, frames: Sequence[SensoryDecisionFrameR10]) -> TypedSensoryBatchR10:
        if not frames: raise ValueError("empty frames")
        mi = np.stack([f.micro_1m_60x5 for f in frames]).astype(np.float32)
        mis = np.stack([f.micro_stamps_60x5 for f in frames]).astype(np.float32)
        ma = np.stack([f.macro_1h_32x5 for f in frames]).astype(np.float32)
        mas = np.stack([f.macro_stamps_32x5 for f in frames]).astype(np.float32)
        close = np.stack([f.medium_close_64 for f in frames]).astype(np.float32)
        risk = np.stack([f.ordered4h30 for f in frames]).astype(np.float32)
        op = self.operator.encode_batch(mi, mis, ma, mas)
        med = self.medium.encode_batch(close)
        return TypedSensoryBatchR10(op, med, risk, tuple(f.symbol for f in frames), np.asarray([f.decision_time_ms for f in frames], dtype=np.int64))
