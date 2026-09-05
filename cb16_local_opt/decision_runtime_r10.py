from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import torch

from .binance_archive_input_r10 import SensoryDecisionFrameR10
from .frozen_sensory_stack_r10 import FrozenSensoryStackR10, TypedSensoryBatchR10
from .typed_central_brain_r10 import TypedCentralBrainR10, build_g0_brain_r10

@dataclass(frozen=True)
class DecisionBatchR10:
    direction: np.ndarray
    requested_risk: np.ndarray
    direction_probs: np.ndarray
    operator48: np.ndarray
    medium48: np.ndarray
    ordered4h30_shadow: np.ndarray

class CB16DecisionRuntimeR10:
    def __init__(self, package_root: str|Path, *, device: str="cuda", brain_tier: str="TIER_1", brain_seed: int=0, brain_state_dict: str|Path|None=None, verify_assets: bool=True):
        self.device=torch.device(device)
        self.sensory=FrozenSensoryStackR10(package_root,device=device,verify_hashes=verify_assets)
        self.brain=build_g0_brain_r10(brain_tier,seed=brain_seed,device=device)
        if brain_state_dict:
            self.brain.load_state_dict(torch.load(brain_state_dict,map_location=self.device),strict=True)
        self.brain.eval()

    def decide(self, frames: Sequence[SensoryDecisionFrameR10], account_state6: np.ndarray, account_to_frame: np.ndarray|None=None) -> DecisionBatchR10:
        s=self.sensory.encode_frames(frames)
        A=np.asarray(account_state6,dtype=np.float32)
        if A.ndim!=2 or A.shape[1]!=6: raise ValueError("account_state6 must be [N,6]")
        if account_to_frame is None:
            if len(A)!=len(frames): raise ValueError("account_to_frame required when N != M")
            idx=np.arange(len(A),dtype=np.int64)
        else:
            idx=np.asarray(account_to_frame,dtype=np.int64)
        if idx.shape!=(len(A),) or idx.min(initial=0)<0 or idx.max(initial=-1)>=len(frames): raise ValueError("bad account_to_frame")
        op=torch.from_numpy(s.operator48[idx]).to(self.device); med=torch.from_numpy(s.medium48[idx]).to(self.device); ac=torch.from_numpy(A).to(self.device)
        with torch.inference_mode():
            out=self.brain(op,med,ac); act=self.brain.compose_action(out)
        return DecisionBatchR10(
            direction=act["direction"].cpu().numpy(), requested_risk=act["requested_risk"].cpu().numpy().astype(np.float32),
            direction_probs=out["direction_probs"].cpu().numpy().astype(np.float32), operator48=s.operator48, medium48=s.medium48,
            ordered4h30_shadow=s.ordered4h30,
        )
