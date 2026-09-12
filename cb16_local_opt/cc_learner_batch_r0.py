from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class LearnerBatch:
    sequence_ids: tuple[str,...]; observations:torch.Tensor; critic_observations:torch.Tensor
    actions:torch.Tensor; rewards:torch.Tensor; discounts:torch.Tensor; log_mu:torch.Tensor; mask:torch.Tensor

def build_batch(records:list[dict])->LearnerBatch:
    records=sorted(records,key=lambda r:(r["sequence_id"],r["step"]))
    if not records: raise ValueError("EMPTY_BATCH")
    ids=tuple(dict.fromkeys(r["sequence_id"] for r in records))
    def t(k,dtype=torch.float32): return torch.tensor([r[k] for r in records],dtype=dtype)
    return LearnerBatch(ids,t("observation"),t("critic_observation"),t("action",torch.long),t("reward"),t("discount"),t("log_mu"),t("mask"))
