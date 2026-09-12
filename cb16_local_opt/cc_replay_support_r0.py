from __future__ import annotations
from dataclasses import dataclass
import math
from collections import Counter
from typing import Sequence

@dataclass(frozen=True)
class ReplaySupportSample:
    log_mu: float
    log_pi: float
    generation: str
    source: str

@dataclass(frozen=True)
class ReplaySupportHealth:
    ratios: tuple[float, ...]
    clipped_ratios: tuple[float, ...]
    clipping_fraction: float
    nonfinite_count: int
    low_support_frequency: float
    effective_support: float
    generation_contribution: dict[str, int]
    source_contribution: dict[str, int]


def support_health(samples: Sequence[ReplaySupportSample], *, rho_clip: float = 1.0, low_support_ratio: float = 0.05) -> ReplaySupportHealth:
    if not samples: raise ValueError("samples required")
    ratios=[]; nonfinite=0
    for s in samples:
        if not math.isfinite(s.log_mu) or not math.isfinite(s.log_pi):
            nonfinite += 1; ratios.append(float("nan")); continue
        try: r=math.exp(s.log_pi-s.log_mu)
        except OverflowError: r=float("inf")
        if not math.isfinite(r): nonfinite += 1
        ratios.append(r)
    finite=[r for r in ratios if math.isfinite(r)]
    clipped=[min(r,rho_clip) if math.isfinite(r) else 0.0 for r in ratios]
    clipping=sum(1 for r in finite if r>rho_clip)/len(samples)
    low=sum(1 for r in finite if r<low_support_ratio)/len(samples)
    sw=sum(clipped); sw2=sum(r*r for r in clipped)
    ess=(sw*sw/sw2) if sw2 else 0.0
    return ReplaySupportHealth(tuple(ratios), tuple(clipped), clipping, nonfinite, low, ess,
                               dict(Counter(s.generation for s in samples)), dict(Counter(s.source for s in samples)))
