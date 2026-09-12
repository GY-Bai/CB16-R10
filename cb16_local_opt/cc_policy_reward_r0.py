from __future__ import annotations
import math

def arithmetic_equity_reward(equity_t:float,equity_tp1:float,e_ref:float)->float:
    a,b,r=map(float,(equity_t,equity_tp1,e_ref))
    if not all(map(math.isfinite,(a,b,r))) or r<=0: raise ValueError("BAD_EQUITY_REWARD_INPUT")
    return (b-a)/r

def path_rewards(equities:list[float]|tuple[float,...],e_ref:float)->list[float]:
    if len(equities)<2: raise ValueError("PATH_TOO_SHORT")
    return [arithmetic_equity_reward(a,b,e_ref) for a,b in zip(equities,equities[1:])]
