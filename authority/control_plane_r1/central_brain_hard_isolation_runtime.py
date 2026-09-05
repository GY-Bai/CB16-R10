from __future__ import annotations
import numpy as np

SCHEMA="CB16_HARD_ISOLATED_TYPED_CONSUMER_RUNTIME_V1"

def consume(operator, state, account, account_to_market_index):
    """Typed transport consumer. Gathers market rows internally; never concatenates lanes."""
    O=np.asarray(operator,dtype=np.float32); S=np.asarray(state,dtype=np.float32)
    A=np.asarray(account,dtype=np.float32); I=np.asarray(account_to_market_index,dtype=np.int64)
    if O.ndim!=2 or O.shape[1]!=48: raise ValueError("Operator must be [M,48]")
    if S.ndim!=2 or S.shape[1]!=48 or S.shape[0]!=O.shape[0]: raise ValueError("State must be [M,48] aligned with Operator")
    if A.ndim!=2 or A.shape[1]!=6: raise ValueError("Account must be [N,6]")
    if I.ndim!=1 or len(I)!=len(A): raise ValueError("index must be [N]")
    if len(I) and (I.min()<0 or I.max()>=len(O)): raise IndexError("account_to_market_index out of bounds")
    return {
      "OperatorPacket48": np.ascontiguousarray(O[I]),
      "StateContextPacket48": np.ascontiguousarray(S[I]),
      "AccountStatePacket6": np.ascontiguousarray(A),
    }
