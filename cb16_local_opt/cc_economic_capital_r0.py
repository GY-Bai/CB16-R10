from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json

@dataclass(frozen=True)
class CapitalFlow:
    account_lineage_id: str
    kind: str
    amount: float
    flow_id: str

@dataclass(frozen=True)
class CapitalLedger:
    initial_allocated_capital: float
    flows: tuple[CapitalFlow, ...] = ()

    def validate(self) -> "CapitalLedger":
        if self.initial_allocated_capital <= 0: raise ValueError("initial capital must be positive")
        valid={"EXTERNAL_DEPOSIT","EXTERNAL_WITHDRAWAL","NEW_ACCOUNT_ALLOCATION","LINEAGE_RESTART"}
        seen=set()
        for f in self.flows:
            if f.kind not in valid or f.amount < 0 or not f.flow_id: raise ValueError("invalid capital flow")
            if f.flow_id in seen: raise ValueError("duplicate capital flow")
            seen.add(f.flow_id)
        return self

    @property
    def gross_capital_denominator(self) -> float:
        self.validate()
        return self.initial_allocated_capital + sum(f.amount for f in self.flows if f.kind in {"EXTERNAL_DEPOSIT","NEW_ACCOUNT_ALLOCATION"})

    @property
    def external_withdrawals(self) -> float:
        return sum(f.amount for f in self.flows if f.kind=="EXTERNAL_WITHDRAWAL")

    @property
    def denominator_id(self) -> str:
        payload={"initial":self.initial_allocated_capital,"flows":[f.__dict__ for f in self.flows],"gross":self.gross_capital_denominator}
        return sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
