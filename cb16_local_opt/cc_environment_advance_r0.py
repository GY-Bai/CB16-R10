from __future__ import annotations
from dataclasses import dataclass, replace
from .account_economics_r0 import AccountEconomicsStateR0
from .cc_runtime_boundary_r0 import CONTINUE
@dataclass(frozen=True)
class CCEnvironmentIntervalR0:
    mark_price_after:float
    funding_cashflow:float=0.0
    mechanical_fee:float=0.0
    liability_delta:float=0.0
    external_capital_flow:float=0.0
    external_capital_flow_ref:str|None=None
    force_liquidate:bool=False
    boundary_type:str=CONTINUE
    def validate(self):
        import math
        vals=(self.mark_price_after,self.funding_cashflow,self.mechanical_fee,self.liability_delta,self.external_capital_flow)
        if any(not math.isfinite(float(v)) for v in vals) or self.mark_price_after<=0 or self.mechanical_fee<0: raise RuntimeError("CCENV_INTERVAL_INVALID")
        if self.external_capital_flow and not self.external_capital_flow_ref: raise RuntimeError("CCENV_CAPITAL_FLOW_REF_REQUIRED")
def advance_environment_r0(state:AccountEconomicsStateR0,interval:CCEnvironmentIntervalR0)->AccountEconomicsStateR0:
    state.validate(); interval.validate(); marked=replace(state,mark_price=interval.mark_price_after); marked.validate()
    if interval.force_liquidate and marked.position_quantity!=0.0:
        realized=marked.unrealized_pnl
        marked=replace(marked,cash=marked.cash+marked.margin_collateral+realized,position_quantity=0.0,position_cost_basis=0.0,realized_pnl_cumulative=marked.realized_pnl_cumulative+realized,margin_collateral=0.0)
    liabilities=marked.liabilities+interval.liability_delta
    if liabilities<0: raise RuntimeError("CCENV_LIABILITY_UNDERFLOW")
    out=replace(marked,cash=marked.cash+interval.funding_cashflow-interval.mechanical_fee+interval.external_capital_flow,fees_cumulative=marked.fees_cumulative+interval.mechanical_fee,funding_cumulative=marked.funding_cumulative+interval.funding_cashflow,liabilities=liabilities,external_capital_flows_cumulative=marked.external_capital_flows_cumulative+interval.external_capital_flow)
    out.validate(); return out
