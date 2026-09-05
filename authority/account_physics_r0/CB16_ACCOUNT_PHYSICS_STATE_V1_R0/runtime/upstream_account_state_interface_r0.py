from __future__ import annotations
import math
import numpy as np
FIELD_ORDER = [
    "signed_exposure_fraction_of_contract_cap",
    "entry_price_log_ratio",
    "remaining_holding_fraction",
    "current_drawdown_fraction",
    "risk_budget_remaining_fraction",
    "margin_utilization_fraction",
]

def encode_account(raw: dict) -> dict:
    eq=float(raw["equity"]); peak=float(raw["peak_equity"]); signed_notional=float(raw["signed_position_notional"])
    max_lev=float(raw["max_gross_leverage_contract"]); cur=float(raw["current_price"]); entry=raw.get("entry_price")
    held=float(raw.get("holding_bars",0.0)); max_hold=float(raw["max_holding_bars_contract"])
    risk_rem=float(raw["risk_budget_remaining"]); risk_cap=float(raw["risk_budget_capacity"])
    margin_used=float(raw["margin_used"]); margin_cap=float(raw["margin_capacity"])
    base_valid = eq>0 and peak>0 and max_lev>0 and cur>0 and max_hold>0 and risk_cap>0 and margin_cap>0
    signed_exp=(signed_notional/eq)/max_lev if base_valid else 0.0
    has_position=abs(signed_notional)>0.0
    entry_valid=bool(base_valid and has_position and entry is not None and float(entry)>0)
    entry_log=math.log(cur/float(entry)) if entry_valid else 0.0
    hold_valid=bool(base_valid and has_position)
    remaining=(max_hold-held)/max_hold if hold_valid else 1.0
    drawdown=(peak-eq)/peak if base_valid else 0.0
    risk_frac=risk_rem/risk_cap if base_valid else 0.0
    margin_util=margin_used/margin_cap if base_valid else 0.0
    payload=np.asarray([signed_exp,entry_log,remaining,drawdown,risk_frac,margin_util],dtype=np.float32)
    flags=[bool(base_valid),bool(base_valid),entry_valid,hold_valid,bool(base_valid),bool(base_valid),bool(base_valid)]
    return {"protocol_version":"AccountStatePacketV1","account_state_type":"SELF_ACCOUNT_OBSERVATION","validity_flags":flags,"payload_dim":6,"payload":[float(x) for x in payload]}
