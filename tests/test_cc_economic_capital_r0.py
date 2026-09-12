from cb16_local_opt.cc_economic_capital_r0 import *
def test_restart_and_deposit_are_not_free_denominator_replenishment():
    s=summarize_capital({"a":100},[CapitalEvent("d","a","deposit",20),CapitalEvent("w","a","withdrawal",10),CapitalEvent("r","a2","lineage_restart",50)]); assert s.initial_allocated_capital==100 and s.external_withdrawals==10 and s.gross_contributed_denominator==170
