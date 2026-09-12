from cb16_local_opt.cc_economic_capital_r0 import CapitalLedger, CapitalFlow

def test_restart_new_money_is_explicit_and_not_free():
    l=CapitalLedger(100,(CapitalFlow('a','EXTERNAL_WITHDRAWAL',20,'w'),CapitalFlow('a2','NEW_ACCOUNT_ALLOCATION',50,'n'),CapitalFlow('a2','LINEAGE_RESTART',0,'r'))).validate()
    assert l.gross_capital_denominator==150 and l.external_withdrawals==20 and len(l.denominator_id)==64
