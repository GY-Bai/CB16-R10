from cb16_local_opt.cc_experience_store_r0 import RawFactStore

def test_all_protected_failure_classes_roundtrip(tmp_path):
    kinds=['liquidation','negative_equity_liability','reversal_second_leg_failure','rejected_action_with_market_loss','pending_settlement','economic_terminal','data_end_truncation']
    s=RawFactStore(tmp_path)
    for k in kinds: s.put(k,{'failure_classification':k,'post_equity':-10 if k=='negative_equity_liability' else 1})
    assert [s.get(k)['failure_classification'] for k in kinds]==kinds and s.count()==len(kinds)
