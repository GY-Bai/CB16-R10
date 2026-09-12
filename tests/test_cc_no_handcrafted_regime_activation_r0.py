import pathlib
def test_no_handcrafted_activation_engine():
 root=pathlib.Path(__file__).parents[1]/'cb16_local_opt'; forbidden=('manual_regime','calendar_switch','regime_switch_table','cycle_activation_rule')
 text='\n'.join(p.read_text() for p in root.glob('cc_*.py')); assert not any(x in text for x in forbidden)
