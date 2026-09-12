import pytest
from dataclasses import replace
from tests.cc_thread_c_fixtures import tx

def test_self_auditable_transition_has_policy_rng_market_and_signed_economics():
    t=tx(0)
    assert t.log_mu<0 and t.rng_stream_id and t.market_lineage_id and t.environment.post_account_truth_hash

def test_nonfinite_log_mu_fails_closed():
    with pytest.raises(ValueError): replace(tx(),log_mu=float('nan')).validate()
