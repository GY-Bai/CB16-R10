import json
from pathlib import Path


def test_baseline_binds_exact_base_wire_and_firewall():
    p = Path(__file__).parents[1] / "authority/rearchitecture_r11/CB16_R11_CC_THREAD_C_BASELINE_V1.json"
    x = json.loads(p.read_text())
    assert x["implementation_base_sha"] == "89d62bf966f476e598f0e2f5c5e8e03c15a8db51"
    assert x["sibling_cc_dependencies"] == []
    assert x["planning_docs_source"]["blob_sha"] == "322d629fc1eaa5a3ba64c1d408253997b4967285"
    assert x["frozen_inputs"]["r1_science_identity"]["blob_sha"] == "a0433af7ed0e3ab22772a873e60dbc811136e6a9"
    assert x["frozen_inputs"]["raw_store_reference"]["blob_sha"] == "bb3b72950cb39bdc64de490505490aee4404b6a5"
    assert x["frozen_inputs"]["demonstration_store_reference"]["blob_sha"] == "581518d6cdcc1466a8bd7d6553730090262a693c"
    assert x["wire_contracts"] == {
        "W-02": "CCEnvironmentTransitionV1",
        "W-03": "CCExperienceSequenceV1",
        "W-05": "CCEconomicResultV1",
    }
    assert x["formal_owner_decisions"]["conflicting_buy_hold_vs_flat_master_ranking"] == "UNRESOLVED_OWNER_DECISION"
    assert x["firewall"] == {"final_holdout_opened": False, "fresh_data_downloaded": False}
