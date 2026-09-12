import json, pathlib
P=pathlib.Path(__file__).parents[1]/"authority/rearchitecture_r11/CB16_R11_CC_THREAD_B_BASELINE_V1.json"
def test_baseline_binding_and_actor_audit():
 d=json.loads(P.read_text()); assert d["base_sha"]=="89d62bf966f476e598f0e2f5c5e8e03c15a8db51"
 assert len(d["dependencies"])==8 and all(len(x["git_blob_sha"])==40 for x in d["dependencies"])
 assert all(d["actor_audit"][f"BC-0{i}"] .startswith("PRESENT") for i in range(34,38))
 assert d["sibling_cc_dependencies"]==[] and d["final_holdout"]=="SEALED" and d["fresh_market_data"]=="FORBIDDEN"
