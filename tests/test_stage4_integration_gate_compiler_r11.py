from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cb16_local_opt.stage4_integration_gate_compiler_r11 import (
    EXPECTED_GUARDS,
    EXPECTED_TASK_BRANCHES,
    FORBIDDEN_FINAL_VERDICT,
    FREEZE_PATH,
    INTEGRATION_SEED_SHA,
    READY_VERDICT,
    SCIENTIFIC_STATUS,
    IntegrationGateFailure,
    ReceiptInput,
    compile_integration_receipts,
    validate_final_cutover_report,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/stage4_fixtures/valid_integration_bundle_v1.json"
RECEIPT_SCHEMA = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_INTEGRATION_RECEIPT_SCHEMA_V1.json"
FINAL_SCHEMA = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_FINAL_CUTOVER_REPORT_SCHEMA_V1.json"


def _fixture():
    return json.loads(FIXTURE.read_text())


def _receipt(row):
    inv = {"synthetic_fixture": True}
    if row["task_id"] == "INTG":
        inv["integrated_runtime_qualification_claimed"] = False
    return {
        "schema": "CB16_R11_STAGE4_INTEGRATION_RECEIPT_V1",
        "task_id": row["task_id"], "status": "PASS",
        "integration_seed_sha": INTEGRATION_SEED_SHA,
        "task_head_sha": row["task_head_sha"], "branch": row["branch"],
        "owned_paths": list(row["touched_paths"]), "touched_paths": list(row["touched_paths"]),
        "tests": {"commands": ["python -m pytest -q"], "passed": True,
                  "evidence": ["GitHub Actions run id=9001", "conclusion=success"]},
        "semantic_guards": copy.deepcopy(EXPECTED_GUARDS),
        "scientific_status": SCIENTIFIC_STATUS,
        "correctness_invariants": inv,
        "known_limitations": ["synthetic fixture only"],
        "final_consolidation_notes": ["final adjudicator binds consolidated runtime"],
    }


def _bundle():
    f = _fixture()
    rows = []
    for x in f["tasks"]:
        rows.append({
            "task_id": x["task_id"],
            "path": f"authority/rearchitecture_r11/stage4_integration_receipts/{x['task_id']}_RECEIPT_V1.json",
            "receipt_commit_sha": x["receipt_commit_sha"],
            "branch_tip_sha": x["receipt_commit_sha"],
            "document": _receipt(x),
        })
    return {**f, "receipts": rows}


class Repo:
    def __init__(self, b):
        self.freeze = b["semantic_freeze_blob_sha"]
        self.commits = {INTEGRATION_SEED_SHA}
        self.parents, self.changed, self.commit_paths = {}, {}, {}
        self.status, self.path_changes, self.merges, self.freeze_override = {}, {}, {}, {}
        self.seed_existing = {FREEZE_PATH, "cb16_local_opt/integration_adapters_r11.py"}
        for x in b["receipts"]:
            h, rc, p = x["document"]["task_head_sha"], x["receipt_commit_sha"], x["path"]
            self.commits |= {h, rc}; self.parents[h] = (INTEGRATION_SEED_SHA,); self.parents[rc] = (h,)
            self.changed[(INTEGRATION_SEED_SHA, h)] = set(x["document"]["touched_paths"])
            self.commit_paths[rc] = {p}; self.status[(rc, p)] = "A"; self.path_changes[(h, rc, p)] = (rc,)
            self.merges[(INTEGRATION_SEED_SHA, h)] = ()

    def object_is_commit(self, s): return s in self.commits
    def is_ancestor(self, a, d):
        todo, seen = [d], set()
        while todo:
            c = todo.pop()
            if c == a: return True
            if c in seen: continue
            seen.add(c); todo.extend(self.parents.get(c, ()))
        return False
    def merge_commits_between(self, b, h): return self.merges.get((b, h), ())
    def changed_paths(self, b, h): return set(self.changed.get((b, h), set()))
    def path_exists_at(self, c, p): return p in self.seed_existing if c == INTEGRATION_SEED_SHA else p in self.seed_existing or p in self.changed.get((INTEGRATION_SEED_SHA, c), set())
    def blob_sha(self, c, p): return self.freeze_override.get(c, self.freeze) if p == FREEZE_PATH else f"blob:{c}:{p}"
    def commit_parents(self, c): return self.parents.get(c, ())
    def commit_changed_paths(self, c): return set(self.commit_paths.get(c, set()))
    def path_status_in_commit(self, c, p): return self.status.get((c, p))
    def path_change_commits(self, b, h, p): return self.path_changes.get((b, h, p), ())


def _gatework(b):
    return {"scientific_status": SCIENTIFIC_STATUS, "semantic_freeze_blob_sha": b["semantic_freeze_blob_sha"],
            "integration_tasks": [{"id": t, "branch": br, "exclusive_existing_file_ownership": (["cb16_local_opt/integration_adapters_r11.py"] if t == "INTB" else [])} for t, br in EXPECTED_TASK_BRANCHES.items()]}


def _inputs(b):
    return [ReceiptInput(x["task_id"], x["path"], x["document"], x["receipt_commit_sha"], x["branch_tip_sha"]) for x in b["receipts"]]


def _compile(b=None, r=None):
    b = _bundle() if b is None else b; r = Repo(b) if r is None else r
    return compile_integration_receipts(gatework=_gatework(b), receipt_schema=json.loads(RECEIPT_SCHEMA.read_text()), receipts=_inputs(b), repo=r)


def _row(b, task): return next(x for x in b["receipts"] if x["task_id"] == task)

def _fails(code, fn):
    with pytest.raises(IntegrationGateFailure) as e: fn()
    assert e.value.code == code


def test_positive_is_preconsolidation_only():
    out = _compile(); assert out["verdict"] == READY_VERDICT; assert out["final_cutover_verdict_emitted"] is False
    assert out["reserved_final_cutover_verdict"] == FORBIDDEN_FINAL_VERDICT


def test_count_and_duplicate_fail_closed():
    b = _bundle(); b["receipts"].pop(); _fails("INTEGRATION_RECEIPT_COUNT_MISMATCH", lambda: _compile(b))
    b = _bundle(); b["receipts"][-1]["task_id"] = "INTA"; _fails("INTEGRATION_RECEIPT_DUPLICATE_TASK", lambda: _compile(b))


@pytest.mark.parametrize("task,field,value,code", [
    ("INTC", "branch", "ai/wrong", "INTEGRATION_RECEIPT_BRANCH_MISMATCH"),
    ("INTD", "integration_seed_sha", "f"*40, "INTEGRATION_RECEIPT_SEED_MISMATCH"),
    ("INTE", "status", "FAIL", "INTEGRATION_RECEIPT_NOT_PASS"),
])
def test_receipt_identity_failures(task, field, value, code):
    b = _bundle(); _row(b, task)["document"][field] = value; _fails(code, lambda: _compile(b))


def test_ancestry_and_merge_fail_closed():
    b = _bundle(); r = Repo(b); h = _row(b,"INTE")["document"]["task_head_sha"]; r.parents[h] = (); _fails("INTEGRATION_TASK_HEAD_NOT_DESCENDANT_OF_SEED", lambda: _compile(b,r))
    b = _bundle(); r = Repo(b); h = _row(b,"INTF")["document"]["task_head_sha"]; r.merges[(INTEGRATION_SEED_SHA,h)] = ("e"*40,); _fails("INTEGRATION_TASK_HEAD_HAS_MERGE", lambda: _compile(b,r))


def test_two_commit_protocol_failures():
    b = _bundle(); x = _row(b,"INTA"); x["receipt_commit_sha"] = x["document"]["task_head_sha"]; x["branch_tip_sha"] = x["receipt_commit_sha"]; r=Repo(b); r.parents[x["document"]["task_head_sha"]]=(INTEGRATION_SEED_SHA,); _fails("INTEGRATION_RECEIPT_NOT_DISTINCT_COMMIT", lambda:_compile(b,r))
    b = _bundle(); r=Repo(b); x=_row(b,"INTB"); r.parents[x["receipt_commit_sha"]] = ("d"*40,); _fails("INTEGRATION_RECEIPT_NOT_DIRECTLY_AFTER_TASK_HEAD", lambda:_compile(b,r))
    b = _bundle(); r=Repo(b); x=_row(b,"INTC"); r.commit_paths[x["receipt_commit_sha"]].add("oops.py"); _fails("INTEGRATION_RECEIPT_COMMIT_MUTATES_OTHER_PATHS", lambda:_compile(b,r))
    b = _bundle(); r=Repo(b); x=_row(b,"INTD"); r.status[(x["receipt_commit_sha"],x["path"])]="M"; _fails("INTEGRATION_RECEIPT_NOT_ADDED_EXACTLY_ONCE", lambda:_compile(b,r))
    b = _bundle(); r=Repo(b); x=_row(b,"INTE"); r.path_changes[(x["document"]["task_head_sha"],x["receipt_commit_sha"],x["path"])]=(x["receipt_commit_sha"],"c"*40); _fails("INTEGRATION_RECEIPT_MUTATED_AFTER_CREATION", lambda:_compile(b,r))


def test_path_and_ownership_failures():
    b=_bundle(); r=Repo(b); x=_row(b,"INTF"); r.changed[(INTEGRATION_SEED_SHA,x["document"]["task_head_sha"])].add("hidden.py"); _fails("INTEGRATION_TOUCHED_PATH_MISMATCH",lambda:_compile(b,r))
    b=_bundle(); _row(b,"INTG")["document"]["owned_paths"]=[]; _fails("INTEGRATION_TOUCHED_OUTSIDE_OWNED_PATHS",lambda:_compile(b))
    b=_bundle(); x=_row(b,"INTA"); x["document"]["touched_paths"]=x["document"]["owned_paths"]=["cb16_local_opt/integration_adapters_r11.py"]; r=Repo(b); _fails("INTEGRATION_EXISTING_FILE_OWNERSHIP_VIOLATION",lambda:_compile(b,r))
    assert "cb16_local_opt/integration_adapters_r11.py" in _compile()["accepted_tasks"]["INTB"]["touched_paths"]
    b=_bundle(); x=_row(b,"INTA"); x["document"]["touched_paths"]=x["document"]["owned_paths"]=["tests/final_holdout_2025-09.txt"]; r=Repo(b); _fails("INTEGRATION_FORBIDDEN_PATH_CHANGED",lambda:_compile(b,r))


def test_freeze_science_guards_and_ci_fail_closed():
    b=_bundle(); r=Repo(b); h=_row(b,"INTB")["document"]["task_head_sha"]; r.freeze_override[h]="0"*40; _fails("INTEGRATION_SEMANTIC_FREEZE_MISMATCH",lambda:_compile(b,r))
    b=_bundle(); _row(b,"INTC")["document"]["scientific_status"]="NEW"; _fails("INTEGRATION_SCHEMA_INVALID",lambda:_compile(b))
    b=_bundle(); _row(b,"INTD")["document"]["semantic_guards"]["new_scientific_verdict"]=True; _fails("INTEGRATION_SCHEMA_INVALID",lambda:_compile(b))
    b=_bundle(); _row(b,"INTE")["document"]["tests"]["passed"]=False; _fails("INTEGRATION_TESTS_NOT_PASS",lambda:_compile(b))
    b=_bundle(); _row(b,"INTF")["document"]["tests"]["evidence"]=["local only"]; _fails("INTEGRATION_MACHINE_EVIDENCE_MISSING",lambda:_compile(b))
    b=_bundle(); _row(b,"INTG")["document"]["semantic_guards"]["sibling_integration_dependency_used"]=True; _fails("INTEGRATION_SCHEMA_INVALID",lambda:_compile(b))


def test_intg_overclaim_and_cross_task_overlap_fail():
    b=_bundle(); _row(b,"INTG")["document"]["correctness_invariants"]["integrated_runtime_qualification_claimed"]=True; _fails("INTEGRATION_INTG_RUNTIME_QUALIFICATION_OVERCLAIM",lambda:_compile(b))
    b=_bundle(); a=_row(b,"INTA"); c=_row(b,"INTC"); c["document"]["touched_paths"]=c["document"]["owned_paths"]=list(a["document"]["touched_paths"]); r=Repo(b); _fails("INTEGRATION_TOUCHED_PATH_OVERLAP",lambda:_compile(b,r))


def _final_report(pass_gates=True):
    gates={k:pass_gates for k in [
        "all_accepted_integration_receipts","exact_seed_ancestry","controlled_consolidation_only","semantic_freeze_exact","scientific_status_unchanged","s4a_writer_registry_rebuilt_and_reaudited_at_final_head","unknown_authority_zero","all_stage4_integration_correctness_tests_pass","real_s4h_h01_h20_against_consolidated_runtime","exactly_one_live_canonical_authority_writer","stale_writer_cannot_mutate","legacy_writer_cannot_mutate","permission_cannot_be_bypassed","replay_cannot_become_new_evidence","short_canonical_machine_startup_steady_drain_shutdown_pass","fp32_canonical","amp_false","final_holdout_2025_09_untouched","no_fresh_market_data","no_new_scientific_verdict"]}; gates["long_endurance_not_required"]=True
    accepted={t:{"task_head_sha":f"{i:040x}","receipt_commit_sha":f"{100+i:040x}","branch":br} for i,(t,br) in enumerate(EXPECTED_TASK_BRANCHES.items(),1)}
    return {"schema":"CB16_R11_STAGE4_FINAL_CUTOVER_REPORT_V1","verdict":FORBIDDEN_FINAL_VERDICT,"integration_seed_sha":INTEGRATION_SEED_SHA,"consolidation_head_sha":"f"*40,"scientific_status":SCIENTIFIC_STATUS,"accepted_integration_receipts":accepted,"qualification_gates":gates,"writer_registry":{"rebuilt_at_head":True,"unknown_authority_count":0,"report_identity":"registry@final"},"hostile_matrix":{"report_schema":"CB16_R11_STAGE4_HOSTILE_CUTOVER_REPORT_V1","integrated_runtime_qualification_claimed":True,"h01_h20_all_passed":True},"canonical_machine_smoke":{"runner_labels":["self-hosted","shanxi","cb16-wss-qualification"],"python_resolution_status":"READY","python_resolution_mode":"VERIFIED_CANONICAL_R104_VENV_REUSE","numeric_mode":"FP32","amp":False,"lifecycle_passed":True,"long_endurance_run":False},"semantic_guards":{k:v for k,v in EXPECTED_GUARDS.items() if k in {"semantic_freeze_unchanged","final_holdout_untouched","fresh_market_data_downloaded","historical_market_data_mutated","new_scientific_verdict","scientific_semantics_changed","replay_reinterpreted_as_new_evidence"}},"reserved_verdict_issuer":"FINAL_CONSOLIDATED_ADJUDICATOR_ONLY"}


def test_final_contract_reserved_verdict_requires_every_gate():
    schema=json.loads(FINAL_SCHEMA.read_text()); validate_final_cutover_report(_final_report(),schema)
    x=_final_report(); x["qualification_gates"]["real_s4h_h01_h20_against_consolidated_runtime"]=False; _fails("FINAL_REPORT_GATE_NOT_PASS",lambda:validate_final_cutover_report(x,schema))
    x=_final_report(); x["writer_registry"]["unknown_authority_count"]=1; _fails("FINAL_REPORT_UNKNOWN_AUTHORITY_NONZERO",lambda:validate_final_cutover_report(x,schema))
    x=_final_report(); x["canonical_machine_smoke"]["runner_labels"]=["self-hosted","shanxi","wrong"]; _fails("INTEGRATION_SCHEMA_INVALID",lambda:validate_final_cutover_report(x,schema))


def test_fail_report_does_not_claim_positive_gates():
    schema=json.loads(FINAL_SCHEMA.read_text()); x=_final_report(False); x["verdict"]="FAIL_CLOSED"; validate_final_cutover_report(x,schema)
