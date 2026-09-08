from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cb16_local_opt.stage4_gate_compiler_r11 import (
    EXPECTED_GATEWORK_BASE,
    FREEZE_PATH,
    GateFailure,
    ReceiptInput,
    READY_VERDICT,
    compile_wave1_receipts,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/stage4_fixtures/S4I_SYNTHETIC_WAVE1_RECEIPTS_V1.json"
SCIENCE = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"

BRANCHES = {
    "S4A": "ai/r11-stage4-task-a-authority-inventory-r0",
    "S4B": "ai/r11-stage4-task-b-canonical-entrypoint-r0",
    "S4C": "ai/r11-stage4-task-c-authority-adoption-r0",
    "S4D": "ai/r11-stage4-task-d-singleton-fencing-r0",
    "S4E": "ai/r11-stage4-task-e-permission-closure-r0",
    "S4F": "ai/r11-stage4-task-f-state-roots-r0",
    "S4G": "ai/r11-stage4-task-g-legacy-retirement-r0",
    "S4H": "ai/r11-stage4-task-h-hostile-cutover-r0",
    "S4I": "ai/r11-stage4-task-i-qualification-gates-r0",
}

GUARDS = {
    "semantic_freeze_unchanged": True,
    "final_holdout_untouched": True,
    "fresh_market_data_downloaded": False,
    "historical_market_data_mutated": False,
    "new_scientific_verdict": False,
    "scientific_semantics_changed": False,
    "replay_reinterpreted_as_new_evidence": False,
    "sibling_dependency_used": False,
}


def manifest():
    tasks = []
    for task_id, branch in BRANCHES.items():
        tasks.append({
            "id": task_id,
            "branch": branch,
            "exclusive_existing_file_ownership": ["cb16_local_opt/integration_adapters_r11.py"] if task_id == "S4E" else [],
        })
    return {
        "wave1_tasks": tasks,
        "upstream": {"semantic_freeze_blob_sha": FREEZE_BLOB},
        "scientific_authority": {"scientific_status": SCIENCE},
    }


def schema():
    guard_props = {key: {"const": value} for key, value in GUARDS.items()}
    return {
        "type": "object",
        "required": ["schema", "task_id", "status", "gatework_base_sha", "task_head_sha", "branch", "owned_paths", "touched_paths", "tests", "semantic_guards", "scientific_status", "integration_notes"],
        "properties": {
            "schema": {"const": "CB16_R11_STAGE4_TASK_RECEIPT_V1"},
            "task_id": {"enum": list(BRANCHES)},
            "status": {"enum": ["PASS", "FAIL", "BLOCKED", "NOT_RUN"]},
            "gatework_base_sha": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
            "task_head_sha": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
            "branch": {"type": "string", "minLength": 1},
            "owned_paths": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "touched_paths": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "tests": {
                "type": "object",
                "required": ["commands", "passed"],
                "properties": {
                    "commands": {"type": "array", "items": {"type": "string"}},
                    "passed": {"type": "boolean"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
            "semantic_guards": {
                "type": "object",
                "required": list(GUARDS),
                "properties": guard_props,
            },
            "scientific_status": {"const": SCIENCE},
            "integration_notes": {"type": "array", "items": {"type": "string"}},
        },
    }


class FakeRepo:
    def __init__(self):
        self.commits = {EXPECTED_GATEWORK_BASE, "HEAD"}
        self.ancestors = set()
        self.merges = {}
        self.changed = {}
        self.existing = set()
        self.blobs = {("HEAD", FREEZE_PATH): FREEZE_BLOB}
        self.additions = {}

    def object_is_commit(self, sha): return sha in self.commits
    def is_ancestor(self, ancestor, descendant): return ancestor == descendant or (ancestor, descendant) in self.ancestors
    def merge_commits_between(self, base, head): return self.merges.get((base, head), ())
    def changed_paths(self, base, head): return set(self.changed.get((base, head), set()))
    def path_exists_at(self, commit, path): return (commit, path) in self.existing
    def blob_sha(self, commit, path): return self.blobs[(commit, path)]
    def receipt_addition_commits(self, path): return self.additions.get(path, ())
    def path_clean_at_head(self, path): return True


def valid_inputs():
    fixture = json.loads(FIXTURE.read_text())
    repo = FakeRepo()
    receipts = []
    for index, row in enumerate(fixture["receipts"], start=1):
        task_id = row["task_id"]
        head = f"{index:040x}"
        receipt_commit = f"{index + 100:040x}"
        path = f"authority/rearchitecture_r11/stage4_receipts/{task_id}_RECEIPT_V1.json"
        impl_path = f"cb16_local_opt/stage4_{task_id.lower()}_synthetic_r11.py"
        owned = [impl_path]
        touched = [impl_path]
        repo.commits.update({head, receipt_commit})
        repo.ancestors.update({(EXPECTED_GATEWORK_BASE, head), (head, receipt_commit), (receipt_commit, "HEAD")})
        repo.changed[(EXPECTED_GATEWORK_BASE, head)] = {impl_path}
        repo.additions[path] = (receipt_commit,)
        repo.blobs[(receipt_commit, path)] = f"blob:{task_id}"
        repo.blobs[("HEAD", path)] = f"blob:{task_id}"
        receipt = {
            "schema": "CB16_R11_STAGE4_TASK_RECEIPT_V1",
            "task_id": task_id,
            "status": "PASS",
            "gatework_base_sha": EXPECTED_GATEWORK_BASE,
            "task_head_sha": head,
            "branch": row["branch"],
            "owned_paths": owned,
            "touched_paths": touched,
            "tests": {"commands": [f"pytest {task_id}"], "passed": True, "evidence": [f"synthetic:{task_id}:pass"]},
            "semantic_guards": copy.deepcopy(GUARDS),
            "scientific_status": SCIENCE,
            "integration_notes": ["synthetic fixture only"],
        }
        receipts.append(ReceiptInput(path, receipt))
    return repo, receipts


def adjudicate(repo, receipts, m=None, s=None):
    return compile_wave1_receipts(
        manifest=m or manifest(), receipt_schema=s or schema(), receipts=receipts,
        repo=repo, repository_head="HEAD", expected_gatework_base=EXPECTED_GATEWORK_BASE,
    )


def mutate(receipts, task_id, key, value):
    out = copy.deepcopy(receipts)
    for item in out:
        if item.document["task_id"] == task_id:
            item.document[key] = value
            return out
    raise AssertionError(task_id)


def expect_code(code, repo, receipts, m=None, s=None):
    with pytest.raises(GateFailure) as exc:
        adjudicate(repo, receipts, m=m, s=s)
    assert exc.value.code == code


def test_valid_synthetic_receipts_emit_readiness_only():
    repo, receipts = valid_inputs()
    result = adjudicate(repo, receipts)
    assert result["status"] == "PASS"
    assert result["verdict"] == READY_VERDICT
    assert result["final_cutover_qualified"] is False
    assert "R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED" not in json.dumps(result)


def test_missing_task_rejected():
    repo, receipts = valid_inputs()
    expect_code("STAGE4_RECEIPT_COUNT_MISMATCH", repo, receipts[:-1])


def test_duplicate_task_rejected():
    repo, receipts = valid_inputs()
    dup = copy.deepcopy(receipts)
    dup[-1] = copy.deepcopy(dup[0])
    expect_code("STAGE4_RECEIPT_DUPLICATE_TASK", repo, dup)


@pytest.mark.parametrize("status", ["FAIL", "BLOCKED", "NOT_RUN"])
def test_non_pass_status_rejected(status):
    repo, receipts = valid_inputs()
    expect_code("STAGE4_RECEIPT_NOT_PASS", repo, mutate(receipts, "S4A", "status", status))


def test_wrong_gatework_base_rejected():
    repo, receipts = valid_inputs()
    expect_code("STAGE4_RECEIPT_GATEWORK_BASE_MISMATCH", repo, mutate(receipts, "S4A", "gatework_base_sha", "f" * 40))


def test_wrong_branch_rejected():
    repo, receipts = valid_inputs()
    expect_code("STAGE4_RECEIPT_BRANCH_MISMATCH", repo, mutate(receipts, "S4B", "branch", "wrong"))


def test_malformed_sha_rejected_by_schema():
    repo, receipts = valid_inputs()
    expect_code("STAGE4_RECEIPT_SCHEMA_INVALID", repo, mutate(receipts, "S4C", "task_head_sha", "bad"))


def test_changed_scientific_status_rejected():
    repo, receipts = valid_inputs()
    expect_code("STAGE4_RECEIPT_SCHEMA_INVALID", repo, mutate(receipts, "S4D", "scientific_status", "CHANGED"))


def test_final_holdout_path_rejected():
    repo, receipts = valid_inputs()
    out = copy.deepcopy(receipts)
    item = next(x for x in out if x.document["task_id"] == "S4A")
    head = item.document["task_head_sha"]
    bad = "tests/stage4_final_holdout_2025-09_probe.txt"
    item.document["touched_paths"].append(bad)
    item.document["owned_paths"].append(bad)
    repo.changed[(EXPECTED_GATEWORK_BASE, head)].add(bad)
    expect_code("STAGE4_FORBIDDEN_PATH_CHANGE", repo, out)


def test_fresh_data_download_guard_rejected():
    repo, receipts = valid_inputs()
    out = copy.deepcopy(receipts)
    next(x for x in out if x.document["task_id"] == "S4B").document["semantic_guards"]["fresh_market_data_downloaded"] = True
    expect_code("STAGE4_RECEIPT_SCHEMA_INVALID", repo, out)


def test_sibling_dependency_true_rejected():
    repo, receipts = valid_inputs()
    out = copy.deepcopy(receipts)
    next(x for x in out if x.document["task_id"] == "S4C").document["semantic_guards"]["sibling_dependency_used"] = True
    expect_code("STAGE4_RECEIPT_SCHEMA_INVALID", repo, out)


def test_duplicate_path_ownership_rejected():
    repo, receipts = valid_inputs()
    out = copy.deepcopy(receipts)
    a = next(x for x in out if x.document["task_id"] == "S4A")
    b = next(x for x in out if x.document["task_id"] == "S4B")
    b.document["owned_paths"].append(a.document["owned_paths"][0])
    expect_code("STAGE4_OVERLAPPING_TASK_OWNERSHIP", repo, out)


def test_undeclared_changed_path_rejected():
    repo, receipts = valid_inputs()
    a = next(x for x in receipts if x.document["task_id"] == "S4A")
    repo.changed[(EXPECTED_GATEWORK_BASE, a.document["task_head_sha"])].add("cb16_local_opt/stage4_extra_r11.py")
    expect_code("STAGE4_TOUCHED_PATH_MISMATCH", repo, receipts)


def test_semantic_freeze_mismatch_rejected():
    repo, receipts = valid_inputs()
    repo.blobs[("HEAD", FREEZE_PATH)] = "0" * 40
    expect_code("STAGE4_SEMANTIC_FREEZE_MISMATCH", repo, receipts)


def test_fake_pass_tests_false_rejected():
    repo, receipts = valid_inputs()
    out = copy.deepcopy(receipts)
    next(x for x in out if x.document["task_id"] == "S4D").document["tests"]["passed"] = False
    expect_code("STAGE4_PASS_WITHOUT_TEST_PASS", repo, out)


def test_missing_test_evidence_rejected():
    repo, receipts = valid_inputs()
    out = copy.deepcopy(receipts)
    next(x for x in out if x.document["task_id"] == "S4E").document["tests"].pop("evidence")
    expect_code("STAGE4_PASS_WITHOUT_TEST_EVIDENCE", repo, out)


def test_task_head_must_be_real_commit_and_descend_from_gatework():
    repo, receipts = valid_inputs()
    a = next(x for x in receipts if x.document["task_id"] == "S4A")
    repo.commits.remove(a.document["task_head_sha"])
    expect_code("STAGE4_TASK_HEAD_NOT_COMMIT", repo, receipts)


def test_task_head_must_precede_receipt_commit():
    repo, receipts = valid_inputs()
    a = next(x for x in receipts if x.document["task_id"] == "S4A")
    receipt_commit = repo.additions[a.path][0]
    repo.ancestors.remove((a.document["task_head_sha"], receipt_commit))
    expect_code("STAGE4_TASK_HEAD_NOT_ANCESTOR_OF_RECEIPT", repo, receipts)


def test_receipt_must_have_unique_addition_commit_evidence():
    repo, receipts = valid_inputs()
    a = next(x for x in receipts if x.document["task_id"] == "S4A")
    repo.additions[a.path] = ()
    expect_code("STAGE4_RECEIPT_COMMIT_EVIDENCE_AMBIGUOUS", repo, receipts)


def test_existing_file_modification_rejected_outside_s4e():
    repo, receipts = valid_inputs()
    a = next(x for x in receipts if x.document["task_id"] == "S4A")
    path = a.document["touched_paths"][0]
    repo.existing.add((EXPECTED_GATEWORK_BASE, path))
    expect_code("STAGE4_EXISTING_FILE_OWNERSHIP_VIOLATION", repo, receipts)


def test_s4e_is_only_task_with_existing_integration_adapter_exception():
    repo, receipts = valid_inputs()
    out = copy.deepcopy(receipts)
    e = next(x for x in out if x.document["task_id"] == "S4E")
    new = "cb16_local_opt/integration_adapters_r11.py"
    e.document["touched_paths"] = [new]
    e.document["owned_paths"] = [new]
    head = e.document["task_head_sha"]
    repo.changed[(EXPECTED_GATEWORK_BASE, head)] = {new}
    repo.existing.add((EXPECTED_GATEWORK_BASE, new))
    result = adjudicate(repo, out)
    assert result["status"] == "PASS"


def test_s4e_cannot_modify_unapproved_existing_file():
    repo, receipts = valid_inputs()
    e = next(x for x in receipts if x.document["task_id"] == "S4E")
    path = e.document["touched_paths"][0]
    repo.existing.add((EXPECTED_GATEWORK_BASE, path))
    expect_code("STAGE4_EXISTING_FILE_OWNERSHIP_VIOLATION", repo, receipts)


def test_manifest_cannot_grant_existing_file_exception_to_other_task():
    repo, receipts = valid_inputs()
    m = manifest()
    next(x for x in m["wave1_tasks"] if x["id"] == "S4A")["exclusive_existing_file_ownership"] = ["cb16_local_opt/integration_adapters_r11.py"]
    expect_code("STAGE4_UNEXPECTED_EXISTING_FILE_EXCEPTION", repo, receipts, m=m)


def test_semantic_guards_must_be_exact_no_extra_keys():
    repo, receipts = valid_inputs()
    out = copy.deepcopy(receipts)
    next(x for x in out if x.document["task_id"] == "S4F").document["semantic_guards"]["invented_override"] = False
    expect_code("STAGE4_SEMANTIC_GUARDS_NOT_EXACT", repo, out)


def test_receipt_changed_after_creation_rejected():
    repo, receipts = valid_inputs()
    a = next(x for x in receipts if x.document["task_id"] == "S4A")
    repo.blobs[("HEAD", a.path)] = "different"
    expect_code("STAGE4_RECEIPT_CHANGED_AFTER_CREATION", repo, receipts)


def test_dirty_receipt_worktree_rejected():
    repo, receipts = valid_inputs()
    a = next(x for x in receipts if x.document["task_id"] == "S4A")
    original = repo.path_clean_at_head
    repo.path_clean_at_head = lambda path: False if path == a.path else original(path)
    expect_code("STAGE4_RECEIPT_WORKTREE_DIRTY", repo, receipts)
