"""Fail-closed Stage-4 Integration receipt compiler for INTA-INTH.

INTH emits pre-consolidation readiness only; the final cutover verdict is reserved
for the later consolidated adjudicator.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

INTEGRATION_SEED_SHA = "86a4ac8a9080cd8382600cb998059e9585495a11"
SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
READY_VERDICT = "STAGE4_INTEGRATION_FORK_RECEIPTS_READY_FOR_CONSOLIDATION"
FORBIDDEN_FINAL_VERDICT = "R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED"
RECEIPT_SCHEMA_NAME = "CB16_R11_STAGE4_INTEGRATION_RECEIPT_V1"
FREEZE_PATH = "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
INTB_EXISTING_FILE = "cb16_local_opt/integration_adapters_r11.py"
RECEIPT_PREFIX = "authority/rearchitecture_r11/stage4_integration_receipts"
SHA40 = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_TASK_BRANCHES = {
    "INTA": "ai/r11-stage4-int-a-runtime-spine-r0",
    "INTB": "ai/r11-stage4-int-b-persistence-fencing-r0",
    "INTC": "ai/r11-stage4-int-c-engine-authority-r0",
    "INTD": "ai/r11-stage4-int-d-permission-execution-r0",
    "INTE": "ai/r11-stage4-int-e-adoption-recovery-r0",
    "INTF": "ai/r11-stage4-int-f-worker-lifecycle-r0",
    "INTG": "ai/r11-stage4-int-g-hostile-harness-r0",
    "INTH": "ai/r11-stage4-int-h-final-gates-r0",
}
EXPECTED_GUARDS = {
    "semantic_freeze_unchanged": True,
    "final_holdout_untouched": True,
    "fresh_market_data_downloaded": False,
    "historical_market_data_mutated": False,
    "new_scientific_verdict": False,
    "scientific_semantics_changed": False,
    "replay_reinterpreted_as_new_evidence": False,
    "sibling_integration_dependency_used": False,
    "pushed_to_seed_or_final_branch": False,
    "long_endurance_run": False,
}
FINAL_CONSOLIDATION_REQUIREMENTS = (
    "all_accepted_integration_receipts", "exact_seed_ancestry", "controlled_consolidation_only",
    "semantic_freeze_exact", "scientific_status_unchanged",
    "s4a_writer_registry_rebuilt_and_reaudited_at_final_head", "unknown_authority_zero",
    "all_stage4_integration_correctness_tests_pass", "real_s4h_h01_h20_against_consolidated_runtime",
    "exactly_one_live_canonical_authority_writer", "stale_writer_cannot_mutate",
    "legacy_writer_cannot_mutate", "permission_cannot_be_bypassed",
    "replay_cannot_become_new_evidence", "short_canonical_machine_startup_steady_drain_shutdown_pass",
    "fp32_canonical", "amp_false", "final_holdout_2025_09_untouched", "no_fresh_market_data",
    "no_new_scientific_verdict", "long_endurance_not_required",
)


class IntegrationGateFailure(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code, self.detail = code, detail
        super().__init__(f"{code}{':' + detail if detail else ''}")


@dataclass(frozen=True)
class ReceiptInput:
    task_id: str
    path: str
    document: Mapping[str, Any]
    receipt_commit_sha: str
    branch_tip_sha: str


class RepositoryEvidence(Protocol):
    def object_is_commit(self, sha: str) -> bool: ...
    def is_ancestor(self, ancestor: str, descendant: str) -> bool: ...
    def merge_commits_between(self, base: str, head: str) -> Sequence[str]: ...
    def changed_paths(self, base: str, head: str) -> set[str]: ...
    def path_exists_at(self, commit: str, path: str) -> bool: ...
    def blob_sha(self, commit: str, path: str) -> str: ...
    def commit_parents(self, commit: str) -> Sequence[str]: ...
    def commit_changed_paths(self, commit: str) -> set[str]: ...
    def path_status_in_commit(self, commit: str, path: str) -> str | None: ...
    def path_change_commits(self, base: str, head: str, path: str) -> Sequence[str]: ...


class SubprocessRepositoryEvidence:
    """Read-only git adapter for the later final-adjudicator checkout."""
    def __init__(self, root: Path) -> None: self.root = Path(root).resolve()
    def _r(self, *a: str, check: bool = True):
        return subprocess.run(["git", *a], cwd=self.root, text=True, capture_output=True, check=check)
    def object_is_commit(self, s): return self._r("cat-file", "-e", f"{s}^{{commit}}", check=False).returncode == 0
    def is_ancestor(self, a, d): return self._r("merge-base", "--is-ancestor", a, d, check=False).returncode == 0
    def merge_commits_between(self, b, h): return tuple(x for x in self._r("rev-list", "--merges", f"{b}..{h}").stdout.splitlines() if x)
    def changed_paths(self, b, h): return {x for x in self._r("diff", "--name-only", f"{b}..{h}").stdout.splitlines() if x}
    def path_exists_at(self, c, p): return self._r("cat-file", "-e", f"{c}:{p}", check=False).returncode == 0
    def blob_sha(self, c, p): return self._r("rev-parse", f"{c}:{p}").stdout.strip()
    def commit_parents(self, c): return tuple(self._r("rev-list", "--parents", "-n", "1", c).stdout.strip().split()[1:])
    def commit_changed_paths(self, c): return {x for x in self._r("diff-tree", "--no-commit-id", "--name-only", "-r", c).stdout.splitlines() if x}
    def path_status_in_commit(self, c, p):
        out = self._r("diff-tree", "--no-commit-id", "--name-status", "-r", c, "--", p).stdout.strip()
        return out.split()[0] if out else None
    def path_change_commits(self, b, h, p): return tuple(x for x in self._r("log", "--format=%H", f"{b}..{h}", "--", p).stdout.splitlines() if x)


def _fail(ok: bool, code: str, detail: str = "") -> None:
    if not ok: raise IntegrationGateFailure(code, detail)


def _path(raw: Any) -> str:
    _fail(isinstance(raw, str) and raw and "\\" not in raw, "INTEGRATION_INVALID_REPO_PATH", repr(raw))
    p = PurePosixPath(raw); _fail(not p.is_absolute() and ".." not in p.parts and "." not in p.parts, "INTEGRATION_INVALID_REPO_PATH", raw)
    _fail(p.as_posix() == raw.rstrip("/"), "INTEGRATION_INVALID_REPO_PATH", raw)
    return p.as_posix()


def _type(v: Any, k: str) -> bool:
    return {"object": lambda: isinstance(v, Mapping), "array": lambda: isinstance(v, list),
            "string": lambda: isinstance(v, str), "boolean": lambda: isinstance(v, bool),
            "integer": lambda: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda: isinstance(v, (int, float)) and not isinstance(v, bool),
            "null": lambda: v is None}.get(k, lambda: False)()


def validate_schema(x: Any, s: Mapping[str, Any], where: str = "$", root_schema: Mapping[str, Any] | None = None) -> None:
    """Dependency-free validator for the keywords used by Stage-4 schemas."""
    root = s if root_schema is None else root_schema
    if "$ref" in s:
        ref = s["$ref"]; _fail(isinstance(ref, str) and ref.startswith("#/"), "INTEGRATION_SCHEMA_REF_UNSUPPORTED", str(ref))
        target: Any = root
        for part in ref[2:].split("/"):
            _fail(isinstance(target, Mapping) and part in target, "INTEGRATION_SCHEMA_REF_MISSING", ref); target = target[part]
        _fail(isinstance(target, Mapping), "INTEGRATION_SCHEMA_REF_INVALID", ref); validate_schema(x, target, where, root); return
    if "const" in s: _fail(x == s["const"], "INTEGRATION_SCHEMA_INVALID", f"{where}:const")
    if "enum" in s: _fail(x in s["enum"], "INTEGRATION_SCHEMA_INVALID", f"{where}:enum")
    if "type" in s: _fail(_type(x, s["type"]), "INTEGRATION_SCHEMA_INVALID", f"{where}:type")
    if isinstance(x, str):
        if "minLength" in s: _fail(len(x) >= int(s["minLength"]), "INTEGRATION_SCHEMA_INVALID", f"{where}:minLength")
        if "pattern" in s: _fail(re.search(s["pattern"], x) is not None, "INTEGRATION_SCHEMA_INVALID", f"{where}:pattern")
    if isinstance(x, list):
        if "minItems" in s: _fail(len(x) >= int(s["minItems"]), "INTEGRATION_SCHEMA_INVALID", f"{where}:minItems")
        if "maxItems" in s: _fail(len(x) <= int(s["maxItems"]), "INTEGRATION_SCHEMA_INVALID", f"{where}:maxItems")
        if s.get("uniqueItems"):
            enc = [json.dumps(v, sort_keys=True, separators=(",", ":")) for v in x]; _fail(len(enc) == len(set(enc)), "INTEGRATION_SCHEMA_INVALID", f"{where}:uniqueItems")
        if isinstance(s.get("items"), Mapping):
            for i, v in enumerate(x): validate_schema(v, s["items"], f"{where}[{i}]", root)
    if isinstance(x, Mapping):
        for k in s.get("required", []): _fail(k in x, "INTEGRATION_SCHEMA_INVALID", f"{where}:missing:{k}")
        props = s.get("properties", {})
        for k, sub in props.items():
            if k in x and isinstance(sub, Mapping): validate_schema(x[k], sub, f"{where}.{k}", root)
        if s.get("additionalProperties") is False: _fail(not (set(x) - set(props)), "INTEGRATION_SCHEMA_INVALID", f"{where}:additional:{sorted(set(x)-set(props))}")


def _specs(g: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = g.get("integration_tasks"); _fail(isinstance(rows, list), "INTEGRATION_GATEWORK_INVALID", "integration_tasks")
    out = {}
    for r in rows:
        _fail(isinstance(r, Mapping) and r.get("id") in EXPECTED_TASK_BRANCHES, "INTEGRATION_GATEWORK_TASK_SET_MISMATCH", str(r))
        t = r["id"]; _fail(t not in out, "INTEGRATION_GATEWORK_DUPLICATE_TASK", t); _fail(r.get("branch") == EXPECTED_TASK_BRANCHES[t], "INTEGRATION_GATEWORK_BRANCH_MISMATCH", t); out[t] = r
    _fail(list(out) == list(EXPECTED_TASK_BRANCHES), "INTEGRATION_GATEWORK_TASK_ORDER_MISMATCH", repr(list(out))); return out


def _forbidden(p: str) -> bool:
    q = p.lower(); return p == FREEZE_PATH or "final_holdout" in q or "2025-09" in q or q.startswith(("data/", "market_data/", "provision/assets/binance_usdm_1m_funding_2020_2026"))


def _machine_evidence(e: Any) -> bool:
    text = "\n".join(v for v in e if isinstance(v, str)) if isinstance(e, list) else ""
    return bool(re.search(r"(?:actions/runs/|github[_ -]?action(?:s)?[_ -]?run(?:[_ -]?id)?\s*[:=]\s*)\d+", text, re.I) and re.search(r"(?:conclusion|result|status)\s*[:=]\s*(?:success|pass(?:ed)?)\b", text, re.I))


def _intg_ok(r: Mapping[str, Any]) -> bool:
    if r.get("task_id") != "INTG": return True
    inv = r.get("correctness_invariants") or {}
    if isinstance(inv, Mapping) and inv.get("integrated_runtime_qualification_claimed") is False: return True
    notes = "\n".join(str(v).replace(" ", "").lower() for v in (r.get("final_consolidation_notes") or []))
    return "integrated_runtime_qualification_claimed=false" in notes


def validate_final_cutover_report(report: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Contract validator for the later final consolidated adjudicator."""
    validate_schema(report, schema)
    verdict = report.get("verdict"); _fail(verdict in {FORBIDDEN_FINAL_VERDICT, "FAIL_CLOSED"}, "FINAL_REPORT_VERDICT_INVALID", str(verdict))
    if verdict != FORBIDDEN_FINAL_VERDICT: return
    gates = report.get("qualification_gates") or {}
    for k in FINAL_CONSOLIDATION_REQUIREMENTS: _fail(gates.get(k) is True, "FINAL_REPORT_GATE_NOT_PASS", k)
    _fail(report.get("integration_seed_sha") == INTEGRATION_SEED_SHA, "FINAL_REPORT_SEED_MISMATCH")
    _fail(report.get("scientific_status") == SCIENTIFIC_STATUS, "FINAL_REPORT_SCIENTIFIC_STATUS_CHANGED")
    accepted = report.get("accepted_integration_receipts") or {}; _fail(set(accepted) == set(EXPECTED_TASK_BRANCHES), "FINAL_REPORT_ACCEPTED_RECEIPT_SET_INVALID")
    for t, b in EXPECTED_TASK_BRANCHES.items():
        x = accepted.get(t) or {}; _fail(x.get("branch") == b and SHA40.fullmatch(str(x.get("task_head_sha", ""))) is not None and SHA40.fullmatch(str(x.get("receipt_commit_sha", ""))) is not None, "FINAL_REPORT_RECEIPT_IDENTITY_INVALID", t)
    reg = report.get("writer_registry") or {}; _fail(reg.get("rebuilt_at_head") is True, "FINAL_REPORT_WRITER_REGISTRY_NOT_REBUILT"); _fail(reg.get("unknown_authority_count") == 0, "FINAL_REPORT_UNKNOWN_AUTHORITY_NONZERO")
    h = report.get("hostile_matrix") or {}; _fail(h.get("report_schema") == "CB16_R11_STAGE4_HOSTILE_CUTOVER_REPORT_V1" and h.get("integrated_runtime_qualification_claimed") is True and h.get("h01_h20_all_passed") is True, "FINAL_REPORT_HOSTILE_MATRIX_NOT_PASS")
    m = report.get("canonical_machine_smoke") or {}; _fail(m.get("runner_labels") == ["self-hosted", "shanxi", "cb16-wss-qualification"], "FINAL_REPORT_RUNNER_LABELS_MISMATCH"); _fail(m.get("python_resolution_status") == "READY" and m.get("python_resolution_mode") == "VERIFIED_CANONICAL_R104_VENV_REUSE", "FINAL_REPORT_PYTHON_MODE_MISMATCH"); _fail(m.get("numeric_mode") == "FP32" and m.get("amp") is False and m.get("lifecycle_passed") is True and m.get("long_endurance_run") is False, "FINAL_REPORT_MACHINE_SMOKE_INVALID")
    keys = {"semantic_freeze_unchanged","final_holdout_untouched","fresh_market_data_downloaded","historical_market_data_mutated","new_scientific_verdict","scientific_semantics_changed","replay_reinterpreted_as_new_evidence"}
    _fail(dict(report.get("semantic_guards") or {}) == {k:v for k,v in EXPECTED_GUARDS.items() if k in keys}, "FINAL_REPORT_SEMANTIC_GUARDS_NOT_EXACT")
    _fail(report.get("reserved_verdict_issuer") == "FINAL_CONSOLIDATED_ADJUDICATOR_ONLY", "FINAL_REPORT_VERDICT_ISSUER_INVALID")


def compile_integration_receipts(*, gatework: Mapping[str, Any], receipt_schema: Mapping[str, Any], receipts: Sequence[ReceiptInput], repo: RepositoryEvidence, seed_sha: str = INTEGRATION_SEED_SHA) -> dict[str, Any]:
    """Adjudicate exactly INTA-INTH and emit only pre-consolidation readiness."""
    _fail(seed_sha == INTEGRATION_SEED_SHA and repo.object_is_commit(seed_sha), "INTEGRATION_SEED_MISMATCH", seed_sha)
    _fail(gatework.get("scientific_status") == SCIENTIFIC_STATUS, "INTEGRATION_GATEWORK_SCIENCE_MISMATCH")
    freeze = gatework.get("semantic_freeze_blob_sha"); _fail(isinstance(freeze, str) and SHA40.fullmatch(freeze) is not None, "INTEGRATION_GATEWORK_FREEZE_IDENTITY_MISSING")
    specs = _specs(gatework); _fail(len(receipts) == len(specs), "INTEGRATION_RECEIPT_COUNT_MISMATCH", f"got={len(receipts)} expected={len(specs)}")
    by = {}
    for i in receipts:
        _fail(i.task_id in specs, "INTEGRATION_RECEIPT_UNKNOWN_TASK", i.task_id); _fail(i.task_id not in by, "INTEGRATION_RECEIPT_DUPLICATE_TASK", i.task_id)
        validate_schema(i.document, receipt_schema); _fail(i.document.get("task_id") == i.task_id, "INTEGRATION_RECEIPT_TASK_MISMATCH", i.task_id); by[i.task_id] = i
    _fail(set(by) == set(specs), "INTEGRATION_RECEIPT_TASK_SET_MISMATCH"); _fail(repo.blob_sha(seed_sha, FREEZE_PATH) == freeze, "INTEGRATION_SEED_FREEZE_BLOB_MISMATCH")
    owners, accepted = {}, {}
    for t in EXPECTED_TASK_BRANCHES:
        i, r = by[t], by[t].document; rp = _path(i.path); expected = f"{RECEIPT_PREFIX}/{t}_RECEIPT_V1.json"
        _fail(rp == expected, "INTEGRATION_RECEIPT_PATH_MISMATCH", t); _fail(r.get("schema") == RECEIPT_SCHEMA_NAME, "INTEGRATION_RECEIPT_SCHEMA_NAME_MISMATCH", t); _fail(r.get("status") == "PASS", "INTEGRATION_RECEIPT_NOT_PASS", t)
        _fail(r.get("branch") == EXPECTED_TASK_BRANCHES[t], "INTEGRATION_RECEIPT_BRANCH_MISMATCH", t); _fail(r.get("integration_seed_sha") == seed_sha, "INTEGRATION_RECEIPT_SEED_MISMATCH", t); _fail(r.get("scientific_status") == SCIENTIFIC_STATUS, "INTEGRATION_SCIENTIFIC_STATUS_CHANGED", t); _fail(dict(r.get("semantic_guards") or {}) == EXPECTED_GUARDS, "INTEGRATION_SEMANTIC_GUARDS_NOT_EXACT", t); _fail(_intg_ok(r), "INTEGRATION_INTG_RUNTIME_QUALIFICATION_OVERCLAIM")
        tests = r.get("tests") or {}; _fail(tests.get("passed") is True, "INTEGRATION_TESTS_NOT_PASS", t); _fail(isinstance(tests.get("commands"), list) and bool(tests["commands"]), "INTEGRATION_TEST_COMMANDS_MISSING", t); _fail(_machine_evidence(tests.get("evidence")), "INTEGRATION_MACHINE_EVIDENCE_MISSING", t)
        h = r.get("task_head_sha"); _fail(isinstance(h, str) and SHA40.fullmatch(h) is not None and repo.object_is_commit(h), "INTEGRATION_TASK_HEAD_BAD_SHA", t); _fail(repo.is_ancestor(seed_sha, h), "INTEGRATION_TASK_HEAD_NOT_DESCENDANT_OF_SEED", t); _fail(not repo.merge_commits_between(seed_sha, h), "INTEGRATION_TASK_HEAD_HAS_MERGE", t)
        rc, tip = i.receipt_commit_sha, i.branch_tip_sha; _fail(SHA40.fullmatch(rc or "") is not None and repo.object_is_commit(rc), "INTEGRATION_RECEIPT_COMMIT_INVALID", t); _fail(SHA40.fullmatch(tip or "") is not None and repo.object_is_commit(tip), "INTEGRATION_BRANCH_TIP_INVALID", t); _fail(rc != h, "INTEGRATION_RECEIPT_NOT_DISTINCT_COMMIT", t); _fail(tuple(repo.commit_parents(rc)) == (h,), "INTEGRATION_RECEIPT_NOT_DIRECTLY_AFTER_TASK_HEAD", t); _fail(tip == rc, "INTEGRATION_RECEIPT_NOT_FINAL_BRANCH_COMMIT", t); _fail(repo.commit_changed_paths(rc) == {rp}, "INTEGRATION_RECEIPT_COMMIT_MUTATES_OTHER_PATHS", t); _fail(repo.path_status_in_commit(rc, rp) == "A", "INTEGRATION_RECEIPT_NOT_ADDED_EXACTLY_ONCE", t); _fail(tuple(repo.path_change_commits(h, tip, rp)) == (rc,), "INTEGRATION_RECEIPT_MUTATED_AFTER_CREATION", t)
        declared, owned, actual = {_path(x) for x in r.get("touched_paths", [])}, {_path(x) for x in r.get("owned_paths", [])}, {_path(x) for x in repo.changed_paths(seed_sha, h)}; _fail(declared <= owned, "INTEGRATION_TOUCHED_OUTSIDE_OWNED_PATHS", t); _fail(actual == declared, "INTEGRATION_TOUCHED_PATH_MISMATCH", t)
        for p in sorted(actual):
            _fail(not _forbidden(p), "INTEGRATION_FORBIDDEN_PATH_CHANGED", f"{t}:{p}")
            if repo.path_exists_at(seed_sha, p): _fail(t == "INTB" and p == INTB_EXISTING_FILE, "INTEGRATION_EXISTING_FILE_OWNERSHIP_VIOLATION", f"{t}:{p}")
            _fail(p not in owners, "INTEGRATION_TOUCHED_PATH_OVERLAP", f"{p}:{owners.get(p)}:{t}"); owners[p] = t
        _fail(repo.blob_sha(h, FREEZE_PATH) == freeze and repo.blob_sha(rc, FREEZE_PATH) == freeze, "INTEGRATION_SEMANTIC_FREEZE_MISMATCH", t)
        accepted[t] = {"branch": EXPECTED_TASK_BRANCHES[t], "task_head_sha": h, "receipt_commit_sha": rc, "receipt_path": rp, "touched_paths": sorted(actual)}
    return {"schema": "CB16_R11_STAGE4_INTEGRATION_PRECONSOLIDATION_REPORT_V1", "verdict": READY_VERDICT, "integration_seed_sha": seed_sha, "scientific_status": SCIENTIFIC_STATUS, "accepted_tasks": accepted, "final_cutover_verdict_emitted": False, "reserved_final_cutover_verdict": FORBIDDEN_FINAL_VERDICT, "final_consolidation_requirements": list(FINAL_CONSOLIDATION_REQUIREMENTS)}
