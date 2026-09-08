"""Fail-closed Stage-4 Wave-1 receipt compiler.

This module adjudicates infrastructure receipts only.  It cannot emit the final
R11 canonical-authority cutover verdict and does not implement runtime authority.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Protocol, Sequence

EXPECTED_GATEWORK_BASE = "0f18e08ec7250b9b4e45c62803c25be966834390"
READY_VERDICT = "STAGE4_WAVE1_RECEIPTS_READY_FOR_INTEGRATION"
FORBIDDEN_FINAL_VERDICT = "R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED"
RECEIPT_SCHEMA_NAME = "CB16_R11_STAGE4_TASK_RECEIPT_V1"
FREEZE_PATH = "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
INTEGRATION_ADAPTER_PATH = "cb16_local_opt/integration_adapters_r11.py"
RAW_DATA_MANIFEST_PREFIX = "provision/assets/binance_usdm_1m_funding_2020_2026"
SHA40 = re.compile(r"^[0-9a-f]{40}$")


class GateFailure(RuntimeError):
    """Deterministic fail-closed qualification error."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}{':' + detail if detail else ''}")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    branch: str
    exclusive_existing_file_ownership: frozenset[str]


@dataclass(frozen=True)
class ReceiptInput:
    path: str
    document: Mapping[str, Any]


class RepositoryEvidence(Protocol):
    def object_is_commit(self, sha: str) -> bool: ...
    def is_ancestor(self, ancestor: str, descendant: str) -> bool: ...
    def merge_commits_between(self, base: str, head: str) -> Sequence[str]: ...
    def changed_paths(self, base: str, head: str) -> set[str]: ...
    def path_exists_at(self, commit: str, path: str) -> bool: ...
    def blob_sha(self, commit: str, path: str) -> str: ...
    def receipt_addition_commits(self, path: str) -> Sequence[str]: ...
    def path_clean_at_head(self, path: str) -> bool: ...


class SubprocessRepositoryEvidence:
    """Read-only git evidence adapter used by the CLI/integration gate."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=self.root, text=True, capture_output=True, check=check
        )

    def object_is_commit(self, sha: str) -> bool:
        return self._run("cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return self._run("merge-base", "--is-ancestor", ancestor, descendant, check=False).returncode == 0

    def merge_commits_between(self, base: str, head: str) -> Sequence[str]:
        out = self._run("rev-list", "--merges", f"{base}..{head}").stdout
        return tuple(line for line in out.splitlines() if line)

    def changed_paths(self, base: str, head: str) -> set[str]:
        out = self._run("diff", "--name-only", f"{base}...{head}").stdout
        return {line for line in out.splitlines() if line}

    def path_exists_at(self, commit: str, path: str) -> bool:
        return self._run("cat-file", "-e", f"{commit}:{path}", check=False).returncode == 0

    def blob_sha(self, commit: str, path: str) -> str:
        return self._run("rev-parse", f"{commit}:{path}").stdout.strip()

    def receipt_addition_commits(self, path: str) -> Sequence[str]:
        out = self._run("log", "--diff-filter=A", "--format=%H", "--", path).stdout
        return tuple(line for line in out.splitlines() if line)

    def path_clean_at_head(self, path: str) -> bool:
        unstaged = self._run("diff", "--quiet", "--", path, check=False).returncode
        staged = self._run("diff", "--cached", "--quiet", "--", path, check=False).returncode
        return unstaged == 0 and staged == 0


def _fail(cond: bool, code: str, detail: str = "") -> None:
    if not cond:
        raise GateFailure(code, detail)


def _type_matches(value: Any, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, Mapping)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "null":
        return value is None
    return False


def validate_schema(instance: Any, schema: Mapping[str, Any], where: str = "$") -> None:
    """Validate the JSON-Schema subset used by the frozen Stage-4 receipt schema.

    This intentionally avoids a runtime dependency on jsonschema. Unknown schema
    keywords are ignored; every keyword currently present in the frozen receipt
    schema that constrains values is enforced here.
    """
    if "const" in schema:
        _fail(instance == schema["const"], "STAGE4_RECEIPT_SCHEMA_INVALID", f"{where}:const")
    if "enum" in schema:
        _fail(instance in schema["enum"], "STAGE4_RECEIPT_SCHEMA_INVALID", f"{where}:enum")
    if "type" in schema:
        _fail(_type_matches(instance, schema["type"]), "STAGE4_RECEIPT_SCHEMA_INVALID", f"{where}:type")
    if isinstance(instance, str):
        if "minLength" in schema:
            _fail(len(instance) >= int(schema["minLength"]), "STAGE4_RECEIPT_SCHEMA_INVALID", f"{where}:minLength")
        if "pattern" in schema:
            _fail(re.search(schema["pattern"], instance) is not None, "STAGE4_RECEIPT_SCHEMA_INVALID", f"{where}:pattern")
    if isinstance(instance, list):
        if schema.get("uniqueItems"):
            encoded = [json.dumps(x, sort_keys=True, separators=(",", ":")) for x in instance]
            _fail(len(encoded) == len(set(encoded)), "STAGE4_RECEIPT_SCHEMA_INVALID", f"{where}:uniqueItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(instance):
                validate_schema(item, item_schema, f"{where}[{index}]")
    if isinstance(instance, Mapping):
        for key in schema.get("required", []):
            _fail(key in instance, "STAGE4_RECEIPT_SCHEMA_INVALID", f"{where}:missing:{key}")
        properties = schema.get("properties", {})
        for key, sub_schema in properties.items():
            if key in instance and isinstance(sub_schema, Mapping):
                validate_schema(instance[key], sub_schema, f"{where}.{key}")
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(properties)
            _fail(not extra, "STAGE4_RECEIPT_SCHEMA_INVALID", f"{where}:additional:{sorted(extra)}")


def _normalize_path(raw: Any, code: str = "STAGE4_INVALID_REPO_PATH") -> str:
    _fail(isinstance(raw, str) and bool(raw), code, repr(raw))
    _fail("\\" not in raw, code, raw)
    p = PurePosixPath(raw)
    _fail(not p.is_absolute(), code, raw)
    _fail(".." not in p.parts and "." not in p.parts, code, raw)
    normalized = p.as_posix()
    _fail(normalized == raw.rstrip("/"), code, raw)
    return normalized


def _paths_overlap(left: str, right: str) -> bool:
    a = PurePosixPath(left).parts
    b = PurePosixPath(right).parts
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def _forbidden_path(path: str) -> bool:
    lower = path.lower()
    return (
        path == FREEZE_PATH
        or "final_holdout" in lower
        or "2025-09" in lower
        or path.startswith(RAW_DATA_MANIFEST_PREFIX)
    )


def _task_specs(manifest: Mapping[str, Any]) -> dict[str, TaskSpec]:
    rows = manifest.get("wave1_tasks")
    _fail(isinstance(rows, list), "STAGE4_GATEWORK_MANIFEST_INVALID", "wave1_tasks")
    specs: dict[str, TaskSpec] = {}
    for row in rows:
        _fail(isinstance(row, Mapping), "STAGE4_GATEWORK_MANIFEST_INVALID", "task_row")
        task_id = row.get("id")
        branch = row.get("branch")
        _fail(isinstance(task_id, str) and task_id, "STAGE4_GATEWORK_MANIFEST_INVALID", "task_id")
        _fail(isinstance(branch, str) and branch, "STAGE4_GATEWORK_MANIFEST_INVALID", f"{task_id}:branch")
        _fail(task_id not in specs, "STAGE4_GATEWORK_DUPLICATE_TASK", task_id)
        allowed = frozenset(_normalize_path(x) for x in row.get("exclusive_existing_file_ownership", []))
        specs[task_id] = TaskSpec(task_id, branch, allowed)
    expected = [f"S4{letter}" for letter in "ABCDEFGHI"]
    _fail(list(specs) == expected, "STAGE4_GATEWORK_TASK_SET_MISMATCH", repr(list(specs)))
    return specs


def _expected_guards(receipt_schema: Mapping[str, Any]) -> dict[str, Any]:
    try:
        guard_schema = receipt_schema["properties"]["semantic_guards"]
    except (KeyError, TypeError) as exc:
        raise GateFailure("STAGE4_RECEIPT_SCHEMA_GUARDS_MISSING") from exc
    exact: dict[str, Any] = {}
    for key in guard_schema.get("required", []):
        prop = guard_schema.get("properties", {}).get(key, {})
        _fail("const" in prop, "STAGE4_RECEIPT_SCHEMA_GUARD_NOT_EXACT", key)
        exact[key] = prop["const"]
    return exact


def compile_wave1_receipts(
    *,
    manifest: Mapping[str, Any],
    receipt_schema: Mapping[str, Any],
    receipts: Sequence[ReceiptInput],
    repo: RepositoryEvidence,
    repository_head: str = "HEAD",
    expected_gatework_base: str = EXPECTED_GATEWORK_BASE,
) -> dict[str, Any]:
    """Adjudicate all Wave-1 receipts and return only the integration-readiness verdict."""
    specs = _task_specs(manifest)
    _fail(SHA40.fullmatch(expected_gatework_base) is not None, "STAGE4_BAD_EXPECTED_GATEWORK_BASE")
    _fail(repo.object_is_commit(expected_gatework_base), "STAGE4_GATEWORK_BASE_NOT_COMMIT")

    upstream = manifest.get("upstream") or {}
    expected_freeze_blob = upstream.get("semantic_freeze_blob_sha")
    _fail(isinstance(expected_freeze_blob, str) and len(expected_freeze_blob) == 40,
          "STAGE4_GATEWORK_FREEZE_IDENTITY_MISSING")
    _fail(repo.blob_sha(repository_head, FREEZE_PATH) == expected_freeze_blob,
          "STAGE4_SEMANTIC_FREEZE_MISMATCH")

    expected_science = (manifest.get("scientific_authority") or {}).get("scientific_status")
    _fail(isinstance(expected_science, str) and expected_science,
          "STAGE4_GATEWORK_SCIENTIFIC_STATUS_MISSING")
    exact_guards = _expected_guards(receipt_schema)

    _fail(len(receipts) == len(specs), "STAGE4_RECEIPT_COUNT_MISMATCH", f"got={len(receipts)} expected={len(specs)}")
    by_task: dict[str, ReceiptInput] = {}
    for item in receipts:
        _fail(isinstance(item.document, Mapping), "STAGE4_RECEIPT_NOT_OBJECT", item.path)
        validate_schema(item.document, receipt_schema)
        task_id = item.document.get("task_id")
        _fail(task_id in specs, "STAGE4_RECEIPT_UNKNOWN_TASK", str(task_id))
        _fail(task_id not in by_task, "STAGE4_RECEIPT_DUPLICATE_TASK", str(task_id))
        by_task[str(task_id)] = item
    missing = sorted(set(specs) - set(by_task))
    _fail(not missing, "STAGE4_RECEIPT_MISSING_TASK", repr(missing))

    ownership: list[tuple[str, str]] = []
    touched_owner: dict[str, str] = {}
    receipt_commits: dict[str, str] = {}
    task_heads: dict[str, str] = {}

    for task_id, spec in specs.items():
        item = by_task[task_id]
        receipt = item.document
        receipt_path = _normalize_path(item.path)
        expected_receipt_path = f"authority/rearchitecture_r11/stage4_receipts/{task_id}_RECEIPT_V1.json"
        _fail(receipt_path == expected_receipt_path,
              "STAGE4_RECEIPT_PATH_TASK_MISMATCH", f"{task_id}:{receipt_path}")
        _fail(receipt.get("schema") == RECEIPT_SCHEMA_NAME, "STAGE4_RECEIPT_SCHEMA_MISMATCH", task_id)
        _fail(receipt.get("branch") == spec.branch, "STAGE4_RECEIPT_BRANCH_MISMATCH", task_id)
        _fail(receipt.get("gatework_base_sha") == expected_gatework_base, "STAGE4_RECEIPT_GATEWORK_BASE_MISMATCH", task_id)
        _fail(receipt.get("status") == "PASS", "STAGE4_RECEIPT_NOT_PASS", f"{task_id}:{receipt.get('status')}")
        _fail(receipt.get("scientific_status") == expected_science, "STAGE4_SCIENTIFIC_STATUS_CHANGED", task_id)

        guards = receipt.get("semantic_guards") or {}
        _fail(dict(guards) == exact_guards, "STAGE4_SEMANTIC_GUARDS_NOT_EXACT", task_id)
        _fail(guards.get("sibling_dependency_used") is False, "STAGE4_SIBLING_DEPENDENCY_USED", task_id)

        tests = receipt.get("tests") or {}
        _fail(tests.get("passed") is True, "STAGE4_PASS_WITHOUT_TEST_PASS", task_id)
        _fail(isinstance(tests.get("commands"), list) and len(tests["commands"]) > 0,
              "STAGE4_PASS_WITHOUT_TEST_COMMAND", task_id)
        _fail(isinstance(tests.get("evidence"), list) and len(tests["evidence"]) > 0,
              "STAGE4_PASS_WITHOUT_TEST_EVIDENCE", task_id)

        head = receipt.get("task_head_sha")
        _fail(isinstance(head, str) and SHA40.fullmatch(head) is not None, "STAGE4_RECEIPT_BAD_SHA", task_id)
        _fail(repo.object_is_commit(head), "STAGE4_TASK_HEAD_NOT_COMMIT", f"{task_id}:{head}")
        _fail(repo.is_ancestor(expected_gatework_base, head), "STAGE4_TASK_HEAD_NOT_DESCENDANT_OF_GATEWORK", task_id)
        _fail(not repo.merge_commits_between(expected_gatework_base, head), "STAGE4_TASK_HEAD_HAS_MERGE", task_id)
        task_heads[task_id] = head

        declared_touched = {_normalize_path(x) for x in receipt.get("touched_paths", [])}
        actual_touched = {_normalize_path(x) for x in repo.changed_paths(expected_gatework_base, head)}
        _fail(declared_touched == actual_touched, "STAGE4_TOUCHED_PATH_MISMATCH",
              f"{task_id}:actual={sorted(actual_touched)} declared={sorted(declared_touched)}")
        _fail(not [p for p in actual_touched if _forbidden_path(p)], "STAGE4_FORBIDDEN_PATH_CHANGE", task_id)

        owned = [_normalize_path(x) for x in receipt.get("owned_paths", [])]
        _fail(len(owned) == len(set(owned)), "STAGE4_DUPLICATE_OWNED_PATH", task_id)
        _fail(actual_touched <= set(owned), "STAGE4_CHANGED_PATH_NOT_OWNED", task_id)

        for path in actual_touched:
            prior = touched_owner.get(path)
            _fail(prior is None, "STAGE4_DUPLICATE_TOUCHED_PATH", f"{path}:{prior},{task_id}")
            touched_owner[path] = task_id
            existed = repo.path_exists_at(expected_gatework_base, path)
            if existed:
                _fail(path in spec.exclusive_existing_file_ownership,
                      "STAGE4_EXISTING_FILE_OWNERSHIP_VIOLATION", f"{task_id}:{path}")
            else:
                _fail("stage4" in path.lower(), "STAGE4_NEW_PATH_NOT_NAMESPACED", f"{task_id}:{path}")
            if path == INTEGRATION_ADAPTER_PATH:
                _fail(task_id == "S4E", "STAGE4_INTEGRATION_ADAPTER_OWNER_VIOLATION", task_id)

        for path in owned:
            ownership.append((task_id, path))

        additions = tuple(repo.receipt_addition_commits(receipt_path))
        _fail(len(additions) == 1, "STAGE4_RECEIPT_COMMIT_EVIDENCE_AMBIGUOUS", f"{task_id}:{len(additions)}")
        receipt_commit = additions[0]
        _fail(repo.object_is_commit(receipt_commit), "STAGE4_RECEIPT_COMMIT_NOT_COMMIT", task_id)
        _fail(receipt_commit != head, "STAGE4_RECEIPT_NOT_SEPARATE_COMMIT", task_id)
        _fail(repo.is_ancestor(head, receipt_commit), "STAGE4_TASK_HEAD_NOT_ANCESTOR_OF_RECEIPT", task_id)
        _fail(repo.is_ancestor(receipt_commit, repository_head), "STAGE4_RECEIPT_COMMIT_NOT_IN_EVIDENCE_HEAD", task_id)
        _fail(repo.blob_sha(receipt_commit, receipt_path) == repo.blob_sha(repository_head, receipt_path),
              "STAGE4_RECEIPT_CHANGED_AFTER_CREATION", task_id)
        _fail(repo.path_clean_at_head(receipt_path), "STAGE4_RECEIPT_WORKTREE_DIRTY", task_id)
        receipt_commits[task_id] = receipt_commit

    for index, (task_a, path_a) in enumerate(ownership):
        for task_b, path_b in ownership[index + 1 :]:
            if task_a != task_b and _paths_overlap(path_a, path_b):
                raise GateFailure("STAGE4_OVERLAPPING_TASK_OWNERSHIP", f"{task_a}:{path_a} <> {task_b}:{path_b}")

    for task_id, spec in specs.items():
        if task_id == "S4E":
            _fail(spec.exclusive_existing_file_ownership == frozenset({INTEGRATION_ADAPTER_PATH}),
                  "STAGE4_S4E_EXISTING_FILE_EXCEPTION_MISMATCH")
        else:
            _fail(not spec.exclusive_existing_file_ownership,
                  "STAGE4_UNEXPECTED_EXISTING_FILE_EXCEPTION", task_id)

    result = {
        "schema": "CB16_R11_STAGE4_WAVE1_GATE_ADJUDICATION_V1",
        "status": "PASS",
        "verdict": READY_VERDICT,
        "gatework_base_sha": expected_gatework_base,
        "task_heads": task_heads,
        "receipt_commits": receipt_commits,
        "scientific_status": expected_science,
        "final_cutover_qualified": False,
        "final_cutover_verdict_emitted": False,
    }
    _fail(FORBIDDEN_FINAL_VERDICT not in json.dumps(result, sort_keys=True),
          "STAGE4_FINAL_VERDICT_FORBIDDEN_IN_WAVE1_COMPILER")
    return result


def load_receipts(paths: Iterable[Path], root: Path) -> list[ReceiptInput]:
    root = Path(root).resolve()
    loaded: list[ReceiptInput] = []
    for path in paths:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise GateFailure("STAGE4_RECEIPT_OUTSIDE_REPO", str(path)) from exc
        try:
            doc = json.loads(resolved.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise GateFailure("STAGE4_RECEIPT_UNREADABLE", relative) from exc
        loaded.append(ReceiptInput(relative, doc))
    return loaded
