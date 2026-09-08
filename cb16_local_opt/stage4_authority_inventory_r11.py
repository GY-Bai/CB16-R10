from __future__ import annotations

"""Stage-4 S4A fail-closed authority mutation surface inventory.

This module is deliberately read-only. It discovers repository-local code paths that
can directly or indirectly mutate an authority-bearing R11/R10 state, then requires
an explicit registry classification for every discovered path+symbol. Call
reachability never upgrades a legacy or qualification surface into canonical authority.
"""

import ast
import copy
import json
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

GATEWORK_BASE_SHA = "0f18e08ec7250b9b4e45c62803c25be966834390"
SEMANTIC_FREEZE_REL = "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
SEMANTIC_FREEZE_BLOB_SHA = "3c401a0a350984381912f7860181e3e96eb8d7cf"
SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
REGISTRY_SCHEMA = "CB16_R11_STAGE4_AUTHORITY_WRITER_REGISTRY_V1"
AUDIT_SCHEMA = "CB16_R11_STAGE4_AUTHORITY_SURFACE_AUDIT_V1"


class Classification(str, Enum):
    R11_CANONICAL_CANDIDATE = "R11_CANONICAL_CANDIDATE"
    LEGACY_REFERENCE_ONLY = "LEGACY_REFERENCE_ONLY"
    QUALIFICATION_ONLY = "QUALIFICATION_ONLY"
    TEST_ONLY = "TEST_ONLY"
    NON_AUTHORITATIVE_READ_ONLY = "NON_AUTHORITATIVE_READ_ONLY"
    UNKNOWN_AUTHORITY = "UNKNOWN_AUTHORITY"


class AuthorityDomain(str, Enum):
    EVIDENCE_MINT_ADMISSION = "EVIDENCE_MINT_ADMISSION"
    EVENT_JOURNAL = "EVENT_JOURNAL"
    TRAINING_SNAPSHOT_SEAL = "TRAINING_SNAPSHOT_SEAL"
    CHALLENGER_CREATION = "CHALLENGER_CREATION"
    TOURNAMENT_RESULT = "TOURNAMENT_RESULT"
    CHAMPION_COMMIT_PROMOTION_REJECTION = "CHAMPION_COMMIT_PROMOTION_REJECTION"
    CHECKPOINT_SEAL = "CHECKPOINT_SEAL"
    GENERATION_ADVANCEMENT_RELEASE = "GENERATION_ADVANCEMENT_RELEASE"
    RUNTIME_AUTHORITY_OWNERSHIP = "RUNTIME_AUTHORITY_OWNERSHIP"
    PERMISSION_GRANT = "PERMISSION_GRANT"
    PHYSICS_ACCOUNT_TRANSITION = "PHYSICS_ACCOUNT_TRANSITION"


CLASSIFICATIONS = frozenset(x.value for x in Classification)
DOMAINS = frozenset(x.value for x in AuthorityDomain)
PRODUCTION_ROLES = frozenset({"PRODUCTION_AUTHORITY", "CANONICAL_RUNTIME_AUTHORITY"})

# Explicit APIs are mutation discovery signals, never authority declarations.
_CALL_DOMAIN_MAP: Mapping[str, tuple[str, ...]] = {
    "put_evidence": (AuthorityDomain.EVIDENCE_MINT_ADMISSION.value,),
    "put_once": (AuthorityDomain.EVIDENCE_MINT_ADMISSION.value,),
    "seal_evidence_set": (AuthorityDomain.EVIDENCE_MINT_ADMISSION.value,),
    "_append_payload": (AuthorityDomain.EVIDENCE_MINT_ADMISSION.value,),
    "_seal_active": (AuthorityDomain.EVIDENCE_MINT_ADMISSION.value,),
    "seal_all_active_segments": (AuthorityDomain.EVIDENCE_MINT_ADMISSION.value,),
    "recover_active_tails": (AuthorityDomain.EVIDENCE_MINT_ADMISSION.value,),
    "accept_teacher_evidence": (AuthorityDomain.EVIDENCE_MINT_ADMISSION.value,),
    "materialize_trace_evidence": (AuthorityDomain.EVIDENCE_MINT_ADMISSION.value,),
    "seal_trace_batch": (AuthorityDomain.EVENT_JOURNAL.value,),
    "seal_generation_outcome": (
        AuthorityDomain.EVENT_JOURNAL.value,
        AuthorityDomain.CHAMPION_COMMIT_PROMOTION_REJECTION.value,
        AuthorityDomain.GENERATION_ADVANCEMENT_RELEASE.value,
    ),
    "append_once": (AuthorityDomain.EVENT_JOURNAL.value,),
    "_persist": (AuthorityDomain.EVENT_JOURNAL.value,),
    "seal_snapshot": (AuthorityDomain.TRAINING_SNAPSHOT_SEAL.value,),
    "seal_training_snapshot": (AuthorityDomain.TRAINING_SNAPSHOT_SEAL.value,),
    "train_challenger": (AuthorityDomain.CHALLENGER_CREATION.value,),
    "train_challenger_r2": (AuthorityDomain.CHALLENGER_CREATION.value,),
    "complete_training": (AuthorityDomain.CHALLENGER_CREATION.value,),
    "_train_step": (AuthorityDomain.CHALLENGER_CREATION.value,),
    "train_one_step": (AuthorityDomain.CHALLENGER_CREATION.value,),
    "decide_tournament": (AuthorityDomain.TOURNAMENT_RESULT.value,),
    "atomic_commit": (AuthorityDomain.CHAMPION_COMMIT_PROMOTION_REJECTION.value,),
    "commit_tournament": (AuthorityDomain.CHAMPION_COMMIT_PROMOTION_REJECTION.value,),
    "put_state_dict": (AuthorityDomain.CHECKPOINT_SEAL.value,),
    "seal_generation_checkpoint": (
        AuthorityDomain.CHECKPOINT_SEAL.value,
        AuthorityDomain.CHAMPION_COMMIT_PROMOTION_REJECTION.value,
    ),
    "seal_checkpoint": (AuthorityDomain.CHECKPOINT_SEAL.value,),
    "release_next_generation": (AuthorityDomain.GENERATION_ADVANCEMENT_RELEASE.value,),
    "acquire_authority": (AuthorityDomain.RUNTIME_AUTHORITY_OWNERSHIP.value,),
    "release_authority": (AuthorityDomain.RUNTIME_AUTHORITY_OWNERSHIP.value,),
    "validate_fencing_token": (AuthorityDomain.RUNTIME_AUTHORITY_OWNERSHIP.value,),
    "assert_fencing_token": (AuthorityDomain.RUNTIME_AUTHORITY_OWNERSHIP.value,),
    "grant_permission": (AuthorityDomain.PERMISSION_GRANT.value,),
    "authorize_action": (AuthorityDomain.PERMISSION_GRANT.value,),
    "apply_permission": (AuthorityDomain.PERMISSION_GRANT.value,),
    "account_transition": (AuthorityDomain.PHYSICS_ACCOUNT_TRANSITION.value,),
    "physics_transition": (AuthorityDomain.PHYSICS_ACCOUNT_TRANSITION.value,),
    "execute_transition": (AuthorityDomain.PHYSICS_ACCOUNT_TRANSITION.value,),
}

_MUTATION_VERB_RE = re.compile(
    r"(?:^|_)(?:put|append|seal|commit|promote|reject|release|advance|train|materialize|"
    r"accept|grant|authorize|transition|step|execute|recover|write|persist|consume)(?:_|$)",
    re.IGNORECASE,
)
_CONTEXT_DOMAIN_TERMS: Mapping[str, tuple[str, ...]] = {
    AuthorityDomain.EVIDENCE_MINT_ADMISSION.value: ("evidence",),
    AuthorityDomain.EVENT_JOURNAL.value: ("journal", "event"),
    AuthorityDomain.TRAINING_SNAPSHOT_SEAL.value: ("snapshot",),
    AuthorityDomain.CHALLENGER_CREATION.value: ("challenger", "training"),
    AuthorityDomain.TOURNAMENT_RESULT.value: ("tournament",),
    AuthorityDomain.CHAMPION_COMMIT_PROMOTION_REJECTION.value: ("champion", "promotion", "promote", "reject"),
    AuthorityDomain.CHECKPOINT_SEAL.value: ("checkpoint",),
    AuthorityDomain.GENERATION_ADVANCEMENT_RELEASE.value: ("generation",),
    AuthorityDomain.RUNTIME_AUTHORITY_OWNERSHIP.value: ("fencing", "lease", "runtime_authority", "active_authority"),
    AuthorityDomain.PERMISSION_GRANT.value: ("permission", "allowed_action", "supervisor"),
    AuthorityDomain.PHYSICS_ACCOUNT_TRANSITION.value: ("physics", "account_transition"),
}

_RELEVANT_SUFFIXES = frozenset({".py", ".yml", ".yaml", ".sh"})
_EXCLUDED_DIR_PARTS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", "third_party"})


@dataclass(frozen=True)
class DiscoveredSurface:
    path: str
    symbol: str
    line: int
    authority_domains: tuple[str, ...]
    mutation_capabilities: tuple[str, ...]
    discovery_evidence: tuple[str, ...]
    caller_evidence: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.path, self.symbol

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "line": self.line,
            "authority_domains": list(self.authority_domains),
            "mutation_capabilities": list(self.mutation_capabilities),
            "discovery_evidence": list(self.discovery_evidence),
            "caller_evidence": list(self.caller_evidence),
        }


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _iter_scannable_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _RELEVANT_SUFFIXES:
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in _EXCLUDED_DIR_PARTS for part in rel_parts):
            continue
        yield path


def _attribute_leaf(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _parent_symbols(tree: ast.AST) -> dict[ast.AST, str]:
    result: dict[ast.AST, str] = {}

    def visit(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = child.name if not prefix else f"{prefix}.{child.name}"
                result[child] = name
                visit(child, name)
            else:
                visit(child, prefix)

    visit(tree)
    return result


def _context_domains(text: str) -> set[str]:
    low = text.lower()
    out: set[str] = set()
    for domain, terms in _CONTEXT_DOMAIN_TERMS.items():
        if any(term in low for term in terms):
            out.add(domain)
    return out


def _python_surfaces(path: Path, root: Path) -> tuple[list[DiscoveredSurface], list[tuple[str, int, str]]]:
    rel = _rel(root, path)
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=rel)
    except (UnicodeDecodeError, SyntaxError) as exc:
        return [DiscoveredSurface(rel, "<UNPARSEABLE_PYTHON>", 1, tuple(), ("UNPARSEABLE_PYTHON",), (repr(exc),), tuple())], []

    symbols = _parent_symbols(tree)
    calls: list[tuple[str, int, str]] = []
    surfaces: list[DiscoveredSurface] = []
    for node, symbol in symbols.items():
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        leaf = node.name
        called: list[tuple[str, int]] = []
        string_constants: list[str] = []
        has_optimizer_step = False
        persistent_write = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_leaf = _attribute_leaf(child.func)
                if call_leaf:
                    called.append((call_leaf, getattr(child, "lineno", node.lineno)))
                    calls.append((call_leaf, getattr(child, "lineno", node.lineno), symbol))
                    if call_leaf == "step" and isinstance(child.func, ast.Attribute):
                        base = child.func.value
                        if isinstance(base, ast.Name) and "optim" in base.id.lower():
                            has_optimizer_step = True
                    if call_leaf in {"write", "write_text", "write_bytes", "replace", "save", "dump", "execute", "executemany"}:
                        persistent_write = True
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                string_constants.append(child.value)

        domains: set[str] = set()
        evidence: set[str] = set()
        caps: set[str] = set()
        if leaf in _CALL_DOMAIN_MAP:
            domains.update(_CALL_DOMAIN_MAP[leaf])
            evidence.add(f"SYMBOL_NAME:{leaf}")
            caps.add(leaf)
        for call_leaf, line in called:
            if call_leaf in _CALL_DOMAIN_MAP:
                domains.update(_CALL_DOMAIN_MAP[call_leaf])
                evidence.add(f"CALL:{call_leaf}@L{line}")
                caps.add(call_leaf)
        if has_optimizer_step:
            domains.add(AuthorityDomain.CHALLENGER_CREATION.value)
            evidence.add("CALL:optimizer.step")
            caps.add("MODEL_PARAMETER_MUTATION")

        context = " ".join([rel, symbol, leaf, *string_constants])
        context_domains = _context_domains(context)
        if persistent_write and context_domains:
            domains.update(context_domains)
            evidence.add("PERSISTENT_WRITE_WITH_AUTHORITY_CONTEXT")
            caps.add("PERSISTENT_STATE_MUTATION")
        if _MUTATION_VERB_RE.search(leaf) and context_domains:
            domains.update(context_domains)
            evidence.add("MUTATION_VERB_WITH_AUTHORITY_CONTEXT")
            caps.add(leaf)

        if domains:
            surfaces.append(
                DiscoveredSurface(
                    path=rel,
                    symbol=symbol,
                    line=int(getattr(node, "lineno", 1)),
                    authority_domains=tuple(sorted(domains)),
                    mutation_capabilities=tuple(sorted(caps)),
                    discovery_evidence=tuple(sorted(evidence)),
                    caller_evidence=tuple(),
                )
            )
    return surfaces, calls


def _workflow_surface(path: Path, root: Path) -> DiscoveredSurface | None:
    rel = _rel(root, path)
    text = path.read_text(encoding="utf-8", errors="replace")
    low = text.lower()
    if not (
        "run_r11_stage2" in low
        or "run_r11_stage3" in low
        or "run_r11_task_f" in low
        or "run_r11_teacher" in low
        or "r11-stage2" in rel.lower()
        or "r11-stage3" in rel.lower()
    ):
        return None
    domains = _context_domains(text)
    if "integration" in low or "burst" in low or "crash" in low or "soak" in low:
        domains.update(
            {
                AuthorityDomain.EVIDENCE_MINT_ADMISSION.value,
                AuthorityDomain.EVENT_JOURNAL.value,
                AuthorityDomain.TRAINING_SNAPSHOT_SEAL.value,
                AuthorityDomain.CHALLENGER_CREATION.value,
                AuthorityDomain.TOURNAMENT_RESULT.value,
                AuthorityDomain.CHAMPION_COMMIT_PROMOTION_REJECTION.value,
                AuthorityDomain.CHECKPOINT_SEAL.value,
                AuthorityDomain.GENERATION_ADVANCEMENT_RELEASE.value,
            }
        )
    if not domains:
        domains.add(AuthorityDomain.RUNTIME_AUTHORITY_OWNERSHIP.value)
    return DiscoveredSurface(
        path=rel,
        symbol=f"workflow::{path.name}",
        line=1,
        authority_domains=tuple(sorted(domains)),
        mutation_capabilities=("INVOKES_R11_QUALIFICATION_EXECUTION",),
        discovery_evidence=("R11_STAGE2_STAGE3_QUALIFICATION_WORKFLOW",),
        caller_evidence=("GITHUB_ACTIONS_ENTRYPOINT",),
    )


def discover_authority_surfaces(repo_root: str | Path) -> list[DiscoveredSurface]:
    root = Path(repo_root).resolve()
    raw: list[DiscoveredSurface] = []
    callsites: dict[str, list[str]] = {}
    for path in _iter_scannable_files(root):
        if path.suffix.lower() == ".py":
            surfaces, calls = _python_surfaces(path, root)
            raw.extend(surfaces)
            rel = _rel(root, path)
            for call_leaf, line, caller_symbol in calls:
                callsites.setdefault(call_leaf, []).append(f"{rel}:L{line}:{caller_symbol}")
        elif path.suffix.lower() in {".yml", ".yaml"}:
            surface = _workflow_surface(path, root)
            if surface is not None:
                raw.append(surface)

    out: list[DiscoveredSurface] = []
    for surface in raw:
        leaf = surface.symbol.rsplit(".", 1)[-1]
        callers = tuple(sorted(set(callsites.get(leaf, ()))))
        if not callers:
            callers = surface.caller_evidence or ("EXPORTED_OR_DIRECT_ENTRYPOINT",)
        out.append(
            DiscoveredSurface(
                path=surface.path,
                symbol=surface.symbol,
                line=surface.line,
                authority_domains=surface.authority_domains,
                mutation_capabilities=surface.mutation_capabilities,
                discovery_evidence=surface.discovery_evidence,
                caller_evidence=callers,
            )
        )

    unique: dict[tuple[str, str], DiscoveredSurface] = {}
    for surface in out:
        prior = unique.get(surface.key)
        if prior is not None and prior != surface:
            raise RuntimeError(f"STAGE4_DUPLICATE_DISCOVERY_CONFLICT:{surface.key}")
        unique[surface.key] = surface
    return [unique[k] for k in sorted(unique)]


def load_registry(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("STAGE4_REGISTRY_NOT_OBJECT")
    return obj


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, str]:
    return str(entry.get("path", "")), str(entry.get("symbol", ""))


def audit_registry(
    repo_root: str | Path,
    registry: Mapping[str, Any],
    *,
    verify_git_guards: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    problems: list[str] = []
    if registry.get("schema") != REGISTRY_SCHEMA:
        problems.append("REGISTRY_SCHEMA_MISMATCH")
    if registry.get("gatework_base_sha") != GATEWORK_BASE_SHA:
        problems.append("GATEWORK_BASE_MISMATCH")
    if registry.get("semantic_freeze_blob_sha") != SEMANTIC_FREEZE_BLOB_SHA:
        problems.append("SEMANTIC_FREEZE_IDENTITY_MISMATCH")
    if registry.get("scientific_status") != SCIENTIFIC_STATUS:
        problems.append("SCIENTIFIC_STATUS_CHANGED")

    discoveries = discover_authority_surfaces(root)
    discovered = {x.key: x for x in discoveries}
    entries_raw = registry.get("entries")
    if not isinstance(entries_raw, list):
        entries_raw = []
        problems.append("REGISTRY_ENTRIES_NOT_LIST")
    entries: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries_raw:
        if not isinstance(entry, Mapping):
            problems.append("REGISTRY_ENTRY_NOT_OBJECT")
            continue
        key = _entry_key(entry)
        if not all(key):
            problems.append(f"REGISTRY_ENTRY_KEY_MISSING:{key}")
            continue
        if key in entries:
            problems.append(f"MULTIPLE_CLASSIFICATIONS:{key[0]}:{key[1]}")
            continue
        entries[key] = entry
        classification = str(entry.get("classification", ""))
        if classification not in CLASSIFICATIONS:
            problems.append(f"INVALID_CLASSIFICATION:{key[0]}:{key[1]}:{classification}")
        if classification == Classification.UNKNOWN_AUTHORITY.value:
            problems.append(f"UNKNOWN_AUTHORITY:{key[0]}:{key[1]}")
        domains = entry.get("authority_domains")
        if not isinstance(domains, list) or not domains or any(str(x) not in DOMAINS for x in domains):
            problems.append(f"INVALID_AUTHORITY_DOMAINS:{key[0]}:{key[1]}")
        callers = entry.get("caller_callsite_evidence")
        if not isinstance(callers, list) or not callers:
            problems.append(f"MISSING_CALLSITE_EVIDENCE:{key[0]}:{key[1]}")
        role = str(entry.get("current_role", ""))
        eligible = entry.get("eligible_for_canonical_authority") is True
        if classification in {Classification.QUALIFICATION_ONLY.value, Classification.TEST_ONLY.value}:
            if role in PRODUCTION_ROLES or eligible:
                problems.append(f"NONPRODUCTION_WRITER_ESCALATION:{key[0]}:{key[1]}")
        if classification == Classification.LEGACY_REFERENCE_ONLY.value:
            if role in PRODUCTION_ROLES or eligible:
                problems.append(f"LEGACY_REACHABILITY_ESCALATION:{key[0]}:{key[1]}")
        if classification == Classification.NON_AUTHORITATIVE_READ_ONLY.value and eligible:
            problems.append(f"READ_ONLY_ESCALATION:{key[0]}:{key[1]}")
        if classification == Classification.R11_CANONICAL_CANDIDATE.value and role == "CANONICAL_RUNTIME_AUTHORITY":
            problems.append(f"PREMATURE_STAGE4_CANONICAL_CLAIM:{key[0]}:{key[1]}")

    for key, surface in discovered.items():
        entry = entries.get(key)
        if entry is None:
            problems.append(f"UNKNOWN_AUTHORITY:{key[0]}:{key[1]}")
            continue
        reg_domains = set(str(x) for x in entry.get("authority_domains", ()))
        if not set(surface.authority_domains).issubset(reg_domains):
            problems.append(
                f"DOMAIN_COVERAGE_MISMATCH:{key[0]}:{key[1]}:"
                f"discovered={sorted(surface.authority_domains)}:registered={sorted(reg_domains)}"
            )
    for key in sorted(set(entries) - set(discovered)):
        problems.append(f"STALE_OR_UNDISCOVERED_REGISTRY_ENTRY:{key[0]}:{key[1]}")

    if verify_git_guards:
        try:
            freeze_blob = subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{SEMANTIC_FREEZE_REL}"], cwd=root, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            problems.append(f"GIT_GUARD_UNAVAILABLE:{exc}")
        else:
            if freeze_blob != SEMANTIC_FREEZE_BLOB_SHA:
                problems.append("SEMANTIC_FREEZE_BLOB_CHANGED")
            try:
                changed = subprocess.check_output(
                    ["git", "diff", "--name-only", f"{GATEWORK_BASE_SHA}...HEAD"], cwd=root, text=True
                ).splitlines()
            except subprocess.CalledProcessError as exc:
                problems.append(f"GIT_DIFF_GUARD_FAILED:{exc}")
            else:
                for path in changed:
                    low = path.lower()
                    if path == SEMANTIC_FREEZE_REL or "final_holdout" in low or "2025-09" in low:
                        problems.append(f"FORBIDDEN_STAGE4_PATH_CHANGE:{path}")
                    if path.startswith("provision/assets/binance_usdm_1m_funding_2020_2026"):
                        problems.append(f"FROZEN_MARKET_DATA_PATH_CHANGE:{path}")
                    exists_at_base = subprocess.run(
                        ["git", "cat-file", "-e", f"{GATEWORK_BASE_SHA}:{path}"],
                        cwd=root,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    ).returncode == 0
                    if exists_at_base:
                        problems.append(f"S4A_EXISTING_FILE_MODIFIED:{path}")
                    elif "stage4" not in low:
                        problems.append(f"S4A_NEW_PATH_NOT_STAGE4_NAMESPACED:{path}")

    counts: dict[str, int] = {x.value: 0 for x in Classification}
    for entry in entries.values():
        classification = str(entry.get("classification", ""))
        if classification in counts:
            counts[classification] += 1

    unknowns = sorted(x for x in problems if x.startswith("UNKNOWN_AUTHORITY:"))
    return {
        "schema": AUDIT_SCHEMA,
        "status": "PASS" if not problems else "FAIL",
        "pass": not problems,
        "gatework_base_sha": GATEWORK_BASE_SHA,
        "semantic_freeze_blob_sha": SEMANTIC_FREEZE_BLOB_SHA,
        "scientific_status": SCIENTIFIC_STATUS,
        "scientific_semantics_changed": False,
        "new_scientific_verdict": False,
        "new_scientific_evidence_created": False,
        "final_holdout_opened": False,
        "fresh_market_data_downloaded": False,
        "discovered_writer_count": len(discoveries),
        "registered_writer_count": len(entries),
        "classification_counts": counts,
        "unknown_authority_count": len(unknowns),
        "problems": sorted(set(problems)),
        "discoveries": [x.as_dict() for x in discoveries],
    }


def registry_without_entry(registry: Mapping[str, Any], key: tuple[str, str]) -> dict[str, Any]:
    out = copy.deepcopy(dict(registry))
    out["entries"] = [x for x in out.get("entries", []) if _entry_key(x) != key]
    return out


def require_pass(report: Mapping[str, Any]) -> None:
    if report.get("pass") is not True:
        raise RuntimeError("STAGE4_AUTHORITY_SURFACE_AUDIT_FAILED:" + json.dumps(report.get("problems", []), sort_keys=True))


__all__ = [
    "AUDIT_SCHEMA",
    "AuthorityDomain",
    "Classification",
    "DiscoveredSurface",
    "GATEWORK_BASE_SHA",
    "REGISTRY_SCHEMA",
    "SCIENTIFIC_STATUS",
    "SEMANTIC_FREEZE_BLOB_SHA",
    "audit_registry",
    "discover_authority_surfaces",
    "load_registry",
    "registry_without_entry",
    "require_pass",
]
