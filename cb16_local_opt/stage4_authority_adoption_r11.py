from __future__ import annotations

"""Stage-4 one-time authority adoption for CB16 R11.

This module binds an already accepted authority state to a target canonical R11
runtime identity.  It does not create evidence, advance a generation, mutate source
roots, or adjudicate scientific truth.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping

ADOPTION_RECEIPT_SCHEMA = "CB16_R11_STAGE4_AUTHORITY_ADOPTION_RECEIPT_V1"
CANONICAL_HASH_ALGORITHM = "sha256"
FAIL_AFTER_TEMP_FSYNC_BEFORE_PUBLISH = "AFTER_TEMP_FSYNC_BEFORE_PUBLISH"
FAIL_AFTER_PUBLISH_BEFORE_DIRECTORY_FSYNC = "AFTER_PUBLISH_BEFORE_DIRECTORY_FSYNC"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REPO = re.compile(r"^[^/\s]+/[^/\s]+$")
FaultHook = Callable[[str], None]


class AuthorityAdoptionError(RuntimeError):
    """Base class for fail-closed Stage-4 adoption failures."""


class SourceAuthorityMismatch(AuthorityAdoptionError):
    """Observed source authority does not match the accepted source contract."""


class AdoptionReceiptCorrupt(AuthorityAdoptionError):
    """An existing receipt is malformed, non-canonical, or self-inconsistent."""


class AuthorityAdoptionConflict(AuthorityAdoptionError):
    """A different legitimate adoption already occupies the one-time receipt path."""


@dataclass(frozen=True)
class SourceAuthorityIdentity:
    source_repo: str
    source_sha: str
    semantic_freeze_identity: str
    source_generation: int
    champion_identity: str
    champion_hash: str
    checkpoint_identity: str
    checkpoint_hash: str
    evidence_root_identity: str
    journal_head_identity: str
    checkpoint_root_identity: str

    def validate(self) -> None:
        if not _REPO.fullmatch(self.source_repo):
            raise SourceAuthorityMismatch("STAGE4_ADOPTION_SOURCE_REPO_INVALID")
        if not _HEX40.fullmatch(self.source_sha):
            raise SourceAuthorityMismatch("STAGE4_ADOPTION_SOURCE_SHA_INVALID")
        if not _HEX40.fullmatch(self.semantic_freeze_identity):
            raise SourceAuthorityMismatch("STAGE4_ADOPTION_FREEZE_IDENTITY_INVALID")
        if isinstance(self.source_generation, bool) or int(self.source_generation) < 0:
            raise SourceAuthorityMismatch("STAGE4_ADOPTION_SOURCE_GENERATION_INVALID")
        for name in (
            "champion_identity",
            "checkpoint_identity",
            "evidence_root_identity",
            "journal_head_identity",
            "checkpoint_root_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SourceAuthorityMismatch(f"STAGE4_ADOPTION_{name.upper()}_MISSING")
        for name in ("champion_hash", "checkpoint_hash"):
            if not _HEX64.fullmatch(getattr(self, name)):
                raise SourceAuthorityMismatch(f"STAGE4_ADOPTION_{name.upper()}_INVALID")


@dataclass(frozen=True)
class AuthorityAdoptionContract:
    """Exact source state already accepted upstream plus target runtime identity."""

    accepted_source: SourceAuthorityIdentity
    target_r11_authority_identity: str

    def validate(self) -> None:
        self.accepted_source.validate()
        if not isinstance(self.target_r11_authority_identity, str) or not self.target_r11_authority_identity.strip():
            raise SourceAuthorityMismatch("STAGE4_ADOPTION_TARGET_AUTHORITY_IDENTITY_MISSING")

    @classmethod
    def from_mapping(cls, obj: Mapping[str, Any]) -> "AuthorityAdoptionContract":
        try:
            source_obj = obj["accepted_source"]
            target = obj["target_r11_authority_identity"]
            if not isinstance(source_obj, Mapping):
                raise TypeError("accepted_source")
            source = SourceAuthorityIdentity(**dict(source_obj))
            contract = cls(source, str(target))
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceAuthorityMismatch("STAGE4_ADOPTION_CONTRACT_MALFORMED") from exc
        contract.validate()
        return contract


@dataclass(frozen=True)
class AdoptionResult:
    status: str
    canonical_content_hash: str
    receipt_path: str
    adoption_generation: int


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def _source_mapping(source: SourceAuthorityIdentity) -> dict[str, Any]:
    return {
        "source_repo": source.source_repo,
        "source_sha": source.source_sha,
        "semantic_freeze_identity": source.semantic_freeze_identity,
        "source_generation": int(source.source_generation),
        "champion": {
            "identity": source.champion_identity,
            "sha256": source.champion_hash,
        },
        "checkpoint": {
            "identity": source.checkpoint_identity,
            "sha256": source.checkpoint_hash,
        },
        "evidence_root_identity": source.evidence_root_identity,
        "journal_head_identity": source.journal_head_identity,
        "checkpoint_root_identity": source.checkpoint_root_identity,
    }


def _identity_payload(source: SourceAuthorityIdentity, target_r11_authority_identity: str) -> dict[str, Any]:
    """Return the complete authority identity; runtime timestamp is deliberately absent."""

    return {
        "schema": ADOPTION_RECEIPT_SCHEMA,
        "source": _source_mapping(source),
        "target": {
            "r11_authority_identity": target_r11_authority_identity,
            "adoption_generation": int(source.source_generation),
        },
        "semantic_guards": {
            "scientific_history_rewritten": False,
            "new_evidence_created": False,
            "new_scientific_verdict": False,
            "generation_advanced_by_adoption": False,
        },
        "identity_rules": {
            "adoption_timestamp_is_identity_authority": False,
            "canonical_content_hash_algorithm": CANONICAL_HASH_ALGORITHM,
        },
    }


def _normalize_timestamp(timestamp: str | None) -> str:
    if timestamp is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(timestamp, str) or not timestamp:
        raise AuthorityAdoptionError("STAGE4_ADOPTION_TIMESTAMP_INVALID")
    candidate = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise AuthorityAdoptionError("STAGE4_ADOPTION_TIMESTAMP_INVALID") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AuthorityAdoptionError("STAGE4_ADOPTION_TIMESTAMP_MUST_BE_OFFSET_AWARE")
    return timestamp


def build_adoption_receipt(
    source: SourceAuthorityIdentity,
    *,
    target_r11_authority_identity: str,
    adoption_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    source.validate()
    if not isinstance(target_r11_authority_identity, str) or not target_r11_authority_identity.strip():
        raise SourceAuthorityMismatch("STAGE4_ADOPTION_TARGET_AUTHORITY_IDENTITY_MISSING")
    identity = _identity_payload(source, target_r11_authority_identity)
    return {
        **identity,
        "metadata": {"adoption_timestamp_utc": _normalize_timestamp(adoption_timestamp_utc)},
        "canonical_content_hash": _sha256_obj(identity),
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise AdoptionReceiptCorrupt(f"STAGE4_ADOPTION_DUPLICATE_JSON_KEY:{key}")
        out[key] = value
    return out


def _parse_receipt_bytes(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        obj = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, AdoptionReceiptCorrupt) as exc:
        if isinstance(exc, AdoptionReceiptCorrupt):
            raise
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_RECEIPT_JSON_INVALID") from exc
    if not isinstance(obj, dict):
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_RECEIPT_NOT_OBJECT")
    if raw != canonical_json_bytes(obj) + b"\n":
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_RECEIPT_NOT_CANONICALLY_SERIALIZED")
    return obj


def _source_from_receipt(obj: Mapping[str, Any]) -> SourceAuthorityIdentity:
    try:
        source = obj["source"]
        champion = source["champion"]
        checkpoint = source["checkpoint"]
        result = SourceAuthorityIdentity(
            source_repo=source["source_repo"],
            source_sha=source["source_sha"],
            semantic_freeze_identity=source["semantic_freeze_identity"],
            source_generation=source["source_generation"],
            champion_identity=champion["identity"],
            champion_hash=champion["sha256"],
            checkpoint_identity=checkpoint["identity"],
            checkpoint_hash=checkpoint["sha256"],
            evidence_root_identity=source["evidence_root_identity"],
            journal_head_identity=source["journal_head_identity"],
            checkpoint_root_identity=source["checkpoint_root_identity"],
        )
    except (KeyError, TypeError) as exc:
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_RECEIPT_SOURCE_INCOMPLETE") from exc
    try:
        result.validate()
    except SourceAuthorityMismatch as exc:
        raise AdoptionReceiptCorrupt(str(exc)) from exc
    return result


def verify_adoption_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: AuthorityAdoptionContract | None = None,
) -> str:
    """Fail closed unless a receipt is complete, self-consistent and optionally contract-bound."""

    expected_top = {
        "schema",
        "source",
        "target",
        "semantic_guards",
        "identity_rules",
        "metadata",
        "canonical_content_hash",
    }
    if set(receipt) != expected_top:
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_RECEIPT_TOP_LEVEL_FIELDS_INVALID")
    if receipt.get("schema") != ADOPTION_RECEIPT_SCHEMA:
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_RECEIPT_SCHEMA_INVALID")
    source = _source_from_receipt(receipt)
    try:
        target = receipt["target"]
        guards = receipt["semantic_guards"]
        rules = receipt["identity_rules"]
        metadata = receipt["metadata"]
        target_identity = target["r11_authority_identity"]
        adoption_generation = target["adoption_generation"]
        timestamp = metadata["adoption_timestamp_utc"]
    except (KeyError, TypeError) as exc:
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_RECEIPT_REQUIRED_FIELD_MISSING") from exc
    if set(target) != {"r11_authority_identity", "adoption_generation"}:
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_RECEIPT_TARGET_FIELDS_INVALID")
    if set(guards) != {
        "scientific_history_rewritten",
        "new_evidence_created",
        "new_scientific_verdict",
        "generation_advanced_by_adoption",
    }:
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_RECEIPT_GUARD_FIELDS_INVALID")
    if any(guards.values()) or any(value is not False for value in guards.values()):
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_SEMANTIC_GUARD_VIOLATION")
    if rules != {
        "adoption_timestamp_is_identity_authority": False,
        "canonical_content_hash_algorithm": CANONICAL_HASH_ALGORITHM,
    }:
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_IDENTITY_RULES_INVALID")
    if not isinstance(target_identity, str) or not target_identity.strip():
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_TARGET_AUTHORITY_IDENTITY_MISSING")
    if isinstance(adoption_generation, bool) or int(adoption_generation) != int(source.source_generation):
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_GENERATION_CHANGED")
    try:
        _normalize_timestamp(timestamp)
    except AuthorityAdoptionError as exc:
        raise AdoptionReceiptCorrupt(str(exc)) from exc
    identity = _identity_payload(source, target_identity)
    expected_hash = _sha256_obj(identity)
    if receipt.get("canonical_content_hash") != expected_hash:
        raise AdoptionReceiptCorrupt("STAGE4_ADOPTION_CANONICAL_CONTENT_HASH_MISMATCH")
    if contract is not None:
        contract.validate()
        if source != contract.accepted_source:
            raise SourceAuthorityMismatch("STAGE4_ADOPTION_RECEIPT_SOURCE_CONTRACT_MISMATCH")
        if target_identity != contract.target_r11_authority_identity:
            raise SourceAuthorityMismatch("STAGE4_ADOPTION_RECEIPT_TARGET_CONTRACT_MISMATCH")
    return expected_hash


def load_adoption_receipt(
    path: str | Path,
    *,
    contract: AuthorityAdoptionContract | None = None,
) -> dict[str, Any]:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise AdoptionReceiptCorrupt(f"STAGE4_ADOPTION_RECEIPT_READ_FAILED:{p}") from exc
    obj = _parse_receipt_bytes(raw)
    verify_adoption_receipt(obj, contract=contract)
    return obj


def _verify_observed_source(
    observed_source: SourceAuthorityIdentity,
    contract: AuthorityAdoptionContract,
) -> None:
    observed_source.validate()
    contract.validate()
    if observed_source != contract.accepted_source:
        expected = asdict(contract.accepted_source)
        observed = asdict(observed_source)
        mismatches = sorted(k for k in expected if expected[k] != observed[k])
        raise SourceAuthorityMismatch(
            "STAGE4_ADOPTION_OBSERVED_SOURCE_MISMATCH:" + ",".join(mismatches)
        )


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise AuthorityAdoptionError("STAGE4_ADOPTION_DIRECTORY_FSYNC_OPEN_FAILED") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise AuthorityAdoptionError("STAGE4_ADOPTION_DIRECTORY_FSYNC_FAILED") from exc
    finally:
        os.close(fd)


def _publish_no_replace(
    path: Path,
    raw: bytes,
    *,
    fault_hook: FaultHook | None = None,
) -> bool:
    """Publish one complete receipt without ever replacing an existing authority history."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage4-", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if fault_hook is not None:
            fault_hook(FAIL_AFTER_TEMP_FSYNC_BEFORE_PUBLISH)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        except OSError as exc:
            # No replace/rename fallback is allowed: an atomic create-if-absent primitive
            # is part of the one-history safety contract.
            raise AuthorityAdoptionError("STAGE4_ADOPTION_ATOMIC_PUBLISH_UNAVAILABLE") from exc
        if fault_hook is not None:
            fault_hook(FAIL_AFTER_PUBLISH_BEFORE_DIRECTORY_FSYNC)
        _fsync_directory(path.parent)
        return True
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            # An orphan temp is non-authoritative; inability to clean it must not alter
            # the already-published receipt. Startup verification ignores temp files.
            pass


def adopt_authority(
    observed_source: SourceAuthorityIdentity,
    *,
    contract: AuthorityAdoptionContract,
    receipt_path: str | Path,
    adoption_timestamp_utc: str | None = None,
    _fault_hook: FaultHook | None = None,
) -> AdoptionResult:
    """Bind accepted state to canonical R11 without changing scientific generation.

    The only write performed is the adoption receipt itself (and a non-authoritative
    temporary sibling during atomic publication).  Source roots are represented only by
    opaque identities, so this API has no capability to mutate Evidence, Journal, or
    checkpoint storage.
    """

    _verify_observed_source(observed_source, contract)
    candidate = build_adoption_receipt(
        observed_source,
        target_r11_authority_identity=contract.target_r11_authority_identity,
        adoption_timestamp_utc=adoption_timestamp_utc,
    )
    candidate_hash = str(candidate["canonical_content_hash"])
    path = Path(receipt_path)
    raw = canonical_json_bytes(candidate) + b"\n"

    if path.exists():
        existing = load_adoption_receipt(path, contract=contract)
        if existing["canonical_content_hash"] != candidate_hash:
            raise AuthorityAdoptionConflict("STAGE4_ADOPTION_CONFLICTING_SECOND_ADOPTION")
        return AdoptionResult(
            "ALREADY_ADOPTED",
            candidate_hash,
            str(path),
            int(observed_source.source_generation),
        )

    published = _publish_no_replace(path, raw, fault_hook=_fault_hook)
    if published:
        # Re-read exact published bytes before declaring success.
        persisted = load_adoption_receipt(path, contract=contract)
        if persisted["canonical_content_hash"] != candidate_hash:
            raise AuthorityAdoptionConflict("STAGE4_ADOPTION_PUBLISHED_IDENTITY_CONFLICT")
        return AdoptionResult(
            "ADOPTED",
            candidate_hash,
            str(path),
            int(observed_source.source_generation),
        )

    existing = load_adoption_receipt(path, contract=contract)
    if existing["canonical_content_hash"] != candidate_hash:
        raise AuthorityAdoptionConflict("STAGE4_ADOPTION_CONFLICTING_CONCURRENT_ADOPTION")
    return AdoptionResult(
        "ALREADY_ADOPTED",
        candidate_hash,
        str(path),
        int(observed_source.source_generation),
    )
