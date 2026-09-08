#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from cb16_local_opt.stage4_authority_adoption_r11 import (
    AuthorityAdoptionContract,
    SourceAuthorityIdentity,
    adopt_authority,
    canonical_json_bytes,
    load_adoption_receipt,
)

REPO = "GY-Bai/CB16-R10"
R11_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
QUALIFIED_ADOPTION_MODULE_BLOB = "d67d567d2a9a5bc18cbb04a78bb3307c00847b08"
TARGET_AUTHORITY = "CB16_R11_CANONICAL_RUNTIME"
R11_GENERATION = 0

LEGACY_SOURCE_SHA = "f056ae6a0722e3e92d71793024a6e6d3fe9af003"
LEGACY_ACTIVE_AUTHORITY_BLOB = "22aaa0b5bd816e5271a281ae0ad55fe1624497c7"
LEGACY_CAMPAIGN_BLOB = "09fe3c58b5d1d6ba1541367d9591dbff22fd2bdf"
LEGACY_GENERATION = 66
LEGACY_CHAMPION_SEMANTIC_SHA256 = "d0e8c01cc58a1e70936f94f58eeca794eae2e4f89036ec26ccb49ae5d5219886"
LEGACY_CHECKPOINT_SHA256 = "799167bd0a2a820922c03d77563ed3fd22170666819dfbda048cc036141bccfb"
LEGACY_GENERATION_RESULT_SHA256 = "7e60d93f55e2617d561dbd9e3fc835b7d3ac3d6a84b08b72b39a01d49bc41d8d"
LEGACY_SNAPSHOT_CONSUMPTION_SHA256 = "dcbac35f673aa8ef96a9dfd79f5b48bfae47653890b63e402f7fbcd5dcaf8620"
LEGACY_TRAINING_RECEIPT_SHA256 = "0a7c6f27217a6a60c999f90d2a5e0234a8a5d9650c4ade613baa0e39a28a6b03"
LEGACY_ON_POLICY_RECEIPT_SHA256 = "1c7e9819ab325ee34872eb558ab30d9efc41f8f72ff3886ef750e9daa140a07c"
LEGACY_SNAPSHOT_ID = "R102_G66_TRAINING_SNAPSHOT"
LEGACY_SNAPSHOT_SHA256 = "9c800cea6a3c21f0884e529e22a5aaf0fa310f3a3d9688604151d4706271f6ae"
LEGACY_EVIDENCE_OBJECTS = 9714

DATASET_SEAL_SHA256 = "94bc3288a7e6097c4301c93d836607edee207ac3b2c7cf3e7833c85ea0f9546a"
DERIVED_CACHE_LINEAGE_SHA256 = "69e3bb82ba717104422dd45608d6a05725ccc53a08bded621c8f52095ccbb1d6"

LEGACY_ROOT = Path("/data/cb16_hdd/cb16_runtime/R10_4")
R11_ROOT = Path(os.environ.get("CB16_R11_G0_ROOT", "/cb16_r11_runtime/G0"))
AUTHORITY_DIR = R11_ROOT / "authority"
BOOTSTRAP_DIR = R11_ROOT / "bootstrap"
LINEAGE_RECEIPT = AUTHORITY_DIR / "CB16_R11_G0_DERIVED_CACHE_LINEAGE.json"
DATASET_RECEIPT = AUTHORITY_DIR / "CB16_R11_G0_DATASET_SEAL.json"
GENESIS_RECORD = AUTHORITY_DIR / "CB16_R11_G0_GENESIS_SOURCE_RECORD.json"
CONTRACT_PATH = AUTHORITY_DIR / "CB16_R11_G0_AUTHORITY_ADOPTION_CONTRACT.json"
ADOPTION_RECEIPT_PATH = AUTHORITY_DIR / "CB16_R11_G0_AUTHORITY_ADOPTION_RECEIPT.json"
BOOTSTRAP_CHECKPOINT = BOOTSTRAP_DIR / "R11_G0_INITIAL_CHAMPION.pt"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json_bytes(obj) + b"\n"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError(code)


def publish_exact_no_replace(path: Path, raw: bytes, *, readonly_after: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file(), f"R11_G0_EXISTING_PATH_NOT_FILE:{path}")
        require(path.read_bytes() == raw, f"R11_G0_IMMUTABLE_CONFLICT:{path}")
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            require(path.is_file() and path.read_bytes() == raw, f"R11_G0_PUBLISH_RACE_CONFLICT:{path}")
        if readonly_after:
            os.chmod(path, 0o444)
    finally:
        tmp.unlink(missing_ok=True)


def verify_repo_authority() -> str:
    head = git("rev-parse", "HEAD")
    require(len(head) == 40, "R11_G0_HEAD_INVALID")
    require(
        git("rev-parse", "HEAD:authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json") == R11_FREEZE_BLOB,
        "R11_G0_SEMANTIC_FREEZE_DRIFT",
    )
    require(
        git("rev-parse", "HEAD:cb16_local_opt/stage4_authority_adoption_r11.py") == QUALIFIED_ADOPTION_MODULE_BLOB,
        "R11_G0_ADOPTION_MODULE_NOT_QUALIFIED_BLOB",
    )
    subprocess.check_call(["git", "cat-file", "-e", f"{LEGACY_SOURCE_SHA}^{{commit}}"])
    require(
        git("rev-parse", f"{LEGACY_SOURCE_SHA}:ACTIVE_AUTHORITY_R10_2.json") == LEGACY_ACTIVE_AUTHORITY_BLOB,
        "R11_G0_LEGACY_ACTIVE_AUTHORITY_BLOB_DRIFT",
    )
    require(
        git("rev-parse", f"{LEGACY_SOURCE_SHA}:cb16_local_opt/r102_campaign.py") == LEGACY_CAMPAIGN_BLOB,
        "R11_G0_LEGACY_CAMPAIGN_BLOB_DRIFT",
    )
    return head


def verify_strong_lineage() -> dict[str, Any]:
    require(LINEAGE_RECEIPT.is_file(), "R11_G0_STRONG_LINEAGE_RECEIPT_MISSING")
    require(DATASET_RECEIPT.is_file(), "R11_G0_DATASET_RECEIPT_MISSING")
    x = json.loads(LINEAGE_RECEIPT.read_text(encoding="utf-8"))
    require(x.get("schema") == "CB16_R11_G0_DERIVED_CACHE_LINEAGE_V1", "R11_G0_LINEAGE_SCHEMA_MISMATCH")
    require(x.get("status") == "STRONG_LINEAGE", "R11_G0_LINEAGE_NOT_STRONG")
    require(x.get("lineage_identity_sha256") == DERIVED_CACHE_LINEAGE_SHA256, "R11_G0_LINEAGE_IDENTITY_MISMATCH")
    require(x.get("dataset_seal_sha256") == DATASET_SEAL_SHA256, "R11_G0_LINEAGE_DATASET_SEAL_MISMATCH")
    require(x.get("final_holdout_payload_opened") is False, "R11_G0_LINEAGE_HOLDOUT_VIOLATION")
    require(int(x.get("network_reads", -1)) == 0, "R11_G0_LINEAGE_NETWORK_READ_VIOLATION")
    require(len(x.get("symbols", [])) == 10, "R11_G0_LINEAGE_SYMBOL_COUNT_MISMATCH")
    seal = json.loads(DATASET_RECEIPT.read_text(encoding="utf-8"))
    require(seal.get("canonical_seal_sha256") == DATASET_SEAL_SHA256, "R11_G0_DATASET_SEAL_MISMATCH")
    require(seal.get("holdout_payload_files_opened") == 0, "R11_G0_DATASET_HOLDOUT_OPENED")
    require(seal.get("network_reads") == 0, "R11_G0_DATASET_NETWORK_READ_VIOLATION")
    return x


def verify_legacy_g66() -> dict[str, Any]:
    require(LEGACY_ROOT.is_dir(), "R11_G0_LEGACY_ROOT_MISSING")
    require(not os.access(LEGACY_ROOT, os.W_OK), "R11_G0_LEGACY_ROOT_MUST_BE_EFFECTIVELY_READ_ONLY")
    gd = LEGACY_ROOT / "generations" / "G66"
    g67 = LEGACY_ROOT / "generations" / "G67"
    paths = {
        "generation_result": gd / "GENERATION_RESULT.json",
        "snapshot_consumption": gd / "SNAPSHOT_CONSUMPTION_G66.json",
        "training_receipt": gd / "CHALLENGER_TRAINING_RECEIPT_G66.json",
        "on_policy_receipt": gd / "ON_POLICY_REAL_TRACE_RECEIPT.json",
        "checkpoint": gd / "champion_after.pt",
    }
    for p in paths.values():
        require(p.is_file(), f"R11_G0_LEGACY_G66_FILE_MISSING:{p}")
    expected_hashes = {
        "generation_result": LEGACY_GENERATION_RESULT_SHA256,
        "snapshot_consumption": LEGACY_SNAPSHOT_CONSUMPTION_SHA256,
        "training_receipt": LEGACY_TRAINING_RECEIPT_SHA256,
        "on_policy_receipt": LEGACY_ON_POLICY_RECEIPT_SHA256,
        "checkpoint": LEGACY_CHECKPOINT_SHA256,
    }
    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    require(actual_hashes == expected_hashes, f"R11_G0_LEGACY_G66_HASH_MISMATCH:{actual_hashes}")

    gr = json.loads(paths["generation_result"].read_text(encoding="utf-8"))
    sc = json.loads(paths["snapshot_consumption"].read_text(encoding="utf-8"))
    require(gr.get("schema") == "CB16_R10_2_GENERATION_RESULT_V1", "R11_G0_G66_RESULT_SCHEMA_MISMATCH")
    require(gr.get("generation_attempt") == LEGACY_GENERATION, "R11_G0_G66_GENERATION_MISMATCH")
    require(gr.get("tournament", {}).get("decision") == "REJECT", "R11_G0_G66_NOT_REJECT")
    require(gr.get("frozen_authority_unchanged") is True, "R11_G0_G66_FROZEN_AUTHORITY_CHANGED")
    require(gr.get("parent_champion_semantic_sha256") == LEGACY_CHAMPION_SEMANTIC_SHA256, "R11_G0_G66_PARENT_CHAMPION_MISMATCH")
    require(gr.get("champion_after", {}).get("semantic_sha256") == LEGACY_CHAMPION_SEMANTIC_SHA256, "R11_G0_G66_CHAMPION_SEMANTIC_MISMATCH")
    require(gr.get("champion_after", {}).get("file_sha256") == LEGACY_CHECKPOINT_SHA256, "R11_G0_G66_CHECKPOINT_RECEIPT_MISMATCH")
    require(gr.get("training_snapshot", {}).get("snapshot_id") == LEGACY_SNAPSHOT_ID, "R11_G0_G66_SNAPSHOT_ID_MISMATCH")
    require(gr.get("training_snapshot", {}).get("snapshot_hash") == LEGACY_SNAPSHOT_SHA256, "R11_G0_G66_SNAPSHOT_HASH_MISMATCH")
    require(gr.get("training_snapshot", {}).get("evidence_objects") == LEGACY_EVIDENCE_OBJECTS, "R11_G0_G66_EVIDENCE_COUNT_MISMATCH")
    require(gr.get("training", {}).get("amp") is False, "R11_G0_G66_AMP_NOT_FALSE")
    require(sc.get("schema") == "CB16_R10_2_SNAPSHOT_CONSUMPTION_V1", "R11_G0_G66_CONSUMPTION_SCHEMA_MISMATCH")
    require(sc.get("generation") == LEGACY_GENERATION, "R11_G0_G66_CONSUMPTION_GENERATION_MISMATCH")
    require(sc.get("status") == "CONSUMED_EXACTLY_ONCE", "R11_G0_G66_SNAPSHOT_NOT_CONSUMED_EXACTLY_ONCE")
    require(sc.get("snapshot_hash") == LEGACY_SNAPSHOT_SHA256, "R11_G0_G66_CONSUMPTION_HASH_MISMATCH")
    require(g67.is_dir() and not any(g67.iterdir()), "R11_G0_G67_NOT_EMPTY")
    return {"paths": {k: str(v) for k, v in paths.items()}, "hashes": actual_hashes, "generation_result": gr, "snapshot_consumption": sc}


def materialize_bootstrap_checkpoint(source_checkpoint: Path) -> dict[str, Any]:
    raw = source_checkpoint.read_bytes()
    require(len(raw) == 763553, "R11_G0_BOOTSTRAP_CHECKPOINT_SIZE_MISMATCH")
    require(sha256_bytes(raw) == LEGACY_CHECKPOINT_SHA256, "R11_G0_BOOTSTRAP_SOURCE_CHECKPOINT_HASH_MISMATCH")
    publish_exact_no_replace(BOOTSTRAP_CHECKPOINT, raw, readonly_after=True)
    require(sha256_file(BOOTSTRAP_CHECKPOINT) == LEGACY_CHECKPOINT_SHA256, "R11_G0_BOOTSTRAP_CHECKPOINT_POSTWRITE_HASH_MISMATCH")
    root_basis = {
        "schema": "CB16_R11_G0_BOOTSTRAP_CHECKPOINT_ROOT_V1",
        "generation": R11_GENERATION,
        "checkpoint_file": BOOTSTRAP_CHECKPOINT.name,
        "checkpoint_sha256": LEGACY_CHECKPOINT_SHA256,
        "checkpoint_bytes": len(raw),
        "champion_semantic_sha256": LEGACY_CHAMPION_SEMANTIC_SHA256,
        "legacy_predecessor_generation": LEGACY_GENERATION,
        "storage_role": "BOOTSTRAP_ONLY__NOT_YET_STAGE4_DATA_ROOT",
    }
    return {
        "path": str(BOOTSTRAP_CHECKPOINT),
        "root_basis": root_basis,
        "root_identity_sha256": sha256_bytes(canonical_json_bytes(root_basis)),
    }


def main() -> int:
    head = verify_repo_authority()
    lineage = verify_strong_lineage()
    legacy = verify_legacy_g66()
    bootstrap = materialize_bootstrap_checkpoint(Path(legacy["paths"]["checkpoint"]))

    evidence_root_identity = f"legacy-r104-g66-sealed-training-snapshot:sha256:{LEGACY_SNAPSHOT_SHA256}"
    journal_head_identity = f"legacy-r104-g66-last-durable-generation-result:sha256:{LEGACY_GENERATION_RESULT_SHA256}"
    checkpoint_root_identity = f"r11-g0-bootstrap-checkpoint-root:sha256:{bootstrap['root_identity_sha256']}"
    champion_identity = f"r11-g0-champion:semantic-sha256:{LEGACY_CHAMPION_SEMANTIC_SHA256}"
    checkpoint_identity = f"r11-g0-bootstrap-checkpoint:sha256:{LEGACY_CHECKPOINT_SHA256}"

    source = SourceAuthorityIdentity(
        source_repo=REPO,
        source_sha=head,
        semantic_freeze_identity=R11_FREEZE_BLOB,
        source_generation=R11_GENERATION,
        champion_identity=champion_identity,
        champion_hash=LEGACY_CHAMPION_SEMANTIC_SHA256,
        checkpoint_identity=checkpoint_identity,
        checkpoint_hash=LEGACY_CHECKPOINT_SHA256,
        evidence_root_identity=evidence_root_identity,
        journal_head_identity=journal_head_identity,
        checkpoint_root_identity=checkpoint_root_identity,
    )
    contract = AuthorityAdoptionContract(source, TARGET_AUTHORITY)
    contract.validate()

    genesis_core = {
        "schema": "CB16_R11_G0_GENESIS_SOURCE_RECORD_V1",
        "status": "ACCEPTED_R11_G0_SOURCE_STATE",
        "r11": {
            "source_repo": REPO,
            "source_sha": head,
            "semantic_freeze_git_blob": R11_FREEZE_BLOB,
            "generation": R11_GENERATION,
            "target_authority": TARGET_AUTHORITY,
            "champion_identity": champion_identity,
            "champion_semantic_sha256": LEGACY_CHAMPION_SEMANTIC_SHA256,
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_sha256": LEGACY_CHECKPOINT_SHA256,
            "checkpoint_root_identity": checkpoint_root_identity,
            "evidence_root_identity": evidence_root_identity,
            "journal_head_identity": journal_head_identity,
        },
        "legacy_predecessor": {
            "runtime": "R10_4",
            "source_repo": REPO,
            "source_sha": LEGACY_SOURCE_SHA,
            "source_active_authority_git_blob": LEGACY_ACTIVE_AUTHORITY_BLOB,
            "source_campaign_git_blob": LEGACY_CAMPAIGN_BLOB,
            "last_committed_generation": LEGACY_GENERATION,
            "champion_semantic_sha256": LEGACY_CHAMPION_SEMANTIC_SHA256,
            "checkpoint_sha256": LEGACY_CHECKPOINT_SHA256,
            "generation_result_sha256": LEGACY_GENERATION_RESULT_SHA256,
            "snapshot_consumption_sha256": LEGACY_SNAPSHOT_CONSUMPTION_SHA256,
            "training_receipt_sha256": LEGACY_TRAINING_RECEIPT_SHA256,
            "on_policy_receipt_sha256": LEGACY_ON_POLICY_RECEIPT_SHA256,
            "sealed_training_snapshot_id": LEGACY_SNAPSHOT_ID,
            "sealed_training_snapshot_sha256": LEGACY_SNAPSHOT_SHA256,
            "sealed_training_snapshot_evidence_objects": LEGACY_EVIDENCE_OBJECTS,
            "next_generation_G67_empty": True,
            "campaign_completed_100_generations": False,
        },
        "legacy_source_provenance_adjudication": {
            "status": "VERIFIED_FOR_LINEAGE_NOT_USED_AS_R11_SOURCE_SHA",
            "github_actions_run_id": 34075942291,
            "github_actions_job_id": 101612700658,
            "workflow_source_authority_sha": LEGACY_SOURCE_SHA,
            "gated_execute_started_utc": "2026-09-07T03:29:21Z",
            "g66_first_durable_artifact_utc": "2026-09-07T05:50:06.147735Z",
            "g66_generation_result_utc": "2026-09-07T06:03:41.511831Z",
            "continuous_g59_through_g66_sequence_observed": True,
            "competing_r104_github_run_in_0602_0610_utc_observed": False,
            "note": "legacy source SHA is predecessor provenance only; canonical S4C source SHA is the R11 genesis commit above",
        },
        "historical_data": {
            "dataset_seal_sha256": DATASET_SEAL_SHA256,
            "derived_cache_lineage_sha256": DERIVED_CACHE_LINEAGE_SHA256,
            "derived_cache_status": lineage.get("status"),
            "symbols": lineage.get("symbols"),
            "stride_hours": lineage.get("stride_hours"),
            "prehistory_hours": lineage.get("prehistory_hours"),
            "final_holdout_payload_opened": False,
            "network_reads": 0,
        },
        "bootstrap_checkpoint": bootstrap,
        "semantic_guards": {
            "scientific_history_rewritten": False,
            "new_evidence_created": False,
            "new_scientific_verdict": False,
            "generation_advanced_by_initialization": False,
            "legacy_generation_relabelled_as_r11_generation": False,
            "r11_generation_is_fresh_genesis_zero": True,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "training_started": False,
        },
        "scientific_status": "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE",
    }
    genesis = {**genesis_core, "genesis_identity_sha256": sha256_bytes(canonical_json_bytes(genesis_core))}
    publish_exact_no_replace(GENESIS_RECORD, canonical_bytes(genesis))

    contract_mapping = {
        "accepted_source": {
            "source_repo": source.source_repo,
            "source_sha": source.source_sha,
            "semantic_freeze_identity": source.semantic_freeze_identity,
            "source_generation": source.source_generation,
            "champion_identity": source.champion_identity,
            "champion_hash": source.champion_hash,
            "checkpoint_identity": source.checkpoint_identity,
            "checkpoint_hash": source.checkpoint_hash,
            "evidence_root_identity": source.evidence_root_identity,
            "journal_head_identity": source.journal_head_identity,
            "checkpoint_root_identity": source.checkpoint_root_identity,
        },
        "target_r11_authority_identity": TARGET_AUTHORITY,
    }
    publish_exact_no_replace(CONTRACT_PATH, canonical_bytes(contract_mapping))

    adoption = adopt_authority(source, contract=contract, receipt_path=ADOPTION_RECEIPT_PATH)
    receipt = load_adoption_receipt(ADOPTION_RECEIPT_PATH, contract=contract)
    require(receipt["source"]["source_sha"] == head, "R11_G0_ADOPTION_SOURCE_SHA_MISMATCH")
    require(receipt["source"]["semantic_freeze_identity"] == R11_FREEZE_BLOB, "R11_G0_ADOPTION_FREEZE_MISMATCH")
    require(receipt["source"]["source_generation"] == R11_GENERATION, "R11_G0_ADOPTION_GENERATION_MISMATCH")
    require(receipt["target"]["adoption_generation"] == R11_GENERATION, "R11_G0_ADOPTION_TARGET_GENERATION_MISMATCH")
    require(receipt["semantic_guards"]["new_evidence_created"] is False, "R11_G0_ADOPTION_CREATED_EVIDENCE")
    require(receipt["semantic_guards"]["new_scientific_verdict"] is False, "R11_G0_ADOPTION_CREATED_VERDICT")
    require(receipt["semantic_guards"]["generation_advanced_by_adoption"] is False, "R11_G0_ADOPTION_ADVANCED_GENERATION")

    print(json.dumps({
        "schema": "CB16_R11_G0_GENESIS_ADOPTION_RESULT_V1",
        "status": "R11_G0_GENESIS_AUTHORITY_READY",
        "r11_source_sha": head,
        "r11_generation": R11_GENERATION,
        "r11_semantic_freeze_blob": R11_FREEZE_BLOB,
        "target_authority": TARGET_AUTHORITY,
        "legacy_predecessor_source_sha": LEGACY_SOURCE_SHA,
        "legacy_predecessor_generation": LEGACY_GENERATION,
        "champion_semantic_sha256": LEGACY_CHAMPION_SEMANTIC_SHA256,
        "checkpoint_sha256": LEGACY_CHECKPOINT_SHA256,
        "dataset_seal_sha256": DATASET_SEAL_SHA256,
        "derived_cache_lineage_sha256": DERIVED_CACHE_LINEAGE_SHA256,
        "genesis_identity_sha256": genesis["genesis_identity_sha256"],
        "adoption_status": adoption.status,
        "adoption_canonical_content_hash": adoption.canonical_content_hash,
        "new_evidence_created": False,
        "new_scientific_verdict": False,
        "generation_advanced": False,
        "training_started": False,
        "final_holdout_payload_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
