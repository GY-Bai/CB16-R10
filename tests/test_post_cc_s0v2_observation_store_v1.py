from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from cb16_local_opt.post_cc_observation_store_v1 import (
    ImmutableContentStore,
    ObservationSemanticConflict,
    ObservationStoreCorruption,
    ObservationStoreV1,
)
from tests.cc_s0v2_support import make_canonical_observation_v1


def test_store_round_trip_is_idempotent_and_reopen_safe(tmp_path: Path):
    store = ObservationStoreV1(tmp_path)
    fact = make_canonical_observation_v1()
    first = store.put(fact)
    second = store.put(fact)
    assert first == second
    assert store.count() == 1
    store.close()
    reopened = ObservationStoreV1(tmp_path)
    assert reopened.count() == 1
    assert reopened.get(first.logical_id) == fact
    assert reopened.verify_all() == 1


def test_conflicting_content_for_existing_logical_identity_is_rejected(tmp_path: Path):
    content = ImmutableContentStore(tmp_path, "conflict_test")
    logical_id = "logical-1"
    content.put_bytes(logical_id, b'{"a":1}')
    content.put_bytes(logical_id, b'{"a":1}')
    with pytest.raises(ObservationSemanticConflict, match="LOGICAL_ID_ALREADY_BOUND"):
        content.put_bytes(logical_id, b'{"a":2}')


def test_checksum_corruption_fails_closed(tmp_path: Path):
    store = ObservationStoreV1(tmp_path)
    fact = make_canonical_observation_v1()
    receipt = store.put(fact)
    object_files = list((tmp_path / "observations" / "objects").rglob("*.bin"))
    assert len(object_files) == 1
    object_files[0].write_bytes(b'{"corrupted":true}')
    with pytest.raises(ObservationStoreCorruption, match="CONTENT_OBJECT_(SIZE|HASH)_MISMATCH"):
        store.get(receipt.logical_id)


def test_no_mutation_after_commit_and_deterministic_representation(tmp_path: Path):
    store_a = ObservationStoreV1(tmp_path / "a")
    store_b = ObservationStoreV1(tmp_path / "b")
    fact = make_canonical_observation_v1()
    receipt_a = store_a.put(fact)
    receipt_b = store_b.put(fact)
    assert receipt_a.content_sha256 == receipt_b.content_sha256
    bytes_a = (tmp_path / "a" / "observations" / "objects" / receipt_a.content_sha256[:2] / f"{receipt_a.content_sha256}.bin").read_bytes()
    bytes_b = (tmp_path / "b" / "observations" / "objects" / receipt_b.content_sha256[:2] / f"{receipt_b.content_sha256}.bin").read_bytes()
    assert bytes_a == bytes_b
    store_a.put(fact)
    after = (tmp_path / "a" / "observations" / "objects" / receipt_a.content_sha256[:2] / f"{receipt_a.content_sha256}.bin").read_bytes()
    assert after == bytes_a


def test_process_restart_reconstruction_reads_only_durable_bytes(tmp_path: Path):
    store = ObservationStoreV1(tmp_path)
    fact = make_canonical_observation_v1()
    receipt = store.put(fact)
    del store
    code = (
        "import sys;"
        "from cb16_local_opt.post_cc_observation_store_v1 import ObservationStoreV1;"
        f"s=ObservationStoreV1({str(tmp_path)!r});"
        f"f=s.get({receipt.logical_id!r});"
        "print(f.observation_hash)"
    )
    output = subprocess.check_output([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    assert output == fact.observation_hash
