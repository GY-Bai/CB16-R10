import pytest

from cb16_local_opt.scientific_qualification_artifact_v1 import (
    QualificationArtifactError,
    build_artifact_manifest_v1,
    verify_artifact_manifest_v1,
)


def test_artifact_manifest_detects_tamper(tmp_path):
    target = tmp_path / "proof.json"
    target.write_text('{"ok":true}\n', encoding="utf-8")
    manifest = build_artifact_manifest_v1(tmp_path, ["proof.json"])
    assert verify_artifact_manifest_v1(tmp_path, manifest)["all_bytes_match"] is True

    target.write_text('{"ok":false}\n', encoding="utf-8")
    result = verify_artifact_manifest_v1(tmp_path, manifest)
    assert result["all_bytes_match"] is False
    assert "ARTIFACT_SHA256_MISMATCH:proof.json" in result["contract_violations"]


def test_artifact_manifest_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-proof.json"
    outside.write_text('{"outside":true}\n', encoding="utf-8")
    link = tmp_path / "escape.json"
    link.symlink_to(outside)
    with pytest.raises(QualificationArtifactError, match="ARTIFACT_ESCAPES_ROOT"):
        build_artifact_manifest_v1(tmp_path, ["escape.json"])


def test_artifact_manifest_rejects_null_required_path(tmp_path):
    (tmp_path / "None").write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(QualificationArtifactError, match="ARTIFACT_PATH_INVALID"):
        build_artifact_manifest_v1(tmp_path, [None])


def test_artifact_manifest_rejects_plain_string_container(tmp_path):
    (tmp_path / "proof.json").write_text("proof\n", encoding="utf-8")
    with pytest.raises(QualificationArtifactError, match="ARTIFACT_REQUIREMENTS_CONTAINER_INVALID"):
        build_artifact_manifest_v1(tmp_path, "proof.json")


def test_artifact_verifier_rejects_null_entry_path(tmp_path):
    (tmp_path / "None").write_text("sentinel\n", encoding="utf-8")
    manifest = {
        "schema": "CB16_QUALIFICATION_ARTIFACT_BYTE_MANIFEST_V1",
        "entries": [{"path": None, "size_bytes": 9, "sha256": "not-used"}],
    }
    with pytest.raises(QualificationArtifactError, match="ARTIFACT_PATH_INVALID"):
        verify_artifact_manifest_v1(tmp_path, manifest)
