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
