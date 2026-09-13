"""CI-D record-binding verification must pass at the candidate head."""

from __future__ import annotations

import scripts.verify_r11_post_cc_s1_record_binding as binding


def test_record_binding_verification_passes_at_current_head():
    result = binding.verify_record_binding_v1(".")
    assert result["status"] == "PASS", result["checks"]
    assert result["checks"]["implementation_candidate_is_ancestor_of_head"] is True
    assert result["checks"]["execution_manifest_file_sha256_matches"] is True
    assert result["checks"]["execution_manifest_payload_sha256_matches"] is True
    assert result["checks"]["only_review_metadata_changed_since_implementation_candidate"] is True
