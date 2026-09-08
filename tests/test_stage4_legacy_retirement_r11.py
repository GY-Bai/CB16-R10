from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cb16_local_opt.stage4_legacy_retirement_r11 import (
    AUTHORITATIVE_CAPABILITIES,
    NON_AUTHORITATIVE_ROLES,
    ActiveCapabilityClaim,
    ActiveLegacyWriterDetected,
    Capability,
    CapabilityDenied,
    DeclaredCapabilityClaim,
    InvalidRetirementToken,
    InvalidRuntimeIdentity,
    LegacyAuthorityDenied,
    LegacyFallbackDenied,
    LegacyRetirementGuard,
    RuntimeRole,
    SCIENTIFIC_STATUS,
    audit_declared_claims,
    policy_manifest,
    self_audit_policy,
)

ROOT = Path(__file__).resolve().parents[1]
KEY = b"stage4-s4g-unit-test-signing-key-32bytes-plus"
EXPECTED_SCIENCE = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"


def guard() -> LegacyRetirementGuard:
    return LegacyRetirementGuard(KEY, issuer_id="S4G_UNIT_TEST_ISSUER")


def test_authoritative_capability_set_is_complete_and_exactly_nine():
    assert {x.value for x in AUTHORITATIVE_CAPABILITIES} == {
        "ACQUIRE_CANONICAL_RUNTIME_AUTHORITY",
        "MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE",
        "APPEND_AUTHORITATIVE_EVENT_JOURNAL",
        "SEAL_TRAINING_SNAPSHOT",
        "TRANSITION_CHALLENGER_CHAMPION",
        "SEAL_CHECKPOINT",
        "RELEASE_GENERATION",
        "GRANT_PERMISSION",
        "EXECUTE_ACCOUNT_PHYSICS_TRANSITION",
    }


@pytest.mark.parametrize("role", sorted(NON_AUTHORITATIVE_ROLES, key=lambda x: x.value))
@pytest.mark.parametrize("capability", sorted(AUTHORITATIVE_CAPABILITIES, key=lambda x: x.value))
def test_every_noncanonical_role_is_denied_every_authoritative_capability(role, capability):
    g = guard()
    identity = g.issue_identity(f"subject-{role.value}", role)
    with pytest.raises(LegacyAuthorityDenied):
        g.issue_token(identity, capability)


def test_legacy_reference_and_oracle_remain_readable_and_comparable():
    g = guard()
    reference = g.issue_identity("legacy-ref", RuntimeRole.LEGACY_REFERENCE)
    oracle = g.issue_identity("legacy-oracle", RuntimeRole.LEGACY_ORACLE)
    ref_token = g.issue_token(reference, Capability.READ_LEGACY_REFERENCE)
    oracle_token = g.issue_token(oracle, Capability.COMPARE_LEGACY_ORACLE)
    assert g.assert_token(reference, Capability.READ_LEGACY_REFERENCE, ref_token).authority_granted is False
    assert g.assert_token(oracle, Capability.COMPARE_LEGACY_ORACLE, oracle_token).authority_granted is False


def test_legacy_migration_module_remains_importable_as_non_authoritative_reference():
    module = importlib.import_module("cb16_local_opt.r2_legacy_migration")
    assert hasattr(module, "semantic_projection")


def test_legacy_replay_is_engineering_only_and_cannot_mint_evidence():
    g = guard()
    identity = g.issue_identity("legacy-replay", RuntimeRole.LEGACY_REPLAY)
    replay_token = g.issue_token(identity, Capability.ENGINEERING_REPLAY)
    decision = g.assert_token(identity, Capability.ENGINEERING_REPLAY, replay_token)
    assert decision.authority_granted is False
    with pytest.raises(LegacyAuthorityDenied):
        g.issue_token(identity, Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE)


def test_compatibility_reader_can_translate_read_only_but_not_write():
    g = guard()
    identity = g.issue_identity("legacy-compat", RuntimeRole.LEGACY_COMPATIBILITY_READER)
    token = g.issue_token(identity, Capability.READ_HISTORICAL_COMPATIBILITY)
    assert g.assert_token(identity, Capability.READ_HISTORICAL_COMPATIBILITY, token).authority_granted is False
    with pytest.raises(LegacyAuthorityDenied):
        g.issue_token(identity, Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL)


@pytest.mark.parametrize("role", [RuntimeRole.DIAGNOSTIC, RuntimeRole.TEST])
def test_diagnostic_and_test_roles_cannot_escalate_to_production_authority(role):
    g = guard()
    identity = g.issue_identity(role.value.lower(), role)
    diagnostic = g.issue_token(identity, Capability.RUN_DIAGNOSTICS)
    assert g.assert_token(identity, Capability.RUN_DIAGNOSTICS, diagnostic).authority_granted is False
    for capability in AUTHORITATIVE_CAPABILITIES:
        with pytest.raises(LegacyAuthorityDenied):
            g.issue_token(identity, capability)


def test_role_specific_safe_capability_is_still_fail_closed():
    g = guard()
    reference = g.issue_identity("legacy-ref", RuntimeRole.LEGACY_REFERENCE)
    with pytest.raises(CapabilityDenied):
        g.issue_token(reference, Capability.ENGINEERING_REPLAY)


def test_failed_r11_startup_never_falls_back_to_legacy_authority():
    g = guard()
    legacy = g.issue_identity("old-runtime", RuntimeRole.LEGACY_REFERENCE)
    with pytest.raises(LegacyFallbackDenied):
        g.require_canonical_startup(
            canonical_startup_succeeded=False,
            proposed_fallback=legacy,
        )


def test_successful_r11_startup_rejects_proposed_legacy_fallback_too():
    g = guard()
    legacy = g.issue_identity("old-runtime", RuntimeRole.LEGACY_ORACLE)
    with pytest.raises(LegacyFallbackDenied):
        g.require_canonical_startup(
            canonical_startup_succeeded=True,
            proposed_fallback=legacy,
        )


def test_successful_r11_startup_only_continues_to_independent_authority_gates():
    g = guard()
    result = g.require_canonical_startup(canonical_startup_succeeded=True)
    assert "INDEPENDENT_AUTHORITY_GATES" in result


def test_forged_legacy_identity_cannot_change_role_to_canonical():
    g = guard()
    legacy = g.issue_identity("legacy", RuntimeRole.LEGACY_REPLAY)
    forged = replace(legacy, role=RuntimeRole.R11_CANONICAL_RUNTIME.value)
    with pytest.raises(InvalidRuntimeIdentity):
        g.issue_token(forged, Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL)


def test_forged_capability_token_fails_binding_check():
    g = guard()
    canonical = g.issue_identity("r11", RuntimeRole.R11_CANONICAL_RUNTIME)
    token = g.issue_token(canonical, Capability.READ_LEGACY_REFERENCE)
    forged = replace(token, capability=Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL.value)
    with pytest.raises(InvalidRetirementToken):
        g.assert_token(canonical, Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL, forged)


def test_token_is_bound_to_exact_identity_and_cannot_be_replayed_by_other_subject():
    g = guard()
    first = g.issue_identity("r11-a", RuntimeRole.R11_CANONICAL_RUNTIME)
    second = g.issue_identity("r11-b", RuntimeRole.R11_CANONICAL_RUNTIME)
    token = g.issue_token(first, Capability.SEAL_CHECKPOINT)
    with pytest.raises(InvalidRetirementToken):
        g.assert_token(second, Capability.SEAL_CHECKPOINT, token)


def test_canonical_retirement_token_is_admission_only_not_positive_authority():
    g = guard()
    canonical = g.issue_identity("r11", RuntimeRole.R11_CANONICAL_RUNTIME)
    token = g.issue_token(canonical, Capability.RELEASE_GENERATION)
    decision = g.assert_token(canonical, Capability.RELEASE_GENERATION, token)
    assert token.admission_only is True
    assert token.authority_granted is False
    assert decision.authority_granted is False
    assert decision.requires_independent_runtime_authority is True


def test_active_legacy_authoritative_writer_detection_fails_closed_before_token_need():
    g = guard()
    legacy = g.issue_identity("legacy-writer", RuntimeRole.LEGACY_REFERENCE)
    claim = ActiveCapabilityClaim(
        identity=legacy,
        capability=Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL.value,
        token=None,
        active=True,
    )
    with pytest.raises(ActiveLegacyWriterDetected):
        g.audit_active_claims([claim])


def test_active_canonical_claim_needs_valid_retirement_token_and_still_gets_no_authority():
    g = guard()
    canonical = g.issue_identity("r11", RuntimeRole.R11_CANONICAL_RUNTIME)
    with pytest.raises(InvalidRetirementToken):
        g.audit_active_claims(
            [
                ActiveCapabilityClaim(
                    canonical,
                    Capability.SEAL_TRAINING_SNAPSHOT.value,
                    None,
                    True,
                )
            ]
        )
    token = g.issue_token(canonical, Capability.SEAL_TRAINING_SNAPSHOT)
    report = g.audit_active_claims(
        [ActiveCapabilityClaim(canonical, Capability.SEAL_TRAINING_SNAPSHOT.value, token, True)]
    )
    assert report["canonical_authoritative_claim_count"] == 1
    assert report["canonical_claims_require_independent_runtime_authority"] is True


def test_forged_active_identity_fails_closed():
    g = guard()
    legacy = g.issue_identity("legacy", RuntimeRole.LEGACY_REFERENCE)
    forged = replace(legacy, signature="0" * 64)
    with pytest.raises(InvalidRuntimeIdentity):
        g.audit_active_claims(
            [ActiveCapabilityClaim(forged, Capability.READ_LEGACY_REFERENCE.value, None, True)]
        )


def test_inactive_historical_legacy_writer_claim_does_not_become_live_authority():
    report = audit_declared_claims(
        [
            DeclaredCapabilityClaim(
                "historical-runtime",
                RuntimeRole.LEGACY_REFERENCE.value,
                Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL.value,
                active=False,
            )
        ]
    )
    assert report["inactive_claim_count"] == 1
    assert report["active_legacy_authoritative_writer_count"] == 0


def test_static_declared_active_legacy_writer_fails_closed():
    with pytest.raises(ActiveLegacyWriterDetected):
        audit_declared_claims(
            [
                DeclaredCapabilityClaim(
                    "legacy-runtime",
                    RuntimeRole.LEGACY_REPLAY.value,
                    Capability.SEAL_CHECKPOINT.value,
                    True,
                )
            ]
        )


def test_unknown_declared_role_and_capability_fail_closed():
    from cb16_local_opt.stage4_legacy_retirement_r11 import UnknownCapability, UnknownRuntimeRole

    with pytest.raises(UnknownRuntimeRole):
        audit_declared_claims([DeclaredCapabilityClaim("x", "UNKNOWN", "RUN_DIAGNOSTICS", True)])
    with pytest.raises(UnknownCapability):
        audit_declared_claims([DeclaredCapabilityClaim("x", "DIAGNOSTIC", "ROOT_EVERYTHING", True)])


def test_self_audit_denies_complete_noncanonical_authoritative_matrix():
    report = self_audit_policy()
    expected = len(NON_AUTHORITATIVE_ROLES) * len(AUTHORITATIVE_CAPABILITIES)
    assert report["self_audit"]["noncanonical_authoritative_escalations_tested"] == expected
    assert report["self_audit"]["noncanonical_authoritative_escalations_denied"] == expected
    assert report["self_audit"]["canonical_admission_token_authority_granted"] is False


def test_policy_preserves_stage4_scientific_guards():
    report = policy_manifest()
    assert SCIENTIFIC_STATUS == EXPECTED_SCIENCE
    assert report["scientific_status"] == EXPECTED_SCIENCE
    assert report["legacy_python_is_runtime_authority"] is False
    assert report["legacy_replay_is_engineering_only"] is True
    assert report["legacy_replay_may_mint_new_evidence"] is False
    assert report["legacy_fallback_on_r11_startup_failure"] is False
    assert report["scientific_semantics_changed"] is False
    assert report["new_scientific_verdict"] is False
    assert report["final_holdout_accessed"] is False
    assert report["fresh_market_data_downloaded"] is False
    assert report["historical_market_data_mutated"] is False


def test_s4g_module_does_not_import_sibling_stage4_modules():
    source = (ROOT / "cb16_local_opt/stage4_legacy_retirement_r11.py").read_text()
    import_lines = [line.strip() for line in source.splitlines() if line.lstrip().startswith(("import ", "from "))]
    assert not [line for line in import_lines if "stage4_" in line]


def test_cli_self_check_passes():
    proc = subprocess.run(
        [sys.executable, "scripts/audit_r11_stage4_legacy_authority.py", "--self-check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["audit_status"] == "PASS"


def test_cli_rejects_active_legacy_writer(tmp_path):
    claims = tmp_path / "claims.json"
    claims.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "subject_id": "legacy-cli",
                        "role": "LEGACY_REPLAY",
                        "capability": "RELEASE_GENERATION",
                        "active": True,
                    }
                ]
            }
        )
    )
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_r11_stage4_legacy_authority.py",
            "--claims-json",
            str(claims),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 2
    report = json.loads(proc.stdout)
    assert report["audit_status"] == "FAIL"
    assert "ACTIVE_LEGACY_WRITER_DETECTED" in report["error"]
