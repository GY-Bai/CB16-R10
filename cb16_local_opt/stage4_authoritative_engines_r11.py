from __future__ import annotations

"""Stage-4 authority-aware wrappers for already-qualified R11 compute engines.

INTC owns execution admission only.  It does not redefine Teacher, Trace/H72,
Champion inference, training, validation, tournament, Evidence, persistence,
Permission/Physics, or worker lifecycle semantics.

The wrappers are deliberately structural around ``EngineBundle``.  Every accepted
execution is bound to an S4B ``RuntimeContext``, an R11 ``AuthorityStamp``, S4G
negative-eligibility identities/tokens, and an injected live-fence assertion.
S4G admission never grants positive authority; the fence remains independent.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .runtime_events_r11 import (
    AuthorityStamp,
    PolicyResult,
    TournamentResult,
    TrainingResult,
    ValidationResult,
    WorkCompletion,
    WorkItem,
    WorkKind,
)
from .runtime_protocols_r11 import EngineBundle
from .stage4_canonical_runtime_r11 import AuthorityLease, RuntimeContext
from .stage4_legacy_retirement_r11 import (
    Capability,
    LegacyRetirementGuard,
    RetirementCapabilityToken,
    RuntimeIdentity,
    RuntimeRole,
)


SCHEMA = "CB16_R11_STAGE4_INTC_AUTHORITATIVE_ENGINE_WRAPPER_V1"
SCIENTIFIC_STATUS = (
    "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
)

# S4G has no scientific-computation capabilities and INTC must not reinterpret
# mutation capabilities as computation semantics.  Canonical-runtime acquisition
# eligibility is therefore the single negative admission used for engine/runtime
# identities.  Positive execution authority still comes only from the live fence.
ENGINE_EXECUTION_ELIGIBILITY_CAPABILITY = Capability.ACQUIRE_CANONICAL_RUNTIME_AUTHORITY


class Stage4EngineAuthorityError(RuntimeError):
    """Base class for INTC fail-closed admission/lineage errors."""


class InvalidEngineAuthorityContext(Stage4EngineAuthorityError):
    pass


class EngineRoleEscalationDenied(Stage4EngineAuthorityError):
    pass


class EngineLineageMismatch(Stage4EngineAuthorityError):
    pass


class EngineCompletionRejected(Stage4EngineAuthorityError):
    pass


class TrainingNumericIdentityUnproven(Stage4EngineAuthorityError):
    pass


@runtime_checkable
class LiveFenceAssertionBoundaryR11(Protocol):
    """Injected positive authority boundary, compatible with S4B lease providers."""

    def assert_current(self, lease: AuthorityLease) -> None: ...


@runtime_checkable
class ExecutableEngineR11(Protocol):
    def execute(self, work: WorkItem) -> WorkCompletion: ...


@dataclass(frozen=True)
class EngineIdentityBindingR11:
    """S4G negative-eligibility binding for one concrete engine identity."""

    identity: RuntimeIdentity
    admission_token: RetirementCapabilityToken


@dataclass(frozen=True)
class CanonicalEngineAuthorityContextR11:
    """Injected accepted runtime/Champion lineage; this object mints no authority."""

    runtime_context: RuntimeContext
    authority_stamp: AuthorityStamp
    runtime_identity: RuntimeIdentity
    runtime_admission_token: RetirementCapabilityToken


class IntrospectiveTrainingNumericAssertionR11:
    """Fail closed unless an already-qualified training runtime exposes its config.

    The assertion does not set dtype, AMP, gradients, optimizer parameters, or model
    state.  It only observes a direct ``config`` or a narrow wrapper's
    ``training_runtime.config`` / ``runtime.config``.
    """

    _CANDIDATE_ATTRS = ("training_runtime", "runtime")

    def assert_canonical(self, engine: ExecutableEngineR11) -> None:
        candidates = [engine]
        for name in self._CANDIDATE_ATTRS:
            obj = getattr(engine, name, None)
            if obj is not None:
                candidates.append(obj)
        for obj in candidates:
            config = getattr(obj, "config", None)
            if config is None:
                continue
            if not hasattr(config, "amp_enabled") or not hasattr(config, "dtype"):
                continue
            if getattr(config, "amp_enabled") is not False:
                raise TrainingNumericIdentityUnproven(
                    "R11_INTC_TRAINING_AMP_MUST_REMAIN_FALSE"
                )
            if str(getattr(config, "dtype")) != "torch.float32":
                raise TrainingNumericIdentityUnproven(
                    "R11_INTC_TRAINING_DTYPE_MUST_REMAIN_TORCH_FLOAT32"
                )
            return
        raise TrainingNumericIdentityUnproven(
            "R11_INTC_TRAINING_NUMERIC_IDENTITY_NOT_OBSERVABLE"
        )


_ENGINE_ATTR_BY_KIND: Mapping[WorkKind, str] = {
    WorkKind.TEACHER_ACQUIRE: "teacher",
    WorkKind.TRACE_EXECUTE: "trace",
    WorkKind.POLICY_INFER: "inference",
    WorkKind.TRAIN_CHALLENGER: "training",
    WorkKind.VALIDATE: "validation",
    WorkKind.TOURNAMENT: "tournament",
}


def _assert_s4g_canonical_eligibility(
    *,
    guard: LegacyRetirementGuard,
    identity: RuntimeIdentity,
    token: RetirementCapabilityToken,
    label: str,
) -> None:
    role = guard.validate_identity(identity)
    if role is not RuntimeRole.R11_CANONICAL_RUNTIME:
        raise EngineRoleEscalationDenied(
            f"R11_INTC_NONCANONICAL_{label}_DENIED:{role.value}"
        )
    decision = guard.assert_token(
        identity, ENGINE_EXECUTION_ELIGIBILITY_CAPABILITY, token
    )
    if (
        not decision.eligible_under_legacy_retirement_policy
        or decision.authority_granted
        or not decision.requires_independent_runtime_authority
    ):
        raise InvalidEngineAuthorityContext(
            f"R11_INTC_S4G_ADMISSION_INVARIANT_FAILED:{label}"
        )


def _validate_authority_context(
    context: CanonicalEngineAuthorityContextR11,
    guard: LegacyRetirementGuard,
) -> None:
    if not isinstance(context, CanonicalEngineAuthorityContextR11):
        raise InvalidEngineAuthorityContext("R11_INTC_AUTHORITY_CONTEXT_TYPE_INVALID")
    runtime = context.runtime_context
    if not isinstance(runtime, RuntimeContext):
        raise InvalidEngineAuthorityContext("R11_INTC_RUNTIME_CONTEXT_TYPE_INVALID")
    stamp = context.authority_stamp
    if not isinstance(stamp, AuthorityStamp):
        raise InvalidEngineAuthorityContext("R11_INTC_AUTHORITY_STAMP_TYPE_INVALID")

    authority_id = runtime.binding.authority_id
    if (
        not authority_id
        or runtime.recovered.authority_id != authority_id
        or runtime.lease.authority_id != authority_id
    ):
        raise InvalidEngineAuthorityContext("R11_INTC_RUNTIME_AUTHORITY_IDENTITY_MISMATCH")
    if (
        runtime.binding.generation != runtime.recovered.generation
        or stamp.generation != runtime.binding.generation
    ):
        raise InvalidEngineAuthorityContext("R11_INTC_RUNTIME_GENERATION_MISMATCH")
    if not stamp.champion_id or not stamp.champion_hash:
        raise InvalidEngineAuthorityContext("R11_INTC_CHAMPION_IDENTITY_MISSING")
    if not stamp.teacher_authority_id or not stamp.physics_authority_id:
        raise InvalidEngineAuthorityContext("R11_INTC_UPSTREAM_AUTHORITY_IDENTITY_MISSING")
    if context.runtime_identity.subject_id != authority_id:
        raise InvalidEngineAuthorityContext(
            "R11_INTC_RUNTIME_S4G_SUBJECT_NOT_BOUND_TO_AUTHORITY_ID"
        )

    _assert_s4g_canonical_eligibility(
        guard=guard,
        identity=context.runtime_identity,
        token=context.runtime_admission_token,
        label="RUNTIME_IDENTITY",
    )


def _validate_work_lineage(
    work: WorkItem,
    *,
    expected_kind: WorkKind,
    context: CanonicalEngineAuthorityContextR11,
) -> None:
    if not isinstance(work, WorkItem):
        raise EngineLineageMismatch("R11_INTC_WORK_ITEM_TYPE_INVALID")
    if work.kind is not expected_kind:
        raise EngineLineageMismatch(
            f"R11_INTC_WORK_KIND_MISMATCH:{expected_kind.value}:{work.kind.value}"
        )
    stamp = context.authority_stamp
    if work.generation != stamp.generation:
        raise EngineLineageMismatch(
            f"R11_INTC_WORK_GENERATION_MISMATCH:{stamp.generation}:{work.generation}"
        )
    if work.parent_champion_id != stamp.champion_id:
        raise EngineLineageMismatch(
            f"R11_INTC_WORK_CHAMPION_LINEAGE_MISMATCH:{stamp.champion_id}:{work.parent_champion_id}"
        )


def _validate_typed_result_lineage(
    result: Any,
    work: WorkItem,
    context: CanonicalEngineAuthorityContextR11,
) -> None:
    stamp = context.authority_stamp
    if work.kind is WorkKind.POLICY_INFER:
        if not isinstance(result, PolicyResult):
            raise EngineCompletionRejected("R11_INTC_POLICY_RESULT_TYPE_INVALID")
        if result.work_id != work.work_id:
            raise EngineCompletionRejected("R11_INTC_POLICY_RESULT_WORK_ID_MISMATCH")
        if result.generation != stamp.generation or result.champion_id != stamp.champion_id:
            raise EngineCompletionRejected("R11_INTC_POLICY_RESULT_LINEAGE_MISMATCH")
        if result.poison_bits:
            raise EngineCompletionRejected("R11_INTC_POLICY_RESULT_POISONED")
        return

    if work.kind is WorkKind.TRAIN_CHALLENGER:
        if not isinstance(result, TrainingResult):
            raise EngineCompletionRejected("R11_INTC_TRAINING_RESULT_TYPE_INVALID")
        if result.work_id != work.work_id:
            raise EngineCompletionRejected("R11_INTC_TRAINING_RESULT_WORK_ID_MISMATCH")
        if result.generation != stamp.generation or result.parent_champion_id != stamp.champion_id:
            raise EngineCompletionRejected("R11_INTC_TRAINING_RESULT_LINEAGE_MISMATCH")
        if work.snapshot_id is not None and result.snapshot_id != work.snapshot_id:
            raise EngineCompletionRejected("R11_INTC_TRAINING_SNAPSHOT_MISMATCH")
        if result.poison_bits:
            raise EngineCompletionRejected("R11_INTC_TRAINING_RESULT_POISONED")
        return

    if work.kind is WorkKind.VALIDATE:
        if not isinstance(result, ValidationResult):
            raise EngineCompletionRejected("R11_INTC_VALIDATION_RESULT_TYPE_INVALID")
        if result.work_id != work.work_id:
            raise EngineCompletionRejected("R11_INTC_VALIDATION_RESULT_WORK_ID_MISMATCH")
        if result.generation != stamp.generation or result.parent_champion_id != stamp.champion_id:
            raise EngineCompletionRejected("R11_INTC_VALIDATION_RESULT_LINEAGE_MISMATCH")
        if work.snapshot_id is not None and result.snapshot_id != work.snapshot_id:
            raise EngineCompletionRejected("R11_INTC_VALIDATION_SNAPSHOT_MISMATCH")
        if result.poison_bits:
            raise EngineCompletionRejected("R11_INTC_VALIDATION_RESULT_POISONED")
        return

    if work.kind is WorkKind.TOURNAMENT:
        if not isinstance(result, TournamentResult):
            # In particular, a TournamentCommitProposal / CommitReceipt is not an
            # accepted engine output. Promotion remains a later persistence mutation.
            raise EngineCompletionRejected("R11_INTC_TOURNAMENT_RESULT_TYPE_INVALID")
        if result.work_id != work.work_id:
            raise EngineCompletionRejected("R11_INTC_TOURNAMENT_RESULT_WORK_ID_MISMATCH")
        if result.generation != stamp.generation or result.parent_champion_id != stamp.champion_id:
            raise EngineCompletionRejected("R11_INTC_TOURNAMENT_RESULT_LINEAGE_MISMATCH")
        if result.poison_bits:
            raise EngineCompletionRejected("R11_INTC_TOURNAMENT_RESULT_POISONED")


def _validate_completion(
    completion: WorkCompletion,
    *,
    work: WorkItem,
    context: CanonicalEngineAuthorityContextR11,
) -> None:
    if not isinstance(completion, WorkCompletion):
        raise EngineCompletionRejected("R11_INTC_ENGINE_COMPLETION_TYPE_INVALID")
    if completion.work_id != work.work_id or completion.kind is not work.kind:
        raise EngineCompletionRejected("R11_INTC_ENGINE_COMPLETION_WORK_IDENTITY_MISMATCH")
    if completion.generation != work.generation:
        raise EngineCompletionRejected("R11_INTC_ENGINE_COMPLETION_GENERATION_MISMATCH")
    if completion.poison_bits:
        raise EngineCompletionRejected("R11_INTC_ENGINE_COMPLETION_POISONED")
    _validate_typed_result_lineage(completion.result, work, context)


class _AuthorityAwareEngineR11:
    """One domain adapter. It has intentionally no promotion/persistence methods."""

    def __init__(
        self,
        *,
        expected_kind: WorkKind,
        engine: ExecutableEngineR11,
        engine_identity: EngineIdentityBindingR11,
        authority_context: CanonicalEngineAuthorityContextR11,
        legacy_guard: LegacyRetirementGuard,
        fence: LiveFenceAssertionBoundaryR11,
    ) -> None:
        self._expected_kind = expected_kind
        self._engine = engine
        self._engine_identity = engine_identity
        self._authority_context = authority_context
        self._legacy_guard = legacy_guard
        self._fence = fence
        self._training_numeric_assertion = IntrospectiveTrainingNumericAssertionR11()

    def execute(self, work: WorkItem) -> WorkCompletion:
        _validate_authority_context(self._authority_context, self._legacy_guard)
        _validate_work_lineage(
            work, expected_kind=self._expected_kind, context=self._authority_context
        )
        _assert_s4g_canonical_eligibility(
            guard=self._legacy_guard,
            identity=self._engine_identity.identity,
            token=self._engine_identity.admission_token,
            label=f"{self._expected_kind.value}_ENGINE_IDENTITY",
        )
        if self._expected_kind is WorkKind.TRAIN_CHALLENGER:
            self._training_numeric_assertion.assert_canonical(self._engine)

        # Positive authority is checked immediately before computation and again
        # before its output can be accepted.  S4G is never substituted for this.
        lease = self._authority_context.runtime_context.lease
        self._fence.assert_current(lease)
        completion = self._engine.execute(work)
        self._fence.assert_current(lease)

        _validate_completion(
            completion, work=work, context=self._authority_context
        )
        return completion


def build_authoritative_engine_bundle_r11(
    *,
    engines: EngineBundle,
    engine_identities: Mapping[WorkKind, EngineIdentityBindingR11],
    authority_context: CanonicalEngineAuthorityContextR11,
    legacy_guard: LegacyRetirementGuard,
    fence: LiveFenceAssertionBoundaryR11,
) -> EngineBundle:
    """Wrap present R11 compute engines without changing their result objects.

    Missing engines remain missing.  Every present authoritative engine requires
    its own canonical S4G identity binding; sharing a Python method or being
    importable is never enough to become canonical.
    """

    if not isinstance(engines, EngineBundle):
        raise TypeError("R11_INTC_ENGINE_BUNDLE_TYPE_INVALID")
    _validate_authority_context(authority_context, legacy_guard)
    wrapped: dict[str, ExecutableEngineR11 | None] = {
        "teacher": None,
        "trace": None,
        "inference": None,
        "training": None,
        "validation": None,
        "tournament": None,
    }
    for kind, attr in _ENGINE_ATTR_BY_KIND.items():
        engine = getattr(engines, attr)
        if engine is None:
            continue
        binding = engine_identities.get(kind)
        if not isinstance(binding, EngineIdentityBindingR11):
            raise InvalidEngineAuthorityContext(
                f"R11_INTC_ENGINE_IDENTITY_BINDING_REQUIRED:{kind.value}"
            )
        wrapped[attr] = _AuthorityAwareEngineR11(
            expected_kind=kind,
            engine=engine,
            engine_identity=binding,
            authority_context=authority_context,
            legacy_guard=legacy_guard,
            fence=fence,
        )
    return EngineBundle(**wrapped)


__all__ = [
    "CanonicalEngineAuthorityContextR11",
    "ENGINE_EXECUTION_ELIGIBILITY_CAPABILITY",
    "EngineCompletionRejected",
    "EngineIdentityBindingR11",
    "EngineLineageMismatch",
    "EngineRoleEscalationDenied",
    "IntrospectiveTrainingNumericAssertionR11",
    "InvalidEngineAuthorityContext",
    "LiveFenceAssertionBoundaryR11",
    "SCHEMA",
    "SCIENTIFIC_STATUS",
    "Stage4EngineAuthorityError",
    "TrainingNumericIdentityUnproven",
    "build_authoritative_engine_bundle_r11",
]
