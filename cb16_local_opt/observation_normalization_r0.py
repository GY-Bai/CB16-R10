from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from .account_observation_r0 import AccountPolicyObservationR0
from .execution_observation_r0 import ExecutionObservationR0


OBSERVATION_NORMALIZER_SCHEMA_R0 = "CB16_R11_BC_OBSERVATION_NORMALIZER_V1_R0"
OBSERVATION_NORMALIZER_VERSION_R0 = "CB16_R11_BC_OBSERVATION_NORMALIZATION_V1_R0"
NORMALIZATION_CONTEXT_SCHEMA_R0 = "CB16_R11_BC_NORMALIZATION_CONTEXT_V1_R0"
NORMALIZED_POLICY_STATE_SCHEMA_R0 = "CB16_R11_BC_NORMALIZED_POLICY_STATE_V1_R0"
MONEY_SCALE_SEMANTICS_R0 = "DECLARED_ACCOUNT_MONETARY_SCALE_NOT_REWARD_E_REF"
MARKET_TRANSFORM_R0 = "PASSTHROUGH_FROZEN_CAUSAL_SENSORY"
ACCOUNT_TRANSFORM_R0 = "MONEY_OVER_DECLARED_SCALE_AND_PRICE_UNIT_INVARIANT_POSITION"
EXECUTION_TRANSFORM_R0 = "MECHANICAL_NOTIONAL_OVER_SCALE_WITH_DIMENSIONLESS_PASSTHROUGH"

ACCOUNT_SCALED_FIELDS_R0 = (
    "cash_scale",
    "position_notional_scale",
    "cost_basis_price_ratio",
    "unrealized_pnl_scale",
    "equity_scale",
    "margin_collateral_scale",
    "liabilities_scale",
    "economic_responsibility_open",
)
EXECUTION_SCALED_FIELDS_R0 = (
    "terminated",
    "truncated",
    "legal_short",
    "legal_flat",
    "legal_long",
    "max_permitted_target_risk",
    "margin_capacity_scale",
    "available_margin_for_new_exposure_scale",
    "max_gross_leverage",
    "initial_margin_rate",
    "maintenance_margin_rate",
    "maintenance_collateral_scale",
    "declared_max_legal_notional_scale",
    "lot_step_notional_scale",
    "lot_min_notional_scale",
    "lot_max_notional_scale",
    "min_notional_scale",
)


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _positive(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(code)
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise RuntimeError(code)
    return out


def _finite_tuple(values: tuple[float, ...]) -> tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in out):
        raise RuntimeError("ACNORM_R0_MARKET_FEATURE_INVALID")
    return out


def _scaled_optional_notional(value: float | None, *, price: float | None, scale: float) -> float | None:
    if value is None:
        return None
    raw = float(value)
    if not math.isfinite(raw):
        raise RuntimeError("ACNORM_R0_OPTIONAL_RESOURCE_INVALID")
    notional = raw if price is None else raw * price
    return notional / scale


@dataclass(frozen=True)
class ObservationNormalizerSpecR0:
    schema_version: str
    normalizer_version: str
    money_scale_semantics: str
    market_transform: str
    account_transform: str
    execution_transform: str

    def validate(self) -> None:
        if self.schema_version != OBSERVATION_NORMALIZER_SCHEMA_R0:
            raise RuntimeError("ACNORM_R0_SPEC_SCHEMA_MISMATCH")
        if self.normalizer_version != OBSERVATION_NORMALIZER_VERSION_R0:
            raise RuntimeError("ACNORM_R0_VERSION_MISMATCH")
        if self.money_scale_semantics != MONEY_SCALE_SEMANTICS_R0:
            raise RuntimeError("ACNORM_R0_MONEY_SCALE_SEMANTICS_MISMATCH")
        if self.market_transform != MARKET_TRANSFORM_R0:
            raise RuntimeError("ACNORM_R0_MARKET_TRANSFORM_MISMATCH")
        if self.account_transform != ACCOUNT_TRANSFORM_R0:
            raise RuntimeError("ACNORM_R0_ACCOUNT_TRANSFORM_MISMATCH")
        if self.execution_transform != EXECUTION_TRANSFORM_R0:
            raise RuntimeError("ACNORM_R0_EXECUTION_TRANSFORM_MISMATCH")

    @property
    def semantic_sha256(self) -> str:
        self.validate()
        return _sha(dict(self.__dict__))


OBSERVATION_NORMALIZER_SPEC_R0 = ObservationNormalizerSpecR0(
    schema_version=OBSERVATION_NORMALIZER_SCHEMA_R0,
    normalizer_version=OBSERVATION_NORMALIZER_VERSION_R0,
    money_scale_semantics=MONEY_SCALE_SEMANTICS_R0,
    market_transform=MARKET_TRANSFORM_R0,
    account_transform=ACCOUNT_TRANSFORM_R0,
    execution_transform=EXECUTION_TRANSFORM_R0,
)


@dataclass(frozen=True)
class ObservationNormalizationContextR0:
    schema_version: str
    context_id: str
    money_scale: float

    def validate(self) -> None:
        if self.schema_version != NORMALIZATION_CONTEXT_SCHEMA_R0:
            raise RuntimeError("ACNORM_R0_CONTEXT_SCHEMA_MISMATCH")
        if not self.context_id:
            raise RuntimeError("ACNORM_R0_CONTEXT_ID_INVALID")
        _positive(self.money_scale, code="ACNORM_R0_MONEY_SCALE_INVALID")

    @property
    def context_sha256(self) -> str:
        self.validate()
        return _sha(
            {
                "schema_version": self.schema_version,
                "context_id": self.context_id,
                "money_scale": float(self.money_scale),
            }
        )


def make_normalization_context_r0(*, context_id: str, money_scale: float) -> ObservationNormalizationContextR0:
    context = ObservationNormalizationContextR0(
        schema_version=NORMALIZATION_CONTEXT_SCHEMA_R0,
        context_id=context_id,
        money_scale=_positive(money_scale, code="ACNORM_R0_MONEY_SCALE_INVALID"),
    )
    context.validate()
    return context


@dataclass(frozen=True)
class NormalizedPolicyStateR0:
    schema_version: str
    normalizer_sha256: str
    normalization_context_sha256: str
    market_causal_features: tuple[float, ...]
    account_scaled_features: tuple[tuple[str, object], ...]
    execution_scaled_features: tuple[tuple[str, object], ...]

    def validate(self) -> None:
        if self.schema_version != NORMALIZED_POLICY_STATE_SCHEMA_R0:
            raise RuntimeError("ACNORM_R0_STATE_SCHEMA_MISMATCH")
        if self.normalizer_sha256 != OBSERVATION_NORMALIZER_SPEC_R0.semantic_sha256:
            raise RuntimeError("ACNORM_R0_NORMALIZER_IDENTITY_MISMATCH")
        for value, code in (
            (self.normalizer_sha256, "ACNORM_R0_NORMALIZER_HASH_INVALID"),
            (self.normalization_context_sha256, "ACNORM_R0_CONTEXT_HASH_INVALID"),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise RuntimeError(code)
        _finite_tuple(self.market_causal_features)
        if tuple(name for name, _ in self.account_scaled_features) != ACCOUNT_SCALED_FIELDS_R0:
            raise RuntimeError("ACNORM_R0_ACCOUNT_FIELDS_MISMATCH")
        if tuple(name for name, _ in self.execution_scaled_features) != EXECUTION_SCALED_FIELDS_R0:
            raise RuntimeError("ACNORM_R0_EXECUTION_FIELDS_MISMATCH")

    def model_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "market_causal_features": self.market_causal_features,
            "account_scaled_features": self.account_scaled_features,
            "execution_scaled_features": self.execution_scaled_features,
        }


def normalize_policy_state_r0(
    *,
    market_causal_features: tuple[float, ...],
    account_observation: AccountPolicyObservationR0,
    execution_observation: ExecutionObservationR0,
    context: ObservationNormalizationContextR0,
) -> NormalizedPolicyStateR0:
    """Normalize causal policy state without hiding real exchange mechanics.

    Market sensory features are treated as an already-frozen causal organ output
    and are passed through.  Monetary/account quantities are expressed relative
    to an explicit account normalization scale.  Quantity constraints are first
    converted to current-price notional so a pure asset-unit redenomination does
    not create an identity shortcut; if an exchange constraint is *not* scaled
    with that redenomination, its changed economic effect remains visible.

    The normalization scale is operational observation context, not reward
    ``E_ref`` and not an economic objective.  The normalizer semantic hash and
    context hash are metadata for later trajectory/checkpoint binding and are
    excluded from the model payload.
    """

    account_observation.validate()
    execution_observation.validate()
    context.validate()
    market = _finite_tuple(market_causal_features)
    scale = float(context.money_scale)
    price = float(execution_observation.current_price)

    if float(account_observation.position_quantity) != float(execution_observation.current_quantity):
        raise RuntimeError("ACNORM_R0_POSITION_AUTHORITY_MISMATCH")

    quantity = float(account_observation.position_quantity)
    cost_basis_ratio = 0.0 if quantity == 0.0 else float(account_observation.position_cost_basis) / price
    account_features = (
        ("cash_scale", float(account_observation.cash) / scale),
        ("position_notional_scale", quantity * price / scale),
        ("cost_basis_price_ratio", cost_basis_ratio),
        ("unrealized_pnl_scale", float(account_observation.unrealized_pnl) / scale),
        ("equity_scale", float(account_observation.equity) / scale),
        ("margin_collateral_scale", float(account_observation.margin_collateral) / scale),
        ("liabilities_scale", float(account_observation.liabilities) / scale),
        ("economic_responsibility_open", account_observation.economic_responsibility_open),
    )
    execution_features = (
        ("terminated", execution_observation.terminated),
        ("truncated", execution_observation.truncated),
        ("legal_short", execution_observation.legal_short),
        ("legal_flat", execution_observation.legal_flat),
        ("legal_long", execution_observation.legal_long),
        ("max_permitted_target_risk", float(execution_observation.max_permitted_target_risk)),
        ("margin_capacity_scale", float(execution_observation.margin_capacity) / scale),
        (
            "available_margin_for_new_exposure_scale",
            float(execution_observation.available_margin_for_new_exposure) / scale,
        ),
        ("max_gross_leverage", float(execution_observation.max_gross_leverage)),
        ("initial_margin_rate", float(execution_observation.initial_margin_rate)),
        ("maintenance_margin_rate", float(execution_observation.maintenance_margin_rate)),
        ("maintenance_collateral_scale", float(execution_observation.maintenance_collateral) / scale),
        (
            "declared_max_legal_notional_scale",
            _scaled_optional_notional(execution_observation.declared_max_legal_notional, price=None, scale=scale),
        ),
        (
            "lot_step_notional_scale",
            _scaled_optional_notional(execution_observation.lot_step_size, price=price, scale=scale),
        ),
        (
            "lot_min_notional_scale",
            _scaled_optional_notional(execution_observation.lot_min_qty, price=price, scale=scale),
        ),
        (
            "lot_max_notional_scale",
            _scaled_optional_notional(execution_observation.lot_max_qty, price=price, scale=scale),
        ),
        (
            "min_notional_scale",
            _scaled_optional_notional(execution_observation.min_notional, price=None, scale=scale),
        ),
    )

    state = NormalizedPolicyStateR0(
        schema_version=NORMALIZED_POLICY_STATE_SCHEMA_R0,
        normalizer_sha256=OBSERVATION_NORMALIZER_SPEC_R0.semantic_sha256,
        normalization_context_sha256=context.context_sha256,
        market_causal_features=market,
        account_scaled_features=account_features,
        execution_scaled_features=execution_features,
    )
    state.validate()
    return state
