"""S0-v2 generation switch and same-account continuation.

A child checkpoint may become the running policy only at an authorized runtime
boundary, and the logical account ledger/lineage must continue unchanged.
Constructing a fresh flat account to make the child "act" is rejected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping

import torch
from torch import nn

from .account_economics_r0 import AccountEconomicsStateR0
from .cc_environment_lifecycle_r0 import POST_STATE_PUBLISHED
from .cc_experience_wire_r0 import canonical_json_bytes
from .cc_runtime_account_loop_r0 import CCContinuousAccountRuntimeR0
from .cc_runtime_boundary_r0 import (
    COMPUTE_CHUNK,
    CONTINUE,
    DATA_END_TRUNCATION,
    ECONOMIC_TERMINAL,
    OBJECTIVE_HORIZON_REACHED,
    PAUSE,
    PROCESS_FAILURE,
    TRADING_DISABLED_PENDING_SETTLEMENT,
)
from .cc_runtime_generation_switch_r0 import HEX64, switch_generation_r0
from .post_cc_update_transaction_v1 import STATUS_COMMITTED, DurableUpdateStoreV1


class GenerationContinuityError(RuntimeError):
    pass


CHILD_POLICY_IDENTITY_SCHEMA_V1 = "CB16_R11_S0V2_CHILD_POLICY_IDENTITY_V1"


AUTHORIZED_SWITCH_BOUNDARIES = frozenset(
    {CONTINUE, COMPUTE_CHUNK, PAUSE, DATA_END_TRUNCATION, OBJECTIVE_HORIZON_REACHED}
)
FORBIDDEN_SWITCH_BOUNDARIES = frozenset(
    {ECONOMIC_TERMINAL, PROCESS_FAILURE, TRADING_DISABLED_PENDING_SETTLEMENT}
)


@dataclass(frozen=True)
class GenerationContinuityReceiptV1:
    switch_id: str
    old_policy_generation: str
    old_policy_id: str
    old_policy_sha256: str
    new_policy_generation: str
    new_policy_id: str
    new_policy_sha256: str
    child_checkpoint_sha256: str
    account_lineage_id: str
    account_truth_hash_before: str
    account_truth_hash_after: str
    boundary_type: str
    environment_time: int
    policy_decision_index: int
    parent_checkpoint_mutated_in_place: bool
    account_fully_preserved: bool
    authorized_boundary: bool

    def validate(self) -> "GenerationContinuityReceiptV1":
        if len(self.switch_id) != 64:
            raise ValueError("switch_id must be 64-hex")
        for name in ("old_policy_id", "new_policy_id", "account_lineage_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        for name in ("old_policy_sha256", "new_policy_sha256", "child_checkpoint_sha256", "account_truth_hash_before", "account_truth_hash_after"):
            if not isinstance(getattr(self, name), str) or HEX64.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be 64-hex text")
        if self.parent_checkpoint_mutated_in_place:
            raise ValueError("parent checkpoint must not be mutated in place")
        if not self.account_fully_preserved:
            raise ValueError("account must be fully preserved across switch")
        if not self.authorized_boundary:
            raise ValueError("switch boundary is not authorized")
        return self


def parameter_state_sha256_v1(module: nn.Module) -> str:
    payload = {}
    for name, tensor in sorted(module.state_dict().items()):
        payload[name] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "values": [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()],
        }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _account_truth_hash_v1(account: AccountEconomicsStateR0) -> str:
    account.validate()
    payload = asdict(account)
    payload["unrealized_pnl"] = float(account.unrealized_pnl)
    payload["equity"] = float(account.equity)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def child_policy_identity_v1(
    *,
    child_checkpoint_sha256: str,
    policy_generation: str,
    policy_id: str,
) -> str:
    """Canonical runtime policy identity bound to a committed child checkpoint."""
    if not isinstance(child_checkpoint_sha256, str) or HEX64.fullmatch(child_checkpoint_sha256) is None:
        raise ValueError("child_checkpoint_sha256 must be 64-hex")
    for name, value in (("policy_generation", policy_generation), ("policy_id", policy_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty text")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": CHILD_POLICY_IDENTITY_SCHEMA_V1,
                "child_checkpoint_sha256": child_checkpoint_sha256,
                "policy_generation": str(policy_generation),
                "policy_id": str(policy_id),
            }
        )
    ).hexdigest()


def assert_same_logical_account_continuation_v1(
    *,
    account_lineage_id_before: str,
    account_lineage_id_after: str,
    account_before: AccountEconomicsStateR0,
    account_after: AccountEconomicsStateR0,
) -> None:
    """Fail closed on lineage change, ledger reset or flat-account substitution."""
    if account_lineage_id_before != account_lineage_id_after:
        raise GenerationContinuityError("GENERATION_ACCOUNT_LINEAGE_CHANGED")
    account_before.validate()
    account_after.validate()
    if asdict(account_before) != asdict(account_after):
        raise GenerationContinuityError("GENERATION_ACCOUNT_LEDGER_MUTATED_OR_REPLACED")


def commit_child_generation_v1(
    runtime: CCContinuousAccountRuntimeR0,
    *,
    update_store: DurableUpdateStoreV1,
    update_id: str,
    new_policy_generation: str,
    new_policy_id: str,
    new_policy_sha256: str,
    expected_account_snapshot: AccountEconomicsStateR0,
    boundary_type: str,
) -> GenerationContinuityReceiptV1:
    """Bind a durably COMMITTED child checkpoint at an authorized boundary.

    Missing/PREPARED/STAGED authority is rejected.  The runtime policy SHA must
    equal the canonical ``child_policy_identity_v1`` mapping of the committed
    child checkpoint and the new generation/id; an unrelated hash is rejected.
    """
    if not isinstance(runtime, CCContinuousAccountRuntimeR0):
        raise TypeError("runtime must be CCContinuousAccountRuntimeR0")
    if not isinstance(update_store, DurableUpdateStoreV1):
        raise TypeError("update_store must be DurableUpdateStoreV1")
    if not isinstance(update_id, str) or not update_id.strip():
        raise ValueError("update_id must be non-empty text")
    if runtime.phase != POST_STATE_PUBLISHED:
        raise GenerationContinuityError("GENERATION_SWITCH_REQUIRES_PUBLISHED_BOUNDARY")
    if boundary_type in FORBIDDEN_SWITCH_BOUNDARIES:
        raise GenerationContinuityError("GENERATION_SWITCH_BOUNDARY_FORBIDDEN")
    if boundary_type not in AUTHORIZED_SWITCH_BOUNDARIES:
        raise GenerationContinuityError("GENERATION_SWITCH_BOUNDARY_NOT_AUTHORIZED")
    expected_account_snapshot.validate()

    committed_update = update_store.get_record(update_id)
    if committed_update is None:
        raise GenerationContinuityError("COMMITTED_CHILD_AUTHORITY_MISSING")
    committed_update.validate()
    if committed_update.commit_status != STATUS_COMMITTED:
        raise GenerationContinuityError(
            f"CHILD_GENERATION_REQUIRES_COMMITTED_UPDATE:{committed_update.commit_status}"
        )
    child_checkpoint_sha256 = committed_update.child_checkpoint_sha256
    if child_checkpoint_sha256 is None or HEX64.fullmatch(child_checkpoint_sha256) is None:
        raise GenerationContinuityError("COMMITTED_CHILD_CHECKPOINT_IDENTITY_INVALID")
    # Prove the committed child checkpoint bytes are still durable and intact.
    update_store.load_child_checkpoint(update_id)
    expected_policy_sha256 = child_policy_identity_v1(
        child_checkpoint_sha256=child_checkpoint_sha256,
        policy_generation=new_policy_generation,
        policy_id=new_policy_id,
    )
    if new_policy_sha256 != expected_policy_sha256:
        raise GenerationContinuityError("RUNTIME_POLICY_IDENTITY_NOT_BOUND_TO_COMMITTED_CHILD")

    account_before = runtime.account
    if asdict(account_before) != asdict(expected_account_snapshot):
        raise GenerationContinuityError("GENERATION_EXPECTED_ACCOUNT_MISMATCH")
    lineage_before = runtime.account_lineage_id
    old_policy_generation = str(runtime.policy_generation)
    old_policy_id = str(runtime.policy_id)
    old_policy_sha256 = str(runtime.policy_sha256)
    truth_before = _account_truth_hash_v1(account_before)
    switch = switch_generation_r0(
        runtime,
        new_policy_generation=new_policy_generation,
        new_policy_id=new_policy_id,
        new_policy_sha256=new_policy_sha256,
    )
    account_after = runtime.account
    assert_same_logical_account_continuation_v1(
        account_lineage_id_before=lineage_before,
        account_lineage_id_after=runtime.account_lineage_id,
        account_before=account_before,
        account_after=account_after,
    )
    truth_after = _account_truth_hash_v1(account_after)
    switch_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "old_policy_generation": old_policy_generation,
                "new_policy_generation": str(new_policy_generation),
                "new_policy_id": str(new_policy_id),
                "new_policy_sha256": str(new_policy_sha256),
                "child_checkpoint_sha256": str(child_checkpoint_sha256),
                "environment_time": int(switch.environment_time),
                "policy_decision_index": int(switch.decision_index),
                "boundary_type": str(boundary_type),
            }
        )
    ).hexdigest()
    receipt = GenerationContinuityReceiptV1(
        switch_id=switch_id,
        old_policy_generation=old_policy_generation,
        old_policy_id=old_policy_id,
        old_policy_sha256=old_policy_sha256,
        new_policy_generation=str(new_policy_generation),
        new_policy_id=str(new_policy_id),
        new_policy_sha256=str(new_policy_sha256),
        child_checkpoint_sha256=str(child_checkpoint_sha256),
        account_lineage_id=lineage_before,
        account_truth_hash_before=truth_before,
        account_truth_hash_after=truth_after,
        boundary_type=str(boundary_type),
        environment_time=int(switch.environment_time),
        policy_decision_index=int(switch.decision_index),
        parent_checkpoint_mutated_in_place=False,
        account_fully_preserved=True,
        authorized_boundary=True,
    )
    return receipt.validate()


def assert_behavior_checkpoint_immutable_v1(
    behavior_actor: nn.Module, *, expected_parameter_state_sha256: str
) -> None:
    actual = parameter_state_sha256_v1(behavior_actor)
    if actual != expected_parameter_state_sha256:
        raise GenerationContinuityError("PARENT_BEHAVIOR_CHECKPOINT_MUTATED_IN_PLACE")
