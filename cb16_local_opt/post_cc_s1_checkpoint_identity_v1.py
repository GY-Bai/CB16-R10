"""S1 frozen initial-checkpoint identity and isolated initialization paths.

B4 fixes:

* the declared ``SORTED_STATE_DICT_NAME_SHAPE_DTYPE_FLOAT_VALUE_JSON_V1``
  codec is implemented here and actually evaluated against the frozen authority
  hash instead of merely restating the hash string;
* the canonical target Actor+Critic initialization is isolated from behavior
  actor construction, so target Critic state can never depend on task/seed RNG
  order.

If the frozen authority hash cannot be reproduced under the declared codec,
the verifier reports a CONTRACT_MISMATCH.  It never substitutes another hash in
place of the authority value.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .cc_policy_brain_r0 import CCCentralBrain
from .cc_critic_value_r0 import SeparateCritic

CHECKPOINT_CODEC_ID_V1 = "SORTED_STATE_DICT_NAME_SHAPE_DTYPE_FLOAT_VALUE_JSON_V1"
FROZEN_AUTHORITY_INITIAL_CHECKPOINT_SHA256_V1 = (
    "32928a6b2fea2346d303c9b61e8f86dee66099e973e7391372836b4f8a706021"
)
FROZEN_INITIALIZATION_SEED_V1 = 1701
S1_BINDINGS_FOR_INIT_V1 = None  # delayed import to avoid cycles


class S1InitialCheckpointContractError(RuntimeError):
    pass


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_state_dict_sha256_v1(modules: Sequence[tuple[str, nn.Module]]) -> str:
    """Declared codec: sorted state dict name -> {dtype, shape, float values} JSON."""
    payload: dict[str, Any] = {}
    for label, module in modules:
        if not isinstance(module, nn.Module):
            raise TypeError("semantic_state_dict_sha256_v1 expects nn.Module entries")
        for name, tensor in sorted(module.state_dict().items()):
            key = name if len(modules) == 1 else f"{label}.{name}"
            payload[key] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "values": [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()],
            }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _brain_bindings_v1():
    from .post_cc_s1_tasks_v1 import S1_BRAIN_BINDINGS

    return S1_BRAIN_BINDINGS


def initialize_canonical_target_actor_critic_v1() -> tuple[CCCentralBrain, SeparateCritic]:
    """Frozen target initialization: one isolated seed, Actor then Critic."""
    torch.manual_seed(FROZEN_INITIALIZATION_SEED_V1)
    actor = CCCentralBrain(2, 3, 2, 8, _brain_bindings_v1())
    critic = SeparateCritic(7, 8)
    actor.assert_gradient_ownership()
    trainable = sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in actor.parameters() if not parameter.requires_grad)
    critic_parameters = sum(parameter.numel() for parameter in critic.parameters())
    if (trainable, frozen, critic_parameters) != (241, 16, 73):
        raise S1InitialCheckpointContractError("MODEL_PARAMETER_COUNT_CONTRACT_MISMATCH")
    return actor, critic


def initialize_isolated_behavior_actor_v1(seed: int) -> CCCentralBrain:
    """Behavior actor init that cannot perturb the target initialization stream."""
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("behavior seed must be a non-negative integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        brain = CCCentralBrain(2, 3, 2, 8, _brain_bindings_v1())
    brain.assert_gradient_ownership()
    for parameter in brain.parameters():
        parameter.requires_grad_(False)
    return brain


def initial_checkpoint_identity_v1() -> Mapping[str, Any]:
    """Compute the canonical initial actor/critic identity under the declared codec."""
    actor, critic = initialize_canonical_target_actor_critic_v1()
    actor_only = semantic_state_dict_sha256_v1((("actor", actor),))
    actor_critic = semantic_state_dict_sha256_v1((("actor", actor), ("critic", critic)))
    reproduced = FROZEN_AUTHORITY_INITIAL_CHECKPOINT_SHA256_V1 in (actor_only, actor_critic)
    return {
        "codec_id": CHECKPOINT_CODEC_ID_V1,
        "initialization_seed": FROZEN_INITIALIZATION_SEED_V1,
        "construction_order": "ACTOR_THEN_CRITIC_SINGLE_FROZEN_SEED_STREAM",
        "declared_frozen_sha256": FROZEN_AUTHORITY_INITIAL_CHECKPOINT_SHA256_V1,
        "computed_actor_only_sha256": actor_only,
        "computed_actor_plus_critic_sha256": actor_critic,
        "declared_hash_reproduced": bool(reproduced),
        "contract_state": "MATCH" if reproduced else "CONTRACT_MISMATCH",
    }


def write_initial_checkpoint_identity_artifact_v1(path: str | Path) -> Mapping[str, Any]:
    """Emit the machine-readable exact-SHA Shanxi checkpoint identity artifact."""
    import platform

    identity = dict(initial_checkpoint_identity_v1())
    payload = {
        "schema": "CB16_R11_POST_CC_S1_INITIAL_CHECKPOINT_IDENTITY_V1",
        "status": "EMITTED",
        "role": "PRE_QUALIFICATION_VERSIONED_AUTHORITY_CORRECTION_EVIDENCE",
        "codec_id": identity["codec_id"],
        "initialization_seed": identity["initialization_seed"],
        "construction_order": identity["construction_order"],
        "modules": ["actor", "critic"],
        "declared_frozen_sha256": identity["declared_frozen_sha256"],
        "computed_actor_only_sha256": identity["computed_actor_only_sha256"],
        "computed_actor_plus_critic_sha256": identity["computed_actor_plus_critic_sha256"],
        "declared_hash_reproduced": identity["declared_hash_reproduced"],
        "contract_state": identity["contract_state"],
        "historical_v1_mutated": False,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def load_corrective_authority_v1(repo_root: str | Path) -> Mapping[str, Any] | None:
    path = (
        Path(repo_root)
        / "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_INITIAL_CHECKPOINT_IDENTITY_V1.json"
    )
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def effective_initial_checkpoint_contract_v1(repo_root: str | Path) -> Mapping[str, Any]:
    """Resolve the effective S1 initial-checkpoint contract.

    * historical V1 remains the declared fallback and stays non-reproducible;
    * a Sol-frozen versioned corrective authority can supersede it, but only if
      it binds the exact computed Actor+Critic hash/codec/seed/order.
    """
    identity = dict(initial_checkpoint_identity_v1())
    authority = load_corrective_authority_v1(repo_root)
    if authority is None:
        return {
            "contract_state": identity["contract_state"],
            "authority_source": "HISTORICAL_V1_DECLARED",
            "authority_status": "MISSING_VERSIONED_CORRECTIVE_AUTHORITY",
            "effective_frozen_sha256": identity["declared_frozen_sha256"],
            "computed_actor_plus_critic_sha256": identity["computed_actor_plus_critic_sha256"],
            "codec_id": identity["codec_id"],
        }
    frozen_sha = authority.get("frozen_actor_plus_critic_sha256")
    matches = bool(
        frozen_sha == identity["computed_actor_plus_critic_sha256"]
        and authority.get("codec_id") == identity["codec_id"]
        and int(authority.get("initialization_seed", -1)) == int(identity["initialization_seed"])
        and authority.get("construction_order") == identity["construction_order"]
    )
    frozen_by_sol = authority.get("status") == "FROZEN_BY_SOL"
    return {
        "contract_state": "MATCH" if (frozen_by_sol and matches) else "CONTRACT_MISMATCH",
        "authority_source": "VERSIONED_CORRECTIVE_AUTHORITY",
        "authority_status": authority.get("status"),
        "effective_frozen_sha256": frozen_sha,
        "computed_actor_plus_critic_sha256": identity["computed_actor_plus_critic_sha256"],
        "codec_id": identity["codec_id"],
        "frozen_by_sol": bool(frozen_by_sol),
        "binding_matches_computed_identity": bool(matches),
    }


def assert_frozen_initial_checkpoint_v1() -> Mapping[str, Any]:
    """Fail closed for qualification when the authority hash is not reproduced."""
    identity = dict(initial_checkpoint_identity_v1())
    if not identity["declared_hash_reproduced"]:
        raise S1InitialCheckpointContractError(
            "FROZEN_INITIAL_CHECKPOINT_SHA256_NOT_REPRODUCED_UNDER_DECLARED_CODEC"
        )
    return identity
