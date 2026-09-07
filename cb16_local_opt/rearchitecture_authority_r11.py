from __future__ import annotations

"""R11 semantic authority guard.

R11 deliberately stops treating the legacy Python runtime as scientific authority.
The immutable authority is instead the market-data identity, frozen sensory taps/assets,
Central-Brain functional architecture, authority separation, frozen Physics, and the
probabilistic learning contracts declared in CB16_SEMANTIC_FREEZE_V1.json.

This module contains two kinds of checks:

1. dependency-free static contract checks suitable for lightweight CI;
2. Shanxi/runtime checks and a read-only dataset sealing mechanism.

The data sealer NEVER mutates market data and NEVER opens archive payloads at or after
the unopened holdout boundary.  For those archives it binds filename/size to Binance's
pre-existing CHECKSUM sidecar and defers payload SHA verification until a scientifically
authorized opening.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


SEMANTIC_FREEZE_REL = "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
LAYER_CONTRACT_REL = "assets/FROZEN_LAYER_EXTRACTION_CONTRACT_R10.json"
PROVISION_REL = "provision/assets/binance_usdm_1m_funding_2020_2026.json"
ACTION_INTENT_REL = "authority/control_plane_r1/ACTION_INTENT_SCHEMA_V1.json"
RISK_AUTHORITY_REL = "authority/control_plane_r1/RISK_AUTHORITY_CONTRACT_V1.json"
PHYSICS_CONTRACT_REL = (
    "authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0/"
    "ACCOUNT_PHYSICS_CONTRACT_V1.json"
)

TEN_SYMBOLS_R11 = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "SOLUSDT",
)
UNOPENED_HOLDOUT_MONTH_R11 = "2025-09"

_KLINE_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)-1m-(?P<ym>\d{4}-\d{2})\.zip$")
_FUNDING_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)-fundingRate-(?P<ym>\d{4}-\d{2})\.zip$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"R11_EXPECTED_JSON_OBJECT:{p}")
    return obj


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_semantic_freeze(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    obj = _load_json(root / SEMANTIC_FREEZE_REL)
    _expect(obj.get("schema") == "CB16_R11_SEMANTIC_FREEZE_V1", "R11_FREEZE_SCHEMA_DRIFT")
    _expect(obj.get("status") == "FROZEN", "R11_FREEZE_NOT_FROZEN")
    _expect(obj.get("scientific_semantics_changed") is False, "R11_FREEZE_SEMANTIC_FLAG_DRIFT")
    return obj


def verify_static_semantic_contracts(repo_root: str | Path) -> dict[str, Any]:
    """Dependency-free verification of the semantic authority declarations.

    This intentionally validates meanings/identities rather than hashes of old Python source.
    It is safe to run in GitHub-hosted lightweight CI without torch/safetensors.
    """

    root = Path(repo_root).resolve()
    freeze = load_semantic_freeze(root)
    imm = freeze["immutable"]

    data_contract = imm["historical_market_dataset"]
    _expect(tuple(data_contract["symbols"]) == TEN_SYMBOLS_R11, "R11_TEN_SYMBOL_SET_DRIFT")
    _expect(data_contract["mutation"] == "FORBIDDEN", "R11_DATA_MUTATION_POLICY_DRIFT")
    _expect(data_contract["unopened_holdout_start"].startswith(UNOPENED_HOLDOUT_MONTH_R11), "R11_HOLDOUT_BOUNDARY_DRIFT")

    provision = _load_json(root / PROVISION_REL)
    _expect(provision.get("asset_id") == "binance_usdm_1m_funding_2020_2026", "R11_PROVISION_ASSET_DRIFT")
    _expect(provision.get("type") == "dataset" and provision.get("required") is True, "R11_PROVISION_REQUIRED_DRIFT")
    _expect(
        provision.get("integrity", {}).get("manifest_sha256") == data_contract["provision_manifest_sha256"],
        "R11_PROVISION_MANIFEST_IDENTITY_DRIFT",
    )
    _expect(
        int(provision.get("integrity", {}).get("size_bytes", -1)) == int(data_contract["discovered_size_bytes"]),
        "R11_PROVISION_DISCOVERED_SIZE_DRIFT",
    )

    sensory = imm["frozen_sensory_organs"]
    layer = _load_json(root / LAYER_CONTRACT_REL)
    _expect(layer.get("status") == "FROZEN", "R11_LAYER_CONTRACT_NOT_FROZEN")
    _expect(layer.get("remote") == "OFF" and sensory.get("remote") == "OFF", "R11_REMOTE_POLICY_DRIFT")
    op = layer["operator48"]
    med = layer["medium48"]
    _expect(op["historical_scientific_tap"] == sensory["operator48"]["tap"], "R11_OPERATOR_TAP_DRIFT")
    _expect(op["frozen_reducer_sha256"] == sensory["operator48"]["reducer_sha256"], "R11_OPERATOR_REDUCER_DRIFT")
    _expect(
        layer["active_prefix_contract"]["kronos_model_l5"]["semantic_sha256"]
        == sensory["operator48"]["kronos_model_prefix_semantic_sha256"],
        "R11_KRONOS_MODEL_PREFIX_DRIFT",
    )
    _expect(
        layer["active_prefix_contract"]["kronos_tokenizer_encode"]["semantic_sha256"]
        == sensory["operator48"]["kronos_tokenizer_prefix_semantic_sha256"],
        "R11_KRONOS_TOKENIZER_PREFIX_DRIFT",
    )
    _expect(med["historical_scientific_tap"] == sensory["medium48"]["tap"], "R11_TIMESFM_TAP_DRIFT")
    _expect(med["adapter_sha256"] == sensory["medium48"]["adapter_sha256"], "R11_MEDIUM_ADAPTER_DRIFT")
    _expect(
        layer["active_prefix_contract"]["timesfm_layer3"]["semantic_sha256"]
        == sensory["medium48"]["timesfm_layer3_prefix_semantic_sha256"],
        "R11_TIMESFM_PREFIX_DRIFT",
    )

    authority = imm["authority_separation"]
    _expect(authority["principle"] == "TRUTH != BELIEF != DECISION != PERMISSION", "R11_AUTHORITY_PRINCIPLE_DRIFT")

    action = _load_json(root / ACTION_INTENT_REL)
    forbidden = set(action.get("forbidden_semantics", []))
    _expect({"predicted_pnl", "confidence", "hidden_market_state_copy"}.issubset(forbidden), "R11_ACTION_INTENT_FORBIDDEN_SEMANTICS_DRIFT")
    risk_prop = action["properties"]["requested_risk_multiplier"]
    _expect(float(risk_prop["minimum"]) == 0.0 and float(risk_prop["maximum"]) == 1.0, "R11_REQUESTED_RISK_RANGE_DRIFT")

    risk = _load_json(root / RISK_AUTHORITY_REL)
    _expect(risk.get("owner") == "CONTROL_PLANE_EXTERNAL_STATE", "R11_RISK_AUTHORITY_OWNER_DRIFT")
    _expect(risk.get("physics_mutation_policy") == "PRESERVE_UNCHANGED", "R11_RISK_PHYSICS_MUTATION_DRIFT")
    _expect(risk.get("step_mutation") == "NONE", "R11_RISK_STEP_MUTATION_DRIFT")

    physics = _load_json(root / PHYSICS_CONTRACT_REL)
    _expect(physics.get("contract_sha256") == authority["physics_contract_sha256"], "R11_PHYSICS_CONTRACT_DRIFT")
    _expect(int(physics["sim_config"]["max_holding_bars"]) == 72, "R11_H72_PHYSICS_DRIFT")
    _expect(physics["action_contract"]["decision"] == {"FLAT": 1, "LONG": 2, "SHORT": 0}, "R11_ACTION_MAPPING_DRIFT")
    _expect(physics["state_separation"]["brain_visible"] == "AccountStatePacketV1 6D observation only", "R11_ACCOUNT_OBSERVATION_DRIFT")

    brain = imm["central_brain_architecture"]
    _expect(brain["tier"] == "TIER_1", "R11_BRAIN_TIER_DRIFT")
    _expect(brain["input_boundary"] == {"Operator48": 48, "Medium48": 48, "AccountState6": 6}, "R11_BRAIN_INPUT_DRIFT")
    _expect(int(brain["parameter_count"]) == 189052, "R11_BRAIN_PARAMETER_CONTRACT_DRIFT")
    _expect(brain["action_composition"]["requested_risk_is_confidence"] is False, "R11_RISK_CONFIDENCE_SEMANTIC_DRIFT")

    learning = imm["probabilistic_learning_semantics"]
    _expect(learning["target"] == "P(U | I_t, a)", "R11_PROBABILISTIC_TARGET_DRIFT")
    _expect(learning["realized_outcome_role"] == "ONE_REALIZATION_SAMPLE_NOT_CORRECT_ACTION_LABEL", "R11_OUTCOME_LABEL_DRIFT")
    _expect(learning["train_teacher"]["mode"] == "BLOCKED_CROSSFIT", "R11_TRAIN_TEACHER_MODE_DRIFT")
    _expect(learning["validation_teacher"]["mode"] == "PREQUENTIAL", "R11_VALIDATION_TEACHER_MODE_DRIFT")
    _expect(learning["champion_challenger"]["F0_F1_F2_F3_are_promotion_drivers"] is False, "R11_CONTROL_PROMOTION_DRIFT")

    history = imm["historical_knowledge_policy"]
    _expect(history["recent_evidence_may_erase_historical_information"] is False, "R11_HISTORY_ERASURE_DRIFT")
    _expect(history["hand_engineered_cycle_or_regime_activation_engine"] == "FORBIDDEN", "R11_HANDCRAFTED_CYCLE_DRIFT")

    return {
        "schema": "CB16_R11_STATIC_SEMANTIC_GUARD_V1",
        "status": "PASS",
        "scientific_semantics_changed": False,
        "symbols": list(TEN_SYMBOLS_R11),
        "unopened_holdout_month": UNOPENED_HOLDOUT_MONTH_R11,
        "legacy_python_is_runtime_authority": False,
    }


def verify_legacy_brain_oracle(repo_root: str | Path) -> dict[str, Any]:
    """Use the current TIER_1 module only as an oracle for the frozen functional shape.

    New R11 implementations do not have to preserve module names or source structure.
    """

    root = Path(repo_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch  # lazy: lightweight static CI does not need torch
    from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10

    model = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    expected_shapes = {
        "operator_encoder.0.weight": (64, 48),
        "operator_encoder.0.bias": (64,),
        "medium_encoder.0.weight": (64, 48),
        "medium_encoder.0.bias": (64,),
        "account_encoder.0.weight": (32, 6),
        "account_encoder.0.bias": (32,),
        "shared_core.0.weight": (256, 160),
        "shared_core.0.bias": (256,),
        "shared_core.2.weight": (256, 256),
        "shared_core.2.bias": (256,),
        "direction_body.0.weight": (128, 256),
        "direction_body.0.bias": (128,),
        "direction_out.weight": (3, 128),
        "direction_out.bias": (3,),
        "direction_embedding.weight": (3, 8),
        "sizing_body.0.weight": (128, 264),
        "sizing_body.0.bias": (128,),
        "sizing_body.2.weight": (64, 128),
        "sizing_body.2.bias": (64,),
        "sizing_out.weight": (1, 64),
        "sizing_out.bias": (1,),
    }
    state = model.state_dict()
    _expect(set(state) == set(expected_shapes), "R11_LEGACY_BRAIN_ORACLE_KEYSET_DRIFT")
    for key, shape in expected_shapes.items():
        _expect(tuple(state[key].shape) == tuple(shape), f"R11_LEGACY_BRAIN_ORACLE_SHAPE_DRIFT:{key}")
    parameter_count = sum(int(p.numel()) for p in model.parameters())
    _expect(parameter_count == 189052, f"R11_LEGACY_BRAIN_PARAMETER_COUNT_DRIFT:{parameter_count}")
    _expect(all(p.dtype == torch.float32 for p in model.parameters()), "R11_LEGACY_BRAIN_NON_FP32_PARAMETER")
    return {
        "schema": "CB16_R11_LEGACY_BRAIN_ORACLE_GUARD_V1",
        "status": "PASS",
        "parameter_count": parameter_count,
        "dtype": "torch.float32",
    }


def verify_installed_frozen_prefixes(package_root: str | Path) -> dict[str, Any]:
    """Deep package check over the exact frozen sensory tensor prefixes.

    Requires torch+safetensors and is intended for Shanxi/runtime qualification, not
    lightweight GitHub-hosted CI.
    """

    root = Path(package_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cb16_local_opt.frozen_layer_contract_r10 import verify_installed_prefix

    rows = {}
    for prefix_id in ("kronos_model_l5", "kronos_tokenizer_encode", "timesfm_layer3"):
        rows[prefix_id] = verify_installed_prefix(root, prefix_id)
    for rel, expected in (
        ("assets/operator/operator_reducers_v1.npz", "c61b20341ea7d859842821bfd401baad17b210d02317da6324be7de9d3d54423"),
        ("assets/medium/CANONICAL_NONLINEAR48_SEED24680_PORTABLE.npz", "897ba29937817b225df73c6ad2b5bbf183d2a18a0488dcdcff5d7b43ade436c2"),
    ):
        actual = sha256_file(root / rel)
        _expect(actual == expected, f"R11_FROZEN_ASSET_HASH_DRIFT:{rel}:{actual}")
    return {
        "schema": "CB16_R11_FROZEN_PREFIX_RUNTIME_GUARD_V1",
        "status": "PASS",
        "prefixes": rows,
    }


def _read_checksum_sidecar(zip_path: Path) -> tuple[str, str]:
    sidecar = Path(str(zip_path) + ".CHECKSUM")
    if not sidecar.is_file():
        raise RuntimeError(f"R11_CHECKSUM_SIDECAR_MISSING:{sidecar}")
    raw = sidecar.read_bytes()
    text = raw.decode("utf-8", errors="strict").strip()
    if not text:
        raise RuntimeError(f"R11_CHECKSUM_SIDECAR_EMPTY:{sidecar}")
    expected = text.split()[0].lower()
    if not _HEX64_RE.match(expected):
        raise RuntimeError(f"R11_CHECKSUM_SIDECAR_INVALID:{sidecar}:{expected}")
    return expected, sha256_bytes(raw)


def _archive_month(path: Path, *, kind: str, symbol: str) -> str:
    rx = _KLINE_RE if kind == "KLINE_1M" else _FUNDING_RE
    m = rx.match(path.name)
    if not m or m.group("symbol") != symbol:
        raise RuntimeError(f"R11_ARCHIVE_NAME_INVALID:{kind}:{path}")
    return str(m.group("ym"))


def _iter_archives(data_root: Path, symbol: str, *, kind: str) -> Iterable[Path]:
    if kind == "KLINE_1M":
        d = data_root / "klines_1m" / symbol
        pattern = f"{symbol}-1m-????-??.zip"
    elif kind == "FUNDING":
        d = data_root / "fundingRate" / symbol
        pattern = f"{symbol}-fundingRate-????-??.zip"
    else:
        raise ValueError(kind)
    if not d.is_dir():
        raise FileNotFoundError(d)
    rows = sorted(d.glob(pattern))
    if not rows:
        raise RuntimeError(f"R11_NO_ARCHIVES:{kind}:{symbol}")
    return rows


def _ensure_exact_kline_symbol_set(data_root: Path) -> None:
    kroot = data_root / "klines_1m"
    if not kroot.is_dir():
        raise FileNotFoundError(kroot)
    actual = tuple(sorted(p.name for p in kroot.iterdir() if p.is_dir()))
    expected = tuple(sorted(TEN_SYMBOLS_R11))
    _expect(actual == expected, f"R11_KLINE_SYMBOL_SET_DRIFT:{actual}!={expected}")


def build_read_only_dataset_seal(data_root: str | Path) -> dict[str, Any]:
    """Build a deterministic seal from already-present Shanxi data without mutation.

    For months before 2025-09, the archive itself is SHA256-read and checked against its
    Binance CHECKSUM sidecar.  For 2025-09 and later, the ZIP payload is never opened or
    hashed; only pathname, byte size and the pre-existing CHECKSUM sidecar are bound.
    """

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    _ensure_exact_kline_symbol_set(root)

    entries: list[dict[str, Any]] = []
    hashed_bytes = 0
    unopened_bytes = 0
    for symbol in TEN_SYMBOLS_R11:
        for kind in ("KLINE_1M", "FUNDING"):
            for path in _iter_archives(root, symbol, kind=kind):
                ym = _archive_month(path, kind=kind, symbol=symbol)
                size = int(path.stat().st_size)
                expected, sidecar_sha = _read_checksum_sidecar(path)
                rel = path.relative_to(root).as_posix()
                if ym < UNOPENED_HOLDOUT_MONTH_R11:
                    actual = sha256_file(path)
                    if actual != expected:
                        raise RuntimeError(f"R11_EXISTING_ARCHIVE_CHECKSUM_MISMATCH:{rel}:{actual}!={expected}")
                    mode = "HASHED_CONSUMED_HISTORY"
                    archive_sha = actual
                    hashed_bytes += size
                else:
                    mode = "UNOPENED_CHECKSUM_BOUND"
                    archive_sha = None
                    unopened_bytes += size
                entries.append({
                    "kind": kind,
                    "symbol": symbol,
                    "month": ym,
                    "relative_path": rel,
                    "size_bytes": size,
                    "mode": mode,
                    "archive_sha256": archive_sha,
                    "official_expected_sha256": expected,
                    "checksum_sidecar_sha256": sidecar_sha,
                })

    entries.sort(key=lambda x: (x["kind"], x["symbol"], x["month"], x["relative_path"]))
    download_manifest = root / "DOWNLOAD_MANIFEST.json"
    manifest_binding = None
    if download_manifest.is_file():
        manifest_binding = {
            "relative_path": "DOWNLOAD_MANIFEST.json",
            "size_bytes": int(download_manifest.stat().st_size),
            "sha256": sha256_file(download_manifest),
        }
    base = {
        "schema": "CB16_R11_SHANXI_DATASET_SEAL_V1",
        "status": "SEALED_EXISTING_BYTES_READ_ONLY",
        "scientific_semantics_changed": False,
        "symbols": list(TEN_SYMBOLS_R11),
        "unopened_holdout_month": UNOPENED_HOLDOUT_MONTH_R11,
        "unopened_payload_bytes_read": 0,
        "archive_count": len(entries),
        "hashed_consumed_history_bytes": hashed_bytes,
        "unopened_checksum_bound_bytes": unopened_bytes,
        "download_manifest": manifest_binding,
        "entries": entries,
    }
    return {**base, "seal_sha256": sha256_obj(base)}


def write_read_only_dataset_seal(data_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(data_root).resolve()
    out = Path(output_path).resolve()
    try:
        out.relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("R11_DATASET_SEAL_OUTPUT_MUST_NOT_BE_INSIDE_MARKET_DATA_ROOT")
    seal = build_read_only_dataset_seal(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(canonical_json_bytes(seal) + b"\n")
    os.replace(tmp, out)
    return seal


def verify_read_only_dataset_seal(data_root: str | Path, seal_path: str | Path) -> dict[str, Any]:
    expected = _load_json(seal_path)
    _expect(expected.get("schema") == "CB16_R11_SHANXI_DATASET_SEAL_V1", "R11_DATASET_SEAL_SCHEMA_DRIFT")
    seal_hash = expected.get("seal_sha256")
    base = dict(expected)
    base.pop("seal_sha256", None)
    _expect(seal_hash == sha256_obj(base), "R11_DATASET_SEAL_SELF_HASH_DRIFT")
    actual = build_read_only_dataset_seal(data_root)
    _expect(actual["seal_sha256"] == seal_hash, f"R11_DATASET_BYTES_OR_BINDINGS_DRIFT:{actual['seal_sha256']}!={seal_hash}")
    return {
        "schema": "CB16_R11_DATASET_SEAL_VERIFICATION_V1",
        "status": "PASS",
        "seal_sha256": seal_hash,
        "archive_count": actual["archive_count"],
        "unopened_payload_bytes_read": 0,
    }


def _print_json(obj: Mapping[str, Any]) -> None:
    print(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("static")
    p.add_argument("--repo-root", default=".")

    p = sub.add_parser("brain-oracle")
    p.add_argument("--repo-root", default=".")

    p = sub.add_parser("prefixes")
    p.add_argument("--package-root", required=True)

    p = sub.add_parser("seal-data")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("verify-data")
    p.add_argument("--data-root", required=True)
    p.add_argument("--seal", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "static":
        result = verify_static_semantic_contracts(args.repo_root)
    elif args.cmd == "brain-oracle":
        result = verify_legacy_brain_oracle(args.repo_root)
    elif args.cmd == "prefixes":
        result = verify_installed_frozen_prefixes(args.package_root)
    elif args.cmd == "seal-data":
        result = write_read_only_dataset_seal(args.data_root, args.out)
    elif args.cmd == "verify-data":
        result = verify_read_only_dataset_seal(args.data_root, args.seal)
    else:
        raise AssertionError(args.cmd)
    _print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
