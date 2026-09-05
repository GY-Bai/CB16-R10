from __future__ import annotations
from dataclasses import dataclass
import numpy as np

CANARY_POLICY_ID_R101 = "CB16_R10_1_FROZEN_SENSORY_CANARY_CONTRACT_V1"
COSINE_MIN_R101 = 0.999999
HIDDEN_METRICS_R101 = frozenset({"micro_d45", "macro_d45", "timesfm_layer3"})
PACKET_METRICS_R101 = frozenset({"operator48", "medium48"})


def max_abs_error_r101(actual, expected) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def min_row_cosine_r101(actual, expected) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return float("-inf")
    x = a.reshape(a.shape[0], -1)
    y = b.reshape(b.shape[0], -1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return float(np.min(np.sum(x * y, axis=1) / np.maximum(den, 1e-30)))


def evaluate_canary_metric_r101(name: str, actual, expected, *, atol: float, cosine_min: float = COSINE_MIN_R101) -> dict:
    err = max_abs_error_r101(actual, expected)
    cos = min_row_cosine_r101(actual, expected)
    abs_pass = bool(err <= atol)
    cosine_pass = bool(cos >= cosine_min)
    if name in HIDDEN_METRICS_R101:
        rule = "ABS_OR_COSINE"
        passed = abs_pass or cosine_pass
    elif name in PACKET_METRICS_R101:
        rule = "ABS_AND_COSINE"
        passed = abs_pass and cosine_pass
    else:
        raise ValueError(f"UNKNOWN_CANARY_METRIC:{name}")
    return {
        "policy_id": CANARY_POLICY_ID_R101,
        "metric": name,
        "rule": rule,
        "atol": float(atol),
        "cosine_min_required": float(cosine_min),
        "max_abs_error": err,
        "min_cosine": cos,
        "abs_pass": abs_pass,
        "cosine_pass": cosine_pass,
        "pass": bool(passed),
    }


def evaluate_frozen_sensory_canary_r101(checks: dict, *, device: str) -> dict:
    if device == "cpu":
        atol = 5e-5
    elif device.startswith("cuda"):
        atol = 5e-4
    else:
        raise ValueError(f"UNSUPPORTED_DEVICE_POLICY:{device}")
    metrics = {name: evaluate_canary_metric_r101(name, actual, expected, atol=atol) for name, (actual, expected) in checks.items()}
    return {
        "schema": "CB16_R10_1_FROZEN_SENSORY_CANARY_RECEIPT_V1",
        "policy_id": CANARY_POLICY_ID_R101,
        "device": device,
        "atol": atol,
        "hidden_rule": "ABS_OR_COSINE",
        "packet_rule": "ABS_AND_COSINE",
        "metrics": metrics,
        "status": "PASS" if all(x["pass"] for x in metrics.values()) else "FAIL",
    }
