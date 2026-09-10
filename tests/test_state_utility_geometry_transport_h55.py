from __future__ import annotations

import inspect
import numpy as np

from cb16_local_opt.state_utility_geometry_transport_h55 import H55_METRICS, adjudicate_h55, spearman_h55, _metric_readout


def test_metric_registry_exact():
    assert tuple(H55_METRICS) == ("FULL102", "MARKET96", "OPERATOR48", "MEDIUM48", "ACCOUNT6")
    assert len(H55_METRICS["FULL102"]) == 102
    assert len(H55_METRICS["MARKET96"]) == 96
    assert len(H55_METRICS["OPERATOR48"]) == 48
    assert len(H55_METRICS["MEDIUM48"]) == 48
    assert len(H55_METRICS["ACCOUNT6"]) == 6


def test_spearman_monotone_and_reverse():
    x = [0,1,2,3,4]
    assert abs(spearman_h55(x, x) - 1.0) < 1e-15
    assert abs(spearman_h55(x, list(reversed(x))) + 1.0) < 1e-15


def test_metric_readout_signature_has_no_teacher_or_future_selector():
    params = set(inspect.signature(_metric_readout).parameters)
    forbidden = {"teacher", "kernel", "top_k", "future_utility_selector", "support_selector"}
    assert not (params & forbidden)


def _fold(fold, aligned, rotated):
    return {
        "fold": fold,
        "aligned": {m: {"rho": aligned[m], "raw_rho": aligned[m], "nearest_decile_centered_mse_ratio": 0.8} for m in H55_METRICS},
        "rotated": {s: {m: {"rho": rotated[m][s]} for m in H55_METRICS} for s in (1,7,13,23,31)},
        "rotation_receipts": {s: {"target_feature_multiset_preserved": True, "train_features_byte_identical": True, "scenario_identity_preserved": True} for s in (1,7,13,23,31)},
    }


def test_full102_pass_classification():
    rows=[]
    for f in range(1,6):
        aligned={m:0.4 for m in H55_METRICS}
        rotated={m:{s:0.1 for s in (1,7,13,23,31)} for m in H55_METRICS}
        rows.append(_fold(f, aligned, rotated))
    out=adjudicate_h55(rows)
    assert out["classification"]=="FULL_STATE_GEOMETRY_TRANSPORT_SUPPORTED"
    assert out["metrics"]["FULL102"]["transport_supported"] is True


def test_market_nontransport_classification():
    rows=[]
    for f in range(1,6):
        aligned={m:(0.3 if m=="ACCOUNT6" else -0.1) for m in H55_METRICS}
        rotated={m:{s:0.0 for s in (1,7,13,23,31)} for m in H55_METRICS}
        rows.append(_fold(f, aligned, rotated))
    out=adjudicate_h55(rows)
    assert out["classification"]=="MARKET_STATE_UTILITY_GEOMETRY_TEMPORAL_NONTRANSPORT"
