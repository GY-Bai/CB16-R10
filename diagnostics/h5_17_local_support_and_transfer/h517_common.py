"""Shared frozen primitives for the H5.17 diagnostic harness.

Every function here is a byte-faithful re-statement of the frozen H5.16 helpers
(cb16_local_opt.full_state_nonlinear_utility_invariance_h516) so that the
diagnostic runs use exactly the same estimator, weights and loss as the
adjudicated H5.16 execution.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import (
    H516_DIM,
    H516_SEED,
    H516_UTILITY_DIM,
)

__all__ = ["frozen_regressor", "row_weights", "fit_predict"]


def frozen_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        criterion="squared_error",
        max_features=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=5,
        bootstrap=True,
        max_samples=None,
        ccp_alpha=0.0,
        random_state=H516_SEED,
        n_jobs=1,
    )


def row_weights(timestamps: Sequence[int]) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    clocks, counts = np.unique(ts, return_counts=True)
    count_by_clock = {int(c): int(n) for c, n in zip(clocks.tolist(), counts.tolist())}
    return np.asarray(
        [1.0 / (len(clocks) * count_by_clock[int(t)] * 6.0) for t in ts for _ in range(6)],
        dtype=np.float64,
    )


def fit_predict(train_x, train_y, train_ts, eval_x) -> np.ndarray:
    ex = np.asarray(eval_x, dtype=np.float64)
    model = frozen_regressor()
    model.fit(
        np.asarray(train_x, dtype=np.float64).reshape(-1, H516_DIM),
        np.asarray(train_y, dtype=np.float64).reshape(-1, H516_UTILITY_DIM),
        sample_weight=row_weights(train_ts),
    )
    return np.asarray(model.predict(ex.reshape(-1, H516_DIM)), dtype=np.float64).reshape(
        ex.shape[0], ex.shape[1], H516_UTILITY_DIM
    )
