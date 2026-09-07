from __future__ import annotations

"""Task F hardware binding for the R11 FP32 training engine.

Task B deliberately keeps strict device identity checks inside the compute engine.
Task F owns placement, so aliases such as ``cuda`` are resolved to the concrete
CUDA device before the engine is constructed. This changes WHERE work executes,
not the training rule or any scientific identity.
"""

import torch

from .gpu_runtime_r11 import TelemetryConfigR11
from .training_runtime_r11 import EvaluationRuntimeR11, R11TrainingConfig, TrainingRuntimeR11


def canonical_runtime_device_r11(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cuda":
        return dev
    if not torch.cuda.is_available():
        raise RuntimeError("R11_CUDA_REQUESTED_BUT_UNAVAILABLE")
    index = torch.cuda.current_device() if dev.index is None else int(dev.index)
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f"R11_CUDA_DEVICE_INDEX_INVALID:{index}")
    return torch.device("cuda", index)


class IntegratedTrainingRuntimeR11(TrainingRuntimeR11):
    """TrainingRuntimeR11 with Task-F-owned concrete device placement."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        config: R11TrainingConfig | None = None,
        evaluation_runtime: EvaluationRuntimeR11 | None = None,
        telemetry_config: TelemetryConfigR11 | None = None,
    ):
        concrete = canonical_runtime_device_r11(device)
        super().__init__(
            device=concrete,
            config=config,
            evaluation_runtime=evaluation_runtime,
            telemetry_config=telemetry_config,
        )
        if self.device != concrete:
            raise RuntimeError("R11_INTEGRATED_TRAINING_DEVICE_BINDING_DRIFT")


__all__ = ["canonical_runtime_device_r11", "IntegratedTrainingRuntimeR11"]
