from __future__ import annotations

import pytest
import torch

from cb16_local_opt.training_integration_r11 import (
    IntegratedTrainingRuntimeR11,
    canonical_runtime_device_r11,
)


def test_cpu_device_binding_is_identity():
    assert canonical_runtime_device_r11("cpu") == torch.device("cpu")
    runtime = IntegratedTrainingRuntimeR11(device="cpu")
    assert runtime.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only placement regression")
def test_cuda_alias_resolves_to_concrete_current_device():
    expected = torch.device("cuda", torch.cuda.current_device())
    assert canonical_runtime_device_r11("cuda") == expected
    assert canonical_runtime_device_r11(expected) == expected
    runtime = IntegratedTrainingRuntimeR11(device="cuda")
    assert runtime.device == expected
