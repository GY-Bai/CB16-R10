from __future__ import annotations

from cb16_local_opt.training_integration_r11 import (
    default_eval_cuda_graph_for_capability_r11,
)


def test_gtx1060_sm61_defaults_to_eager_after_real_machine_microbench():
    assert default_eval_cuda_graph_for_capability_r11((6, 1)) is False


def test_non_cuda_defaults_to_eager_and_other_cuda_is_not_overgeneralized():
    assert default_eval_cuda_graph_for_capability_r11(None) is False
    assert default_eval_cuda_graph_for_capability_r11((7, 5)) is True
    assert default_eval_cuda_graph_for_capability_r11((8, 6)) is True
