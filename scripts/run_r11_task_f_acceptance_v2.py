#!/usr/bin/env python3
from __future__ import annotations

"""Task F acceptance entrypoint with concrete hardware placement binding."""

from cb16_local_opt.training_integration_r11 import IntegratedTrainingRuntimeR11
from scripts import run_r11_task_f_acceptance as acceptance

# Task F owns WHERE computation happens.  Replace only the placement-aware runtime
# constructor; all training math, evidence, optimizer, seeds and gates remain Task B's.
acceptance.TrainingRuntimeR11 = IntegratedTrainingRuntimeR11

if __name__ == "__main__":
    raise SystemExit(acceptance.main())
