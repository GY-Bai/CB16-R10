"""CB16 Shanxi Frozen-Body + Blank-Central-Brain Runtime R10.

Active R10 path: exact frozen Operator48 (Kronos), exact frozen Medium48 (TimesFM),
AccountState6, Ordered4H30 shadow/task-gated risk sidecar, Remote OFF, and a blank
trainable typed Central Brain.  Older R5-R9 modules are retained only as implementation
lineage and learning-infrastructure dependencies; the legacy anonymous Market64 path is
not the R10 active sensory contract.
"""

RUNTIME_VERSION = "CB16_SHANXI_FROZEN_BODY_G0_BRAIN_R10"
RUNTIME_STATUS = "FROZEN_TYPED_SENSORY_BODY_PLUS_BLANK_CENTRAL_BRAIN"

# R10 frozen-body + blank-brain production path.
from .binance_archive_input_r10 import BinanceUSDMArchiveSourceR10, SensoryDecisionFrameR10, aggregate_1m, iter_sensory_frames, ordered4h30_from_hourly
from .frozen_sensory_stack_r10 import FrozenSensoryStackR10, SensoryAssetPathsR10
from .typed_central_brain_r10 import TypedCentralBrainR10, build_g0_brain_r10
from .decision_runtime_r10 import CB16DecisionRuntimeR10
