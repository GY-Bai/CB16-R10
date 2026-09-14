"""R4 sitecustomize hook: install write-path instrumentation when enabled."""

from __future__ import annotations

import os

if os.environ.get("CB16_R4_METRICS_DIR") and os.environ.get("CB16_R4_MONITORED_ROOTS"):
    import cb16_r4_write_path_instrument as _instrument

    _instrument.install()
