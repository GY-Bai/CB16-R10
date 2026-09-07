"""CB16 R10 read-only observability and diagnostic sidecars.

These modules are explicitly non-status-driving and must not mutate canonical scientific state.
"""

__all__ = [
    "model_probe",
    "postrun",
    "runtime_observer",
]
