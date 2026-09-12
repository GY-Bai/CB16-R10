from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Callable, Sequence, TypeVar

from .cc_fast_policy_broker_r0 import PolicyRequest, PolicyResponse

T = TypeVar("T")


def run_with_deadline(fn: Callable[[], T], *, timeout_s: float, failure_code: str) -> T:
    if timeout_s <= 0:
        raise ValueError("RUNTIME_DEADLINE_INVALID")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cc-fast-deadline")
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(failure_code) from exc
    finally:
        if future.done():
            executor.shutdown(wait=True, cancel_futures=True)


def validate_policy_response_alignment(requests: Sequence[PolicyRequest], responses: Sequence[PolicyResponse]) -> None:
    if len(requests) != len(responses):
        raise RuntimeError("BROKER_RESPONSE_COUNT_MISMATCH")
    for request, response in zip(requests, responses):
        if (
            response.account_lineage_id != request.account_lineage_id
            or response.decision_index != request.decision_index
            or response.policy_generation != request.policy_generation
            or response.policy_sha256 != request.policy_sha256
            or response.rng_stream_id != request.rng_stream_id
            or response.rng_counter != request.rng_counter
        ):
            raise RuntimeError("OUT_OF_ORDER_OR_CONTAMINATED_POLICY_RESPONSE")
