"""Bounded retry of LLM transport / server-loss errors (R1 C3 prep, 19 Oct 2026).

Motivation: the rerun-2 STOP (docs/r1/reports/RERUN2_STOP_DIAGNOSIS_20261019.md in
the paper repository) lost seven leaves to a terminated Ollama runner
("llama runner process no longer running", "unexpected EOF").  This module
retries such *transport* failures a bounded number of times:

* at most ``max_attempts`` attempts (default 3) with a fixed backoff;
* only for transport / server-loss errors: connection refused or reset,
  server disconnects and unexpected EOF, a terminated llama runner, HTTP 5xx;
* the same request is re-sent, so the D26 seed is identical on every attempt;
* every attempt is logged (attempt index, error class) and the log is put in
  the ``llm_call`` event by the manager;
* parse failures never reach this module (they happen after a successful
  response) and are therefore never retried by it; timeouts and other errors
  are not retried either.

``TransportRetryPolicy(max_attempts=1)`` (``LEGACY_POLICY``) is the historical
no-retry behaviour.  On a final failure the original exception is re-raised
with the attempt log attached as ``exc.llm_call_attempts``.

ENGINEERING_ONLY.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

POLICY_VERSION = "bounded_transport_retry/v1"


@dataclass(frozen=True)
class TransportRetryPolicy:
    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = (2.0, 4.0)

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts must be an int >= 1")
        if any(float(b) < 0 for b in self.backoff_seconds):
            raise ValueError("backoff must be non-negative")

    def delay(self, failed_attempt: int) -> float:
        """Seconds to wait after the *failed_attempt*-th (1-based) attempt."""
        if not self.backoff_seconds:
            return 0.0
        index = min(failed_attempt - 1, len(self.backoff_seconds) - 1)
        return float(self.backoff_seconds[index])

    @property
    def enabled(self) -> bool:
        return self.max_attempts > 1

    def to_dict(self) -> dict[str, Any]:
        return {"version": POLICY_VERSION, "max_attempts": self.max_attempts,
                "backoff_seconds": [float(b) for b in self.backoff_seconds]}


DEFAULT_POLICY = TransportRetryPolicy()
LEGACY_POLICY = TransportRetryPolicy(max_attempts=1, backoff_seconds=())

_CLASS_NAMES = {
    "ConnectError": "connect_error",                # httpx: connection refused / unreachable
    "RemoteProtocolError": "server_disconnected",   # httpx: server closed without a response
    "ReadError": "read_error",                      # httpx: connection reset while reading
    "WriteError": "write_error",
    "IncompleteRead": "incomplete_read",
    "RemoteDisconnected": "server_disconnected",
}


def _chain(exc: BaseException):
    seen: set[int] = set()
    stack = [exc]
    while stack and len(seen) < 8:
        current = stack.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack += [current.__cause__, current.__context__]


def _status_code(exc: BaseException) -> int | None:
    for candidate in (getattr(exc, "status_code", None),
                      getattr(getattr(exc, "response", None), "status_code", None)):
        if isinstance(candidate, int):
            return candidate
    return None


def classify_transport_error(exc: BaseException) -> str | None:
    """The retryable transport-error class of *exc*, or ``None`` when it must not be retried."""
    for current in _chain(exc):
        message = str(current).lower()
        if "llama runner process no longer running" in message:
            return "runner_terminated"
        if "unexpected eof" in message:
            return "unexpected_eof"
        if isinstance(current, ConnectionRefusedError) or "connection refused" in message:
            return "connection_refused"
        if isinstance(current, ConnectionResetError) or "connection reset" in message:
            return "connection_reset"
        if isinstance(current, (BrokenPipeError, ConnectionAbortedError)):
            return "connection_aborted"
        names = [cls.__name__ for cls in type(current).__mro__]
        if any("Timeout" in name for name in names):
            return None  # timeouts are not server loss
        for name in names:
            if name in _CLASS_NAMES:
                return _CLASS_NAMES[name]
        code = _status_code(current)
        if code is not None and 500 <= code <= 599:
            return "server_5xx"
        if isinstance(current, ConnectionError):
            return "connection_error"
    return None


def _entry(attempt: int, exc: BaseException | None, error_class: str | None, seconds: float) -> dict[str, Any]:
    entry: dict[str, Any] = {"attempt": attempt, "error_class": error_class, "seconds": round(seconds, 6)}
    if exc is not None:
        entry["error_type"] = type(exc).__name__
        entry["error"] = str(exc)[:300]
    return entry


def invoke_with_retry(
    call: Callable[[], Any],
    policy: TransportRetryPolicy = DEFAULT_POLICY,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, list[dict[str, Any]]]:
    """Run *call* under *policy*; return its result and the attempt log.

    The log has one entry per attempt (1-based ``attempt``; ``error_class``
    ``None`` for the successful one, ``"non_retryable"`` for an error that is
    not a transport error).  On a final failure the exception is re-raised
    with ``exc.llm_call_attempts`` set to the log.
    """
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, policy.max_attempts + 1):
        started = time.perf_counter()
        try:
            result = call()
        except Exception as exc:  # noqa: BLE001 - classified below, re-raised unless retryable
            error_class = classify_transport_error(exc)
            attempts.append(_entry(attempt, exc, error_class or "non_retryable", time.perf_counter() - started))
            if error_class is None or attempt == policy.max_attempts:
                try:
                    exc.llm_call_attempts = list(attempts)  # type: ignore[attr-defined]
                    exc.llm_call_retry_policy = policy.to_dict()  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - exotic exceptions without __dict__
                    pass
                raise
            sleep(policy.delay(attempt))
            continue
        attempts.append(_entry(attempt, None, None, time.perf_counter() - started))
        return result, attempts
    raise AssertionError("unreachable")  # pragma: no cover
