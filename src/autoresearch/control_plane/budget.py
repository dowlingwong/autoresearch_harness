from __future__ import annotations

from dataclasses import asdict, dataclass


class BudgetExceededError(RuntimeError):
    """Raised when a campaign tries to consume beyond its fixed trial budget."""


@dataclass(frozen=True)
class BudgetState:
    total_trials: int
    consumed_trials: int = 0

    def __post_init__(self) -> None:
        if self.total_trials < 1:
            raise ValueError("total_trials must be >= 1")
        if self.consumed_trials < 0:
            raise ValueError("consumed_trials must be >= 0")
        if self.consumed_trials > self.total_trials:
            raise BudgetExceededError("consumed_trials cannot exceed total_trials")

    @property
    def next_budget_index(self) -> int:
        return self.consumed_trials + 1

    @property
    def exhausted(self) -> bool:
        return self.consumed_trials >= self.total_trials

    def consume_one(self) -> "BudgetState":
        if self.exhausted:
            raise BudgetExceededError("fixed trial budget exhausted")
        return BudgetState(total_trials=self.total_trials, consumed_trials=self.consumed_trials + 1)

    def to_dict(self) -> dict[str, int | bool]:
        payload = asdict(self)
        payload["next_budget_index"] = self.next_budget_index
        payload["exhausted"] = self.exhausted
        return payload


@dataclass(frozen=True)
class ComputeBudget:
    """Compute + LLM-call budget, tracked next to the trial-count ``BudgetState``.

    Each cap is optional.  When all caps are ``None`` (the default, which is
    what a plain ``--budget N`` campaign uses) the budget is never exhausted
    and behaviour is unchanged.  Counters are always maintained so that the
    controller view can report remaining headroom.

    The campaign loops count every worker run (verify re-runs included), the
    worker seconds, and every LLM call: proposals and revisions from
    non-deterministic managers.  Deterministic managers (``deterministic =
    True``, e.g. baseline and prompt) make no LLM call and consume none.
    """

    max_worker_runs: int | None = None
    max_worker_seconds: float | None = None
    max_llm_calls: int | None = None
    worker_runs: int = 0
    worker_seconds: float = 0.0
    llm_calls: int = 0

    def __post_init__(self) -> None:
        for value, name in (
            (self.max_worker_runs, "max_worker_runs"),
            (self.max_llm_calls, "max_llm_calls"),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.max_worker_seconds is not None and self.max_worker_seconds < 0:
            raise ValueError("max_worker_seconds must be >= 0")
        if self.worker_runs < 0 or self.llm_calls < 0 or self.worker_seconds < 0:
            raise ValueError("compute counters must be >= 0")

    @property
    def exhausted(self) -> bool:
        if self.max_worker_runs is not None and self.worker_runs >= self.max_worker_runs:
            return True
        if self.max_worker_seconds is not None and self.worker_seconds >= self.max_worker_seconds:
            return True
        if self.max_llm_calls is not None and self.llm_calls >= self.max_llm_calls:
            return True
        return False

    def consume_worker_run(self, seconds: float = 0.0) -> "ComputeBudget":
        return ComputeBudget(
            max_worker_runs=self.max_worker_runs,
            max_worker_seconds=self.max_worker_seconds,
            max_llm_calls=self.max_llm_calls,
            worker_runs=self.worker_runs + 1,
            worker_seconds=self.worker_seconds + max(float(seconds), 0.0),
            llm_calls=self.llm_calls,
        )

    def consume_llm_call(self, count: int = 1) -> "ComputeBudget":
        return ComputeBudget(
            max_worker_runs=self.max_worker_runs,
            max_worker_seconds=self.max_worker_seconds,
            max_llm_calls=self.max_llm_calls,
            worker_runs=self.worker_runs,
            worker_seconds=self.worker_seconds,
            llm_calls=self.llm_calls + count,
        )

    def remaining(self) -> dict[str, float | int | None]:
        def _left(cap, used):
            return None if cap is None else max(cap - used, 0)

        return {
            "worker_runs_remaining": _left(self.max_worker_runs, self.worker_runs),
            "worker_seconds_remaining": _left(self.max_worker_seconds, self.worker_seconds),
            "llm_calls_remaining": _left(self.max_llm_calls, self.llm_calls),
            "worker_runs_used": self.worker_runs,
            "worker_seconds_used": self.worker_seconds,
            "llm_calls_used": self.llm_calls,
        }

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["exhausted"] = self.exhausted
        payload.update(self.remaining())
        return payload

