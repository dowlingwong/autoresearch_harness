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

