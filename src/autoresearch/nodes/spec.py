from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

MetricDirection = Literal["maximize", "minimize"]


class NodeSpecError(ValueError):
    """Raised when a node specification is missing required fields."""


@dataclass(frozen=True)
class BudgetSpec:
    trials: int
    max_wall_clock_hours: float | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "BudgetSpec":
        trials = int(payload.get("trials", 0))
        if trials < 1:
            raise NodeSpecError("default_budget.trials must be >= 1")
        hours = payload.get("max_wall_clock_hours")
        return cls(trials=trials, max_wall_clock_hours=float(hours) if hours is not None else None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodeSpec:
    name: str
    description: str
    editable_paths: tuple[str, ...]
    frozen_paths: tuple[str, ...]
    setup_command: str
    run_command: str
    metric_name: str
    metric_direction: MetricDirection
    metric_parser: str
    acceptance_rule: str
    validity_checks: tuple[str, ...]
    default_budget: BudgetSpec
    expected_runtime: str | None = None
    failure_categories: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "NodeSpec":
        required = (
            "name",
            "description",
            "editable_paths",
            "frozen_paths",
            "setup_command",
            "run_command",
            "metric_name",
            "metric_direction",
            "metric_parser",
            "acceptance_rule",
            "validity_checks",
            "default_budget",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise NodeSpecError(f"node spec missing required fields: {', '.join(missing)}")

        direction = str(payload["metric_direction"])
        if direction not in {"maximize", "minimize"}:
            raise NodeSpecError("metric_direction must be 'maximize' or 'minimize'")

        editable_paths = tuple(str(path) for path in payload["editable_paths"])
        if not editable_paths:
            raise NodeSpecError("editable_paths must not be empty")

        return cls(
            name=str(payload["name"]),
            description=str(payload["description"]),
            editable_paths=editable_paths,
            frozen_paths=tuple(str(path) for path in payload["frozen_paths"]),
            setup_command=str(payload["setup_command"]),
            run_command=str(payload["run_command"]),
            metric_name=str(payload["metric_name"]),
            metric_direction=direction,  # type: ignore[arg-type]
            metric_parser=str(payload["metric_parser"]),
            acceptance_rule=str(payload["acceptance_rule"]),
            validity_checks=tuple(str(check) for check in payload["validity_checks"]),
            default_budget=BudgetSpec.from_mapping(dict(payload["default_budget"])),
            expected_runtime=(
                str(payload["expected_runtime"])
                if payload.get("expected_runtime") is not None
                else None
            ),
            failure_categories=tuple(str(category) for category in payload.get("failure_categories", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["editable_paths"] = list(self.editable_paths)
        payload["frozen_paths"] = list(self.frozen_paths)
        payload["validity_checks"] = list(self.validity_checks)
        payload["failure_categories"] = list(self.failure_categories)
        return payload


def load_node_spec(path: str | Path) -> NodeSpec:
    """Load a node spec from JSON-compatible YAML.

    Stage 2 starts dependency-free. The config file uses JSON syntax, which is
    valid YAML, so this loader can use the standard library.
    """

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NodeSpecError("node spec root must be an object")
    return NodeSpec.from_mapping(payload)

