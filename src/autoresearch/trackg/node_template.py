"""Track G node template: ``trackg_node/v1`` (M10 step 0; R1-BX-D5).

A Track G node is declared in a YAML file (see
``configs/trackg/node_template.yaml``).  It carries everything the harness
``NodeSpec`` carries for scope enforcement (``editable_paths``,
``frozen_paths``, run command, metric name and direction) plus:

- ``run_budget``: a **fixed step or token budget** (``kind: steps`` or
  ``kind: tokens`` with a positive integer ``value``).  A wall-clock budget is
  refused (R1-BX-D5): so is any key whose name contains a time word (``time``,
  ``timeout``, ``hour``, ``minute``, ``second``, ``duration``, ``deadline``,
  ``wall``, ``clock``, ``runtime`` …) at any depth -- including the harness's
  ``default_budget.max_wall_clock_hours`` and keys inside ``notes`` and
  ``external_benchmark`` -- and a ``run_command`` that passes a time budget
  (``--time-budget``, ``--timeout``, ``--max-time``, ``TIME_BUDGET=…``, …).
  A campaign that stops on time would make the number of trials depend on
  the host's speed.  Free text (string values) is not scanned.
- ``allow_new_files`` (default ``false``): whether a diff may create files.
- ``workers``: fixed data-loader worker and thread counts.
- ``metric``: the regexes the base parser (:mod:`autoresearch.trackg.outcome`)
  uses.
- ``selection``: whether the task is test-scored; if so, selection is on the
  validation split only (R1-TRACKG-ROLE).
- ``determinism``: the deterministic settings the runner applies.

The loader is strict: unknown top-level keys are refused so a misspelled
field cannot silently fall back to a default.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

import yaml

from autoresearch.nodes.spec import NodeSpec

TRACKG_NODE_SCHEMA = "trackg_node/v1"
BUDGET_KINDS = ("steps", "tokens")
DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
BASE_METRIC_PARSER = "autoresearch.trackg.outcome:parse_run_log"

# Any key that names a time quantity is refused, wherever it appears (including the
# free-form ``notes`` and ``external_benchmark`` mappings).
_WALL_CLOCK_KEY_RE = re.compile(
    r"(wall|clock|time|hour|minute|second|duration|deadline|(^|_)secs?($|_)|(^|_)mins?($|_)|"
    r"(^|_)hrs?($|_))",
    re.IGNORECASE,
)
# A run command that passes a time budget to the task is refused too.
_WALL_CLOCK_COMMAND_RE = re.compile(
    r"(?i)(\b[A-Z_]*(TIMEOUT|SECONDS|MINUTES|HOURS|TIME_LIMIT|MAX_TIME|WALLCLOCK)[A-Z_]*=|time[_-]?budget|--time(?![\w-])|--timeout\b|--?max[_-]?time|wall[_-]?clock|--duration\b|"
    r"--?(seconds|minutes|hours)\b|\btimeout\s+\d)"
)
_WALL_CLOCK_KIND_RE = re.compile(
    r"(wall|clock|time|sec|second|minute|min|hour|duration)", re.IGNORECASE
)

_TOP_LEVEL_KEYS = {
    "schema", "name", "description", "suite", "domain", "editable_paths", "frozen_paths",
    "setup_command", "run_command", "metric_name", "metric_direction", "metric",
    "run_budget", "workers", "selection", "determinism", "default_budget",
    "failure_categories", "external_benchmark", "notes", "seed_env", "allow_new_files",
}
_REQUIRED_KEYS = (
    "schema", "name", "description", "editable_paths", "frozen_paths", "run_command",
    "metric_name", "metric_direction", "metric", "run_budget", "workers", "selection",
)


class TrackGNodeSpecError(ValueError):
    """Raised when a Track G node spec is malformed or declares a wall-clock budget."""


class WallClockBudgetError(TrackGNodeSpecError):
    """Raised when a spec declares a wall-clock budget (R1-BX-D5)."""


@dataclass(frozen=True)
class RunBudget:
    kind: Literal["steps", "tokens"]
    value: int

    @classmethod
    def from_mapping(cls, raw: object) -> "RunBudget":
        if not isinstance(raw, Mapping):
            raise TrackGNodeSpecError("run_budget must be a mapping {kind, value}")
        extra = set(raw) - {"kind", "value"}
        if extra:
            raise TrackGNodeSpecError(f"run_budget has unknown keys: {sorted(extra)}")
        kind = str(raw.get("kind", ""))
        if kind not in BUDGET_KINDS:
            if _WALL_CLOCK_KIND_RE.search(kind):
                raise WallClockBudgetError(
                    f"run_budget.kind {kind!r} is a wall-clock budget; only a fixed step or "
                    "token budget is allowed (R1-BX-D5)"
                )
            raise TrackGNodeSpecError(f"run_budget.kind must be one of {BUDGET_KINDS}, got {kind!r}")
        value = raw.get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TrackGNodeSpecError("run_budget.value must be a positive integer")
        return cls(kind=kind, value=value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class WorkerCounts:
    dataloader_workers: int
    threads: int

    @classmethod
    def from_mapping(cls, raw: object) -> "WorkerCounts":
        if not isinstance(raw, Mapping):
            raise TrackGNodeSpecError("workers must be a mapping {dataloader_workers, threads}")
        extra = set(raw) - {"dataloader_workers", "threads"}
        if extra:
            raise TrackGNodeSpecError(f"workers has unknown keys: {sorted(extra)}")
        out = {}
        for key, minimum in (("dataloader_workers", 0), ("threads", 1)):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise TrackGNodeSpecError(f"workers.{key} must be an integer >= {minimum} (fixed, not 'auto')")
            out[key] = value
        return cls(**out)


@dataclass(frozen=True)
class MetricRules:
    metric_regex: str
    loss_regex: str | None = None
    steps_regex: str | None = None
    test_metric_regex: str | None = None
    infrastructure_log_patterns: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: object) -> "MetricRules":
        if not isinstance(raw, Mapping):
            raise TrackGNodeSpecError("metric must be a mapping")
        allowed = {"metric_regex", "loss_regex", "steps_regex", "test_metric_regex",
                   "infrastructure_log_patterns"}
        extra = set(raw) - allowed
        if extra:
            raise TrackGNodeSpecError(f"metric has unknown keys: {sorted(extra)}")
        if not raw.get("metric_regex"):
            raise TrackGNodeSpecError("metric.metric_regex is required")
        if not raw.get("loss_regex"):
            raise TrackGNodeSpecError(
                "metric.loss_regex is required: a non-finite loss must be caught as divergent "
                "(R1-BX-REVISIONS (c))")
        values: dict[str, Any] = {}
        for key in ("metric_regex", "loss_regex", "steps_regex", "test_metric_regex"):
            pattern = raw.get(key)
            if pattern is None:
                values[key] = None
                continue
            try:
                compiled = re.compile(str(pattern), re.MULTILINE)
            except re.error as error:
                raise TrackGNodeSpecError(f"metric.{key} is not a valid regex: {error}") from error
            if compiled.groups < 1:
                raise TrackGNodeSpecError(f"metric.{key} must have one capture group")
            values[key] = str(pattern)
        patterns = raw.get("infrastructure_log_patterns", ()) or ()
        values["infrastructure_log_patterns"] = tuple(str(p) for p in patterns)
        return cls(**values)


@dataclass(frozen=True)
class SelectionPolicy:
    test_scored: bool
    selection_split: Literal["validation", "train", "none"] = "validation"
    test_metric_name: str | None = None

    @classmethod
    def from_mapping(cls, raw: object) -> "SelectionPolicy":
        if not isinstance(raw, Mapping):
            raise TrackGNodeSpecError("selection must be a mapping")
        extra = set(raw) - {"test_scored", "selection_split", "test_metric_name"}
        if extra:
            raise TrackGNodeSpecError(f"selection has unknown keys: {sorted(extra)}")
        if not isinstance(raw.get("test_scored"), bool):
            raise TrackGNodeSpecError("selection.test_scored must be true or false")
        split = str(raw.get("selection_split", "validation"))
        if split not in {"validation", "train", "none"}:
            raise TrackGNodeSpecError("selection.selection_split must be validation, train or none")
        test_metric = raw.get("test_metric_name")
        return cls(
            test_scored=bool(raw["test_scored"]),
            selection_split=split,  # type: ignore[arg-type]
            test_metric_name=str(test_metric) if test_metric is not None else None,
        )


@dataclass(frozen=True)
class DeterminismSettings:
    cudnn_benchmark: bool = False
    deterministic_algorithms: bool = True
    cublas_workspace_config: str = DEFAULT_CUBLAS_WORKSPACE_CONFIG
    pythonhashseed: str = "0"
    deterministic_warn_only: bool = False

    @classmethod
    def from_mapping(cls, raw: object) -> "DeterminismSettings":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise TrackGNodeSpecError("determinism must be a mapping")
        allowed = {"cudnn_benchmark", "deterministic_algorithms", "cublas_workspace_config", "pythonhashseed",
                   "deterministic_warn_only"}
        extra = set(raw) - allowed
        if extra:
            raise TrackGNodeSpecError(f"determinism has unknown keys: {sorted(extra)}")
        settings = cls(
            cudnn_benchmark=_strict_bool(raw.get("cudnn_benchmark", False), "determinism.cudnn_benchmark"),
            deterministic_algorithms=_strict_bool(raw.get("deterministic_algorithms", True),
                                                  "determinism.deterministic_algorithms"),
            cublas_workspace_config=str(raw.get("cublas_workspace_config", DEFAULT_CUBLAS_WORKSPACE_CONFIG)),
            pythonhashseed=str(raw.get("pythonhashseed", "0")),
            deterministic_warn_only=_strict_bool(raw.get("deterministic_warn_only", False),
                                                 "determinism.deterministic_warn_only"),
        )
        if settings.cudnn_benchmark or not settings.deterministic_algorithms:
            raise TrackGNodeSpecError(
                "determinism: cudnn_benchmark must be false and deterministic_algorithms true; "
                "a node that cannot run deterministically records that in the run record instead"
            )
        if settings.cublas_workspace_config not in {":4096:8", ":16:8"}:
            raise TrackGNodeSpecError("determinism.cublas_workspace_config must be ':4096:8' or ':16:8'")
        return settings


@dataclass(frozen=True)
class TrackGNodeSpec:
    name: str
    description: str
    editable_paths: tuple[str, ...]
    frozen_paths: tuple[str, ...]
    run_command: str
    metric_name: str
    metric_direction: Literal["maximize", "minimize"]
    metric: MetricRules
    run_budget: RunBudget
    workers: WorkerCounts
    selection: SelectionPolicy
    determinism: DeterminismSettings = field(default_factory=DeterminismSettings)
    setup_command: str = ""
    suite: str | None = None
    domain: str | None = None
    campaign_trials: int = 1
    failure_categories: tuple[str, ...] = ()
    seed_env: tuple[str, str] = ("AR_TRAIN_SEED", "AR_SPLIT_SEED")
    allow_new_files: bool = False
    raw_sha256: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrackGNodeSpec":
        if not isinstance(payload, Mapping):
            raise TrackGNodeSpecError("node spec root must be a mapping")
        refuse_wall_clock_keys(payload)
        unknown = set(payload) - _TOP_LEVEL_KEYS
        if unknown:
            raise TrackGNodeSpecError(f"unknown top-level keys: {sorted(unknown)}")
        missing = [key for key in _REQUIRED_KEYS if key not in payload]
        if missing:
            raise TrackGNodeSpecError(f"node spec missing required fields: {', '.join(missing)}")
        if payload["schema"] != TRACKG_NODE_SCHEMA:
            raise TrackGNodeSpecError(f"schema must be {TRACKG_NODE_SCHEMA!r}")
        direction = str(payload["metric_direction"])
        if direction not in {"maximize", "minimize"}:
            raise TrackGNodeSpecError("metric_direction must be 'maximize' or 'minimize'")
        editable = _path_tuple(payload["editable_paths"], "editable_paths")
        if not editable:
            raise TrackGNodeSpecError("editable_paths must not be empty")
        frozen = _path_tuple(payload["frozen_paths"], "frozen_paths")
        if not frozen:
            raise TrackGNodeSpecError(
                "frozen_paths must not be empty: declare the evaluator, data loader and metric code"
            )
        norm_editable = [_norm_scope(p) for p in editable]
        norm_frozen = [_norm_scope(p) for p in frozen]
        for raw_path, path in zip(editable, norm_editable):
            if not path:
                raise TrackGNodeSpecError(f"editable_paths: {raw_path!r} names the whole node; list paths")
            covering = [f for f in norm_frozen if path == f or path.startswith(f + "/")]
            if covering:
                raise TrackGNodeSpecError(f"editable path {raw_path!r} is covered by frozen path(s) {covering}")
        if any(not f for f in norm_frozen):
            raise TrackGNodeSpecError("frozen_paths: '.' would freeze the whole node")
        command = str(payload["run_command"])
        found = _WALL_CLOCK_COMMAND_RE.search(command)
        if found:
            raise WallClockBudgetError(
                f"run_command {command!r} contains {found.group(0)!r}, which looks like a time budget; "
                "use run_budget (steps or tokens) and {budget} (R1-BX-D5). Move a flag that is not a "
                "time budget into the wrapper")
        trials = 1
        if "default_budget" in payload:
            budget = payload["default_budget"]
            if not isinstance(budget, Mapping) or set(budget) - {"trials"}:
                raise TrackGNodeSpecError("default_budget may only carry 'trials'")
            trials = budget.get("trials")
            if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
                raise TrackGNodeSpecError("default_budget.trials must be a positive integer")
        seed_env = payload.get("seed_env", {"train": "AR_TRAIN_SEED", "split": "AR_SPLIT_SEED"})
        if not isinstance(seed_env, Mapping) or set(seed_env) != {"train", "split"}:
            raise TrackGNodeSpecError("seed_env must be a mapping {train, split}")
        if str(seed_env["train"]) == str(seed_env["split"]):
            raise TrackGNodeSpecError("seed_env.train and seed_env.split must be different variables")
        selection = SelectionPolicy.from_mapping(payload["selection"])
        metric = MetricRules.from_mapping(payload["metric"])
        spec = cls(
            name=str(payload["name"]),
            description=str(payload["description"]),
            editable_paths=editable,
            frozen_paths=frozen,
            run_command=str(payload["run_command"]),
            metric_name=str(payload["metric_name"]),
            metric_direction=direction,  # type: ignore[arg-type]
            metric=metric,
            run_budget=RunBudget.from_mapping(payload["run_budget"]),
            workers=WorkerCounts.from_mapping(payload["workers"]),
            selection=selection,
            determinism=DeterminismSettings.from_mapping(payload.get("determinism")),
            setup_command=str(payload.get("setup_command", "") or ""),
            suite=str(payload["suite"]) if payload.get("suite") is not None else None,
            domain=str(payload["domain"]) if payload.get("domain") is not None else None,
            campaign_trials=int(trials),
            failure_categories=tuple(str(c) for c in payload.get("failure_categories", ()) or ()),
            seed_env=(str(seed_env["train"]), str(seed_env["split"])),
            allow_new_files=_strict_bool(payload.get("allow_new_files", False), "allow_new_files"),
            raw_sha256=_canonical_sha256(payload),
        )
        # Imported here to keep the dependency one-way (selection imports this module).
        from autoresearch.trackg.selection import check_selection_policy

        check_selection_policy(spec)
        return spec

    def to_base_node_spec(self) -> NodeSpec:
        """The harness ``NodeSpec`` view, so the existing scope validator applies unchanged."""
        comparator = ">" if self.metric_direction == "maximize" else "<"
        return NodeSpec.from_mapping({
            "name": self.name,
            "description": self.description,
            "editable_paths": list(self.editable_paths),
            "frozen_paths": list(self.frozen_paths),
            "setup_command": self.setup_command,
            "run_command": self.run_command,
            "metric_name": self.metric_name,
            "metric_direction": self.metric_direction,
            "metric_parser": BASE_METRIC_PARSER,
            "acceptance_rule": f"candidate_metric {comparator} current_best_metric",
            "validity_checks": ["metric_present", "finite_metric", "editable_scope_only", "command_exit_zero"],
            "default_budget": {"trials": self.campaign_trials},
            "failure_categories": list(self.failure_categories),
        })

    def budgeted_run_command(self) -> str:
        """The run command with ``{budget}`` replaced by the fixed budget value."""
        return self.run_command.replace("{budget}", str(self.run_budget.value))


def refuse_wall_clock_keys(payload: object, *, _path: str = "") -> None:
    """Raise :class:`WallClockBudgetError` for any wall-clock-looking key at any depth."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            here = f"{_path}.{key}" if _path else str(key)
            found = _WALL_CLOCK_KEY_RE.search(str(key))
            if found:
                raise WallClockBudgetError(
                    f"{here}: key contains the time word {found.group(0)!r}; a wall-clock budget is "
                    "refused (R1-BX-D5). Use run_budget (steps or tokens); rename a key that is not a "
                    "time budget (for example min_lr -> lr_min) or move it into prose"
                )
            refuse_wall_clock_keys(value, _path=here)
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            refuse_wall_clock_keys(item, _path=f"{_path}[{index}]")


def load_trackg_node_spec(path: str | Path) -> TrackGNodeSpec:
    """Load a ``trackg_node/v1`` YAML file (JSON is valid YAML too)."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrackGNodeSpec.from_mapping(payload)


def _norm_scope(path: str) -> str:
    """Normalise a scope entry the way the scope validator does, without the glob suffix."""
    text = str(PurePosixPath(path.replace("\\", "/")))
    text = "" if text == "." else text.lstrip("/")
    if text.endswith("/*"):
        text = text[:-2]
    return text.rstrip("/")


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TrackGNodeSpecError(f"{name} must be true or false")
    return value


def _path_tuple(raw: object, name: str) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        raise TrackGNodeSpecError(f"{name} must be a list")
    out = []
    for item in raw:
        text = str(item)
        if not text or text.startswith("/") or ".." in Path(text).parts:
            raise TrackGNodeSpecError(f"{name}: {text!r} must be a relative path inside the node")
        out.append(text)
    return tuple(out)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
