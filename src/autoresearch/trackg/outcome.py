"""Base metric parser with the divergence class (M10 step 0; R1-BX-REVISIONS (c)).

Every run is classified as **exactly one** of:

- ``ok(value)``: exit code 0, no non-finite loss in the log, a finite metric
  reported, and (if the node declares ``steps_regex``) the fixed budget was
  reached;
- ``divergent``: a non-finite loss (also a non-numeric token containing
  ``nan``/``inf`` such as ``tensor(nan)``), or no (finite) metric at the fixed budget
  (R1-BX-PROPOSED-CONFIRM, REVISIONS (c) (E5)).  A run that exits non-zero
  without being killed is in this class (the proposal broke the run);
- ``infrastructure_failure``: the run timed out, was killed by a signal (or
  an exit code the node declares as a kill, default 137 and 143), could not
  be launched, or its log matches one of the node's infrastructure patterns
  (for example ``No space left on device``).

A selection-metric match that overlaps a test-metric match (test-scored
nodes) is ``divergent`` with reason ``metric_regex_matches_test_line``, so a
loose ``metric_regex`` can never select on the test metric.

Infrastructure is checked first, so a run that was killed is never reported
as divergent even if its partial log shows a NaN.  The three constructors
enforce the invariant that only ``ok`` carries a value.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

OK = "ok"
DIVERGENT = "divergent"
INFRASTRUCTURE_FAILURE = "infrastructure_failure"
OUTCOME_KINDS = (OK, DIVERGENT, INFRASTRUCTURE_FAILURE)

DEFAULT_KILL_EXIT_CODES = frozenset({137, 143})  # 128+SIGKILL, 128+SIGTERM (shell-reported)
DEFAULT_INFRASTRUCTURE_LOG_PATTERNS = (
    r"No space left on device",
    r"Input/output error",
    r"slurmstepd: error",
    r"CUDA driver version is insufficient",
    r"\bDUE TO TIME LIMIT\b",
    r"\bDUE TO PREEMPTION\b",
)

_NON_FINITE_IN_TOKEN = re.compile(r"(?i)(?<![A-Za-z])(nan|[+-]?inf(inity)?)(?![A-Za-z])")
_FLOAT_TOKEN = re.compile(r"^[+-]?(nan|inf|infinity|\d+(\.\d*)?([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)$", re.I)


@dataclass(frozen=True)
class RunOutcome:
    kind: str
    value: float | None = None
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in OUTCOME_KINDS:
            raise ValueError(f"unknown outcome kind {self.kind!r}")
        if self.kind == OK:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("ok outcome needs a finite value")
        elif self.value is not None:
            raise ValueError(f"{self.kind} outcome must not carry a value")
        if self.kind != OK and not self.reason:
            raise ValueError(f"{self.kind} outcome needs a reason")

    @classmethod
    def ok(cls, value: float, **detail: Any) -> "RunOutcome":
        return cls(OK, float(value), "", dict(detail))

    @classmethod
    def divergent(cls, reason: str, **detail: Any) -> "RunOutcome":
        return cls(DIVERGENT, None, reason, dict(detail))

    @classmethod
    def infrastructure_failure(cls, reason: str, **detail: Any) -> "RunOutcome":
        return cls(INFRASTRUCTURE_FAILURE, None, reason, dict(detail))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessExit:
    """What the runner observed about the process (no metric knowledge)."""

    returncode: int | None
    timed_out: bool = False
    launch_error: str | None = None

    @property
    def killed_signal(self) -> int | None:
        if self.returncode is not None and self.returncode < 0:
            return -self.returncode
        return None


def parse_float_token(token: str) -> float | None:
    """Parse a number token including ``nan``/``inf``; ``None`` if it is not a number."""
    text = token.strip().rstrip(",;")
    if not _FLOAT_TOKEN.match(text):
        return None
    return float(text)


def classify_run(
    log_text: str,
    process: ProcessExit,
    *,
    metric_regex: str,
    loss_regex: str | None = None,
    steps_regex: str | None = None,
    budget_steps: int | None = None,
    test_metric_regex: str | None = None,
    kill_exit_codes: Iterable[int] = DEFAULT_KILL_EXIT_CODES,
    infrastructure_log_patterns: Iterable[str] = DEFAULT_INFRASTRUCTURE_LOG_PATTERNS,
) -> RunOutcome:
    """Classify one run.  See the module docstring for the rules and their order."""
    # 1. infrastructure: never mixed with the other classes.
    if process.launch_error is not None:
        return RunOutcome.infrastructure_failure("launch_error", error=process.launch_error)
    if process.timed_out:
        return RunOutcome.infrastructure_failure("timeout", returncode=process.returncode)
    if process.killed_signal is not None:
        return RunOutcome.infrastructure_failure("killed", signal=process.killed_signal)
    if process.returncode in set(kill_exit_codes):
        return RunOutcome.infrastructure_failure("killed", returncode=process.returncode)
    if process.returncode is None:
        return RunOutcome.infrastructure_failure("no_exit_status")
    for pattern in infrastructure_log_patterns:
        if re.search(pattern, log_text, re.MULTILINE):
            return RunOutcome.infrastructure_failure("infrastructure_log_pattern", pattern=pattern,
                                                     returncode=process.returncode)

    # 2. divergence: a non-finite loss anywhere in the log.  A token that is not a plain
    #    number (``tensor(nan)``, ``nan%``) counts as non-finite if it contains nan/inf.
    if loss_regex:
        for match in re.finditer(loss_regex, log_text, re.MULTILINE):
            token = match.group(1)
            value = parse_float_token(token)
            if (value is not None and not math.isfinite(value)) or (
                    value is None and _NON_FINITE_IN_TOKEN.search(token)):
                return RunOutcome.divergent("non_finite_loss", token=token.strip())

    # 3. divergence: no finite metric at the budget.  A selection-metric match that
    #    overlaps a test-metric line is refused loudly (validation-only selection).
    test_spans = []
    if test_metric_regex:
        test_spans = [m.span() for m in re.finditer(test_metric_regex, log_text, re.MULTILINE)]
    metric_matches = list(re.finditer(metric_regex, log_text, re.MULTILINE))
    for match in metric_matches:
        start, end = match.span()
        if any(start < t_end and t_start < end for t_start, t_end in test_spans):
            return RunOutcome.divergent("metric_regex_matches_test_line", line=match.group(0).strip())
    metrics = [m.group(1) for m in metric_matches]
    if process.returncode != 0:
        return RunOutcome.divergent("nonzero_exit", returncode=process.returncode,
                                    metric_reported=bool(metrics))
    if not metrics:
        return RunOutcome.divergent("no_metric_at_budget")
    value = parse_float_token(metrics[-1])
    if value is None:
        return RunOutcome.divergent("unparseable_metric", token=metrics[-1].strip())
    if not math.isfinite(value):
        return RunOutcome.divergent("non_finite_metric", token=metrics[-1].strip())
    if steps_regex and budget_steps is not None:
        steps = [parse_float_token(m.group(1)) for m in re.finditer(steps_regex, log_text, re.MULTILINE)]
        steps = [s for s in steps if s is not None and math.isfinite(s)]
        last = int(max(steps)) if steps else None
        if last is None or last < budget_steps:
            return RunOutcome.divergent("budget_not_reached", last_step=last, budget=budget_steps)
    return RunOutcome.ok(value)


def parse_run_log(log_path: str | Path, process: ProcessExit, spec: Any) -> RunOutcome:
    """Classify a run from its log file with the rules of a ``TrackGNodeSpec``."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace") if Path(log_path).exists() else ""
    patterns = tuple(DEFAULT_INFRASTRUCTURE_LOG_PATTERNS) + tuple(spec.metric.infrastructure_log_patterns)
    budget_steps = spec.run_budget.value if spec.run_budget.kind == "steps" else None
    return classify_run(
        text,
        process,
        metric_regex=spec.metric.metric_regex,
        loss_regex=spec.metric.loss_regex,
        steps_regex=spec.metric.steps_regex,
        budget_steps=budget_steps,
        test_metric_regex=spec.metric.test_metric_regex if spec.selection.test_scored else None,
        infrastructure_log_patterns=patterns,
    )
