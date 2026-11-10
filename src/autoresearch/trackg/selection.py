"""Validation-only selection for test-scored tasks (R1-TRACKG-ROLE; M10 step 0).

A test-scored task (one whose suite reports a held-out test score) selects,
keeps and ranks on the **validation** metric only.  The test metric, if a
port records it at all, is kept under a ``sealed`` key that
:func:`manager_view` removes, and :func:`select_best` never reads it.

:func:`check_selection_policy` is called by the spec loader, so a test-scored
spec that would select on the test split cannot be loaded.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from autoresearch.trackg.outcome import OK, RunOutcome, parse_float_token

SEALED_KEY = "sealed"
_TEST_NAME_RE = re.compile(r"(^|[_\-.])test([_\-.]|$)", re.IGNORECASE)


class SelectionPolicyError(ValueError):
    """Raised when a spec or a call would select on a test metric."""


def check_selection_policy(spec: Any) -> None:
    policy = spec.selection
    if not policy.test_scored:
        return
    if policy.selection_split != "validation":
        raise SelectionPolicyError("a test-scored task must select on the validation split only")
    if not policy.test_metric_name:
        raise SelectionPolicyError("a test-scored task must name its test metric (selection.test_metric_name)")
    if policy.test_metric_name == spec.metric_name:
        raise SelectionPolicyError("metric_name (selection metric) must differ from the test metric")
    if _TEST_NAME_RE.search(spec.metric_name):
        raise SelectionPolicyError(
            f"metric_name {spec.metric_name!r} looks like a test metric; select on validation only"
        )
    test_regex = spec.metric.test_metric_regex
    if test_regex is not None and test_regex == spec.metric.metric_regex:
        raise SelectionPolicyError("metric_regex and test_metric_regex must differ")
    # Load-time sanity check: the selection regex must not match typical test-metric lines.
    # (At run time any selection match overlapping a test-metric match is divergent.)
    name = policy.test_metric_name
    for sample in (f"{name}: 0.5", f"{name}=0.5", f"{name} 0.5", f"{name}: 0.5\n"):
        if re.search(spec.metric.metric_regex, sample, re.MULTILINE):
            raise SelectionPolicyError(
                f"metric_regex {spec.metric.metric_regex!r} matches the test-metric line {sample.strip()!r}; "
                "anchor it on the validation metric")
    # Mirror check: the test regex must not match the validation metric's line either.
    if test_regex is not None:
        for sample in (f"{spec.metric_name}: 0.5", f"{spec.metric_name}=0.5", f"{spec.metric_name} 0.5"):
            if re.search(test_regex, sample, re.MULTILINE):
                raise SelectionPolicyError(
                    f"test_metric_regex {test_regex!r} matches the validation-metric line {sample!r}; "
                    "anchor it on the test metric")


def selection_value(outcome: RunOutcome) -> float | None:
    """The value used for selection: the validation metric of an ``ok`` run, else ``None``."""
    return outcome.value if outcome.kind == OK else None


def select_best(candidates: Sequence[tuple[str, RunOutcome]], spec: Any) -> str | None:
    """Return the id of the best ``ok`` candidate by the validation metric (first wins ties)."""
    best_id: str | None = None
    best: float | None = None
    for candidate_id, outcome in candidates:
        if not isinstance(outcome, RunOutcome):
            raise SelectionPolicyError("select_best takes RunOutcome values only (never raw metric dicts)")
        value = selection_value(outcome)
        if value is None:
            continue
        better = best is None or (value > best if spec.metric_direction == "maximize" else value < best)
        if better:
            best_id, best = candidate_id, value
    return best_id


def parse_sealed_test_metric(log_text: str, spec: Any) -> dict[str, Any]:
    """Parse the test metric into a sealed block for the final report; never used for selection."""
    if not spec.selection.test_scored or not spec.metric.test_metric_regex:
        return {}
    matches = [m.group(1) for m in re.finditer(spec.metric.test_metric_regex, log_text, re.MULTILINE)]
    value = parse_float_token(matches[-1]) if matches else None
    return {SEALED_KEY: {spec.selection.test_metric_name: value}}


def manager_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of a run record without the sealed test block, for anything a manager or controller sees."""
    return {key: value for key, value in record.items() if key != SEALED_KEY}


def assert_no_test_metric(names: Iterable[str], spec: Any) -> None:
    """Refuse a manager-visible metric list that contains the test metric."""
    test_name = spec.selection.test_metric_name
    if spec.selection.test_scored and test_name in set(names):
        raise SelectionPolicyError(f"{test_name!r} is a test metric and must not be shown or used")
