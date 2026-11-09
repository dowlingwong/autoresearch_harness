"""M10 step 0: the base parser returns exactly one of ok / divergent / infrastructure_failure."""
from __future__ import annotations

import math

import pytest

from autoresearch.trackg.outcome import (
    DIVERGENT,
    INFRASTRUCTURE_FAILURE,
    OK,
    ProcessExit,
    RunOutcome,
    classify_run,
)

RULES = dict(metric_regex=r"^val:\s*(\S+)", loss_regex=r"^step \d+ loss:\s*(\S+)",
             steps_regex=r"^step (\d+)", budget_steps=3)
GOOD = "step 1 loss: 1.0\nstep 2 loss: 0.5\nstep 3 loss: 0.25\nval: 0.81\n"


def classify(log, **process):
    return classify_run(log, ProcessExit(**{"returncode": 0, **process}), **RULES)


def test_clean_run_is_ok_with_its_value():
    outcome = classify(GOOD)
    assert outcome.kind == OK and outcome.value == pytest.approx(0.81) and outcome.reason == ""


@pytest.mark.parametrize("token", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_non_finite_loss_is_divergent(token):
    outcome = classify(GOOD.replace("loss: 0.5", f"loss: {token}"))
    assert outcome.kind == DIVERGENT and outcome.reason == "non_finite_loss" and outcome.value is None


def test_missing_metric_is_divergent():
    outcome = classify(GOOD.replace("val: 0.81\n", ""))
    assert (outcome.kind, outcome.reason) == (DIVERGENT, "no_metric_at_budget")


def test_non_finite_metric_is_divergent():
    assert classify(GOOD.replace("val: 0.81", "val: nan")).reason == "non_finite_metric"


def test_budget_not_reached_is_divergent():
    outcome = classify("step 1 loss: 1.0\nval: 0.5\n")
    assert (outcome.kind, outcome.reason) == (DIVERGENT, "budget_not_reached")


def test_crash_without_kill_is_divergent():
    outcome = classify("step 1 loss: 1.0\nTraceback ...\n", returncode=1)
    assert (outcome.kind, outcome.reason) == (DIVERGENT, "nonzero_exit")


def test_timeout_is_infrastructure_failure_even_with_nan_in_the_partial_log():
    outcome = classify(GOOD.replace("loss: 0.5", "loss: nan"), returncode=-15, timed_out=True)
    assert (outcome.kind, outcome.reason) == (INFRASTRUCTURE_FAILURE, "timeout") and outcome.value is None


@pytest.mark.parametrize("returncode", [-9, -15, 137, 143])
def test_killed_is_infrastructure_failure(returncode):
    outcome = classify(GOOD, returncode=returncode)
    assert (outcome.kind, outcome.reason) == (INFRASTRUCTURE_FAILURE, "killed")


def test_launch_error_and_infrastructure_log_patterns():
    assert classify("", returncode=None, launch_error="ENOENT").reason == "launch_error"
    outcome = classify("OSError: [Errno 28] No space left on device\n", returncode=1)
    assert (outcome.kind, outcome.reason) == (INFRASTRUCTURE_FAILURE, "infrastructure_log_pattern")


def test_kinds_never_mix():
    with pytest.raises(ValueError):
        RunOutcome("ok", None)
    with pytest.raises(ValueError):
        RunOutcome("ok", math.nan)
    with pytest.raises(ValueError):
        RunOutcome("divergent", 0.5, "non_finite_loss")
    with pytest.raises(ValueError):
        RunOutcome("infrastructure_failure", 1.0, "timeout")
    with pytest.raises(ValueError):
        RunOutcome("divergent", None, "")
    with pytest.raises(ValueError):
        RunOutcome("mixed", None, "x")
