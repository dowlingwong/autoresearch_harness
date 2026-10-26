"""Parse the JSON returned by MLGym's PythonSubmissionTasks evaluator."""
from __future__ import annotations

import json
import math


def parse_score(stdout: str) -> dict[str, float]:
    try:
        payload = json.loads(stdout)
    except (ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    value = payload.get("Score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {}
    try:
        score = float(value)
    except (OverflowError, ValueError):
        return {}
    return {"Score": score} if math.isfinite(score) else {}


def adapt_evaluation(result: tuple[dict, str]) -> dict[str, float]:
    """Translate an upstream task.evaluate() result; never fabricate a score."""
    if not isinstance(result, tuple) or len(result) != 2:
        return {}
    metrics, submission = result
    if not isinstance(metrics, dict) or not isinstance(submission, str) or not submission:
        return {}
    try:
        return parse_score(json.dumps(metrics, allow_nan=False))
    except (ValueError, TypeError, OverflowError):
        return {}
