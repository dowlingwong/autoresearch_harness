from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from autoresearch.memory.schemas import TrialDecision, TrialRecord


@dataclass(frozen=True)
class RepeatedBadStats:
    repeated_bad_count: int
    repeated_bad_rate: float
    repeated_invalid_count: int
    repeated_degraded_count: int
    flagged_trial_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["flagged_trial_ids"] = list(self.flagged_trial_ids)
        return payload


@dataclass(frozen=True)
class SimilarityComparison:
    method: str
    threshold: float
    repeated_bad_count: int
    repeated_bad_rate: float
    repeated_invalid_count: int
    repeated_degraded_count: int
    flagged_trial_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["flagged_trial_ids"] = list(self.flagged_trial_ids)
        return payload


def compute_repeated_bad_stats(records: list[TrialRecord], text_threshold: float = 0.6) -> RepeatedBadStats:
    prior_bad: list[TrialRecord] = []
    flagged: list[TrialRecord] = []
    for record in records:
        if _is_repeated_bad(record, prior_bad, text_threshold=text_threshold):
            flagged.append(record)
        if _is_bad(record):
            prior_bad.append(record)

    total = len(records)
    repeated_invalid = sum(1 for record in flagged if record.decision == TrialDecision.FAILED_INVALID)
    repeated_degraded = sum(1 for record in flagged if record.decision == TrialDecision.DISCARDED)
    return RepeatedBadStats(
        repeated_bad_count=len(flagged),
        repeated_bad_rate=(len(flagged) / total) if total else 0.0,
        repeated_invalid_count=repeated_invalid,
        repeated_degraded_count=repeated_degraded,
        flagged_trial_ids=tuple(record.trial_id for record in flagged),
    )


def compare_repetition_detectors(
    records: list[TrialRecord],
    *,
    text_threshold: float = 0.6,
) -> list[SimilarityComparison]:
    """Compare current hybrid matching with simpler token-set and fuzzy baselines."""
    methods = ("hybrid", "token_set_jaccard", "fuzzy_sequence")
    return [
        _compute_repeated_bad_stats_by_method(records, method=method, threshold=text_threshold)
        for method in methods
    ]


def is_similar_bad_proposal(candidate: TrialRecord, previous: TrialRecord, text_threshold: float = 0.6) -> bool:
    if not _is_bad(previous):
        return False
    candidate_param = extract_parameter_direction(candidate.proposal_summary + " " + candidate.proposal_rationale)
    previous_param = extract_parameter_direction(previous.proposal_summary + " " + previous.proposal_rationale)
    if candidate_param and candidate_param == previous_param:
        return True
    if candidate.failure_category is not None and candidate.failure_category == previous.failure_category:
        return True
    return jaccard_similarity(normalize_text(candidate.proposal_summary), normalize_text(previous.proposal_summary)) >= text_threshold


def token_set_similarity(left: str, right: str) -> float:
    return jaccard_similarity(normalize_text(left), normalize_text(right))


def fuzzy_sequence_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def extract_parameter_direction(text: str) -> tuple[str, str] | None:
    normalized = normalize_text(text)
    params = (
        "learning rate",
        "lr",
        "dropout",
        "weight decay",
        "stage layers",
        "kernel size",
        "batch size",
    )
    directions = {
        "increase": ("increase", "raise", "higher", "larger", "up"),
        "decrease": ("decrease", "lower", "smaller", "reduce", "down"),
    }
    found_param = next((param for param in params if param in normalized), None)
    if found_param is None:
        return None
    for direction, terms in directions.items():
        if any(term in normalized for term in terms):
            return (found_param, direction)
    return (found_param, "change")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_ ]+", " ", text.lower())).strip()


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _is_repeated_bad(candidate: TrialRecord, prior_bad: list[TrialRecord], text_threshold: float) -> bool:
    if not _is_bad(candidate):
        return False
    return any(is_similar_bad_proposal(candidate, previous, text_threshold=text_threshold) for previous in prior_bad)


def _compute_repeated_bad_stats_by_method(
    records: list[TrialRecord],
    *,
    method: str,
    threshold: float,
) -> SimilarityComparison:
    prior_bad: list[TrialRecord] = []
    flagged: list[TrialRecord] = []
    for record in records:
        if _is_repeated_bad_by_method(record, prior_bad, method=method, threshold=threshold):
            flagged.append(record)
        if _is_bad(record):
            prior_bad.append(record)
    total = len(records)
    repeated_invalid = sum(1 for record in flagged if record.decision == TrialDecision.FAILED_INVALID)
    repeated_degraded = sum(1 for record in flagged if record.decision == TrialDecision.DISCARDED)
    return SimilarityComparison(
        method=method,
        threshold=threshold,
        repeated_bad_count=len(flagged),
        repeated_bad_rate=(len(flagged) / total) if total else 0.0,
        repeated_invalid_count=repeated_invalid,
        repeated_degraded_count=repeated_degraded,
        flagged_trial_ids=tuple(record.trial_id for record in flagged),
    )


def _is_repeated_bad_by_method(
    candidate: TrialRecord,
    prior_bad: list[TrialRecord],
    *,
    method: str,
    threshold: float,
) -> bool:
    if not _is_bad(candidate):
        return False
    for previous in prior_bad:
        if method == "hybrid":
            if is_similar_bad_proposal(candidate, previous, text_threshold=threshold):
                return True
        elif method == "token_set_jaccard":
            if token_set_similarity(candidate.proposal_summary, previous.proposal_summary) >= threshold:
                return True
        elif method == "fuzzy_sequence":
            if fuzzy_sequence_similarity(candidate.proposal_summary, previous.proposal_summary) >= threshold:
                return True
        else:
            raise ValueError(f"unknown similarity method: {method}")
    return False


def _is_bad(record: TrialRecord) -> bool:
    return record.decision in {TrialDecision.DISCARDED, TrialDecision.FAILED_INVALID}
