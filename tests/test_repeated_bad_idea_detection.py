"""Tests for autoresearch.memory.similarity — repeated-bad-idea detection."""
from __future__ import annotations

import unittest

from autoresearch.memory.schemas import (
    ExecutionStatus,
    FailureCategory,
    TrialDecision,
    TrialProvenance,
    TrialRecord,
    ValidityStatus,
)
from autoresearch.memory.similarity import (
    RepeatedBadStats,
    compare_repetition_detectors,
    compute_repeated_bad_stats,
    extract_parameter_direction,
    fuzzy_sequence_similarity,
    is_similar_bad_proposal,
    jaccard_similarity,
    normalize_text,
    token_set_similarity,
)


def _provenance() -> TrialProvenance:
    return TrialProvenance(
        proposal_id="p", patch_id="pa", run_id="r", metric_id="m", decision_id="d"
    )


def _record(
    n: int = 1,
    decision: TrialDecision = TrialDecision.KEPT,
    summary: str = "reduce-lr",
    rationale: str = "lr too high",
    failure_category: FailureCategory | None = None,
) -> TrialRecord:
    validity = (
        ValidityStatus.INVALID
        if decision == TrialDecision.FAILED_INVALID
        else ValidityStatus.VALID
    )
    return TrialRecord(
        trial_id=f"c-trial-{n:03d}",
        campaign_id="c",
        node_id="resnet_trigger",
        budget_index=n,
        timestamp_start="2026-01-01T00:00:00Z",
        timestamp_end="2026-01-01T00:10:00Z",
        manager_mode="baseline_manager",
        worker_mode="dry_run_worker",
        memory_mode="none",
        proposal_summary=summary,
        proposal_rationale=rationale,
        targeted_files=("train.py",),
        patch_ref="patch.diff",
        git_commit_before="a",
        git_commit_after="b",
        execution_status=ExecutionStatus.SUCCESS if decision != TrialDecision.FAILED_INVALID else ExecutionStatus.FAILED,
        validity_status=validity,
        failure_category=failure_category if validity == ValidityStatus.INVALID else None,
        raw_log_ref="run.log",
        parsed_metrics={"val_auc": 0.78} if validity == ValidityStatus.VALID else {},
        current_best_before=0.78,
        delta_vs_best=0.001 if decision == TrialDecision.KEPT else -0.001,
        decision=decision,
        decision_rationale="ok",
        wall_clock_seconds=600.0,
        cumulative_budget_consumed=n,
        provenance=_provenance(),
    )


class TestNormalizeText(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(normalize_text("HELLO World"), "hello world")

    def test_removes_punctuation(self):
        result = normalize_text("reduce LR: from 1e-3 to 5e-4!")
        self.assertNotIn(":", result)
        self.assertNotIn("!", result)

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_text("a  b   c"), "a b c")


class TestJaccardSimilarity(unittest.TestCase):
    def test_identical_strings(self):
        self.assertAlmostEqual(jaccard_similarity("a b c", "a b c"), 1.0)

    def test_completely_different(self):
        self.assertAlmostEqual(jaccard_similarity("a b c", "x y z"), 0.0)

    def test_partial_overlap(self):
        score = jaccard_similarity("a b c", "a b d")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_strings(self):
        self.assertAlmostEqual(jaccard_similarity("", ""), 1.0)

    def test_one_empty(self):
        self.assertAlmostEqual(jaccard_similarity("a b", ""), 0.0)


class TestSimilarityDetectorComparison(unittest.TestCase):
    def test_token_set_similarity_ignores_order(self):
        self.assertAlmostEqual(token_set_similarity("reduce learning rate", "rate learning reduce"), 1.0)

    def test_fuzzy_sequence_similarity_detects_close_text(self):
        self.assertGreater(fuzzy_sequence_similarity("lower dropout", "lower drop out"), 0.8)

    def test_compare_repetition_detectors_returns_all_methods(self):
        records = [
            _record(1, decision=TrialDecision.DISCARDED, summary="reduce learning rate"),
            _record(2, decision=TrialDecision.DISCARDED, summary="reduce learning rate"),
        ]
        rows = compare_repetition_detectors(records, text_threshold=0.5)
        self.assertEqual(
            {row.method for row in rows},
            {"hybrid", "token_set_jaccard", "fuzzy_sequence"},
        )
        self.assertTrue(all(row.repeated_bad_count >= 1 for row in rows))


class TestExtractParameterDirection(unittest.TestCase):
    def test_learning_rate_decrease(self):
        result = extract_parameter_direction("reduce learning rate from 1e-3 to 5e-4")
        self.assertIsNotNone(result)
        self.assertIn("learning rate", result[0])
        self.assertEqual(result[1], "decrease")

    def test_dropout_increase(self):
        result = extract_parameter_direction("increase dropout to 0.5")
        self.assertIsNotNone(result)
        self.assertIn("dropout", result[0])
        self.assertEqual(result[1], "increase")

    def test_no_match_returns_none(self):
        result = extract_parameter_direction("some unrelated text about optimizers")
        self.assertIsNone(result)

    def test_change_without_direction(self):
        result = extract_parameter_direction("change learning rate")
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "change")


class TestIsSimilarBadProposal(unittest.TestCase):
    def test_same_param_direction_flagged(self):
        previous = _record(
            1,
            decision=TrialDecision.DISCARDED,
            summary="reduce learning rate from 1e-3",
        )
        candidate = _record(
            2,
            decision=TrialDecision.DISCARDED,
            summary="reduce learning rate from 5e-4",
        )
        self.assertTrue(is_similar_bad_proposal(candidate, previous))

    def test_good_previous_not_flagged(self):
        previous = _record(1, decision=TrialDecision.KEPT, summary="reduce learning rate")
        candidate = _record(2, decision=TrialDecision.DISCARDED, summary="reduce learning rate")
        self.assertFalse(is_similar_bad_proposal(candidate, previous))

    def test_high_text_similarity_flagged(self):
        text = "reduce dropout from 0.5 to 0.3"
        previous = _record(1, decision=TrialDecision.DISCARDED, summary=text)
        candidate = _record(2, decision=TrialDecision.DISCARDED, summary=text)
        self.assertTrue(is_similar_bad_proposal(candidate, previous, text_threshold=0.5))

    def test_completely_different_not_flagged(self):
        previous = _record(1, decision=TrialDecision.DISCARDED, summary="reduce learning rate")
        candidate = _record(2, decision=TrialDecision.DISCARDED, summary="increase dropout regularisation")
        result = is_similar_bad_proposal(candidate, previous, text_threshold=0.9)
        # High threshold — should NOT be flagged as similar.
        self.assertFalse(result)


class TestComputeRepeatedBadStats(unittest.TestCase):
    def test_no_records(self):
        stats = compute_repeated_bad_stats([])
        self.assertEqual(stats.repeated_bad_count, 0)
        self.assertEqual(stats.repeated_bad_rate, 0.0)

    def test_all_kept_no_repeated(self):
        records = [_record(i, decision=TrialDecision.KEPT) for i in range(1, 4)]
        stats = compute_repeated_bad_stats(records)
        self.assertEqual(stats.repeated_bad_count, 0)

    def test_repeated_bad_detected(self):
        summary = "reduce learning rate from high to low"
        records = [
            _record(1, decision=TrialDecision.DISCARDED, summary=summary),
            _record(2, decision=TrialDecision.DISCARDED, summary=summary),
            _record(3, decision=TrialDecision.DISCARDED, summary=summary),
        ]
        stats = compute_repeated_bad_stats(records, text_threshold=0.5)
        self.assertGreater(stats.repeated_bad_count, 0)
        self.assertGreater(stats.repeated_bad_rate, 0.0)

    def test_repeated_invalid_counted_separately(self):
        summary = "reduce learning rate"
        records = [
            _record(
                1,
                decision=TrialDecision.FAILED_INVALID,
                summary=summary,
                failure_category=FailureCategory.METRIC_MISSING,
            ),
            _record(
                2,
                decision=TrialDecision.FAILED_INVALID,
                summary=summary,
                failure_category=FailureCategory.METRIC_MISSING,
            ),
        ]
        stats = compute_repeated_bad_stats(records, text_threshold=0.5)
        # Second invalid record matches first, so repeated_invalid_count >= 1.
        self.assertGreaterEqual(stats.repeated_invalid_count, 1)

    def test_flagged_trial_ids_populated(self):
        summary = "reduce dropout"
        records = [
            _record(1, decision=TrialDecision.DISCARDED, summary=summary),
            _record(2, decision=TrialDecision.DISCARDED, summary=summary),
        ]
        stats = compute_repeated_bad_stats(records, text_threshold=0.5)
        self.assertIn("c-trial-002", stats.flagged_trial_ids)

    def test_to_dict_serialisable(self):
        import json
        records = [_record(1, decision=TrialDecision.DISCARDED, summary="reduce lr")]
        stats = compute_repeated_bad_stats(records)
        json.dumps(stats.to_dict())  # must not raise

    def test_rate_is_fraction_of_total(self):
        summary = "reduce learning rate"
        records = [
            _record(1, decision=TrialDecision.DISCARDED, summary=summary),
            _record(2, decision=TrialDecision.DISCARDED, summary=summary),
            _record(3, decision=TrialDecision.KEPT, summary="something else"),
            _record(4, decision=TrialDecision.DISCARDED, summary=summary),
        ]
        stats = compute_repeated_bad_stats(records, text_threshold=0.5)
        self.assertAlmostEqual(stats.repeated_bad_rate, stats.repeated_bad_count / 4)


if __name__ == "__main__":
    unittest.main()
