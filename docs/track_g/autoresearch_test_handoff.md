# Track G test boundary handoff

On 27 October 2026, while validating adapter commit `eb37602`, I mistakenly
included existing tests that launch `lr_synthetic` subprocesses. The requested
scope allowed stub training only. These tests were not rerun.

1. Command: `/Users/wongdowling/Documents/autoresearch_harness/.venv/bin/python -m pytest -q tests/stage2/test_track_g_autoresearch.py tests/stage2/test_r1_defects.py`.
   Test ID: `tests/stage2/test_r1_defects.py::TestSeedInjection::test_same_seed_reproduces_and_different_seed_changes`.
   At the logger's `2026-10-27T00:19:46`, the captured output said
   `running: python3 train.py`, followed by `success=False metrics={}` and
   the assertion `False is not true : exit code 1`.
2. Command: `/Users/wongdowling/Documents/autoresearch_harness/.venv/bin/python -m pytest -q tests/stage2/test_track_g_autoresearch.py tests/stage2/test_r1_defects.py::TestClawWorkerMetricSign tests/stage2/test_r1_defects.py::TestCurrentBestOnResume tests/stage2/test_r1_l5.py`.
   Test ID: `tests/stage2/test_r1_l5.py::TestDefaultControllerRegression::test_dry_and_local_default_records_match_fixture`.
   At `2026-10-27T00:19:54`, its captured output showed two calls to
   `python3 train.py`, each followed by `success=False metrics={}`; the test
   failed because its real-campaign records differed from the fixture.

The displayed times are logger timestamps, whose timezone was not recorded in
the output. The worker captured subprocess output in temporary directories;
the tests removed those directories, so the subprocess traceback was not
preserved in the visible pytest output. Exit status and empty metrics directly
establish failure, not by themselves the precise stopping point.

The conclusion that neither invocation reached training is an inference from
additional read-only evidence: the node's configured command is `python3
train.py`, its script imports `numpy` before defining or running the training
code, and a subsequent non-training `python3 -c` module-availability check in
the same shell reported `python3 numpy available: False`. The captured test
logs do not contain the missing-module traceback, so they do not independently
prove the reason for the failure. No GPU or autoresearch model training was
invoked by these tests. The attempt itself exceeded the stub-only testing
boundary and should be considered in independent review.
