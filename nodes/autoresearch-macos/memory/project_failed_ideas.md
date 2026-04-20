---
name: Failed ideas
description: Failed or cautionary patterns for the autoresearch system, especially around control-plane validation and worker behavior.
type: project
---

## Read-only inspection treated as success

**Problem:** Early worker runs would read `train.py`, confirm they understood it, and return a `final` answer without making any edit. The harness accepted this as a completed task.

**Fix:** The harness now enforces that `train.py` must appear in `changed_files` after the worker run, verified via git status diff before/after. A run with no file mutation is rejected as `discard` before training is even attempted.

## Full training run during smoke tests

**Problem:** Running real `uv run train.py` during notebook smoke tests takes 5+ minutes and requires the full data cache, making CI impractical.

**Fix:** Smoke tests use a synthetic `run.log` writer (a Python one-liner that writes fake metrics) instead of real training. The `autoresearch_smoke_packet.json` uses this approach.

## One-shot mutations that become no-ops

**Problem:** The worker would produce a syntactically valid edit that changed a comment or whitespace, which passes the syntax check but has no effect on `val_bpb`. The experiment runs for 5 minutes and returns the same metric, resulting in an automatic `discard`.

**Lesson:** The objective in the experiment packet must be specific and mechanistic (e.g. "increase hidden dim from 512 to 576") rather than open-ended. Vague objectives like "improve training efficiency" lead to cosmetic edits.

## Stale state.json after crash mid-discard

**Problem:** If the process crashed between `git reset --hard` and the state-file write during a `discard`, `.autoresearch_state.json` would still claim a pending experiment pointing at a rolled-back commit. The next `run` would then fail with a confusing git error.

**Fix:** `load_autoresearch_state()` now calls `_recover_stale_pending()` which checks whether the pending commit still exists in git (`git cat-file -e`). If not, it auto-clears the pending block and re-derives `best_bpb` from `results.tsv`.

## Worker given the full experiment history

**Problem:** Injecting all previous `results.tsv` rows into the worker prompt caused it to spend most of its turns reading the history rather than editing code, often exhausting the 8-turn limit before making an edit.

**Lesson:** The worker should only receive the current objective and the current `train.py`. Strategic context (what has been tried, what worked) belongs in the manager's reasoning, not the worker's prompt.
