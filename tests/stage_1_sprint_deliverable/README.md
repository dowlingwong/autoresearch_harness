# Stage 1 Sprint Deliverable

This directory defines the Stage 1 sprint acceptance checks for the system described in `autoresearch_director_report.pdf` and `autoresearch_director_report.md`.

The report describes a governed loop:

```text
manager proposal -> bounded worker edit -> experiment execution -> metric parsing -> keep/discard -> append-only memory
```

Current implementation note: the report mentions REST endpoints such as `/run` and `/loop`, but this checkout exposes the working autoresearch control plane through CLI subcommands in `harness/claw-code/src/main.py`.

## Deliverable Structure

The sprint deliverable has two parts:

- Part 1: check that the harness and framework contracts work as expected.
- Part 2: run a complete governed full-loop experiment.

Part 1 should pass before running Part 2. Part 2 invokes the local worker model and may create experiment commits on the node branch.

## Part 1: Harness and Framework Validation

Goal: verify the control-plane functions, node metric parser, state files, branch isolation guard, and bounded worker contract without committing to a long campaign.

### 1. Prerequisites

Run from the repository root:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
```

Verify local dependencies:

```bash
command -v uv
command -v python3
command -v git
curl -s http://localhost:11434/api/tags | head
```

Prepare the node environment:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger
uv sync
uv run python -m py_compile train.py
```

Pass condition: all commands exit successfully, and Ollama lists the local worker model needed for the full loop.

### 2. Run Existing Integration Tests

These tests validate the control-plane functions without doing a full model-training campaign.

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/harness/claw-code
PYTHONPATH=$PWD python3 -m unittest tests.test_autoresearch_integration
```

Pass condition: the unittest command exits with code `0`.

### 3. Run a Node-Only Baseline Smoke Test

This confirms that the ResNet node can train and emit parseable metrics.

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger
RESNET_TRIGGER_FAST_SEARCH=1 \
RESNET_TRIGGER_FAST_N_SIGNAL=1000 \
RESNET_TRIGGER_FAST_N_NOISE=1000 \
RESNET_TRIGGER_FAST_TRACE_LEN=4096 \
RESNET_TRIGGER_FAST_BATCH_SIZE=64 \
RESNET_TRIGGER_FAST_EPOCHS=3 \
RESNET_TRIGGER_FAST_SKIP_TEST=1 \
RESNET_TRIGGER_EARLY_STOP_PATIENCE=2 \
RESNET_TRIGGER_EARLY_STOP_MIN_DELTA=0.002 \
RESNET_TRIGGER_DEVICE=cpu \
uv run train.py > run.log 2>&1
```

Parse the metric:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/harness/claw-code
PYTHONPATH=$PWD python3 -m src.main autoresearch parse-log \
  --root /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger \
  run.log
```

Pass condition: output contains `"success": true` and a numeric `"val_bpb"`.

### 4. Inspect Control-Plane Status

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/harness/claw-code
PYTHONPATH=$PWD python3 -m src.main autoresearch status \
  --root /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger
```

Pass condition: output reports paths for `.autoresearch_state.json`, `experiment_memory.jsonl`, and `results.tsv`.

## Part 2: Full Governed Loop

Goal: verify that the system can run the complete governed experiment lifecycle on `nodes/ResNet_trigger`.

The full loop should:

- create or use an isolated `autoresearch/<tag>` branch
- record or reuse a baseline
- let the worker edit only `train.py`
- run the node training command
- parse `val_bpb` where lower is better, equivalent to higher validation AUC
- automatically keep improved candidates and discard worse or invalid candidates
- write `results.tsv`, `.autoresearch_state.json`, and append-only `experiment_memory.jsonl`

### 1. Create an Isolated Experiment Branch

The loop command requires a branch named `autoresearch/<tag>` unless `--allow-any-branch` is used.

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/harness/claw-code
PYTHONPATH=$PWD python3 -m src.main autoresearch isolate \
  --root /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger \
  --branch autoresearch/stage-1-sprint \
  --create
```

Pass condition: output has `"isolated_branch": true`.

### 2. Run the Full Governed Loop

This runs baseline handling plus one bounded worker experiment with automatic keep/discard.

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/harness/claw-code
PYTHONPATH=$PWD \
RESNET_TRIGGER_FAST_SEARCH=1 \
RESNET_TRIGGER_FAST_N_SIGNAL=1000 \
RESNET_TRIGGER_FAST_N_NOISE=1000 \
RESNET_TRIGGER_FAST_TRACE_LEN=4096 \
RESNET_TRIGGER_FAST_BATCH_SIZE=64 \
RESNET_TRIGGER_FAST_EPOCHS=3 \
RESNET_TRIGGER_FAST_SKIP_TEST=1 \
RESNET_TRIGGER_EARLY_STOP_PATIENCE=2 \
RESNET_TRIGGER_EARLY_STOP_MIN_DELTA=0.002 \
RESNET_TRIGGER_DEVICE=cpu \
python3 -m src.main autoresearch loop \
  --root /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger \
  --packet /Users/wongdowling/Documents/autoresearch_harness/tests/stage_1_sprint_deliverable/loop_packet.json \
  --model qwen2.5-coder:7b \
  --host http://localhost:11434 \
  --iterations 1 \
  --retry-limit 1
```

For a longer report-style test, increase `--iterations` after the one-iteration smoke passes. The director report's Stage 2 target is 5 rounds x 10 worker experiments, which maps to 50 total loop iterations when using the current CLI interface.

### 3. Inspect Acceptance Artifacts

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger
git status --short
tail -n 20 results.tsv
tail -n 20 experiment_memory.jsonl
cat .autoresearch_state.json
```

Pass conditions:

- `results.tsv` has a baseline row and a keep, discard, or crash row for the candidate.
- `experiment_memory.jsonl` contains `candidate_run` and then `keep`, `discard`, or `crash`.
- `.autoresearch_state.json` has no pending experiment after the loop finishes.
- worker changes are limited to `train.py`; any other modified source file is a failure.

### 4. Metrics to Record for the Report

For each campaign, export these values from `results.tsv`, `experiment_memory.jsonl`, and `.autoresearch_state.json`:

- baseline AUC and best AUC, using `AUC = 1 - val_bpb`
- total candidate runs
- kept, discarded, and crash counts
- best improvement over baseline
- average improvement per kept candidate
- wall-clock runtime per trial and total runtime
- repeated bad proposal count after memory modes are implemented
- memory mode and manager mode

## REST API Parity Test After Upgrade

Once the reported service API is restored or implemented, repeat the same lifecycle through:

- `GET /health`
- `GET /status`
- `GET /memory` or `/memory-summary`
- `POST /run`
- `POST /keep` or `POST /discard`
- `POST /loop`

Pass condition: the REST path writes the same kind of trial records and state transitions as the CLI path.

## Cleanup

To leave the node on the original branch after a smoke test:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger
git branch --show-current
git status --short
```

Only reset or delete branches after confirming you do not need the generated experiment commits or logs.
