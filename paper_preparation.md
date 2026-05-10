# Paper Preparation Plan

## Current Status

Most infrastructure, reporting, artifact indexing, and paper-section scaffolding are implemented. The main remaining work is to produce final real-run evidence, regenerate paper artifacts from those runs, and complete the final paper packaging.

The project is not yet only "full loop and writing." The dry-run evidence is useful for governance-contract validation, but the final paper should distinguish dry-run artifacts from real worker campaigns.

## Remaining Work Before Final Deliverables

1. Run real evidence, not only dry-run evidence.
   - Chunk 2.2 main campaign is currently done as dry-run.
   - Chunk 2.4 memory ablation is currently done as dry-run.
   - For a strong paper, rerun these as real campaigns against Ollama and the worker backend.

2. Generate event streams for final campaigns.
   - Chunk 5.4 is implemented, but existing campaign artifacts predate it.
   - Rerun campaigns so `experiments/events/<campaign>_events.jsonl` exists.

3. Regenerate paper artifacts after real runs.
   - `paper/tables/*.csv`
   - `paper/figures/fig*.svg`
   - `paper/tables/artifact_completeness_report.txt`
   - `artifact_manifest.json`

4. Finish writing chunks.
   - Chunk 4.6: Abstract and Introduction.
   - Chunk 5.3: LaTeX structure, if the final deliverable needs a compilable paper.

5. Optional but useful.
   - Chunk 2.5 manager comparison supports manager/backend-agnostic claims.

## Recommended Experiment Design

### Experiment 1: Main Real Campaign

Purpose: demonstrate complete governed lifecycle behavior on a real scientific ML node.

Recommended setup:

- Node: `resnet_trigger`
- Manager: `prompt_manager`
- Memory: `append_only_summary_with_rationale`
- Budget: 5 trials minimum, 10 if time permits
- Backend: current Claw/Ollama worker path

Report:

- kept / discarded / failed-invalid distribution
- acceptance rate
- invalid rate
- provenance completeness
- artifact capture completeness
- event-stream completeness
- AUC trajectory as secondary task evidence

### Experiment 2: Memory Ablation

Purpose: test whether memory mode changes agent behavior.

Modes:

- `none`
- `append_only_summary`
- `append_only_summary_with_rationale`

Controls:

- same node
- same model
- same budget per mode
- same reset procedure
- same editable scope
- same metric parser
- same acceptance rule

Primary metric:

```text
RepeatedBadRate = repeated_bad_proposals / total_proposals
```

Pre-stated hypothesis:

```text
repeated_bad_rate(none) > repeated_bad_rate(append_only_summary)
                        > repeated_bad_rate(append_only_summary_with_rationale)
```

If the result is flat or reversed, report it as a valid negative finding.

### Experiment 3: Forced Failure Stress Trial

Purpose: prove invalid actions become first-class audit objects.

Use the existing invalid-edit-scope stress worker.

Expected result:

- decision: `failed_invalid`
- failure category: `invalid_edit_scope`
- patch reference present
- raw log reference present
- git state unchanged
- pending guard removed
- event stream includes `decision_made` with failure category

### Experiment 4: Optional Manager or Backend Comparison

Purpose: support backend/manager-agnostic governance claims.

Options:

- `baseline_manager` vs `prompt_manager`
- native proposal backend vs `--llm-backend langchain`

Do not frame this as "which manager is smarter." Frame it as: the same governance contract, ledger schema, artifact capture, and metrics are produced across manager/backend choices.

## Deliverable Map

Tables:

- Table 1: main campaign governance and task metrics
- Table 2: failure taxonomy
- Table 3: memory ablation
- Table 4: provenance chain
- Table 5: AIDE comparison

Figures:

- Figure 1: architecture diagram
- Figure 2: repeated-bad rate by memory mode
- Figure 3: decision breakdown
- Figure 4: AUC trajectory

Artifacts:

- campaign ledgers under `experiments/ledgers/`
- event streams under `experiments/events/`
- trial artifacts under `experiments/artifacts/`
- paper tables under `paper/tables/`
- paper figures under `paper/figures/`
- artifact completeness report
- root `artifact_manifest.json`

## Writing Order

1. Experiments section.
   - Freeze protocol before interpreting results.
   - State the memory ablation hypothesis before results.

2. Results section.
   - Lead with governance metrics, not AUC.
   - Present lifecycle distribution, memory ablation, decision breakdown, failure taxonomy, and provenance before task metric.
   - Put AUC trajectory last as secondary evidence.

3. Discussion section.
   - Separate what is proven from what is not proven.
   - Keep dry-run versus real-run evidence explicit.
   - Name the holdout-node limitation directly.

4. Introduction and Abstract.
   - Write these last.
   - Fill in final numbers only after regenerating tables and figures.
   - Use the core framing: Agent = Model + Harness.
   - State that the harness owns lifecycle, scope enforcement, audit, and decisions.

## Immediate Next Step

Run one real main-campaign trial first to validate the full path:

1. Reset node state.
2. Run `scripts/run_kdd_main_campaign.py` with real worker settings.
3. Confirm these exist:
   - ledger JSONL
   - event-stream JSONL
   - artifact directory
   - patch and raw log references
4. If clean, scale to 5 or 10 trials.
5. Then run the three memory-ablation arms.

## Ollama Real-Campaign Runbook

Use this section to wire local Ollama into a real campaign experiment.

### 1. Start or Verify Ollama

In one terminal:

```bash
ollama serve
```

If the port is already in use, Ollama is probably already running.

Check that the local API responds:

```bash
curl http://localhost:11434/api/tags
```

If `qwen2.5-coder:7b` is missing, pull it:

```bash
ollama pull qwen2.5-coder:7b
```

Optional API smoke test:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5-coder:7b",
  "messages": [{"role": "user", "content": "Reply with ok only."}],
  "stream": false
}'
```

### 2. Repo Preflight

From repo root:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness

uv run --extra dev python -m pytest \
  tests/test_provider_resolver.py \
  tests/test_event_stream.py \
  tests/stage2/test_stage2_control_plane.py
```

Before reset, make sure there are no important manual edits in
`nodes/ResNet_trigger/train.py`. The reset command restores editable node files.

### 3. First Real 1-Trial Smoke

Use a non-final campaign id first:

```bash
uv run --extra dev python scripts/reset_node_state.py \
  --node resnet_trigger \
  --campaign-id ollama_real_smoke
```

Run one real trial:

```bash
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
uv run --extra dev python scripts/run_kdd_main_campaign.py \
  --node resnet_trigger \
  --budget 1 \
  --campaign-id ollama_real_smoke \
  --manager prompt_manager \
  --memory-mode append_only_summary_with_rationale \
  --node-root nodes/ResNet_trigger \
  --model ollama/qwen2.5-coder:7b \
  --no-export
```

Expected outputs:

```text
experiments/ledgers/ollama_real_smoke_trials.jsonl
experiments/events/ollama_real_smoke_events.jsonl
experiments/artifacts/ollama_real_smoke/
```

Inspect:

```bash
tail -n 1 experiments/ledgers/ollama_real_smoke_trials.jsonl | python3 -m json.tool
tail -n 20 experiments/events/ollama_real_smoke_events.jsonl
```

### 4. Optional LangChain Proposal Backend Smoke

Use this after the native smoke works:

```bash
uv run --extra dev python scripts/run_kdd_main_campaign.py \
  --node resnet_trigger \
  --budget 1 \
  --campaign-id ollama_langchain_smoke \
  --manager prompt_manager \
  --memory-mode append_only_summary_with_rationale \
  --node-root nodes/ResNet_trigger \
  --model ollama/qwen2.5-coder:7b \
  --llm-backend langchain \
  --no-export
```

This checks that Ollama can also drive proposal generation through the
LangChain backend. LangChain still cannot write ledgers or make decisions; it
only returns `ManagerProposal`.

### 5. Final Main Campaign

When the one-trial smoke is clean, reset and run the canonical 5-trial campaign:

```bash
uv run --extra dev python scripts/reset_node_state.py \
  --node resnet_trigger \
  --campaign-id kdd_main_5trial
```

```bash
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
uv run --extra dev python scripts/run_kdd_main_campaign.py \
  --node resnet_trigger \
  --budget 5 \
  --campaign-id kdd_main_5trial \
  --manager prompt_manager \
  --memory-mode append_only_summary_with_rationale \
  --node-root nodes/ResNet_trigger \
  --model ollama/qwen2.5-coder:7b
```

Monitor events while running:

```bash
tail -f experiments/events/kdd_main_5trial_events.jsonl
```

### 6. If the Campaign Crashes

List pending guards:

```bash
uv run --extra dev python scripts/recover_pending_trial.py --list
```

Inspect the main campaign guard:

```bash
uv run --extra dev python scripts/recover_pending_trial.py \
  --inspect kdd_main_5trial
```

Mark the pending trial failed safely:

```bash
uv run --extra dev python scripts/recover_pending_trial.py \
  --mark-failed kdd_main_5trial \
  --reason "worker interrupted during real campaign" \
  --node resnet_trigger \
  --manager-mode prompt_manager \
  --worker-mode claw_style_worker \
  --memory-mode append_only_summary_with_rationale
```

### 7. After the Main Campaign

Regenerate tables, completeness report, and manifest:

```bash
uv run --extra dev python scripts/export_kdd_tables.py \
  --campaign-id kdd_main_5trial

uv run --extra dev python scripts/check_kdd_artifact_completeness.py \
  --campaigns kdd_main_5trial \
  --output paper/tables/artifact_completeness_report.txt

uv run --extra dev python scripts/generate_artifact_manifest.py \
  --output artifact_manifest.json
```

Only after this path is clean should you run the three real memory-ablation
arms. Run them one at a time, resetting node state before each arm.
