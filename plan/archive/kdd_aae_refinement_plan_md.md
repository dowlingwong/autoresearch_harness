# KDD AAE Refinement Plan for `autoresearch_harness`

## Goal

Turn the current Stage 2 real ResNet-trigger demonstration into a credible KDD AAE submission by framing it as an **agentic AI evaluation and governance paper**, not as a small ML optimization paper.

The current result already proves that the system can execute one real governed agent loop. The KDD-facing version must prove a broader evaluation claim:

> A governed control plane makes autonomous experimentation auditable, bounded, failure-aware, and empirically measurable under repeated trials.

The paper should stay deliberately modest. It should claim a governance/evaluation protocol, not a general autonomous scientist, not a universal optimizer, and not proof of scientific discovery.

---

## 0. Current Status: What We Already Have

### Already strong

- Real ResNet-trigger full-loop demonstration.
- Actual baseline metric and actual post-agent metric.
- Bounded edit from `LEARNING_RATE = 1e-3` to `5e-4`.
- Valid accepted trial.
- Complete artifact chain:
  - proposal;
  - generated worker packet;
  - patch diff;
  - run log;
  - parsed metrics;
  - JSONL ledger;
  - paper-facing CSV exports.
- Governance metrics already exported:
  - acceptance rate;
  - invalid rate;
  - editable-scope violations;
  - command failure rate;
  - metric parsing failure rate;
  - artifact capture completeness.
- Notebook and SVG plots already generated for the real ResNet run.

### Current paper-facing result

| Item | Current status |
|---|---|
| Benchmark node | `resnet_trigger` |
| Campaign id | `resnet_real_incremental` |
| Manager | `baseline_manager` |
| Memory mode | `append_only_summary_with_rationale` |
| Worker | `claw_style_worker` |
| Baseline val_auc | `0.779911` |
| Final val_auc | `0.782756` |
| Net gain | `+0.002845` |
| Records | 2 total: one baseline, one agent trial |
| Decision | `kept` |
| Validity | valid |
| Provenance completeness | complete for current run |

### Current evidence level

The current result supports this claim:

> Stage 2 converts an autoresearch-style worker loop into an auditable, governed experiment protocol with explicit node contracts, artifact capture, validity checks, and paper-facing metrics.

It does **not yet** support this stronger claim:

> The governance mechanism improves agent behavior across repeated trials.

That missing evidence is the reason the real memory ablation and stress/failure trial are essential.

---

## 1. Target Paper Identity

### Bad identity

> A local LLM agent improved a ResNet trigger by changing the learning rate.

This sounds too small and too domain-specific.

### Correct identity

> A harness-mediated control plane evaluates autonomous ML experimentation by separating proposal, execution, validation, decision authority, memory, and audit logging.

### One-sentence thesis

> Autonomous ML experimentation becomes scientifically credible only when the agent loop is governed by an explicit control plane that owns trial lifecycle, enforces bounded execution, records decisions append-only, and reports auditable governance metrics.

### Target contribution type

This is an **evaluation methodology contribution**.

It is not primarily:

- an AutoML method;
- a ResNet-trigger optimization method;
- a coding-agent benchmark;
- a detector-physics result;
- a general autonomous scientist.

---

## 2. Claims, Conditional Claims, and Non-Claims

## 2.1 Claims currently supported

These can be stated confidently based on the current Stage 2 result:

1. **Bounded execution improves auditability for autonomous experiment loops.**
   - The current real run records proposal, bounded edit, training command, metric parsing, validity, decision, and artifacts.

2. **Explicit lifecycle control makes autonomous experiments inspectable.**
   - The control plane owns state transition and records authoritative decisions.

3. **Governed experimentation can execute a real scientific ML node.**
   - The current ResNet-trigger campaign is not a dry run; it executed a real training command and parsed real metrics.

4. **The framework is not tied to a specific coding-agent backend.**
   - Claw-style/Ollama/Qwen is the current worker substrate, but the contribution should be described as backend-agnostic.

## 2.2 Claims that require new evidence

These should be written as planned claims until experiments are completed:

1. **Append-only memory with rationale reduces repeated poor proposals under fixed budget.**
   - Requires real memory ablation.
   - Do not state as proven until the ablation data exists.

2. **Governance behavior holds across multiple trials.**
   - Requires at least a 5-trial real campaign.

3. **The control plane catches invalid or failed actions.**
   - Requires one forced failure or stress trial.

4. **Governance is manager-agnostic.**
   - Requires manager comparison, or should remain a design claim rather than empirical claim.

## 2.3 Explicit non-claims

Include a non-claims paragraph in the paper:

> This work does not claim to build a general autonomous scientist, prove scientific discovery, introduce a universal optimization algorithm, or depend on a specific coding-agent backend. The ResNet-trigger task is used as a real scientific ML case study for evaluating governed autonomous experimentation; it is not claimed to represent all ML optimization tasks.

This protects the paper from reviewer objections.

---

## 3. Core Reviewer Risk and How to Remove It

### Risk 1 — “This is only a one-trial demo.”

Current status: true.

Fix:

- Run a 5–10 trial main campaign.
- Report lifecycle distribution: kept / discarded / failed-invalid.
- Show trajectory and provenance completeness.

### Risk 2 — “The AUC gain is too small.”

Current gain: `+0.002845`.

Fix:

- Do not lead with AUC.
- Treat AUC as secondary evidence that the governed loop can execute meaningful real experiments.
- Lead with governance metrics.

Safe sentence:

> The task metric confirms real execution, but the primary contribution is the governance protocol and its audit metrics.

### Risk 3 — “Memory claim is not proven.”

Current status: memory modes exist, but real memory ablation still needs execution.

Fix:

- Run real ablation across `none`, `append_only_summary`, and `append_only_summary_with_rationale`.
- Make repeated-bad rate the headline result.

### Risk 4 — “Too domain-specific.”

Fix:

- Frame ResNet-trigger as a controlled scientific ML case study.
- Keep detector details minimal.
- Emphasize general lifecycle: proposal → patch → execution → metric → validation → decision → ledger.

### Risk 5 — “Related work comparison is too broad or speculative.”

Fix:

- Treat claw-style systems, LangGraph, Hermes-style systems, `ml-intern`, and `multiautoresearch` as inspiration or future backend options.
- Do not imply they are direct baselines unless you actually compare against them.

---

## 4. Required Experiments Before Final Paper Writing

## Experiment A — Real Main Campaign

### Purpose

Upgrade the current one-agent-trial demonstration into a multi-trial governed campaign.

### Minimum acceptable version

- 5 real trials.
- Same node: `resnet_trigger`.
- Use `prompt_manager`, not only `baseline_manager`.
- Use `append_only_summary_with_rationale`.
- Start from a clean node state.
- Export all paper tables and figure data.

### Stronger version

- 10 real trials.
- Include at least one discarded trial or invalid/failure case.
- Report full trajectory, not only final best metric.

### Why this matters

The current clean campaign has one baseline and one agent trial. That is enough for a systems demonstration but not enough for a KDD AAE evaluation claim.

### What to report

| Metric | Why it matters |
|---|---|
| Initial val_auc | Shows starting point |
| Best val_auc | Shows optimization outcome |
| Final accepted val_auc | Shows final governed state |
| Net gain | Secondary performance evidence |
| Gain per budget unit | Efficiency under fixed budget |
| Kept / discarded / invalid count | Lifecycle behavior |
| Acceptance rate | Decision behavior |
| Invalid rate | Safety behavior |
| Complete provenance rate | Auditability |
| Artifact capture completeness | Reproducibility |
| Total wall-clock | Practical execution cost |

### Success criterion

The campaign does not need a large AUC gain. It must show complete governed lifecycle behavior over multiple trials.

---

## Experiment B — Memory / Governance Ablation

### Purpose

This is the core KDD AAE experiment.

It tests whether memory changes agent behavior, especially repeated poor proposals.

### Required modes

| Mode | Description | Expected role |
|---|---|---|
| `none` | No memory from previous trials | Weak baseline |
| `append_only_summary` | Outcome summary only | Intermediate baseline |
| `append_only_summary_with_rationale` | Outcome + decision rationale | Main method |

### Fixed conditions

All modes must use:

- same node;
- same manager;
- same worker;
- same model;
- same trial budget;
- same fast training configuration;
- same acceptance rule;
- same editable-scope policy;
- same starting node state.

### Minimum budget

- 5 trials per mode.
- 15 total real trials.

### Stronger budget

- 10 trials per mode.
- 30 total real trials.

Use 10 if the 5-trial result is noisy or inconclusive.

### Primary metric

```text
RepeatedBadRate = number_of_repeated_bad_proposals / total_number_of_proposals
```

### Required operational definition

A proposal is repeated-bad if it is semantically equivalent to a previously rejected, invalid, failed, or clearly degraded proposal and does not introduce a new justification, constraint, or corrective change.

Examples:

| Previous bad proposal | Later proposal | Repeated bad? |
|---|---|---|
| Lower LR to already-tested value, no gain | Lower LR to same value again | Yes |
| Edit frozen file, invalid scope | Edit same frozen file again | Yes |
| Change batch size caused failure | Same batch-size change with no fix | Yes |
| Lower LR failed, later lower LR with new patience/seed control | Maybe no |

### Secondary metrics

| Metric | Meaning |
|---|---|
| Acceptance rate | How many proposed changes were kept |
| Invalid rate | How often governance blocked invalid behavior |
| Best val_auc | Secondary task performance |
| Context length | Cost of memory mode |
| Compression ratio | Whether memory stays manageable |
| Recovery-after-failure | Whether the manager avoids failed patterns |

### Desired result pattern

The strongest pattern is:

```text
Repeated-bad rate:
none > append_only_summary > append_only_summary_with_rationale
```

Task metric does not need to follow the same pattern. Governance behavior is the headline.

### What to do if results are noisy

If 5 trials per mode does not show a clean pattern:

1. Increase to 10 trials per mode.
2. Add controlled repeated-failure opportunities.
3. Report confidence intervals or bootstrap intervals.
4. State that memory effects are preliminary and require larger-scale validation.
5. Do not overclaim.

---

## Experiment C — Forced Failure / Stress Trial

### Purpose

KDD AAE reviewers will expect failure behavior, not only success behavior.

A governance system must show that it catches invalid or unsafe actions.

### Required stress cases

Include at least one of the following:

| Stress type | Expected outcome |
|---|---|
| Edit outside allowed scope | `failed_invalid`, `invalid_edit_scope` |
| Produce no parseable metric | `failed_invalid`, `metric_missing` |
| Introduce syntax error | `failed_invalid`, `syntax_error` |
| Command fails | `failed_invalid`, `runtime_error` or `command_failure` |
| Repeat known rejected idea | counted in repeated-bad metric |

### Best single stress case

Use invalid edit scope.

It is the cleanest governance demonstration because the expected behavior is unambiguous:

```text
worker attempts forbidden edit → control plane detects violation → trial marked failed_invalid → ledger records failure category → no state corruption
```

### What to report

| Field | Expected value |
|---|---|
| `decision` | `failed_invalid` |
| `failure_category` | `invalid_edit_scope` or equivalent |
| `patch_ref` | present if patch generated |
| `raw_log_ref` | present if command ran |
| `parsed_metrics_ref` | absent or marked failed if metric missing |
| `state_update` | rejected / not committed |
| `ledger_record` | present |

### Paper framing

Do not hide failure trials. They are evidence.

Write:

> Unlike conventional experiment trackers that mainly record successful runs, the control plane records invalid and failed trials as first-class audit objects.

---

## Experiment D — Manager Comparison

### Purpose

Show that governance is not tied to one manager.

### Minimum version

Compare:

- `baseline_manager`;
- `prompt_manager`.

Use the same:

- node;
- memory mode;
- budget;
- worker;
- editable scope;
- metric parser.

### What to report

Do not emphasize which manager gets better AUC. Emphasize that governance metrics are still produced consistently.

| Manager | Kept | Discarded | Invalid | Provenance completeness | Artifact completeness |
|---|---:|---:|---:|---:|---:|
| `baseline_manager` | X | X | X | 100% | 100% |
| `prompt_manager` | X | X | X | 100% | 100% |

### Status

Optional, but useful if time permits.

This strengthens the claim:

> The framework evaluates agentic experimentation independently of the specific manager or LLM backend.

If not run, keep this as a design property, not an empirical result.

---

## 5. Metrics to Refine

## 5.1 Repeated-Bad Rate

### Add exact implementation definition

Create a short subsection in the paper:

> We define a repeated-bad proposal as a proposal that repeats the same edit target and edit mechanism as a prior rejected, invalid, or degraded trial without adding a new corrective rationale. We compute the rate over all proposals in a fixed-budget campaign.

### Add columns to exported table

Recommended columns for `memory_ablation_summary.csv`:

```text
campaign_id
memory_mode
budget
total_proposals
kept_count
discarded_count
failed_invalid_count
repeated_bad_count
repeated_bad_rate
best_val_auc
final_val_auc
context_chars
compression_ratio
artifact_capture_completeness
provenance_completeness
```

---

## 5.2 Failure Taxonomy

### Add explicit categories

Use a small fixed taxonomy:

| Category | Definition |
|---|---|
| `invalid_edit_scope` | Patch touched a disallowed file or region |
| `syntax_error` | Code cannot be parsed or imported |
| `runtime_error` | Training command exits nonzero |
| `metric_missing` | Run completes but expected metric cannot be parsed |
| `degraded_metric` | Valid run but worse than current best, therefore discarded |
| `no_op_patch` | Worker produced no effective change |

### Important distinction

Separate:

- **failed_invalid**: trial violates validity or cannot produce metric;
- **discarded**: trial is valid but not kept because metric worsened or did not improve;
- **kept**: trial is valid and accepted.

This distinction is important for reviewer trust.

---

## 5.3 Provenance Completeness

### Define as a metric

```text
ProvenanceCompleteness = completed_required_artifact_links / total_required_artifact_links
```

Required fields:

- `proposal_id`;
- `worker_packet_ref`;
- `patch_ref`;
- `run_log_ref`;
- `parsed_metrics_ref`;
- `decision_id`;
- `ledger_record_id`.

### Report separately for

- kept trials;
- discarded trials;
- failed-invalid trials.

This is stronger than reporting only kept trials.

---

## 5.4 Artifact Capture Completeness

### Define as a metric

```text
ArtifactCaptureCompleteness = captured_artifacts / expected_artifacts
```

Expected artifacts:

- proposal JSON;
- generated worker packet;
- patch diff;
- raw run log;
- parsed metrics;
- control-plane decision record;
- ledger entry.

For failed trials, expected artifacts may differ. Define this clearly.

---

## 6. System/Implementation Refinements

## 6.1 Add a No-Op Repeat Guard

### Problem

The current notes mention that the manager should avoid no-op repeats when the first baseline change is already present.

### Why it matters

Repeated no-op proposals make the agent look weak and may contaminate the memory ablation.

### Implementation instruction

Add a pre-execution validation step:

```python
if patch_is_empty or patch_matches_current_state:
    decision = "failed_invalid"
    failure_category = "no_op_patch"
    write_ledger_record(...)
    skip_execution()
```

### Paper benefit

This gives another governance category:

> The control plane distinguishes valid-but-unhelpful changes from invalid, no-op, and failed trials.

---

## 6.2 Make State Reset Reproducible

### Problem

Ablation campaigns must start from comparable states.

### Risk

If one memory mode starts after a previous edit, the comparison becomes unfair.

### Implementation instruction

Before each campaign:

```bash
git checkout -- nodes/ResNet_trigger/train.py
rm -rf experiments/artifacts/<campaign_id>
rm -f experiments/ledgers/<campaign_id>_trials.jsonl
```

Or implement:

```bash
python3 scripts/reset_node_state.py --node resnet_trigger
```

### Required paper statement

> Each ablation campaign was initialized from the same node state and executed under the same fixed budget and training configuration.

---

## 6.3 Add Seed and Configuration Logging

### Problem

AUC changes are small. Reviewers may ask whether the gain is noise.

### Implementation instruction

Log the following in every trial record:

```text
random_seed
training_seed
data_seed
model_seed
fast_config_hash
node_state_hash
patch_hash
command_hash
```

### Paper benefit

You can say:

> Because this paper studies governance rather than optimizer superiority, we report task metrics as secondary evidence and log all seeds/configuration hashes for reproducibility.

---

## 6.4 Add Bootstrap or Repeated Baseline if Time Allows

### Problem

The current net gain `+0.002845` is positive but small.

### Minimum fix

Do not overclaim the gain.

### Stronger fix

Run 3 repeated baselines under the same original config:

```text
baseline_seed_1
baseline_seed_2
baseline_seed_3
```

Then report:

```text
baseline mean ± std
best governed result
```

### Paper framing

If the gain is within baseline variance, write:

> The optimization gain is treated as secondary; the primary result is that the control plane produces complete and auditable lifecycle records under real execution.

This is safer and more credible.

---

## 6.5 Stabilize Real Memory Ablation Runner

### Problem

The limitations note says real memory ablation execution still depends on stable campaign-runner integration.

### Instruction

Before launching final KDD experiments:

1. Run a 1-trial smoke ablation over all three memory modes.
2. Verify each mode creates a ledger.
3. Verify each ledger includes `memory_mode`.
4. Verify repeated-bad export does not silently return zeros due to missing fields.
5. Only then run the 5- or 10-trial version.

### Acceptance criterion

The ablation is trustworthy only if the repeated-bad metric is computed from real proposal history, not manually inferred after the fact.

---

## 7. Related Work and External Repo Lessons

## 7.1 What to learn from each repo

The goal is not to copy these projects. The goal is to extract design patterns that strengthen the KDD AAE plan while preserving your paper's core contribution: **governed evaluation of autonomous experimentation**.

| Repo / system | What it does well | What to add to our plan | What not to claim |
|---|---|---|---|
| `dzhng/deep-research` | Simple iterative research loop with breadth/depth control, generated follow-up directions, accumulated learnings, and final Markdown report | Add explicit **budget parameters** for autonomous experimentation: breadth = number of candidate proposals per round, depth = number of refinement rounds. Add final `campaign_report.md` generated from ledger + artifacts. | Do not claim web research or literature search is your contribution. |
| `burtenshaw/multiautoresearch` | Organizes autonomous research around separate project tracks: pre-training, post-training, inference | Add a **multi-node roadmap**: `resnet_trigger` now, optional future nodes for detector pipeline, inference optimization, or post-training-style benchmark. Use this only to show extensibility. | Do not claim broad multi-benchmark coverage until you actually run more nodes. |
| `huggingface/ml-intern` | ML-focused autonomous engineer with interactive/headless modes, local model support, trace sharing, notification events, and clear architecture | Add **headless campaign mode**, optional approval mode, trace export, and event notifications for `approval_required`, `error`, and `turn_complete`. Add a reviewer-friendly trace viewer or JSONL-to-Markdown renderer. | Do not claim full ML engineer capability; your system is narrower and more governed. |
| `NousResearch/hermes-agent` | Long-lived assistant with persistent memory, skills, routines, gateways, providers, and trajectory compression | Add **skill/routine registry** only as future work or optional implementation: e.g. reusable experiment routines such as `lower_learning_rate`, `increase_regularization`, `change_batch_size`. Add trajectory compression for memory summaries. | Do not frame the system as self-improving in the broad Hermes sense. |
| `langchain-ai/langchain` | Standard model/tool abstractions, integrations, LangGraph orchestration, LangSmith observability/evaluation | Use LangChain for model/tool interfaces and LangGraph for optional stateful orchestration. Keep your control plane as the governance authority. Add LangSmith tracing as optional observability, not as the main ledger. | Do not replace your append-only ledger with LangSmith/LangGraph checkpoints; those are observability/orchestration, not your authoritative governance record. |

---

## 7.2 High-value additions from `deep-research`

`deep-research` is useful because it has a very clear recursive loop:

```text
query → generate search directions → process results → extract learnings → generate next directions → final report
```

For your system, translate this into:

```text
research objective → generate experiment proposals → execute bounded trials → extract learnings → generate next proposals → final campaign report
```

### Add to plan

1. **Breadth parameter**

```text
proposal_breadth = number of candidate experiment proposals generated per round
```

Example:

```bash
--proposal-breadth 3
```

2. **Depth parameter**

```text
campaign_depth = number of sequential refinement rounds
```

Example:

```bash
--campaign-depth 5
```

3. **Learning object**

Every trial should produce a compact learning:

```json
{
  "trial_id": "...",
  "edit_summary": "lowered learning rate from 1e-3 to 5e-4",
  "metric_delta": 0.002845,
  "decision": "kept",
  "lesson": "lower LR improved fast-smoke validation AUC under current config",
  "avoid_next": []
}
```

4. **Final campaign report**

Generate:

```text
paper/notes/<campaign_id>_campaign_report.md
```

Report sections:

- objective;
- budget;
- trial trajectory;
- accepted changes;
- discarded changes;
- failed-invalid trials;
- lessons learned;
- repeated-bad analysis;
- artifact manifest.

### Paper benefit

This makes your framework easier to understand as an iterative experimental research loop.

---

## 7.3 High-value additions from `ml-intern`

`ml-intern` is especially relevant because it is framed as an autonomous ML engineer. Its useful patterns are:

- interactive mode;
- headless mode;
- local model support;
- session traces;
- event notifications;
- approval/error/completion events;
- architecture diagram separating user operations and events.

### Add to plan

## Add two execution modes

```text
interactive mode: human approves risky trial actions
headless mode: fixed-budget benchmark campaign for paper results
```

Commands:

```bash
python3 scripts/run_campaign.py --mode headless ...
python3 scripts/run_campaign.py --mode interactive ...
```

For KDD AAE, use **headless mode** for reported experiments because it is reproducible.

Use **interactive mode** as future work or optional demo.

## Add event taxonomy

Add events emitted by the control plane:

```text
proposal_created
worker_packet_created
patch_generated
scope_check_passed
scope_check_failed
training_started
training_completed
metric_parsed
metric_missing
decision_kept
decision_discarded
decision_failed_invalid
approval_required
campaign_completed
```

## Add optional notification hooks

Add later, not required for the paper:

```text
on_error
on_approval_required
on_campaign_completed
```

Notification destinations:

- terminal;
- JSONL event stream;
- Slack later;
- email later.

### Paper benefit

Event logging strengthens your governance argument:

> The framework does not merely store final results; it emits lifecycle events that make autonomous experimentation monitorable.

---

## 7.4 High-value additions from `Hermes Agent`

Hermes is useful mainly as inspiration for long-term memory, routines, skills, providers, and trajectory compression.

For your paper, use this carefully.

### Add to future-work plan

## Skill/routine registry

Define small reusable experimental routines:

```text
routine.lower_learning_rate
routine.raise_weight_decay
routine.change_batch_size
routine.increase_dropout
routine.adjust_early_stopping
routine.run_repeated_seed_baseline
```

Each routine should include:

```json
{
  "routine_id": "lower_learning_rate",
  "allowed_files": ["train.py"],
  "parameters": {"factor": [0.1, 0.5, 0.8]},
  "risk_level": "low",
  "requires_approval": false,
  "known_failure_modes": ["too_small_lr_undertrains"]
}
```

## Trajectory compression

Add a memory compaction step:

```text
full trial ledger → compact campaign memory → manager context
```

This connects directly to the memory ablation.

### Do not overclaim

Do not say:

> Our system is self-improving like Hermes.

Say:

> Long-term skill/routine formation is future work; the present paper evaluates bounded campaign memory and governance metrics.

---

## 7.5 High-value additions from `multiautoresearch`

The useful idea is project organization across multiple research tracks.

### Add to plan

Structure your repo/paper around benchmark nodes:

```text
nodes/
  ResNet_trigger/
  future_detector_pipeline/
  future_inference_optimization/
  future_post_training_task/
```

Add a `NodeSpec` abstraction if not already clean:

```yaml
node_id: resnet_trigger
objective_metric: val_auc
maximize: true
allowed_edit_paths:
  - train.py
run_command: ...
parse_metric: ...
artifact_expectations: ...
failure_categories: ...
```

### Paper wording

> The present paper evaluates one real node and defines the node contract needed to extend the protocol to additional scientific ML tasks.

This is much safer than claiming broad coverage.

---

## 7.6 LangChain / LangGraph Integration Plan

You said you want to use LangChain. The right design is:

> LangChain provides model/tool abstraction. LangGraph optionally orchestrates the manager-worker flow. Your control plane remains the governance authority.

Do **not** let LangChain own the scientific state. Do **not** make LangSmith the authoritative record. Your append-only JSONL ledger remains the ground truth.

## Recommended architecture

```text
LangChain model wrapper
        |
        v
LangChain proposal chain / manager
        |
        v
[Your Control Plane]
        |
        |-- validates proposal schema
        |-- creates pending trial
        |-- invokes worker tool
        |-- validates edit scope
        |-- runs training command
        |-- parses metric
        |-- decides keep/discard/failed_invalid
        |-- writes append-only ledger
        v
Artifacts + Paper Metrics
```

Optional LangGraph version:

```text
Graph nodes:
  propose_trial
  create_worker_packet
  execute_worker
  validate_patch
  run_training
  parse_metrics
  decide_trial
  update_memory
  export_artifacts

Graph edges:
  propose_trial → create_worker_packet → execute_worker → validate_patch
  validate_patch → run_training if valid
  validate_patch → decide_trial if invalid
  run_training → parse_metrics → decide_trial → update_memory
```

But the `decide_trial` node must call your existing control-plane decision logic.

---

## 7.7 What to implement with LangChain first

### Phase 1 — LangChain model wrapper only

Implement this first. It is low-risk and useful.

```text
src/autoresearch/llm/langchain_client.py
```

Responsibilities:

- initialize chat model;
- support OpenAI-compatible/local endpoints;
- produce structured proposal JSON;
- log raw prompt/response as artifact;
- return result to existing manager interface.

Example interface:

```python
class LangChainProposalBackend:
    def propose(self, campaign_state: CampaignState) -> Proposal:
        ...
```

The rest of your system remains unchanged.

### Phase 2 — LangChain structured output for proposals

Use a strict proposal schema:

```python
class ExperimentProposal(BaseModel):
    edit_target: str
    edit_intent: str
    expected_effect: str
    risk_level: Literal["low", "medium", "high"]
    allowed_paths: list[str]
    rationale: str
    repeat_check: str
```

The manager should not emit free-form text only. It should emit structured proposals that the control plane can validate.

### Phase 3 — LangGraph orchestration backend

Only after Phase 1–2 are stable, add:

```text
src/autoresearch/orchestration/langgraph_runner.py
```

This should be optional:

```bash
python3 scripts/run_campaign.py --orchestrator native
python3 scripts/run_campaign.py --orchestrator langgraph
```

Paper framing:

> We include a LangGraph-compatible orchestration backend to show the control-plane abstraction can be embedded in standard agent engineering frameworks.

Do not make LangGraph required for the main experiments unless it is fully stable.

### Phase 4 — LangSmith tracing as secondary observability

Add optional LangSmith tracing:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=autoresearch-harness-kdd-aae
```

Use LangSmith to inspect model/tool traces, but still export your own paper metrics from your ledger.

Paper framing:

> LangSmith traces are useful for debugging and observability; the authoritative evaluation record is the append-only trial ledger.

---

## 7.8 New experiments enabled by LangChain

## Experiment E — Backend Interoperability Check

### Purpose

Show that the governance protocol is independent of LLM interface.

### Compare

| Backend | Role |
|---|---|
| current Ollama/Qwen direct backend | existing baseline |
| LangChain-wrapped Ollama/Qwen backend | compatibility check |

### Metrics

Do not compare only AUC. Compare:

- proposal schema validity;
- invalid proposal rate;
- artifact completeness;
- provenance completeness;
- repeated-bad rate;
- best/final val_auc as secondary.

### Paper placement

Optional appendix or short robustness subsection.

## Experiment F — Native vs LangGraph Orchestrator

Only run this if the LangGraph runner is stable.

Compare:

| Orchestrator | Governance owner | Ledger owner | Expected use |
|---|---|---|---|
| native control-plane runner | your framework | your framework | main paper results |
| LangGraph runner | your framework inside graph nodes | your framework | interoperability demo |

### Important

The governance result should be identical in both modes. LangGraph should not change the scientific semantics.

---

## 7.9 How to mention LangChain in the paper

### Good wording

> Our framework is designed to sit below common agent orchestration layers. Managers may be implemented using direct LLM calls, LangChain model/tool abstractions, or a LangGraph workflow. In all cases, the control plane remains responsible for trial validity, keep/discard decisions, append-only ledger updates, and paper-facing governance metrics.

### Bad wording

> We use LangChain to implement the agent, so the framework is production-grade.

Why bad: it makes the contribution look like engineering glue.

### Better positioning

> LangChain/LangGraph integration demonstrates compatibility with standard agent engineering infrastructure, while the paper's contribution is the governance protocol and evaluation metrics.

---

## 7.10 Concrete LangChain tasks to add to checklist

## Must-have if using LangChain in the KDD paper

- [ ] Add `LangChainProposalBackend`.
- [ ] Use structured proposal output.
- [ ] Save raw LangChain prompt/response as artifacts.
- [ ] Add `--llm-backend langchain` option.
- [ ] Run one real campaign using LangChain backend.
- [ ] Verify JSONL ledger is identical in schema to native backend.
- [ ] Keep native runner as fallback.

## Should-have

- [ ] Add optional LangSmith tracing.
- [ ] Add trace ID to trial record.
- [ ] Export trace links only if privacy-safe.
- [ ] Add `langchain_backend_smoke` test.

## Nice-to-have

- [ ] Add optional LangGraph runner.
- [ ] Add graph diagram.
- [ ] Compare native vs LangGraph orchestrator.
- [ ] Add human-in-the-loop interrupt for high-risk edits.

---

## 7.11 Updated related-work paragraph

Use this paragraph later:

> Recent open-source agent systems demonstrate several useful patterns for autonomous research workflows: iterative research loops with accumulated learnings, ML-focused coding agents with trace capture, long-lived assistants with persistent memory and routines, and general-purpose orchestration frameworks with model/tool abstractions. These systems motivate our implementation choices, but they do not by themselves define an evaluation protocol for governed autonomous experimentation. Our contribution is the control-plane layer that constrains what an agent may change, records every trial as an append-only audit object, separates proposal/execution from keep-discard authority, and exports governance metrics such as invalid-action rate, repeated-bad rate, and provenance completeness.

---


Use these limitations proactively. They make the paper more credible.

## 8.1 Single benchmark node

Current limitation:

> Stage 2 currently centers on one benchmark node.

Safe paper wording:

> We evaluate on one real scientific ML node. This is sufficient to demonstrate real governed execution, but not enough to claim broad benchmark coverage.

## 8.2 Memory ablation stability

Current limitation:

> Real memory ablation execution still depends on stable campaign runner integration.

Safe paper wording before running ablation:

> The memory ablation is a planned validation of the governance-memory claim.

Safe paper wording after successful ablation:

> We report memory ablation results only after verifying that each mode starts from the same node state and produces real campaign ledgers.

## 8.3 Dry-run tests are not full evidence

Current limitation:

> Dry-run smoke tests validate contracts but do not replace full worker campaigns.

Safe paper wording:

> Dry-run tests validate control-plane contracts, while the reported empirical results use real worker campaigns unless explicitly marked as smoke tests.

## 8.4 Future backend extensions

Current limitation:

> LangGraph, cloud backends, and UI layers are future extensions, not core claims.

Safe paper wording:

> Orchestration backends, cloud execution, and UI layers are future extensions. The present work focuses on the control-plane protocol and its audit metrics.

---

## 9. Paper Structure Refinement

## 9.1 Recommended Title

Best option:

> **Evaluating Governed Autonomous Experimentation with Bounded Execution and Auditable Memory**

Alternative options:

1. **A Governance Harness for Evaluating Agentic AI in Autonomous ML Experimentation**
2. **Harness-Mediated Evaluation of Autonomous Research Agents**
3. **Auditable Autonomous Experimentation with Explicit Lifecycle Control**

Why the best option works:

- “Evaluating” signals KDD AAE fit.
- “Governed” signals trust/safety/lifecycle.
- “Autonomous Experimentation” signals agentic workflow.
- “Bounded Execution” signals safety mechanism.
- “Auditable Memory” signals the memory ablation contribution.

---

## 9.2 Abstract Template Before Final Results

Use this before the full ablation is complete:

```text
Autonomous ML experimentation agents can propose code edits, execute training runs, and select follow-up trials, but their evaluation remains difficult because conventional task metrics do not capture invalid actions, repeated failures, auditability, or lifecycle governance. We present a harness-mediated control plane for governed autonomous experimentation. The framework separates proposal, execution, validation, and decision authority; enforces editable-scope constraints; records append-only trial ledgers; and exports governance metrics including invalid-action rate, repeated-bad rate, artifact completeness, and provenance completeness. We demonstrate the framework on a ResNet-trigger scientific ML benchmark with a real full-loop campaign that executes a bounded code edit and records complete provenance. We further define a memory-ablation protocol comparing no memory, summary memory, and rationale-linked append-only memory under fixed budget. The study argues that agentic experimentation should be evaluated not only by final task performance, but also by whether agent behavior is bounded, auditable, failure-aware, and reproducible.
```

## 9.3 Abstract Template After Final Results

Use this after the full ablation is complete:

```text
Autonomous ML experimentation agents can propose code edits, execute training runs, and select follow-up trials, but their evaluation remains difficult because conventional task metrics do not capture invalid actions, repeated failures, auditability, or lifecycle governance. We present a harness-mediated control plane for governed autonomous experimentation. The framework separates proposal, execution, validation, and decision authority; enforces editable-scope constraints; records append-only trial ledgers; and exports governance metrics including invalid-action rate, repeated-bad rate, artifact completeness, and provenance completeness. On a ResNet-trigger scientific ML benchmark, a real governed campaign achieves [BEST_VAL_AUC] from [INITIAL_VAL_AUC] while maintaining [PROVENANCE]% provenance completeness. A memory ablation across no-memory, summary-memory, and rationale-memory modes shows repeated-bad rate decreasing from [X]% to [Y]% under equal budget. These results show that governance metrics expose agent behavior beyond final task performance and provide a reproducible basis for evaluating autonomous experimentation systems.
```

---

## 9.4 Introduction Hook

Open with the evaluation problem, not your system.

Suggested opening:

> Autonomous agents are increasingly able to plan experiments, edit code, execute training runs, and select follow-up actions. However, evaluating these systems by final task score alone is insufficient: an agent may reach a good result while repeatedly attempting invalid edits, hiding failed runs, corrupting experimental state, or making decisions that cannot be audited. This creates a gap between autonomous experimentation and scientifically credible evaluation.

Then introduce your control plane as the answer.

---

## 9.5 Results Ordering

Do not lead with AUC.

Recommended order:

1. Governance lifecycle demonstration.
2. Memory ablation repeated-bad rate.
3. Failure/stress trial.
4. Provenance completeness.
5. Task metric improvement.

This makes the paper look like KDD AAE, not like a small ML optimization paper.

---

## 10. Figures and Tables to Add

## Required Figure 1 — Architecture Diagram

Create a simple diagram:

```text
Manager
  |
  v
Proposal
  |
  v
Control Plane ----> Memory / Ledger
  |                    ^
  v                    |
Worker ------------> Artifacts
  |
  v
Training Command
  |
  v
Metric Parser
  |
  v
Validity Check
  |
  v
Keep / Discard / Failed-Invalid Decision
```

Caption:

> The control plane owns lifecycle transitions and records append-only provenance. Managers and workers are pluggable and cannot directly commit trial state.

---

## Required Table 1 — Main Campaign

Include both optimization and governance metrics.

```text
Node
Manager
Memory mode
Budget
Initial val_auc
Best val_auc
Final val_auc
Net gain
Kept / discarded / failed_invalid
Acceptance rate
Invalid rate
Complete provenance rate
Artifact capture completeness
```

---

## Required Table 2 — Memory Ablation

This is the main result table.

```text
Memory mode
Budget
Repeated-bad count
Repeated-bad rate
Kept
Discarded
Failed-invalid
Best val_auc
Context chars
Compression ratio
```

---

## Required Table 3 — Failure Taxonomy

```text
Failure category
Count
Example trial id
Control-plane response
State committed? yes/no
```

---

## Required Table 4 — Provenance Chain

```text
Trial type
Proposal present
Patch present
Run log present
Parsed metric present
Decision present
Ledger entry present
Completeness
```

Rows:

- kept;
- discarded;
- failed_invalid.

---

## Required Figure 2 — Repeated-Bad Rate by Memory Mode

This is the visual centerpiece.

X-axis:

```text
none | append_only_summary | append_only_summary_with_rationale
```

Y-axis:

```text
repeated_bad_rate
```

Caption:

> Rationale-linked memory reduces repeated-bad proposals under fixed budget, indicating that governance memory changes agent behavior rather than merely recording outcomes.

If the result is weaker than expected, use this caption instead:

> Repeated-bad rate by memory mode. The ablation measures whether richer append-only memory changes agent proposal behavior under fixed budget.

---

## Required Figure 3 — Decision Breakdown

Stacked bars by memory mode:

```text
kept / discarded / failed_invalid
```

This shows lifecycle distribution.

---

## Optional Figure 4 — Campaign Trajectory

Plot `val_auc` by trial index.

Use this as secondary evidence.

---

## 11. Writing Rules

### Rule 1 — Governance first, metric second

Bad:

> The agent improved AUC by 0.002845.

Better:

> The real campaign demonstrates that the control plane can execute a bounded code edit, parse a real training metric, preserve complete provenance, and record an auditable keep decision. The accepted edit also improved validation AUC by 0.002845.

---

### Rule 2 — Do not overclaim from small gains

Bad:

> The system optimizes ResNet-trigger performance.

Better:

> The task metric confirms that the governed loop can execute meaningful real experiments, but optimization quality is not the primary claim.

---

### Rule 3 — Make failures look valuable

Bad:

> Some trials failed.

Better:

> Failed and invalid trials are recorded as first-class audit objects, allowing the evaluator to measure invalid-action rate, failure category, and recovery behavior.

---

### Rule 4 — Always separate agent and control plane

Bad:

> The agent decides whether to keep the trial.

Better:

> The agent proposes and executes candidate changes; the control plane validates artifacts and owns the keep/discard/failed-invalid decision.

---

### Rule 5 — State non-claims explicitly

Use this paragraph:

> This work does not claim to build a general autonomous scientist, a universal hyperparameter optimizer, or a new ResNet-trigger model. The contribution is an evaluation and governance protocol for autonomous experimentation, demonstrated on a real scientific ML task.

---

## 12. Concrete Implementation Checklist

## Must complete before final paper writing

- [ ] Run real main campaign with at least 5 trials.
- [ ] Use `prompt_manager` for the main KDD-facing campaign.
- [ ] Run a 1-trial smoke ablation to verify runner integration.
- [ ] Run memory ablation across all three modes.
- [ ] Ensure equal budget and same starting node state for all ablation modes.
- [ ] Include one forced invalid-scope or failed-command trial.
- [ ] Export `main_campaign_summary.csv`.
- [ ] Export `governance_metrics.csv`.
- [ ] Export `memory_ablation_summary.csv`.
- [ ] Export `repeated_bad_idea_rates.csv`.
- [ ] Export `accepted_discarded_invalid_counts.csv`.
- [ ] Export `campaign_trajectory.csv`.
- [ ] Commit all real JSONL ledgers.
- [ ] Commit all artifact directories or a compressed artifact package.

## Should complete if time allows

- [ ] Add manager comparison.
- [ ] Add repeated baseline seeds.
- [ ] Add artifact manifest.
- [ ] Add architecture diagram.
- [ ] Add pseudocode for the control-plane lifecycle.

## Nice to have

- [ ] Second benchmark node or toy node.
- [ ] Public GitHub release tag.
- [ ] Zenodo DOI or archived artifact package.
- [ ] Minimal reviewer quickstart.
- [ ] `make reproduce-kdd-aae` command.
- [ ] One-page artifact appendix.

---

## 13. Suggested Repository Changes

## Add scripts

```text
scripts/reset_node_state.py
scripts/run_kdd_main_campaign.py
scripts/run_kdd_memory_ablation.py
scripts/run_kdd_stress_trial.py
scripts/export_kdd_tables.py
scripts/export_kdd_figures.py
scripts/check_kdd_artifact_completeness.py
```

## Add paper files

```text
paper/kdd_aae_2026/main.tex
paper/kdd_aae_2026/sections/introduction.tex
paper/kdd_aae_2026/sections/related_work.tex
paper/kdd_aae_2026/sections/system.tex
paper/kdd_aae_2026/sections/experiments.tex
paper/kdd_aae_2026/sections/results.tex
paper/kdd_aae_2026/sections/discussion.tex
paper/kdd_aae_2026/tables/
paper/kdd_aae_2026/figures/
```

## Add artifact manifest

```text
artifact_manifest.json
```

Recommended fields:

```json
{
  "campaigns": [],
  "ledgers": [],
  "artifacts": [],
  "tables": [],
  "figures": [],
  "environment": {},
  "commands": []
}
```

---

## 14. Recommended Final Paper Outline

# 1. Introduction

- Autonomous experimentation is hard to evaluate.
- Final metric is insufficient.
- Need governance metrics: invalid actions, repeated failures, auditability, bounded execution.
- Introduce control plane.
- State contributions and non-claims.

# 2. Background and Related Work

- Agentic AI evaluation.
- Autonomous coding/research agents.
- Experiment tracking.
- AutoML.
- Claw-style, LangGraph, Hermes-style, `ml-intern`, and `multiautoresearch` as inspiration/future backend options, not direct baselines.
- Gap: lifecycle governance for autonomous experimentation.

# 3. Governed Control Plane

- Manager/worker separation.
- Trial lifecycle.
- Editable-scope enforcement.
- Pending-trial guard.
- Append-only ledger.
- Memory modes.
- Keep/discard/failed-invalid decision authority.

# 4. Evaluation Protocol

- ResNet-trigger benchmark.
- Fixed-budget campaign.
- Memory ablation.
- Stress trial.
- Metrics.
- Failure taxonomy.

# 5. Results

- Main campaign governance demonstration.
- Memory ablation repeated-bad result.
- Failure taxonomy.
- Provenance completeness.
- Secondary task metric trajectory.

# 6. Discussion

- What is proven.
- What is not proven.
- Why governance metrics matter.
- Limitations.
- Generalization path.

# 7. Conclusion

- Restate control-plane contribution.
- Emphasize evaluation and governance.

---

## 15. Acceptance-Oriented Priority Ranking

### Highest priority

1. Real memory ablation.
2. Forced failure/stress trial.
3. Multi-trial real campaign.
4. Repeated-bad rate table and figure.
5. Clear non-claims.

### Medium priority

6. Manager comparison.
7. Repeated baseline seeds.
8. Artifact manifest.
9. Better README quickstart.

### Lower priority

10. Second benchmark node.
11. Extensive optimization tuning.
12. Large AUC improvement.

Do not spend too much time chasing AUC. Spend the time proving governance behavior.

---

## 16. Submission Readiness Levels

## Level 0 — Not ready

Only dry-run ledgers or code-only claims.

## Level 1 — Weak / borderline

One real baseline and one agent trial.

This is where the project currently is.

## Level 2 — Acceptable workshop paper

- 5-trial real campaign.
- Real memory ablation with all three modes.
- One stress trial.
- Tables 1–4 complete.
- Clear governance framing.

## Level 3 — Strong workshop paper

- 10-trial main campaign.
- 10 trials per memory mode.
- Clean repeated-bad reduction pattern.
- Stress/failure taxonomy.
- Manager comparison.
- Reproducibility package.

## Level 4 — Very strong

- Second benchmark node.
- Repeated seed baselines.
- Public artifact package.
- Clear reviewer quickstart.
- Quantitative uncertainty estimates.

Target at least Level 2. Aim for Level 3 if time allows.

---

## 17. Immediate Next Steps

Do these in order:

1. Freeze the current Stage 2 result as the minimal real demo.
2. Stabilize the real memory-ablation runner with a 1-trial smoke test over all three modes.
3. Reset node state.
4. Run the 5-trial main campaign with `prompt_manager`.
5. Run the forced invalid-scope stress trial.
6. Run the memory ablation with 5 trials per mode.
7. Inspect repeated-bad rates.
8. If noisy, increase ablation to 10 trials per mode.
9. Export all KDD tables and figure CSVs.
10. Write the paper starting from governance, not AUC.
11. Add explicit limitations and non-claims.

---

## 18. The Core Sentence to Use as Filter

For every experiment, paragraph, table, and figure, ask:

> Does this help a reviewer evaluate whether the autonomous agent loop is bounded, auditable, failure-aware, reproducible, and behaviorally improved by governance memory?

If yes, keep it.

If it only says “AUC improved slightly,” move it to secondary evidence.

