# Stage 1 Demo Plan And Current Status

## Objective

Stage one is a minimal but credible demonstration of supervised experimentation:

- a manager decides what to try next
- a bounded worker edits one approved file
- a control plane enforces lifecycle and memory
- a real node executes experiments and returns metrics

This stage is not trying to prove large-scale autonomous research. It is trying to prove a governed experimentation loop that is inspectable and repeatable.

## Current Deliverable

The active stage-one deliverable is built around [nodes/ResNet_trigger](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger).

Current proof points:

- worker node with fixed data path and single editable training surface
- control plane with `run`, `keep`, `discard`, `status`, `memory`, and `memory-summary`
- fast-search mode for short worker screening runs
- manager-driven multi-round campaign execution
- saved artifacts and plotting notebooks

Most recent validated campaign:

- campaign shape: `3 manager rounds x 5 worker runs`
- baseline validation AUC: `0.779911`
- best validation AUC: `0.787556`
- improvement: `+0.007645`

Primary saved outputs:

- [results.tsv](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/results.tsv)
- [experiment_memory.jsonl](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/experiment_memory.jsonl)
- [campaign_3x5_summary.json](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/artifacts/campaign_3x5_summary.json)
- [campaign_3x5_auc.svg](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/artifacts/campaign_3x5_auc.svg)
- [campaign_3x5_plot_analysis.ipynb](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/campaign_3x5_plot_analysis.ipynb)

## System Architecture

```text
Manager
  ↓ HTTP
Control plane
  ↓
Worker
  ↓
Node
  ↓
Metrics + memory + keep/discard
```

### Manager

The manager is responsible for:

- reading status and memory
- choosing the next experiment
- evaluating returned metrics
- deciding `keep` or `discard`
- changing search strategy over time

Stage-one manager modes:

- `claw` with a local OpenAI-compatible model
- interactive Codex session acting as the manager manually

### Control Plane

The control plane lives in [harness/claw-code/src](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code/src).

It is responsible for:

- state machine enforcement
- worker invocation
- metric parsing
- pending-run guard
- results and memory persistence

Important files:

- [api_server.py](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code/src/api_server.py)
- [autoresearch_worker.py](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code/src/autoresearch_worker.py)
- [autoresearch_runner.py](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code/src/autoresearch_runner.py)

### Worker

The worker is intentionally narrow:

- edit only [train.py](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/train.py)
- do not manage strategy
- do not edit data preparation
- return one experiment result at a time

### Node

The node contains the real experiment target.

For [nodes/ResNet_trigger](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger):

- [prepare.py](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/prepare.py) is fixed
- [train.py](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/train.py) is editable
- AUC is the scientific target metric
- `val_bpb = 1 - val_auc` is emitted for control-plane compatibility

## Why This Structure Is Good

### Separation Of Concerns

- manager handles strategy
- control plane handles governance
- worker handles implementation
- node handles execution

This keeps the expensive reasoning layer separate from the execution layer.

### Bounded Authority

- worker edits one file
- manager does not directly mutate the node
- control plane mediates every transition

This makes debugging and auditing practical.

### Reproducible Lifecycle

Every experiment follows the same structure:

```text
propose → edit → run → evaluate → keep/discard → persist
```

That is the core property the demo is meant to show.

## What The Demo Should Show

The stage-one demo should show the following clearly:

1. Initial system state is inspectable.
2. The manager reads status and memory before acting.
3. The worker performs one bounded code change.
4. The node executes a real run and returns metrics.
5. The manager makes a keep/discard decision based on metrics.
6. Results and memory are persisted.
7. Multiple rounds produce an interpretable performance trace.

## Stage-One Demo Surfaces

### Worker and manager demo notebook

Use [cnn_smoke_train_round1.ipynb](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/cnn_smoke_train_round1.ipynb) when you want:

- one worker-round demonstration
- manager control-plane execution
- saved records during the notebook flow

### Plotting / campaign analysis notebook

Use [campaign_3x5_plot_analysis.ipynb](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/campaign_3x5_plot_analysis.ipynb) when you want:

- cleaner plots
- campaign-level summaries
- manager-round and worker-slot analysis
- archived epoch-level views

### Harness verification notebook

Use [stage1_sudhir_demo_verification_v2.ipynb](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code/notebooks/stage1_sudhir_demo_verification_v2.ipynb) when you want:

- explicit control-plane contract proof
- lifecycle verification
- bounded worker proof
- failure/discard demonstration
- memory reuse proof

## Acceptance Criteria

Stage one should be considered successful if it demonstrates:

- manager completes a real run/evaluate/decide/persist loop
- worker is bounded to the approved surface
- at least one real metric-based run is executed
- memory is written and readable
- control-plane endpoints are demonstrably usable
- multi-round results can be plotted and interpreted

## What Is Already Proven

Already demonstrated in the current repo state:

- real worker-node training on the `ResNet_trigger` node
- model selection by validation AUC
- best-model checkpoint saving
- fast-search worker mode for short screening runs
- control-plane keep/discard lifecycle
- memory persistence through `experiment_memory.jsonl`
- multi-round manager campaign artifacts and plots

## Remaining Gaps

Stage one is good enough for a credible demo, but these are still open:

- stronger long-horizon memory and retrieval
- more reliable autonomous local manager behavior
- cleaner git layout around upstream-derived nested repos
- richer multi-node orchestration
- Rust-native autoresearch without relying on the Python control plane

## Recommended Demo Narrative

Use this order:

1. Show the architecture briefly.
2. Show the worker node and its bounded editable surface.
3. Show the control-plane endpoints.
4. Run or review one manager-driven worker cycle.
5. Show persisted results and memory.
6. Show the 3x5 campaign plot and highlight the improvement from `0.779911` to `0.787556`.

That sequence makes the system legible before showing performance curves.
