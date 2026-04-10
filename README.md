# autoresearch_harness

Manager-controlled experimentation over bounded worker nodes.

This repository combines:

- a Python control plane in [harness/claw-code](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code)
- one reference autoresearch node in [nodes/autoresearch-macos](/Users/wongdowling/Documents/autoresearch_harness/nodes/autoresearch-macos)
- one active stage-one demo node in [nodes/ResNet_trigger](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger)

The current project focus is a governed experiment loop:

```text
Manager
  ↓
Control plane API
  ↓
Bounded worker
  ↓
Experiment node
  ↓
Metrics, memory, keep/discard
```

## Current Status

The stage-one demo path is working on the `ResNet_trigger` node.

What is implemented now:

- bounded worker edits to `train.py`
- control-plane lifecycle with `run`, `keep`, and `discard`
- append-only experiment memory and results tracking
- manager-driven multi-round campaigns
- notebook-based inspection and plotting

Most recent verified campaign:

- node: [nodes/ResNet_trigger](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger)
- shape: `3 manager rounds x 5 worker runs`
- baseline validation AUC: `0.779911`
- best validation AUC: `0.787556`
- improvement over baseline: `+0.007645`

Saved campaign artifacts:

- [results.tsv](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/results.tsv)
- [campaign_3x5_records.json](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/artifacts/campaign_3x5_records.json)
- [campaign_3x5_summary.json](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/artifacts/campaign_3x5_summary.json)
- [campaign_3x5_auc.svg](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/artifacts/campaign_3x5_auc.svg)
- [campaign_3x5_plot_analysis.ipynb](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/campaign_3x5_plot_analysis.ipynb)

## Repository Layout

```text
autoresearch_harness/
├── README.md
├── stage_1_demo_plan_optimized_for_sudhir.md
├── harness/
│   └── claw-code/                  # upstream-derived harness checkout
│       ├── src/                    # Python control plane
│       ├── rust/                   # Rust claw CLI
│       ├── notebooks/              # harness demos and verification notebooks
│       └── tests/                  # integration tests
├── nodes/
│   ├── autoresearch-macos/         # reference GPT/autoresearch node
│   └── ResNet_trigger/             # active stage-one worker node
│       ├── prepare.py              # fixed data/split pipeline
│       ├── train.py                # only file the worker edits
│       ├── program.md              # worker guidance
│       ├── artifacts/              # checkpoints, histories, plots
│       ├── results.tsv             # experiment ledger
│       └── experiment_memory.jsonl # event memory
└── runs/                           # ad hoc run outputs
```

## Component Model

### Manager

The manager decides what to try next.

Typical responsibilities:

- read `/status`, `/memory`, and `/memory-summary`
- propose a bounded experiment
- inspect returned metrics
- decide `keep` or `discard`
- avoid repeating losing ideas

Manager options used in this repo:

- `claw` with a local OpenAI-compatible model
- an interactive Codex session acting as the manager manually

### Control Plane

The control plane lives in [harness/claw-code/src](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code/src).

It owns:

- experiment state
- worker invocation
- metric parsing
- pending-run guards
- keep/discard transitions
- memory persistence

Important entrypoints:

- [api_server.py](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code/src/api_server.py)
- [autoresearch_worker.py](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code/src/autoresearch_worker.py)
- [main.py](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code/src/main.py)

### Worker

The worker is a bounded code-editing agent.

Current stage-one contract:

- edit only [train.py](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/train.py)
- run one experiment
- return metrics
- do not manage state or strategy

### Node

The node is the experiment target repository.

For the active demo node:

- [prepare.py](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/prepare.py) stays fixed
- [train.py](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/train.py) is the editable surface
- validation AUC is the primary scientific metric
- a compatibility scalar is emitted as `val_bpb = 1 - val_auc`

## Active Demo Path

If you want the shortest path to the current working demo, use the `ResNet_trigger` node.

### 1. Worker node only

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger
uv sync
uv run train.py
```

That writes artifacts into [artifacts](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/artifacts), including:

- `best_model.pt`
- `best_performance.json`
- `history_latest.json`
- `metrics_latest.json`

### 2. Start the control plane

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
python3 -m src.main api-server \
  --root /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger \
  --port 7331 \
  --listen 127.0.0.1 \
  --backend ollama \
  --model qwen2.5-coder:7b \
  --host http://localhost:11434
```

### 3. Drive one manager iteration

Example:

```bash
curl http://127.0.0.1:7331/status
curl http://127.0.0.1:7331/memory-summary

curl -X POST http://127.0.0.1:7331/run \
  -H 'Content-Type: application/json' \
  -d '{
    "packet": {
      "objective": "Change LEARNING_RATE from 1e-3 to 5e-4 and keep the rest unchanged.",
      "description": "lr-screening"
    }
  }'

curl -X POST http://127.0.0.1:7331/keep \
  -H 'Content-Type: application/json' \
  -d '{"rationale":"Validation AUC improved over the current best."}'
```

## API Surface

The control plane is designed for manager-facing HTTP use.

Core endpoints:

- `GET /health`
- `GET /status`
- `GET /memory?limit=N`
- `GET /memory-summary`
- `POST /setup`
- `POST /baseline`
- `POST /run`
- `POST /keep`
- `POST /discard`
- `POST /loop`

Current safeguards:

- pending experiment guard on `POST /run`
- structured memory writes for candidate and decision events
- rationale support on `keep` and `discard`

## Notebooks

Root-level project work is concentrated in two notebook tracks.

### Harness notebooks

Located under [harness/claw-code/notebooks](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code/notebooks):

- `stage1_sudhir_demo_verification_v2.ipynb`: stage-one control-plane verification
- `autoresearch_control_plane_demo.ipynb`: earlier control-plane demo

### Node notebooks

Located under [nodes/ResNet_trigger](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger):

- [cnn_smoke_train_round1.ipynb](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/cnn_smoke_train_round1.ipynb): active worker-node and manager-demo notebook
- [campaign_3x5_plot_analysis.ipynb](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/campaign_3x5_plot_analysis.ipynb): cleaner plotting and campaign analysis

## Upstream / Repository Notes

[harness/claw-code](/Users/wongdowling/Documents/autoresearch_harness/harness/claw-code) is an upstream-derived checkout of [ultraworkers/claw-code](https://github.com/ultraworkers/claw-code), which is still actively changing.

Current practical interpretation of the repo:

- `claw-code` is the harness dependency and manager runtime surface
- `autoresearch-macos` is the original reference node
- `ResNet_trigger` is the current demo node under active adaptation

If you plan to keep pulling upstream `claw-code`, the clean long-term layout is:

- `harness/claw-code` as a submodule
- `nodes/autoresearch-macos` as a submodule if you also want upstream sync there
- `nodes/ResNet_trigger` as a normal folder in this root repo

## Limits

What is proven now:

- governed manager/control-plane/worker loop
- bounded worker editing surface
- metric-based keep/discard
- persisted experiment memory
- multi-round campaign execution and plotting

What is not proven yet:

- strong autonomous strategy over long campaigns
- multi-node orchestration
- robust concurrent manager execution
- pure Rust-native autoresearch without the Python control plane
