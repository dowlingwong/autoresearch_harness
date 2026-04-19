# instruction.md
## Objective

Refactor and extend the current `autoresearch_harness` repository into a cleaner, more modular, and more publishable system for a KDD Agentic AI Evaluation workshop paper.

The target is not to build a flashy general autonomous researcher.
The target is to build a governed autonomous experimentation framework with:

- clear architectural separation
- bounded execution
- explicit keep/discard control
- append-only experiment memory
- reproducible evaluation
- measurable governance and agent-performance metrics

The project should be shaped so it can support:
1. a solid KDD AAE workshop paper now
2. later extensions toward broader autonomous experimentation and AI-for-science work

---

## Core framing

This repository should be reframed as:

**A governed autonomous experimentation framework for evaluating agentic optimization under bounded execution constraints**

or equivalently:

**Auditable autonomous experimentation with modular planner, harness, and experiment-node separation**

The repo should not present itself mainly as:
- a general autonomous researcher
- a thin wrapper around claw code
- a LangChain demo
- a collection of scripts

Instead, it should present itself as a research and evaluation framework.

---

## High-level architecture to converge to

The repo should be reorganized around the following conceptual layers:

### 1. Manager / Planner
Responsible for:
- proposing candidate modifications
- deciding what to try next
- consuming memory / history
- optionally using LangChain / LangGraph abstractions
- issuing bounded requests to the worker

This layer should be swappable:
- simple baseline manager
- current prompt-driven manager
- future LangGraph-based manager

### 2. Control Plane
Responsible for:
- orchestrating experiment rounds
- assigning budgets
- tracking trial lifecycle
- invoking worker and evaluator
- storing append-only state
- enforcing keep/discard transitions
- exposing a clean internal API or service boundary

This is the core governance layer.

### 3. Bounded Worker Harness
Responsible for:
- applying limited code changes
- respecting editable scope
- executing commands
- returning patch / logs / outputs
- operating as a constrained runtime

This may use or wrap `claw-code`, but the project should not be conceptually dependent on claw internals.
`claw-code` should be treated as the worker substrate, not the whole system.

### 4. Experiment Node
Responsible for:
- defining the concrete optimization target
- declaring editable files
- setup command
- run command
- metric extraction
- objective direction
- acceptance rule
- validity checks

The existing scientific ML / trigger node should become the primary benchmark node.

### 5. Memory / Audit Layer
Responsible for:
- append-only trial history
- accepted / discarded proposals
- rationale
- metric history
- repeated-failure detection
- replayability
- experiment summaries

This should become a first-class feature, not a side log.

### 6. Metrics / Evaluation Layer
Responsible for:
- experiment-level performance metrics
- governance metrics
- budget efficiency
- ablation comparison
- exportable summaries / figures / tables for the paper

---

## Main development goals

The repository must be improved toward four immediate goals:

### Goal A — clean system section for paper
The codebase and docs should clearly support a one-page system description with:
- manager
- control plane
- bounded worker
- experiment node
- metrics parser
- memory
- keep/discard loop

There should be a single clear architecture figure derivable from the repo structure.

### Goal B — one real benchmark node
Do not over-expand into many tasks.

The existing scientific ML / trigger optimization task is enough if it is made clean and reproducible.

The benchmark node must explicitly define:
- editable files
- fixed budget
- run command
- metric output
- acceptance rule
- failure criteria

### Goal C — evaluation beyond final AUC
The framework must track not only best final score, but also governance and evaluation metrics.

At minimum implement:
- total proposed modifications
- accepted modifications
- discarded modifications
- invalid or broken runs
- best metric improvement
- average improvement per accepted run
- repeated bad idea count
- wall-clock efficiency
- improvement per unit budget

### Goal D — at least one ablation
Implement at least one meaningful ablation for the paper.

Preferred ablation:
- no memory
- append-only memory
- append-only memory + explicit keep/discard rationale

Measure:
- duplicate bad proposals
- recovery after failed runs
- acceptance efficiency
- final performance under equal budget

Secondary possible ablations:
- bounded worker vs looser worker
- simple manager vs stronger planner
- with vs without explicit keep/discard governance

---

## Design principles

### 1. Separation of concerns
Do not mix:
- planning logic
- worker execution logic
- node definition
- metric parsing
- persistence
- reporting

Each should have its own module boundary.

### 2. Explicit contracts
Every layer should have a clearly defined contract.

Especially define a node contract and a trial record schema.

### 3. Append-only state
Avoid mutable “latest state only” workflows where possible.

Prefer append-only logs or structured records for:
- trial proposal
- patch applied
- run result
- keep/discard decision
- rationale
- metrics

### 4. Reproducibility
Every run should be reproducible or at least replayable enough for analysis.

Store:
- timestamps
- budget index
- manager prompt/config version
- worker mode
- target node
- patch / diff or reference to diff
- command outputs
- parsed metric result
- decision result

### 5. Publishability
Every engineering decision should be evaluated partly on whether it strengthens:
- system clarity
- experimental validity
- evaluation quality
- paper readability

Do not add complexity unless it helps one of those.

---

## Required repository improvements

## A. Repository structure

Refactor toward a cleaner structure such as:

```text
autoresearch_harness/
  README.md
  instruction.md
  paper/
    figures/
    tables/
    notes/
  docs/
    architecture.md
    node_spec.md
    experiment_protocol.md
    metrics.md
  src/
    manager/
    control_plane/
    worker/
    nodes/
    memory/
    evaluation/
    reporting/
    common/
  configs/
    managers/
    workers/
    experiments/
    nodes/
  experiments/
    runs/
    summaries/
  scripts/
    run_campaign.py
    summarize_campaign.py
    export_paper_tables.py
    export_paper_figures.py
  tests/
```

This exact layout is not mandatory, but the repo must clearly separate:
- orchestration
- execution
- node definitions
- logging / memory
- evaluation / reporting

---

## B. Define a formal node specification

Create a formal node spec for experiment targets.

Each node should expose or declare:

- `name`
- `description`
- `editable_paths`
- `setup_command`
- `run_command`
- `metric_name`
- `metric_direction` (`maximize` or `minimize`)
- `metric_parser`
- `acceptance_rule`
- `validity_checks`
- `budget_unit`
- `default_budget`

The primary benchmark node should be the existing scientific ML / trigger optimization task.

The node spec must be easy to serialize, inspect, and document.

---

## C. Define a formal trial record schema

Create a structured schema for one trial or candidate iteration.

Each trial record should include at least:
- trial id
- parent campaign id
- node id
- timestamp
- manager mode
- worker mode
- proposal summary
- files targeted
- patch / diff reference
- execution status
- raw logs reference
- parsed metric
- validity status
- keep/discard decision
- decision rationale
- wall-clock runtime
- cumulative budget consumed

This record should be append-only.

---

## D. Standardize keep/discard workflow

The keep/discard process should be explicit and machine-readable.

A candidate should move through states such as:
- proposed
- patched
- executed
- parsed
- evaluated
- kept
- discarded
- failed-invalid

The control plane should own these transitions.

Avoid ad hoc per-script behavior.

---

## E. Add structured memory modes

Implement configurable memory modes:

### memory_mode = none
- no historical trial context provided to the manager

### memory_mode = append_only
- manager sees append-only historical summaries of prior trials

### memory_mode = append_only_with_rationale
- manager sees summaries plus explicit keep/discard reasons and failure causes

This should be a top-level experimental toggle for ablation.

---

## F. Add manager baselines

Implement at least:
1. **simple baseline manager**
   - heuristic or minimal prompt-based approach
2. **current manager**
   - current main planning logic
3. **optional future manager hook**
   - interface point for LangChain / LangGraph manager integration

Do not force LangChain everywhere.
LangChain should be optional and modular, mainly for planner orchestration.

---

## G. Treat claw-code as worker substrate

Use `claw-code` only where it helps bounded execution.

The repo should abstract the worker interface so future worker backends are possible.

Do not let repo identity collapse into “wrapper around claw-code”.

Define a worker contract around:
- propose/apply patch
- run command
- collect logs
- return artifacts
- enforce editable scope

---

## H. Improve metrics and reporting

Implement campaign summaries that automatically compute:

- total proposals
- accepted proposals
- discarded proposals
- invalid runs
- acceptance rate
- failure rate
- best metric
- initial metric
- net metric gain
- average gain per accepted proposal
- average gain per proposal
- repeated bad proposal count
- wall-clock per trial
- total wall-clock
- gain per hour
- gain per budget unit

These should be exportable to:
- markdown tables
- CSV / JSON
- paper-ready summary tables

---

## I. Implement repeated-bad-idea detection

Add a lightweight mechanism to detect repeated poor proposals.

This can be approximate.

Possible methods:
- textual similarity of proposal summaries
- targeted file overlap + same rationale class
- repeated failure categories
- repeated patch pattern signatures

This metric is important for the governance and memory ablation.

---

## J. Add paper-oriented outputs

Add scripts that generate:
- summary table for main experiment
- summary table for memory ablation
- trial trajectory plot data
- wall-clock vs performance data
- acceptance / discard bar chart data
- failure category counts

Even if the plotting is not fully automated yet, the exported data should be paper-ready.

---

## Paper-aligned experimental plan

Implement the framework so it supports the following minimum paper experiments.

### Experiment 1 — main governed campaign
Run a fixed-budget campaign on the primary scientific ML node.

Report:
- initial score
- best score
- final accepted score
- number of trials
- accepted / discarded / invalid counts
- runtime
- best accepted modification summary

### Experiment 2 — memory ablation
Compare:
- no memory
- append-only memory
- append-only memory + rationale

Under equal budget, report:
- best score
- acceptance rate
- invalid rate
- repeated bad idea count
- recovery after failed run behavior
- runtime

### Experiment 3 — optional planner comparison
If feasible, compare:
- baseline manager
- stronger manager

Only do this if cheap enough after memory ablation is complete.

---

## Recommended docs to create

Create and maintain these docs:

### `docs/architecture.md`
Explain the full system with the core layers and control flow.

### `docs/node_spec.md`
Explain the node abstraction and benchmark contract.

### `docs/experiment_protocol.md`
Define fixed-budget evaluation protocol, campaign structure, acceptance rule, and reproducibility notes.

### `docs/metrics.md`
Define all reported metrics clearly and consistently.

### `paper/notes/contribution_claims.md`
Short bullet list of what the paper claims and what it does not claim.

---

## README rewrite goals

Rewrite the README so it clearly states:

### What this project is
A governed autonomous experimentation framework.

### What it is not
Not just a coding agent, not just a claw wrapper, not a general AI scientist claim.

### Core ideas
- planner / control plane / worker / node separation
- bounded execution
- append-only memory
- explicit keep/discard governance
- reproducible evaluation

### Current benchmark
Scientific ML / trigger optimization node.

### Current result status
Describe current results honestly and conservatively.

### Future roadmap
Show immediate KDD workshop direction and longer-term extensibility.

---

## What to avoid

Do not spend time on:
- unnecessary frontend/UI work
- too many benchmark domains
- large-scale cloud deployment unless required
- overclaiming “autonomous science”
- tightly coupling all logic to LangChain
- tightly coupling all logic to claw-code internals
- vague multi-agent complexity without evaluation benefit

The paper value is in governance + evaluation + modularity, not in adding many agents.

---

## LangChain / LangGraph integration guidance

LangChain or LangGraph may be integrated, but only as an optional planner or orchestration backend.

Use it only if it helps:
- manager modularity
- planner policy experiments
- tool abstraction
- future extensibility

Do not let LangChain become required infrastructure for the whole repository.

Preferred position:
- LangGraph as an optional manager backend
- the rest of the system remains framework-agnostic

---

## claw-code integration guidance

`claw-code` should be retained or wrapped as a bounded worker runtime where useful.

The repository should clearly separate:
- your control plane and evaluation logic
from
- the worker runtime implementation

This project’s intellectual contribution is not claw-code itself.
It is the governed experimentation framework built around a bounded worker.

---

## Additional concepts to incorporate from reference documents

The following concepts are worth incorporating because they strengthen the repository as both a research artifact and a systems artifact.

### From InterDeepResearch — context, provenance, and human steering

#### 1. Hierarchical context architecture
Extend the memory and logging design using three levels:
- **Information level**: raw artifacts such as patches, logs, metric files, summaries, error messages, and references to generated outputs
- **Action level**: discrete operations such as propose patch, apply patch, run experiment, parse metric, evaluate result, summarize trial, keep, discard, or mark invalid
- **Session level**: grouped action sequences such as one trial, one intervention episode, one budget round, or one campaign phase

This should become a first-class design principle for memory, reporting, and future UI or inspection tools.

#### 2. Milestone-based summaries
Introduce milestone actions that produce concise summaries of progress.
Examples:
- after each completed trial
- after each accepted proposal
- after each failed-invalid run cluster
- after a fixed number of action rounds
- at the end of a campaign phase

These milestone summaries should be stored persistently and used for manager context instead of replaying all raw history.

#### 3. Context reduction / memory compression
Implement a mechanism that keeps raw records in append-only storage but feeds compressed context to the manager.
Possible modes:
- raw history
- summarized history
- summarized history with rationale
- summarized history with failure taxonomy

The goal is to avoid context overload while preserving replayability and auditability.

#### 4. Provenance and backtrace
Add explicit dependency links so any final decision can be traced back through:
- accepted or discarded decision
- parsed metric and validity result
- execution logs and artifacts
- patch or code modification
- proposal summary and manager rationale
- earlier related failed or successful trials

The system should support a future capability where any accepted result can be explained through a structured backtrace.

#### 5. Human-agent collaboration hooks
Do not build a full UI now, but define protocol hooks for future human intervention:
- pause campaign
- inject guidance or constraints
- adjust budget allocation
- mark a direction as undesirable
- force keep/discard
- request a summary of current progress

The framework should be designed so human-in-the-loop steering is a natural extension, not an afterthought.

#### 6. Cross-view / cross-record navigation as a future design goal
Even if no interface is built now, the data model should support future inspection across:
- trial timeline
- action dependency graph
- artifact or information cards
- summary milestones

This means records should be linked and queryable rather than only stored as flat logs.

### From the startup technical guide — systems framing, AgentOps, and runtime rigor

#### 1. Standard agent-systems decomposition
Use the following standard vocabulary where helpful in docs and paper writing:
- **model**
- **tools**
- **orchestration**
- **runtime**
- **memory / context**
- **grounding / task environment**
- **evaluation and observability**

Map the current project onto these concepts without changing the core architecture:
- manager = orchestration
- bounded worker = runtime and tool execution substrate
- experiment node = grounded task environment
- memory layer = context and durable state
- control plane = governance and lifecycle orchestration

#### 2. AgentOps-oriented evaluation mindset
The evaluation should explicitly include operational dimensions, not only benchmark improvement.
The framework should support measurement of:
- correctness or task improvement
- reliability
- failure modes
- efficiency
- observability
- auditability
- budget and runtime cost

These should appear both in code outputs and in paper-facing tables.

#### 3. Runtime discipline
The bounded worker should be described and implemented as a runtime substrate with explicit concern for:
- scoped permissions
- reproducible execution
- error handling
- retry policy if any
- log capture
- artifact capture
- clear failure states

Even if the system is local and lightweight, its design should reflect runtime discipline rather than ad hoc script execution.

#### 4. Observability as a first-class requirement
The system should provide enough structured traces to inspect:
- what the manager proposed
- what the worker changed
- what command ran
- what outputs were produced
- what metric was parsed
- why the candidate was kept or discarded

Observability should be treated as part of the contribution, not merely as debugging support.

#### 5. Data architecture thinking
Different classes of stored data should be treated differently:
- long-term memory / campaign history
- short-term working context for the current manager step
- transactional or decision ledger for authoritative keep/discard and state transitions

The implementation can remain simple, but the conceptual separation should be documented.

#### 6. Multi-agent and framework interoperability as optional future paths
Retain room for future support of:
- multiple managers or specialized planner policies
- external tool protocols
- LangGraph-based orchestration
- other worker runtimes

Do not force this now, but preserve clean interfaces so these become straightforward extensions.

## Additional metrics and experiments inspired by the reference documents

Add the following metrics if feasible:
- number of milestone summaries generated
- compression ratio between raw history and manager-fed summarized context
- number of times human-style intervention hooks would have been useful, if approximated retrospectively
- average backtrace depth for accepted results
- proportion of accepted decisions with complete provenance chain
- action-type distribution across a campaign

Potential future experiments:
- raw memory vs summarized memory vs summarized memory with rationale
- effect of milestone summaries on repeated-bad-idea rate
- effect of provenance-aware memory on decision quality
- human-guided intervention simulation vs fully autonomous campaign
- planner comparison under equal runtime budget and equal worker constraints

## Additional documentation to create

Add these if capacity allows:

### `docs/memory_architecture.md`
Explain hierarchical context levels, milestone summaries, compression, and provenance links.

### `docs/provenance.md`
Explain how decisions trace back to patches, logs, metrics, and prior trial context.

### `docs/agentops.md`
Define operational evaluation dimensions such as reliability, auditability, runtime efficiency, and observability.

## Additional implementation priorities

After the existing Priority 1 work, strongly consider the following sub-priorities:
- implement hierarchical trial/action/artifact records
- implement milestone summaries
- implement summarized manager context mode
- add provenance links between proposals, patches, runs, metrics, and decisions
- expose structured traces for later reporting and possible UI inspection

## Future-extendable directions

The codebase should be designed so the following future directions are easy:

### 1. Additional experiment nodes
Support multiple tasks beyond the first scientific ML benchmark.

Examples:
- alternative ML optimization tasks
- toy autoresearch-style training tasks
- scientific reconstruction tasks
- small hyperparameter/search tasks

### 2. Alternative planner policies
Support:
- heuristic baseline manager
- stronger prompt manager
- LangGraph-based planner
- future RL-trained planner

### 3. Alternative worker backends
Support more than one worker substrate over time.

### 4. Richer memory strategies
Support:
- compressed memory
- failure taxonomy memory
- retrieval-based memory
- strategy summaries
- learned proposal selection

### 5. Better evaluation suites
Support:
- cross-seed experiments
- robustness under noisy metrics
- cost-aware optimization
- policy comparison
- constrained budget benchmarking

### 6. Scientific-AI bridge
Extend the system into a broader AI-for-science experimentation platform:
- domain-constrained optimization
- reproducible scientific ML tuning
- audit-friendly experiment governance

### 7. Toward stronger papers
Longer-term paper directions may include:
- benchmark suite for governed autonomous experimentation
- planner vs harness decomposition study
- memory/governance formal ablations
- AI-for-science autonomous experimentation case studies

---

## Priority order

Follow this priority order strictly.

### Priority 1
Refactor repository structure and contracts:
- node spec
- trial schema
- control-plane lifecycle
- memory modes
- metrics

### Priority 2
Stabilize one benchmark node and one main campaign pipeline.

### Priority 3
Implement reporting and paper-friendly outputs.

### Priority 4
Run memory ablation and produce summary tables.

### Priority 5
Only then consider optional LangChain / LangGraph planner integration.

---

## Deliverables expected from the agent

The agent should produce:

1. a cleaner repository structure
2. formal node specification
3. formal trial record schema
4. append-only memory implementation with modes
5. standardized keep/discard lifecycle
6. campaign summary metrics
7. one benchmark node ready for fixed-budget experiments
8. scripts for paper-ready summary outputs
9. rewritten README
10. architecture and experiment docs
11. clear list of future extension hooks

---

## Success criteria

This refinement is successful if, after completion, the repo can support:

- a one-page system description
- one main experiment on a real task
- one memory/governance ablation
- paper-ready metrics and tables
- a clean KDD AAE framing around evaluation and governance

A strong final test is:

**Could this repository plausibly back a workshop paper titled something like “Auditable Autonomous Experimentation with Bounded Execution and Explicit Keep/Discard Control”?**

If yes, the refinement is on the right track.

