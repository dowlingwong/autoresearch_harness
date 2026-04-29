# Stage 3 Checklist

This checklist is based on the current `autoresearch_harness` architecture plus comparison against:

- Hugging Face ML Intern: https://github.com/huggingface/ml-intern
- Multi-Agent Autoresearch: https://github.com/burtenshaw/multiautoresearch
- Open Deep Research: https://github.com/dzhng/deep-research
- Hermes Agent: https://github.com/NousResearch/hermes-agent
- KDD AAE 2026 workshop: https://kdd-eval-workshop.github.io/agenticai-evaluation-kdd2026/

## Current Architecture Read

- [x] The project is already organized around the right paper-facing layers: manager, control plane, worker, node contracts, memory/audit, evaluation, and reporting.
- [x] The core contribution is not a generic agent wrapper. It is the governed experiment control plane: bounded proposals, lifecycle, editable-scope validation, budget accounting, append-only trial records, provenance, deterministic summaries, and paper exports.
- [x] Stage 2 has a real `run_real_campaign` path wired through `ClawWorker`, but the documentation still describes real execution as primarily legacy Stage 1. Stage 3 should make the Stage 2 real path the canonical path and update docs accordingly.
- [x] Stage 2 dry-run campaigns and memory ablations are useful for deterministic tests, but KDD AAE will need real-run evidence, not only synthetic dry-run ledgers.
- [x] The current `pyproject.toml` does not declare runtime dependencies such as `langgraph`, `langchain-core`, or `langchain-ollama`; scripts currently compensate by manually adding `.venv` site-packages. That should be fixed for reproducibility.

## 1. Do We Need Hermes Agent For The Manager?

Short answer: no, not as a required manager dependency.

- [ ] Keep the built-in manager interface small: `ManagerStatus + MemoryContext + NodeSpec -> ManagerProposal`.
- [ ] Do not replace the Stage 2 manager with Hermes. Hermes is a broad operator/delegation/runtime system with persistent memory, messaging gateways, subagents, remote backends, skills, cron, and trajectory tooling. That is too much surface area for the scientific manager contract.
- [ ] Treat Hermes as an optional outer operator for multi-worker orchestration, not as the inner proposal manager.
- [ ] If Hermes is used, add an adapter outside the core control plane:
  - [ ] Hermes parent session dispatches experiment work.
  - [ ] Stage 2 remains the source of truth for budget, lifecycle, validity, provenance, and decisions.
  - [ ] Hermes workers write through the same `WorkerResult` and `TrialRecord` pathway.
  - [ ] Concurrency is explicitly capped by available compute.
- [ ] Take the useful Hermes idea, not the whole runtime: isolated delegated workers plus explicit task payloads.

Decision: Hermes is optional infrastructure for scaling and delegation. It is not necessary for a KDD AAE submission and should not be required to run the core system.

## 2. Is LangGraph Properly Integrated?

Short answer: partially yes. It is properly scoped, but not yet properly packaged or central enough for a polished Stage 3.

- [x] `LangGraphManager` is correctly scoped to proposal generation only.
- [x] It does not own budget, lifecycle, decision logic, trial records, append store, or worker execution.
- [x] The graph path is simple and auditable: `prepare_context -> generate_proposal -> validate_proposal`.
- [x] It returns the same `ManagerProposal` interface as `BaselineManager` and `PromptManager`.
- [x] Existing smoke tests pass with an injected fake chat model.
- [ ] Add dependencies to `pyproject.toml` instead of relying on `.venv` path injection:
  - [ ] `langgraph`
  - [ ] `langchain-core`
  - [ ] `langchain-ollama`
  - [ ] test dependency for fake chat model support, if separated by the dependency graph
- [ ] Add a normal test runner target, for example `pytest`, instead of keeping smoke tests only under `notebooks/stage2_development_smoke_tests`.
- [ ] Add one real non-dry-run LangGraph campaign smoke test with a small budget and stubbed worker, so the manager is tested through the real control-plane function shape without invoking Ollama.
- [ ] Add structured proposal validation beyond JSON parse success:
  - [ ] objective must mention exactly one bounded change
  - [ ] objective must include allowed editable paths
  - [ ] objective must not instruct the worker to edit frozen files
  - [ ] proposal summary should be normalized for duplicate detection
- [ ] Record `context_text` and raw LLM proposal as artifact refs, or record hashes if full prompt capture is too verbose.

Decision: LangGraph is integrated in the right architectural place, but Stage 3 should make it reproducible, testable through standard tooling, and more strongly validated.

## 3. Advantages To Take From The Three Projects

### From Hugging Face ML Intern

- [ ] Add a queue/event model around campaign execution:
  - [ ] campaign submitted
  - [ ] proposal generated
  - [ ] worker started
  - [ ] approval required
  - [ ] trial completed
  - [ ] metric parsed
  - [ ] decision recorded
  - [ ] campaign completed
- [ ] Add human approval hooks for expensive or destructive steps:
  - [ ] launching real GPU jobs
  - [ ] editing outside configured scope
  - [ ] increasing budget
  - [ ] promoting a result
- [ ] Add context compaction for long campaigns, separate from append-only trial records.
- [ ] Add a tool/router boundary for worker capabilities, even if the first version only wraps the existing `ClawWorker`.
- [ ] Add status notifications as optional adapters, not required dependencies.
- [ ] Add reliability checks for known training-script failure patterns before a real run starts.

### From Multi-Agent Autoresearch

- [ ] Add a promoted-master model:
  - [ ] `experiments/live/master.json`
  - [ ] `experiments/live/master_detail.json`
  - [ ] `experiments/live/dag.json`
  - [ ] refresh command that resets an experiment worker to the current promoted baseline
- [ ] Add isolated experiment worktrees:
  - [ ] one worktree per trial or worker
  - [ ] reserved trial state file
  - [ ] cleanup command with dirty-worktree protection
- [ ] Split durable roles without making them all mandatory runtime agents:
  - [ ] planner: proposes campaign hypotheses
  - [ ] experiment worker: edits and runs only allowed files
  - [ ] reviewer: checks validity and risk
  - [ ] memory keeper: updates durable notes
  - [ ] reporter: exports summaries and paper artifacts
- [ ] Add a `do-not-repeat` memory artifact based on rejected, invalid, or duplicate proposals.
- [ ] Add campaign and experiment markdown templates for human-readable audit trails.
- [ ] Add a local promotion command that only promotes a candidate when the Stage 2 decision module says it beats the current best.
- [ ] Add compute-capacity-aware concurrency limits.

### From Open Deep Research

- [ ] Add breadth/depth controls for literature and hypothesis scouting:
  - [ ] breadth = number of independent hypothesis directions
  - [ ] depth = number of follow-up rounds per direction
- [ ] Add a research-scouting mode that turns papers, prior results, and failed attempts into candidate single-change experiment hypotheses.
- [ ] Deduplicate learnings and hypotheses before they reach the manager.
- [ ] Preserve source URLs and paper references in proposal rationale artifacts.
- [ ] Add progress callbacks for long research/campaign runs.
- [ ] Keep the implementation small and inspectable; do not add a broad multi-agent framework where a typed function and JSONL ledger are enough.

## 4. Stage 2 Problem To Fix Before Stage 3

- [ ] Make Stage 2 real campaigns the documented primary path:
  - [ ] update `docs/stage_2_current_structure.md`
  - [ ] update `docs/architecture.md`
  - [ ] add a real campaign quickstart in `README.md`
- [ ] Stop describing real execution as only legacy Stage 1; describe the legacy harness as the current worker backend behind the Stage 2 control plane.
- [ ] Add package dependencies and remove script-level `.venv` site-package path hacks where possible.
- [ ] Add CI-friendly tests:
  - [ ] unit tests for managers
  - [ ] unit tests for worker result extraction
  - [ ] dry-run campaign tests
  - [ ] LangGraph fake-LLM tests
  - [ ] permission/scope violation tests
  - [ ] metric missing and command failure tests
- [ ] Fix real-run artifact completeness:
  - [ ] capture generated packet path
  - [ ] capture patch diff
  - [ ] capture raw log
  - [ ] capture commit before/after
  - [ ] capture parsed metric payload
  - [ ] capture manager raw output or hash
- [ ] Add a recovery path for pending guards:
  - [ ] list pending campaigns
  - [ ] inspect pending guard
  - [ ] mark failed and append failure record
  - [ ] clear stale guard safely

## 5. KDD AAE Submission Readiness

Short answer: after Stage 3 and the Stage 2 real-path fix, it could become a good KDD AAE workshop submission, but only if the paper is framed as evaluation/governance infrastructure for autonomous experimentation and includes real empirical evidence.

The fit is strong because the workshop explicitly emphasizes:

- agentic AI evaluation
- multi-step planning and tool use
- monitoring under evolving conditions
- standardized metrics and logging protocols
- lifecycle and governance frameworks
- auditability and liability attribution
- production-style monitoring

Required before submission:

- [ ] Run real campaigns, not just dry-runs:
  - [ ] baseline manager
  - [ ] prompt manager
  - [ ] langgraph manager
  - [ ] at least one memory ablation
  - [ ] at least one invalid-scope or failed-command stress test
- [ ] Report governance metrics as first-class results:
  - [ ] editable-scope violation rate
  - [ ] metric parsing failure rate
  - [ ] command failure rate
  - [ ] provenance completeness
  - [ ] repeated bad proposal rate
  - [ ] artifact capture completeness
  - [ ] gain per budget unit
- [ ] Include an ablation table:
  - [ ] no memory
  - [ ] append-only summary
  - [ ] append-only summary with rationale
  - [ ] optional `do-not-repeat` memory
- [ ] Include a failure taxonomy:
  - [ ] invalid edit scope
  - [ ] syntax error
  - [ ] runtime error
  - [ ] metric missing
  - [ ] degraded metric
  - [ ] duplicate hypothesis
  - [ ] timeout
- [ ] Include reproducibility assets:
  - [ ] fixed node spec
  - [ ] fixed campaign configs
  - [ ] raw JSONL ledgers
  - [ ] generated tables
  - [ ] generated figure CSVs
  - [ ] exact run commands
  - [ ] dependency lockfile or installation instructions
- [ ] Clarify claims:
  - [ ] The project is not claiming a new general-purpose coding agent.
  - [ ] The claim is a governed, auditable, benchmark-oriented control plane for autonomous ML experimentation.
  - [ ] The evidence is improved auditability, boundedness, reproducibility, and measurable experiment progress under budget.

Submission judgment:

- [ ] Not ready if Stage 2 remains mostly dry-run and docs still imply the real path bypasses the Stage 2 control plane.
- [ ] Borderline if only one ResNet-trigger benchmark is evaluated.
- [ ] Stronger if Stage 3 adds real ledgers, failure stress tests, memory ablations, and one additional node or task family.
- [ ] Good fit for KDD AAE if the final paper foregrounds evaluation methodology, governance, logging standards, lifecycle control, and empirical audit metrics.

## Stage 3 Implementation Order

- [ ] Fix packaging and dependency declaration.
- [ ] Promote Stage 2 real campaign execution to the documented canonical path.
- [ ] Add standard tests and CI-friendly smoke commands.
- [ ] Add real-run artifact capture and pending-trial recovery.
- [ ] Add promoted-master snapshots and isolated worktrees.
- [ ] Add duplicate and do-not-repeat memory.
- [ ] Add breadth/depth research-scouting mode.
- [ ] Add optional Hermes delegation adapter only after the core Stage 2 path is stable.
- [ ] Run real campaign matrix and export paper-ready tables/figures.
- [ ] Update paper notes around KDD AAE framing and limitations.
