# 2. Background and Related Work

Autonomous experimentation sits at the intersection of coding agents, AutoML, experiment tracking, and harness engineering. The key distinction in this paper is that we evaluate the harness that governs the agent loop, not only the model that proposes code edits. In Trivedy's Agent = Model + Harness decomposition, the control plane, ledger, scope validator, memory interface, metric parser, and state machine are the harness components that make an autonomous experiment bounded and auditable.

## Autonomous Code-Space Exploration

AIDE (Jiang et al. 2025) is the closest related academic system. Both AIDE and `autoresearch_harness` use LLMs to propose atomic code changes that are evaluated by a hard-coded metric. The structural difference is where lifecycle authority lives: AIDE couples search policy, evaluator, and summarisation inside one optimisation loop, while this work separates proposal generation from governance.

**Table 5: Comparison with AIDE.**

| Dimension | AIDE | `autoresearch_harness` |
|---|---|---|
| LLM role | Proposes code and selects the next search node | Proposes a bounded experimental change only |
| Control plane | Hard-coded tree search plus evaluator | Governed lifecycle state machine |
| Keep/discard owner | Hard-coded metric function `h(s)` inside the search loop | Control plane with deterministic, auditable decision rule |
| Scope enforcement | No explicit editable-path whitelist | NodeSpec editable-path whitelist and scope validator |
| Audit ledger | No append-only trial ledger | Append-only JSONL ledger with provenance IDs |
| Failure taxonomy | Not a first-class artifact | Six-category experimentation failure taxonomy |
| Memory/context | Tree summarisation operator `Sigma(T)` | Three-mode memory ablation: none, summary, summary with rationale |
| Reproducibility claim | Metric performance across code-space search | Full provenance from proposal to patch, run log, metric, and decision |

AIDE optimises the ML metric; we govern the experimentation process. These are complementary goals. An AIDE-style tree-search manager could be plugged into this harness, but it still should not directly commit trial state.

## Reproducibility and Auditing Pipelines

SHARP (Siegel et al. 2025) is the most relevant system on human-agent collaboration for scientific reproducibility. SHARP uses an agent to reproduce prior ML experiments by re-executing documented procedures, compare outputs against original claims, and surface discrepancies. It is stronger on external validation and full pipeline reproducibility for published work.

This work is complementary but differently positioned. SHARP reproduces existing experiments; `autoresearch_harness` governs autonomous generation of new ones. SHARP's contribution is a reproducibility-checking pipeline; this paper's contribution is a governance and evaluation protocol for the agent loop that runs while experiments are being created. SHARP does not define trial lifecycle states, append-only audit ledgers, scope enforcement, failure taxonomies, or memory ablation protocols. The main gap relative to SHARP is not concept design but clean empirical evidence under a governed protocol.

| Dimension | SHARP | `autoresearch_harness` |
|---|---|---|
| Primary goal | Reproduce and verify prior published experiments | Govern and audit autonomous generation of new experiments |
| Control plane | Agent executes a documented procedure | Governed state machine owns trial lifecycle |
| Decision authority | Human reviewer inspects agent output | Control plane decides keep/discard/failed-invalid deterministically |
| Audit trail | Execution log of reproduction attempt | Append-only JSONL ledger with provenance IDs per trial |
| Failure handling | Discrepancy reported to human | First-class `failed_invalid` records with failure categories |
| Scope enforcement | Procedure document bounds the agent | NodeSpec editable-path whitelist enforced before patch is applied |

Other open-source agent systems also inform the design. `ml-intern` motivates event traces, local-model support, headless operation, and optional interactive handoff. `multiautoresearch` and deep-research-style systems motivate multi-track project organisation and accumulated learnings. Hermes-style systems motivate durable routines and trajectory compression. These systems are useful implementation references, but they do not define an evaluation protocol for governed autonomous experimentation.

Mind2Web (Deng et al. 2023) represents a different evaluation philosophy: broad benchmark-driven evaluation across many web domains. Unlike benchmark-driven evaluation across many domains, we prioritise a single real task node with complete provenance. This trades breadth for inspectability: every proposal, patch, run log, parsed metric, and keep/discard decision can be audited.

## Harness Engineering

The harness engineering literature independently validates this framing. Designing the environments, feedback loops, and control systems is now recognized as the primary engineering challenge for agentic AI. Trivedy (2026) frames the harness as everything around the model: tools, sandboxes, memory, middleware, and execution hooks. Böckeler (2026) argues that trust in agent output requires constraining the solution space with guides and sensors. OpenAI (2026) describes production agent work as an observability and feedback-loop problem: give the agent a map, not a manual, and turn failures into enforceable harness improvements. Anthropic (Young 2025; Rajasekaran 2026) similarly emphasises progress tracking, verifier separation, and skeptical evaluation for long-running agents.

This paper instantiates those ideas in autonomous ML experimentation. The node specification is a computational guide; the metric parser, pending-trial guard, and state machine are computational sensors; the manager prompt and memory context are inferential guides; the repeated-bad detector is an inferential sensor. The key design choice is that inferential components may propose and summarise, but deterministic components own validity and lifecycle transitions.

## Experiment Tracking and AutoML

Experiment trackers such as MLflow and Weights & Biases record outcomes, metrics, and artifacts after a run exists. They do not govern the agent loop, enforce bounded execution before a patch is applied, or decide whether invalid trials should become first-class audit objects.

AutoML systems, including hyperparameter optimisation and neural architecture search, optimise within a predefined search space. They are not designed to audit an agent's self-directed exploration, classify invalid edit attempts, or explain why a proposed code change was kept, discarded, or failed invalid.

None of the above provides the full governed control plane: scope enforcement, append-only audit ledger, failure taxonomy, and memory ablation for evaluating autonomous experimentation.
