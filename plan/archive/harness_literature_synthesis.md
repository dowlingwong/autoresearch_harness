# Harness Literature Synthesis: Lessons for the KDD AAE Paper

*Generated from the 10 uploaded PDFs + `kdd_aae_refinement_plan_md.md`*
*Date: 2026-05-08*

---

## Executive Summary

The ten documents converge on a single insight that directly validates the autoresearch_harness contribution: **reliable autonomous agents require a deterministic governance layer that wraps the LLM, not just a better LLM**. The paper's core claim — that the governed control plane is the contribution, not the ML optimization — is precisely what the harness engineering literature argues and what the field is calling for. The gap between where the field is and where this paper is positioned is smaller than it looks. This document maps each source to the project and distils into concrete paper-strengthening recommendations.

---

## 1. Source-by-Source Lessons

### 1.1 AIDE: AI-Driven Exploration in the Space of Code (arXiv 2502.13138)
**Weco AI, Feb 2025 — the most closely related academic paper**

AIDE frames ML engineering as code-space optimization: a tree search where each node is a Python script evaluated by a hard-coded metric function. The LLM proposes "atomic" changes (draft / debug / improve), a hard-coded search policy selects which node to expand next, and a summarisation operator keeps prompts compact by distilling tree history.

**What this teaches the project:**
- AIDE is the direct academic predecessor. Both systems: (a) use LLMs to propose single-file code changes, (b) evaluate with a hard-coded metric, (c) maintain a history of prior attempts, and (d) propose "atomic" modifications. The paper **must cite and differentiate from AIDE explicitly**.
- The key structural difference: AIDE's control layer (tree-search policy π, evaluator h, summarizer Σ) is a *single hard-coded component* that owns both the search strategy and the lifecycle. autoresearch_harness **separates** proposal generation (the LLM manager) from lifecycle governance (the control plane). This separation is the architecturally novel claim.
- AIDE has no concept of scope enforcement (editable files), no append-only audit ledger, no failure taxonomy, and no keep/discard decision with rationale. These are all governance features that AIDE deliberately omits because its goal is ML performance, not auditability.
- AIDE's summarization operator (Σ) is the tree-based analog of the project's memory modes. The paper should frame the memory ablation as: "does summary + rationale beat Σ(T)-style distillation on the governance metrics we care about?"

**Paper action:** Add a Related Work paragraph on AIDE with a 3-column comparison table: (1) AIDE, (2) autoresearch_harness, (3) what each is optimizing for (performance vs. governed process). Position the contribution as *complementary*, not superior: "AIDE optimises the ML metric; we govern the experimentation process so that any manager — including AIDE-style tree search — can be audited and reproduced."

---

### 1.2 The Anatomy of an Agent Harness (LangChain blog, Vivek Trivedy, March 2026)

The canonical definition: **Agent = Model + Harness**. Everything that is not the model is the harness: system prompts, tools, sandboxes, orchestration logic, hooks/middleware. The article derives each harness component by working backwards from a desired agent behaviour — filesystem for durable state, bash for general-purpose action, sandboxes for safe execution, memory/search for continual learning, planning tools for decomposition.

**What this teaches the project:**
- The project's control plane IS the harness by this definition. The paper should open with this framing: "In the Agent = Model + Harness formulation (Trivedy 2026), our governed control plane is the harness. Its defining property is that it owns trial lifecycle, scope enforcement, and audit — not the LLM manager."
- The article identifies "hooks/middleware for deterministic execution" as a first-class harness primitive. The project's state machine, pending-trial guard, and scope-validation logic are exactly these hooks. Naming them explicitly in the paper will help reviewers recognise the contribution as harness-level engineering, not prompt engineering.
- "The filesystem is arguably the most foundational harness primitive." The project's append-only JSONL ledger is a filesystem-level audit primitive. The paper should frame the ledger not just as an implementation detail but as the harness's state-persistence mechanism.

**Paper action:** Use "governed harness" as the preferred noun throughout. Add a one-sentence positioning statement in the abstract: "We build and evaluate a governed harness for autonomous ML experimentation, where the LLM manager is a replaceable component and the control plane owns auditability."

---

### 1.3 Harness Engineering — First Thoughts (Thoughtworks / Birgitta Böckeler, Feb 2026)

A response to the OpenAI Codex article. Key insight: increasing trust and reliability in agent outputs required **constraining the solution space** — specific architectural patterns, enforced boundaries, standardised structures. "Harnesses — with custom linters, structural tests, basic context and knowledge documentation, and additional context providers — [will become] the new service templates."

**What this teaches the project:**
- Constraint is a feature, not a limitation. The paper should explicitly frame the editable-path restriction (only `train.py` is in scope) and the budget cap as **reliability mechanisms**, not experimental simplifications. The Thoughtworks framing directly supports this: "for maintainable, AI-generated code at scale that we can trust, something has to give."
- The article notes that "the OpenAI team says: 'Our most difficult challenges now center on designing environments, feedback loops, and control systems.'" This is the autoresearch_harness's exact domain. Cite this as external validation that governance/control is the hard open problem, not model capability.
- The future-state imagined here (harnesses as service templates) is exactly what the paper's node specification system provides: a YAML-defined node spec is a harness template for a class of ML experiments.

**Paper action:** In the Introduction, cite Böckeler and the OpenAI team's finding that "designing environments, feedback loops, and control systems" is the hard problem. Frame the paper as addressing this problem in the ML experimentation context.

---

### 1.4 Harness Engineering for Coding Agent Users (Thoughtworks / Birgitta Böckeler, Apr 2026)

The mature follow-up to the memo. Introduces a clean taxonomy: **guides (feedforward controls)** that steer before the agent acts, and **sensors (feedback controls)** that observe after and enable self-correction. Further divided into **computational** (deterministic, fast: tests, linters, type checkers) and **inferential** (semantic, LLM-based: code review, "LLM as judge") variants. The steering loop: when an issue recurs, improve controls rather than blaming the model.

**What this teaches the project:**
- The project's architecture maps cleanly onto this taxonomy, and the paper should use it:
  - **Computational guides**: node spec (defines valid scope), scope validator (enforce editable paths), budget enforcer (cap trials)
  - **Computational sensors**: metric parser (val_bpb → val_AUC), state machine (enforce transition legality), pending-trial guard (detect crash)
  - **Inferential guides**: manager system prompt + memory injection (steer the LLM's next proposal)
  - **Inferential sensors**: repeated-bad detection (Jaccard similarity flags semantically redundant proposals)
- This taxonomy makes the architecture legible to ML systems reviewers who may not immediately see why a "governed control plane" is architecturally interesting. Name the components using guides/sensors vocabulary.
- "Separately, you get either an agent that keeps repeating the same mistakes (feedback-only) or an agent that encodes rules but never finds out whether they worked (feedforward-only)." The memory ablation is testing exactly whether feedforward context (rationale in memory) combines with feedback (repeated-bad sensor) to reduce repeated mistakes. Frame it this way.

**Paper action:** Add a 2×2 table in Section 3 (System Design) labelling each harness component as Computational/Inferential × Guide/Sensor. This makes the architectural contribution legible and citable.

---

### 1.5 My AI Adoption Journey (Mitchell Hashimoto, Feb 2026)

A practitioner's journey from chatbot → agent → harness engineering. "Step 5: Engineer the Harness — anytime you find an agent makes a mistake, you take the time to engineer a solution such that the agent never makes that mistake again." Two forms: (1) AGENTS.md for prompt-level fixes, (2) actual programmed tools (scripts, test runners) for structural fixes.

**What this teaches the project:**
- Hashimoto describes harness engineering as a reactive process: observe failure, fix permanently. The autoresearch_harness does this systematically: the failure taxonomy (METRIC_MISSING, SCOPE_VIOLATION, PATCH_MALFORMED, NO_OP_PATCH) and the repeated-bad detector are the structured form of this intuition.
- "If you give an agent a way to verify its work, it more often than not fixes its own mistakes." The metric parser and state machine transitions give the manager an execution signal. The memory modes give it a recall signal. The paper can cite Hashimoto as practitioner validation for the design choice of keeping verification deterministic (control plane) while keeping proposal generation inferential (manager).
- The three rules Hashimoto derived from first principles are already encoded in the project: (1) "Break sessions into clear, actionable tasks" = one-at-a-time trial lifecycle; (2) "Split planning vs. execution" = manager proposes, control plane executes; (3) "Give the agent a way to verify its work" = metric parser + keep/discard decision.

**Paper action:** In the motivation section, use Hashimoto as a practitioner data point alongside the academic literature. "Practitioners independently discovered the need for deterministic verification loops (Hashimoto 2026); our work formalises this as a governed control plane."

---

### 1.6 Harness Engineering: Leveraging Codex in an Agent-First World (OpenAI, Feb 2026)

The account of building a 1M-LOC product with zero manually-written code in 5 months (3.5 PRs/engineer/day). Key engineering decisions: AGENTS.md as a 100-line table of contents (not a monolithic instruction manual); a structured `docs/` directory as the system of record; per-worktree observability (logs, metrics, traces queryable by the agent); when the agent fails, treat it as a signal — "what capability is missing, and how do we make it legible and enforceable for the agent?"

**What this teaches the project:**
- The "give the agent a map, not a manual" principle directly applies to the memory design. The memory ablation tests three modes: `none`, `append_only_summary`, `append_only_summary_with_rationale`. The OpenAI finding predicts that `none` and `summary` will both underperform `summary_with_rationale` because the agent needs structured signals, not a monolithic history dump. Use this as the a priori hypothesis for the memory ablation.
- "When the agent struggles, we treat it as a signal: identify what is missing — tools, guardrails, documentation — and feed it back into the repository." The memory rationale mode is doing exactly this: when a trial is discarded, the rationale is appended to memory so the manager sees why it failed. This is the formal version of the OpenAI team's ad-hoc process.
- The per-worktree observability stack (ephemeral logs, metrics, traces per agent run) is the analog of the project's artifact capture: each trial gets an artifact directory with patch.diff and run.log. The paper should note this parallel.

**Paper action:** In the memory ablation section, explicitly state the hypothesis informed by the OpenAI finding: "We predict that rationale-augmented memory will reduce the repeated-bad rate because it gives the manager a structured failure signal rather than raw history." If the ablation results confirm this, cite OpenAI as prior supporting evidence. If they don't, that is an interesting negative result worth discussing.

---

### 1.7 Effective Harnesses for Long-Running Agents (Anthropic, Nov 2025)

Addresses the multi-context-window agent problem. Solution: initializer agent (sets up environment, creates `claude-progress.txt` + feature list as JSON) + coding agent (incremental progress, git commits, reads progress file). Four failure modes and their harness fixes:
1. Agent declares victory too early → feature list with all items initially marked failing
2. Agent leaves broken state → git commits + progress notes
3. Agent marks features done prematurely → verifiable JSON feature spec
4. Agent doesn't know how to run the app → `init.sh` script

**What this teaches the project:**
- The four failure modes have direct ML experimentation analogs, and the paper should map them explicitly:
  1. "Declares victory too early" → budget enforcement prevents stopping after one good trial
  2. "Leaves broken state" → pending-trial guard + git commit tracking ensures reproducible state
  3. "Marks features done prematurely" → validity checks (FAILED_INVALID) catch trials that don't produce parseable metrics
  4. "Doesn't know how to run the app" → node spec + editable-scope enforcement defines the execution contract
- The feature list JSON (all items initially failing, only the `passes` field editable) is a perfect analog of the node spec + budget spec: the contract is fixed, only the metric can change.
- "We found that the best way to elicit clean-state behaviour was to ask the model to commit its progress to git with descriptive commit messages." The project already does this (git_commit_before/after in TrialRecord). The paper should explicitly note this as shared best practice.

**Paper action:** Add a "Failure Mode Analysis" subsection or table that maps the Anthropic-identified failure modes to the project's corresponding harness controls. This grounds the design choices in documented prior art.

---

### 1.8 Harness Design for Long-Running Application Development (Anthropic, Mar 2026)

Extends the long-running agent work with a GAN-inspired multi-agent architecture: planner → generator → evaluator. Key findings:
- Self-evaluation is unreliable ("agents respond by confidently praising the work even when quality is obviously mediocre")
- Separating generator from evaluator is the fix: "tuning a standalone evaluator to be skeptical turns out to be far more tractable than making a generator critical of its own work"
- Sprint contracts: generator and evaluator negotiate what "done" means before coding starts
- "Every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress-testing"

**What this teaches the project:**
- "Self-evaluation is unreliable" directly validates the architecture choice of keeping keep/discard decisions in the control plane, not delegated back to the manager. If the manager owned its own keep/discard, it would be a lenient self-evaluator. The Anthropic data is empirical evidence for this design decision.
- The sprint contract (agree on what "done" looks like before writing code) maps to the node spec + budget spec: the success criterion (val_AUC direction, metric name) is defined before the campaign runs, not negotiated by the manager.
- "The most difficult part was tuning the evaluator to be skeptical." This is the same challenge as calibrating the keep/discard threshold. The paper should discuss how the control plane's evaluation criteria (delta_vs_best > 0 to keep) encode skepticism deterministically rather than relying on the model's self-assessment.
- The iterative harness simplification process ("remove one component at a time and review impact") is the methodology for the manager comparison experiment (Priority 13). Frame the manager comparison as this kind of harness ablation.

**Paper action:** Use the Anthropic finding on self-evaluation unreliability as the key motivation for why the keep/discard decision must live in the control plane. "We separate proposal generation from evaluation; this is consistent with Rajasekaran et al. (2026), who found that LLM self-evaluation is systematically lenient."

---

### 1.9 Better Harness: A Recipe for Harness Hill-Climbing with Evals (LangChain, Apr 2026)

Evals are training data for harness engineering. The harness hill-climbing loop: source evals → split optimization/holdout → run baseline → diagnose from traces → propose targeted harness change → validate on holdout → human review. Key: holdout sets prevent the harness from overfitting to the optimization eval set. Tags evals by behavioural category to enable targeted experiments.

**What this teaches the project (most directly applicable paper to immediate gaps):**
- The autoresearch_harness is a harness hill-climbing system: evals = node metrics, optimization set = trial budget, harness changes = hyperparameter proposals. But it is **missing the holdout set**. Every experiment runs on the same ResNet-trigger node with the same training/validation split. This is the biggest methodological gap the paper needs to acknowledge.
- "Evals encode the behavior we want our agent to exhibit in production. They're the 'training data' for harness engineering." The paper should explicitly describe what the "evals" are in this context: the governance metrics (acceptance rate, repeated-bad rate, provenance completeness) serve as the harness-level evals, while the ML metric (val_AUC) serves as the task-level eval.
- The human review step in Better-Harness corresponds to the control plane's keep/discard decision — but in the project, this is automated (deterministic). The paper should note this as a design choice: "We replace human review with a formal keep/discard rule (delta_vs_best > 0) to enable fully automated campaigns. Human review remains available but is not required for the governance record."
- "Even before running an agent over evals, often a team dogfooding our agent will report errors directly in Slack with a Trace link." The paper's failure taxonomy (Section 5 of the refinement plan) is the structured version of this: every failure is categorised and logged to the ledger.

**Paper action (gap to fix):** Add a Limitations section paragraph: "All experiments use a single node (ResNet-trigger) without a separate holdout node. We cannot rule out that reported improvements overfit to this evaluation domain. Generalisability across node types is future work." Then add a Future Work note: "Applying the harness hill-climbing methodology of Trivedy (2026) across multiple evaluation nodes with holdout splits would strengthen the governance claims."

---

### 1.10 Mind2Web: Towards a Generalist Agent for the Web (arXiv 2306.06070, NeurIPS 2023)

Constructs the first dataset for generalist web agents: 2,000+ tasks across 137 websites, 31 domains. Proposes MindAct: a two-stage model using a small LM for HTML filtering followed by an LLM for action selection. Emphasises out-of-distribution generalisation as the key evaluation challenge.

**What this teaches the project:**
- Mind2Web's primary contribution is a benchmark, not a harness. The paper's evaluation methodology section should position the ResNet-trigger node as a focused, reproducible evaluation domain rather than apologising for narrow scope. Mind2Web's scale (137 websites) required crowdsourcing; reproducibility was the casualty. The project makes the opposite trade-off: one real node, full reproducibility.
- The "out-of-distribution generalisation" challenge Mind2Web identifies is the multi-node generalisation gap for autoresearch_harness. This supports the Limitations framing.
- Mind2Web's crowdsourced action sequences as ground truth are analogous to the project's TrialRecord ledger: both create reproducible traces of agent behaviour for post-hoc analysis.

**Paper action:** Cite Mind2Web briefly in Related Work to contrast evaluation philosophy: "Unlike benchmark-driven evaluation across many domains (Deng et al. 2023), we prioritise a single real task node with complete provenance, enabling auditability that broad benchmarks sacrifice."

---

## 2. Cross-Cutting Themes

### Theme A: The Harness IS the Contribution
Every source agrees that the governance layer (harness) is now the primary engineering challenge, not model capability. The paper's existing positioning — "the governed control plane is the contribution, not the ML gain" — is precisely correct and increasingly well-supported in the literature. Reviewers who might dismiss this as "just engineering" will be behind the curve; the field has moved to recognising harness design as a research problem.

### Theme B: Computational Determinism as a Trust Mechanism
Böckeler (Apr 2026) formalises what Anthropic and OpenAI discovered empirically: computational (deterministic) controls are the reliable layer; inferential controls add semantic richness but introduce variance. The autoresearch_harness's architecture — computational state machine, metric parser, scope validator wrapping an inferential LLM manager — is the right decomposition. Make this explicit.

### Theme C: Failure Taxonomy as a First-Class Artifact
OpenAI, Anthropic, Hashimoto, and Böckeler all treat agent failures as signals to engineer against, not noise to average away. The project's failure taxonomy (METRIC_MISSING, SCOPE_VIOLATION, PATCH_MALFORMED, NO_OP_PATCH, RUNNER_TIMEOUT) is a research contribution in itself: it is the first formal taxonomy of autonomous ML experimentation failures grounded in a real campaign. The paper should present it as such.

### Theme D: Separation of Roles (Manager vs. Control Plane)
Anthropic's finding that "self-evaluation is unreliable" and the need to "tune a standalone evaluator to be skeptical" is the empirical justification for keeping the keep/discard decision in the control plane. AIDE does not make this separation — its evaluator h is hard-coded alongside the search policy. The separation is architecturally novel.

### Theme E: Memory as a Structured Failure Signal
OpenAI ("give Codex a map, not a manual"), Böckeler (feedforward guides), and the Better-Harness recipe all converge on structured, targeted context over raw history. The memory ablation tests this hypothesis. The paper should enter the ablation with an explicit prediction (rationale > summary > none on repeated-bad rate) grounded in these sources.

---

## 3. Gaps the Paper Must Address

| Gap | Source Evidence | Recommended Fix |
|-----|----------------|-----------------|
| **No holdout evaluation node** | Better-Harness: holdout sets prevent overfitting | Add Limitations paragraph; cite as future work |
| **Missing AIDE comparison** | AIDE is the closest related work; reviewers will ask | Add Related Work section with 3-column comparison table |
| **Self-evaluation rationale unstated** | Anthropic Mar 2026: self-eval is unreliable | Add one paragraph in Design Section explaining why control plane owns keep/discard |
| **Memory ablation hypothesis not pre-stated** | OpenAI, Better-Harness | State hypothesis explicitly before presenting results |
| **Failure taxonomy not positioned as contribution** | All sources treat failure classification as research | Add taxonomy table as Table 2; present as a contribution |
| **No bootstrap baseline** | AIDE, refinement plan Sec 5 | Add random-proposal baseline; report as governance sanity check |
| **Computational vs inferential split unlabelled** | Böckeler Apr 2026 | Add 2×2 table in System Design section |

---

## 4. Concrete Recommendations for Strengthening the Paper

### R1 — Rewrite the Abstract with Harness-First Framing
Current framing: "we present a governed control plane for autonomous ML experimentation."
Stronger framing: "We build and evaluate a governed harness for autonomous ML experimentation. In the Agent = Model + Harness decomposition, our harness owns trial lifecycle, scope enforcement, and audit. The LLM manager is a replaceable, interchangeable component. We demonstrate the harness on a real ResNet training node and show that governance metrics — acceptance rate, failure taxonomy, provenance completeness — are as informative as the ML metric itself."

### R2 — Add an Explicit AIDE Comparison Table
In Related Work (or a new Section 2.3), add a table:

| Dimension | AIDE (Jiang et al., 2025) | autoresearch_harness |
|-----------|--------------------------|---------------------|
| LLM role | Proposes code + selects next node | Proposes change only |
| Control plane | Hard-coded tree search + evaluator | Governed lifecycle state machine |
| Keep/discard owner | Hard-coded metric h(s) | Control plane (deterministic rule) |
| Scope enforcement | None | Editable-path whitelist |
| Audit ledger | None | Append-only JSONL with provenance |
| Failure taxonomy | None | 5-category taxonomy |
| Memory/context | Summarization operator Σ(T) | 3-mode ablation |
| Reproducibility claim | Performance on Kaggle | Full provenance from proposal to commit |

Caption: "AIDE optimises the ML metric; autoresearch_harness governs the experimentation process. These are complementary goals."

### R3 — Use the 4-Failure-Mode Table as Design Motivation
Adapt the Anthropic table to the ML experimentation context in Section 3 (Design):

| Failure Mode | Agent System | ML Experimentation Analog | Harness Fix |
|---|---|---|---|
| Declares victory too early | Coding agent | Manager stops after one good trial | Budget enforcer + fixed trial count |
| Leaves broken state | Coding agent | Failed patch leaves repo dirty | Pending-trial guard + git commit check |
| Marks feature done prematurely | Coding agent | Manager claims improvement without valid metric | FAILED_INVALID state + metric parser |
| Doesn't know how to run | Coding agent | Manager proposes out-of-scope edit | Node spec + editable-path whitelist |

### R4 — Frame the Memory Ablation as a Hypothesis Test
Before presenting results, state: "We hypothesise, consistent with OpenAI (2026) and Böckeler (2026), that structured failure context (rationale) reduces the repeated-bad rate more than raw trial summaries, because it gives the manager a targeted signal rather than history noise. We test this across three memory modes."

### R5 — Add the Guides/Sensors 2×2 Table to the Architecture Section
| | Guide (feedforward) | Sensor (feedback) |
|---|---|---|
| **Computational** | Node spec, scope validator, budget cap, editable-path whitelist | Metric parser, state machine validator, pending-trial guard |
| **Inferential** | Manager system prompt + memory injection | Repeated-bad detector (Jaccard + parameter direction) |

This table makes the contribution legible to systems-oriented reviewers and directly cites Böckeler's taxonomy.

### R6 — Present the Failure Taxonomy as Table 2
Position the failure taxonomy as a contribution, not a bookkeeping detail:
- **METRIC_MISSING**: worker ran but produced no parseable metric
- **SCOPE_VIOLATION**: proposed edit touched files outside the whitelist
- **PATCH_MALFORMED**: diff was syntactically invalid or did not apply cleanly
- **NO_OP_PATCH**: patch applied but changed nothing (byte-identical output)
- **RUNNER_TIMEOUT**: execution exceeded wall-clock budget

Note that this is the first formal taxonomy of autonomous ML experimentation failure modes derived from real campaigns.

### R7 — Acknowledge and Quantify the Harness Development Cost
OpenAI reports 5 months of harness engineering for 1M LOC. Anthropic reports "several rounds of tuning" to get the evaluator to grade skeptically. The paper should briefly acknowledge the harness development cost (engineering effort to build the control plane) and note that this is a one-time investment amortised across campaigns. This pre-empts the reviewer objection "is the harness harder to build than just doing the experiment manually?"

### R8 — Add a "Harness as Reusable Infrastructure" Point
The node spec YAML pattern means that adding a new experimental node requires only a spec file, not code changes. This is the "harness as service template" vision from Böckeler. Add one sentence: "The node specification system generalises the harness to new ML experiments without code changes; each YAML spec is a harness template for a class of experiments."

---

## 5. Priority Order for Implementation

Given the KDD AAE timeline, implement in this order:

1. **R2 (AIDE comparison table)** — highest reviewer risk; do this first regardless of experiment status
2. **R5 (Guides/Sensors table)** — pure writing; clarifies architecture with no additional experiments
3. **R3 (Failure mode motivation table)** — pure writing; strengthens design section
4. **R4 (Memory ablation hypothesis)** — one paragraph; run the real ablation before writing results
5. **R6 (Failure taxonomy as Table 2)** — pure writing from existing code
6. **R1 (Abstract rewrite)** — do last once all results are in
7. **R7, R8** — optional polish

---

## 6. What the PDFs Do NOT Teach (Intentional Non-Lessons)

- **Do not mimic AIDE's tree search**: AIDE's multi-trial tree structure requires reusing earlier nodes as parents. The project's sequential budget is simpler and more auditable. Do not add tree search to compete with AIDE; instead, position the linear budget as a reproducibility feature.
- **Do not add a multi-agent planner**: Anthropic's three-agent system (planner/generator/evaluator) is designed for open-ended application development. The project's node spec already encodes the planner's output (the constraint set is pre-specified, not generated). Don't add complexity that the governance framework already handles declaratively.
- **Do not hold out on the negative result**: If the memory ablation shows no difference between `none` and `append_only_summary_with_rationale`, report it. Better-Harness explicitly validates negative results as informative: "agents are famous cheaters" — the null result would mean the manager is insensitive to the memory signal, which is a finding.
- **Do not add automatic harness updates**: Better-Harness's autonomous harness improvement loop is a research direction, not something to add to the paper. The current paper's governance metrics provide the signal for manual harness improvement; autonomous meta-improvement is future work.

---

*Synthesis covers: AIDE (arXiv 2502.13138), Mind2Web (arXiv 2306.06070), "The Anatomy of an Agent Harness" (LangChain/Trivedy), "Harness Engineering — First Thoughts" (Thoughtworks/Böckeler), "Harness Engineering for Coding Agent Users" (Thoughtworks/Böckeler), "My AI Adoption Journey" (Hashimoto), "Harness Engineering: Leveraging Codex in an Agent-First World" (OpenAI/Lopopolo), "Effective Harnesses for Long-Running Agents" (Anthropic/Young), "Harness Design for Long-Running Application Development" (Anthropic/Rajasekaran), "Better Harness: A Recipe for Harness Hill-Climbing with Evals" (LangChain/Trivedy).*
