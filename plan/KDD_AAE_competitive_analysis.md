# KDD AAE competitive analysis — autoresearch_harness vs. SHARP

> Generated 2026-05-10. Based on review of `plan/KDD_AAE_refinement_plan_v2.md`, `plan/KDD_AAE_execution_chunks.md`, and arXiv:2604.18752v1 (SHARP).

---

## The SHARP paper, summarized

SHARP (Scientific Human-Agent Reproduction Pipeline) is a human-agent collaboration framework for reproducing scientific data analyses. Its core framing: reproduction is a *translation task* (paper → code), not extrapolation, which makes it uniquely suited for agents. It decomposes the task into discrete steps, uses specialized subagents (Paper Analyst, Code, Test, Statistician, Critic), pauses at human-defined checkpoints, and is demonstrated by reproducing ParticleNet-Lite jet classification within 0.1% of published metrics across 3 independent runs. A secondary contribution is `claude-parser`, a novel tool for characterizing human-agent conversation by message type (essential/optional/meta) and complexity (hard/medium/easy).

Key facts:
- Built on Claude Code v2.1.92 with claude-opus-4-6
- Deployed on NERSC Perlmutter HPC with 1 NVIDIA A100 GPU
- 3 independent runs, each ~1 working day
- External validation by independent human expert (PyTorch definition + weights only)
- No governance layer, no lifecycle state machine, no audit ledger, no failure taxonomy

---

## Is autoresearch_harness "more shallow" than SHARP?

No and yes — but on different dimensions.

### Where autoresearch_harness is deeper

SHARP has no governance layer. There is no lifecycle state machine, no scope enforcement, no append-only audit ledger, no failure taxonomy, no keep/discard authority separated from the agent, no memory ablation. The control plane is the core contribution and it is genuinely more sophisticated as a governance mechanism than anything SHARP built. SHARP's agent can do whatever it wants within the sandbox; the autoresearch_harness cannot touch frozen files, cannot self-approve decisions, and creates an immutable record of every trial including failures.

### Where SHARP is deeper (right now)

SHARP has 3 complete independent runs, external validation, a real HPC environment, and a conversation characterization framework with real data. autoresearch_harness has 1 baseline + 1 agent trial (Level 1). Conceptual depth does not compensate for evidence thinness at review time.

### The main gap is not design or technical realization — it is real runs and results

The architecture is correct. The framing is correct. The plan already specifies every experiment, in order, with acceptance criteria. There is no architectural rethink needed. The only thing standing between current Level 1 and a competitive submission is executing the campaigns and collecting real numbers to fill the tables already planned.

---

## Architecture assessment

### What is strong

- `Agent = Model + Harness` decomposition is the right frame and increasingly well-supported in the harness engineering literature (Trivedy, Böckeler, OpenAI, Anthropic)
- Append-only JSONL ledger is a real scientific contribution — most AutoML systems have no audit trail
- Failure taxonomy (`syntax_error | runtime_error | metric_missing | invalid_edit_scope | degraded_metric`) is the first formal one of its kind for autonomous ML experimentation
- Manager/control-plane separation (manager proposes, control plane decides) addresses the Anthropic self-evaluation leniency finding structurally
- Three-mode memory ablation is a proper experimental design with a pre-stated hypothesis

### Risks for reviewers

| Risk | Status | Fix |
|------|--------|-----|
| Single benchmark node | Same as SHARP; acceptable if framed as controlled case study | Acknowledge in Limitations, cite Better-Harness |
| Memory ablation unrun | Critical gap | Run 5 trials/mode × 3 modes before submission |
| `no_op_patch` guard missing | Hole in failure taxonomy | Implement Chunk 1.3 |
| LangGraph not in pyproject.toml | Reproducibility red flag | Fix Chunk 1.1 |
| One-trial demo | Evidence too thin | Run 5-trial main campaign (Chunk 2.2) |

---

## What to learn from SHARP

| SHARP pattern | Your equivalent | Action |
|---------------|----------------|--------|
| 3 independent runs, median + spread | 3 memory ablation modes, each from same node state | Run ablation as designed |
| External validation by independent expert | Control plane decisions are deterministic, structurally independent of manager | Frame this explicitly in the paper |
| `claude-parser` conversation characterization as novel evaluation framework | Failure taxonomy as novel evaluation framework | Position failure taxonomy as a contribution, say so loudly |
| "Reproduction is a translation task, not extrapolation" — one-sentence framing | "Governed harness separates proposal from decision authority" | Sharpen to a single punchy sentence and use it everywhere |
| Cites related agentic science work heavily | AIDE comparison table already planned | Add SHARP itself to Related Work as complementary framing |

SHARP is now directly citable as related work. Frame the relationship: SHARP is human-in-the-loop + faithful reproduction; autoresearch_harness is autonomous + governed. Complementary, not competing.

---

## KDD AAE acceptance estimates

| Scenario | Estimated chance | What it requires |
|----------|-----------------|-----------------|
| Current (Level 1) | ~25% | — |
| Level 2 | ~55% | 5-trial campaign + real memory ablation + stress trial + Tables 1–4 |
| Level 3 | ~70% | 10 trials/mode + clean repeated-bad pattern + manager comparison |
| SHARP (estimated) | ~60% | Already done |

### Why autoresearch_harness can beat SHARP at this specific workshop

KDD AAE is specifically about *evaluating agentic AI behavior* — not about building agents that do cool things. The governance metrics (invalid-action rate, repeated-bad rate, provenance completeness, failure taxonomy) are evaluation methodology contributions. SHARP is an impressive system paper about human-agent collaboration, but it is not fundamentally about evaluation methodology. autoresearch_harness is — if the empirical evidence exists to support the claims.

This is a direct alignment advantage. Level 2 experiments executed cleanly put this paper in a stronger position for AAE than SHARP, because they answer the question the workshop is actually asking.

---

## Priority order (unchanged from refinement plan)

1. Fix packaging — `langgraph`, `langchain-core`, `langchain-ollama` in `pyproject.toml`
2. Implement `reset_node_state.py` — required for valid ablation
3. Add `no_op_patch` guard — closes failure taxonomy gap
4. Run 1-trial smoke ablation across all 3 memory modes — verify runner
5. Run 5-trial main campaign with `prompt_manager`
6. Run forced invalid-scope stress trial
7. Run memory ablation (5 trials/mode × 3 modes)
8. Export all KDD tables and figure CSVs
9. Write paper — governance first, AUC last
10. Add AIDE + SHARP comparison to Related Work

---

## The core filter (from refinement plan)

For every experiment, paragraph, table, and figure, ask:

> Does this help a reviewer evaluate whether the autonomous agent loop is **bounded, auditable, failure-aware, reproducible, and behaviorally improved by governance memory**?

If yes: keep it. If it only says "AUC improved slightly": move to secondary evidence.
