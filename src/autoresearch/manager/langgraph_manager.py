"""LangGraph-backed manager — scoped strictly to the proposal-generation layer.

Architecture contract:
  - This module may only touch: ManagerStatus, MemoryContext, NodeSpec, ManagerProposal.
  - It must never access: budget, lifecycle state machine, TrialRecord, TrialAppendStore,
    WorkerResult, keep/discard decisions, or anything in control_plane or memory.append_store.

Graph: prepare_context -> generate_proposal -> validate_proposal
  - prepare_context: formats status + memory + node spec into a bounded prompt
  - generate_proposal: calls the LLM (real or injected mock)
  - validate_proposal: parses LLM JSON output; falls back to heuristic on failure
  => returns ManagerProposal through the same interface as BaselineManager / PromptManager.
"""
from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any, TypedDict

from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.memory.summarizer import MemoryContext
from autoresearch.nodes.spec import NodeSpec


class _PlanState(TypedDict):
    """Internal graph state. Never returned outside this module."""
    status: Any          # ManagerStatus
    memory_context: Any  # MemoryContext
    node_spec: Any       # NodeSpec
    context_text: str    # formatted prompt built by prepare_context
    raw_proposal: str    # raw LLM text output
    proposal: Any        # ManagerProposal | None, filled by validate_proposal


# ---------------------------------------------------------------------------
# Graph node functions — pure functions, no side effects on external state
# ---------------------------------------------------------------------------

def _prepare_context(state: _PlanState) -> _PlanState:
    status: ManagerStatus = state["status"]
    memory_context: MemoryContext = state["memory_context"]
    node_spec: NodeSpec = state["node_spec"]

    lines = [
        "You are a research manager proposing one bounded hyperparameter change.",
        "",
        f"Node: {node_spec.name}",
        f"Description: {node_spec.description}",
        f"Metric to optimize: {node_spec.metric_name} (direction: {node_spec.metric_direction})",
        f"Files you may instruct the worker to edit: {', '.join(node_spec.editable_paths)}",
        f"Budget index (trial number): {status.budget_index}",
        f"Current best {node_spec.metric_name}: {status.current_best_metric}",
        "",
    ]

    if memory_context.context_text.strip():
        lines += ["Prior trial memory:", memory_context.context_text, ""]
    else:
        lines.append("No prior trial memory available.")
        lines.append("")

    lines += [
        "Propose exactly one bounded experiment. Respond with ONLY a JSON object "
        "(no markdown fences, no extra text) with these three string fields:",
        '  "summary"   — short slug describing the change (e.g. "reduce-lr-5e-4")',
        '  "rationale" — one sentence on why this might improve the metric',
        '  "objective" — complete, self-contained instruction for the worker',
        "",
        "Constraints the objective must enforce:",
        f"  - edit only: {', '.join(node_spec.editable_paths)}",
        "  - make exactly one change",
        "  - do not run the experiment yourself; the manager will run it after your edit",
        "",
        'Example: {"summary": "reduce-lr-5e-4", '
        '"rationale": "smaller lr often improves convergence", '
        '"objective": "In train.py, change learning_rate from 1e-3 to 5e-4. '
        'Edit only train.py. Make no other changes."}',
    ]

    return {**state, "context_text": "\n".join(lines)}


def _validate_proposal(state: _PlanState) -> _PlanState:
    raw: str = state["raw_proposal"]
    status: ManagerStatus = state["status"]
    node_spec: NodeSpec = state["node_spec"]

    summary = f"langgraph-proposal-{status.budget_index:03d}"
    rationale = ""
    objective = (
        f"Propose exactly one bounded change to {', '.join(node_spec.editable_paths)} "
        f"to improve {node_spec.metric_name}. "
        "Edit only the listed files. Do not run the experiment."
    )

    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE).strip()
        # Handle partial JSON — find first { ... } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        parsed = json.loads(cleaned)
        if parsed.get("summary", "").strip():
            summary = str(parsed["summary"]).strip()
        if parsed.get("rationale", "").strip():
            rationale = str(parsed["rationale"]).strip()
        if parsed.get("objective", "").strip():
            objective = str(parsed["objective"]).strip()
    except Exception:
        rationale = f"parse-failed; raw={raw[:120]!r}"

    proposal = ManagerProposal(
        manager_mode=LangGraphManager.mode,
        proposal_summary=summary,
        proposal_rationale=rationale,
        target_files=node_spec.editable_paths,
        objective=objective,
        extra={
            "context_sha256": sha256(state.get("context_text", "").encode("utf-8")).hexdigest(),
            "raw_proposal_sha256": sha256(raw.encode("utf-8")).hexdigest(),
            "raw_proposal_chars": len(raw),
        },
    )
    return {**state, "proposal": proposal}


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------

class LangGraphManager:
    """Manager that uses a LangGraph planning graph to generate proposals.

    The graph is compiled lazily on first call. The LLM is injected at
    construction time so tests can pass a FakeListChatModel without Ollama.

    To use with a real Ollama model, install langchain-ollama and leave
    llm=None — the default Ollama backend will be created on first call.
    """

    mode = "langgraph_manager"

    def __init__(
        self,
        llm: Any = None,
        model: str = "qwen2.5-coder:7b",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
    ) -> None:
        self._injected_llm = llm
        self._model = model
        self._host = host
        self._temperature = temperature
        self._graph: Any = None

    @classmethod
    def from_config(cls, config_path: str) -> "LangGraphManager":
        cfg = json.loads(__import__("pathlib").Path(config_path).read_text())
        return cls(
            model=cfg.get("model", "qwen2.5-coder:7b"),
            host=cfg.get("host", "http://localhost:11434"),
            temperature=cfg.get("temperature", 0.2),
        )

    # ------------------------------------------------------------------
    # Public interface — same signature as BaselineManager / PromptManager
    # ------------------------------------------------------------------

    def propose_next_trial(
        self,
        status: ManagerStatus,
        memory_context: MemoryContext,
        node_spec: NodeSpec,
    ) -> ManagerProposal:
        graph = self._get_graph()
        result = graph.invoke(
            {
                "status": status,
                "memory_context": memory_context,
                "node_spec": node_spec,
                "context_text": "",
                "raw_proposal": "",
                "proposal": None,
            }
        )
        return result["proposal"]

    # ------------------------------------------------------------------
    # Internal helpers — graph construction and LLM resolution
    # ------------------------------------------------------------------

    def _get_graph(self) -> Any:
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise ImportError(
                "langgraph is required for LangGraphManager. "
                "Install project dependencies: uv pip install -e '.[dev]'"
            ) from exc

        llm = self._resolve_llm()

        def _generate_proposal(state: _PlanState) -> _PlanState:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=state["context_text"])])
            return {**state, "raw_proposal": response.content}

        builder = StateGraph(_PlanState)
        builder.add_node("prepare_context", _prepare_context)
        builder.add_node("generate_proposal", _generate_proposal)
        builder.add_node("validate_proposal", _validate_proposal)
        builder.add_edge(START, "prepare_context")
        builder.add_edge("prepare_context", "generate_proposal")
        builder.add_edge("generate_proposal", "validate_proposal")
        builder.add_edge("validate_proposal", END)
        return builder.compile()

    def _resolve_llm(self) -> Any:
        if self._injected_llm is not None:
            return self._injected_llm
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=self._model,
                base_url=self._host,
                temperature=self._temperature,
            )
        except ImportError as exc:
            raise ImportError(
                "langchain-ollama is required for LangGraphManager with a real LLM. "
                "Install it: uv pip install langchain-ollama. "
                "For tests, inject a FakeListChatModel via LangGraphManager(llm=...)."
            ) from exc
