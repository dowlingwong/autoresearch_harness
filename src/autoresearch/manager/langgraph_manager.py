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
from autoresearch.llm.langchain_client import _build_chat_model
from autoresearch.llm.providers import LLMConfig, resolve_llm_config


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

    editable_file = node_spec.editable_paths[0] if node_spec.editable_paths else "train.py"

    # List editable values so the LLM can use exact names and current values.
    current_constants = getattr(status, "current_constants", {}) or {}
    if current_constants:
        lines.append(f"Editable values currently in {editable_file} (use exact names):")
        for k, v in current_constants.items():
            lines.append(f"  {k} = {v}")
        lines.append("")

    if memory_context.context_text.strip():
        lines += ["Prior trial memory:", memory_context.context_text, ""]
        lines += [
            "AVOIDANCE RULE (mandatory): Study the trial history above carefully.",
            "Identify every hyperparameter that was already tried in the same direction "
            "(e.g. increasing BATCH_SIZE, reducing LEARNING_RATE) and produced no "
            "improvement or a failure. You MUST NOT propose the same parameter change "
            "in the same direction again.",
            "Choose a DIFFERENT hyperparameter or a DIFFERENT direction than what has "
            "already been tried. If every direction for a parameter has been explored, "
            "switch to a completely different parameter.",
            "",
        ]
    else:
        lines.append("No prior trial memory available.")
        lines.append("")

    lines += [
        "Propose exactly one bounded experiment. Respond with ONLY a JSON object "
        "(no markdown fences, no extra text) with these six string fields:",
        '  "summary"    — short slug (e.g. "reduce-lr-2e-4")',
        '  "rationale"  — one sentence on why this might improve the metric',
        '  "objective"  — complete worker instruction: '
        f'"In {editable_file}, change PARAM from OLD to NEW. Edit only {editable_file}."',
        '  "param"      — exact editable value name from the list above (e.g. "LEARNING_RATE")',
        '  "old_value"  — current value as shown above (e.g. "5e-4")',
        '  "new_value"  — proposed replacement value (e.g. "2e-4")',
        "",
        "Constraints:",
        f"  - edit only: {', '.join(node_spec.editable_paths)}",
        "  - change exactly one listed value",
        "  - param must be one of the editable values listed above",
        "  - do not run the experiment; the harness runs it after your edit",
        "",
        'Example: {"summary": "reduce-lr-2e-4", '
        '"rationale": "smaller lr often improves convergence on noisy signals", '
        f'"objective": "In {editable_file}, change LEARNING_RATE from 5e-4 to 2e-4. Edit only {editable_file}.", '
        '"param": "LEARNING_RATE", "old_value": "5e-4", "new_value": "2e-4"}',
    ]

    return {**state, "context_text": "\n".join(lines)}


def _extract_structured_edit(
    parsed: dict[str, Any],
    objective: str,
    current_constants: dict[str, str],
    node_spec: NodeSpec,
) -> dict[str, str] | None:
    """Try to extract {symbol, old, new, path} for the deterministic patch path.

    Priority:
    1. Explicit ``param`` / ``old_value`` / ``new_value`` fields in the parsed JSON.
    2. Regex fallback on the ``objective`` string: "change SYMBOL from OLD to NEW".

    Returns a structured-edit dict or None if extraction fails.
    """
    if not current_constants:
        return None

    # Case-insensitive lookup: "learning_rate" -> "LEARNING_RATE"
    const_lookup: dict[str, str] = {k.lower(): k for k in current_constants}

    def _resolve(raw_sym: str) -> str | None:
        raw_sym = raw_sym.strip()
        if raw_sym in current_constants:
            return raw_sym
        return const_lookup.get(raw_sym.lower())

    # Strategy 1: explicit JSON fields
    p_raw = str(parsed.get("param") or "").strip()
    o_raw = str(parsed.get("old_value") or "").strip()
    n_raw = str(parsed.get("new_value") or "").strip()
    edit_path = node_spec.editable_paths[0] if node_spec.editable_paths else "train.py"
    edit_type = "config_value" if edit_path.endswith((".yaml", ".yml")) else "python_constant"

    if p_raw and n_raw:  # old_value optional — worker validates against live file
        sym = _resolve(p_raw)
        if sym:
            return {
                "type": edit_type,
                "symbol": sym,
                "old": o_raw or current_constants.get(sym, ""),
                "new": n_raw,
                "path": edit_path,
            }

    # Strategy 2: regex on objective string
    # Values may contain dots (0.02), hyphens (5e-4), so capture non-whitespace
    # and strip trailing sentence punctuation afterward.
    pat = re.compile(
        r"change\s+([A-Za-z_][A-Za-z0-9_]*)\s+from\s+(\S+)\s+to\s+(\S+)",
        re.IGNORECASE,
    )
    m = pat.search(objective)
    if m:
        old_val = m.group(2).rstrip(".,;:")
        new_val = m.group(3).rstrip(".,;:")
        sym = _resolve(m.group(1))
        if sym:
            return {
                "type": edit_type,
                "symbol": sym,
                "old": old_val,
                "new": new_val,
                "path": edit_path,
            }

    return None


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
    parsed: dict[str, Any] = {}

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

    extra: dict[str, Any] = {
        "context_sha256": sha256(state.get("context_text", "").encode("utf-8")).hexdigest(),
        "raw_proposal_sha256": sha256(raw.encode("utf-8")).hexdigest(),
        "raw_proposal_chars": len(raw),
    }

    # Attempt to extract a structured edit so claw_style_worker can apply
    # the proposal via its deterministic patch path (no coding agent required).
    current_constants = getattr(status, "current_constants", {}) or {}
    structured_edit = _extract_structured_edit(parsed, objective, current_constants, node_spec)
    if structured_edit:
        extra["deterministic_patch"] = True
        extra["structured_edit"] = structured_edit

    proposal = ManagerProposal(
        manager_mode=LangGraphManager.mode,
        proposal_summary=summary,
        proposal_rationale=rationale,
        target_files=node_spec.editable_paths,
        objective=objective,
        extra=extra,
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
        model: str = "ollama/qwen2.5-coder:7b",
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
        config = resolve_llm_config(self._model)
        if config.provider == "legacy":
            config = LLMConfig(
                provider="ollama",
                model_name=self._model,
                base_url=self._host,
                api_key="",
                model_id=f"ollama/{self._model}",
            )
        return _build_chat_model(config, self._temperature)
