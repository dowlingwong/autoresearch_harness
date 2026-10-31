"""LangGraph-backed manager — scoped strictly to the proposal-generation layer.

Architecture contract:
  - This module may only touch: ManagerStatus, MemoryContext, NodeSpec, ManagerProposal.
  - It must never access: budget, lifecycle state machine, TrialRecord, TrialAppendStore,
    WorkerResult, keep/discard decisions, or anything in control_plane or memory.append_store.

Graph: prepare_context -> generate_proposal -> validate_proposal
  - prepare_context: formats status + memory + node spec into a bounded prompt
    (plus, only when the control plane requests an E1 revision, a feedback
    block; the default prompt is unchanged)
  - generate_proposal: calls the LLM (real or injected mock)
  - validate_proposal: parses LLM JSON output; falls back to heuristic on failure
  => returns ManagerProposal through the same interface as BaselineManager / PromptManager.
"""
from __future__ import annotations

import json
import re
import time
from hashlib import sha256
from typing import Any, TypedDict

from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.memory.summarizer import MemoryContext
from autoresearch.nodes.active_keys import resolve_active_keys
from autoresearch.nodes.spec import NodeSpec
from autoresearch.llm.langchain_client import _build_chat_model
from autoresearch.llm.providers import LLMConfig, resolve_llm_config, manager_response_format
from autoresearch.llm.transport_retry import (
    DEFAULT_POLICY,
    POLICY_VERSION,
    TransportRetryPolicy,
    invoke_with_retry,
)


class _PlanState(TypedDict):
    """Internal graph state. Never returned outside this module."""
    status: Any          # ManagerStatus
    memory_context: Any  # MemoryContext
    node_spec: Any       # NodeSpec
    context_text: str    # formatted prompt built by prepare_context
    raw_proposal: str    # raw LLM text output
    proposal: Any        # ManagerProposal | None, filled by validate_proposal
    revision_feedback: Any  # str | None — E1 revise feedback from the control plane
    proposal_schema: Any  # optional structure-only JSON schema version
    llm_meta: Any        # dict | None — per-call latency/usage telemetry
    llm_seed: Any        # int | None — D26 deterministic sampling seed
    llm: Any             # LLM for this call (constructor-seeded, or the injected stub)
    llm_seed_as_kwarg: Any  # bool — pass the seed to llm.invoke (injected stubs only)
    transport_retry: Any  # TransportRetryPolicy | None — bounded transport retry


def manager_seed(
    campaign_id: str,
    budget_index: int,
    repetition: int = 0,
    revision_attempt: int = 0,
) -> int:
    """D26: deterministic Ollama seed from (campaign/instance, trial, repetition, revision).

    Stable across processes and platforms (SHA-256 of the canonical string),
    so a branch can be re-run to the same proposals on the same model, Ollama
    version and hardware.  Bit-reproducibility across hardware is not claimed.
    """
    payload = "|".join(
        str(part)
        for part in (campaign_id, int(budget_index), int(repetition), int(revision_attempt))
    )
    return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:4], "big")


def _accepts_seed_kwarg(llm: Any) -> bool:
    """True when ``llm.invoke`` accepts an extra ``seed`` keyword argument."""
    import inspect

    method = getattr(llm, "invoke", None)
    if method is None:
        return False
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return "seed" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _usage_from_response(response: Any) -> tuple[int | None, int | None]:
    """Extract (prompt_tokens, completion_tokens) from a LangChain response.

    Returns ``(None, None)`` when the backend does not report usage.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("output_tokens")
    meta = getattr(response, "response_metadata", None) or {}
    if prompt_tokens is None:
        prompt_tokens = meta.get("prompt_eval_count")
    if completion_tokens is None:
        completion_tokens = meta.get("eval_count")
    return prompt_tokens, completion_tokens


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
    # D25: when the node declares active keys, the prompt shows only the keys
    # active for the current configuration.  The status itself is unfiltered
    # (the scripted stand-in's proposal law depends on reading it unchanged).
    current_constants = getattr(status, "current_constants", {}) or {}
    active = resolve_active_keys(node_spec, current_constants)
    if active is not None:
        active_set = set(active)
        current_constants = {k: v for k, v in current_constants.items() if k in active_set}
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

    revision_feedback = state.get("revision_feedback")
    if revision_feedback:
        # Only present when the control plane asked for an E1 revision; the
        # default prompt (no feedback) is byte-for-byte unchanged.
        lines += [
            "REVISION REQUEST: the control plane did not accept your previous proposal "
            "for this trial.",
            f"Feedback: {revision_feedback}",
            "Propose a different bounded change that addresses this feedback.",
            "",
        ]

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
    *, strict: bool = False,
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

    if strict:
        return None

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
    parse_ok = True

    schema_version = state.get("proposal_schema")
    try:
        if schema_version:
            schema = manager_response_format(schema_version)["json_schema"]["schema"]
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or set(parsed) != set(schema["required"]) or any(
                not isinstance(value, str) for value in parsed.values()
            ):
                raise ValueError("manager_proposal/v1 requires exactly six string fields")
            summary, rationale, objective = parsed["summary"], parsed["rationale"], parsed["objective"]
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
        parse_ok = False
        rationale = f"parse-failed; raw={raw[:120]!r}"
        if schema_version:
            parsed, objective = {}, ""  # never substitute the free-path heuristic


    extra: dict[str, Any] = {
        "context_sha256": sha256(state.get("context_text", "").encode("utf-8")).hexdigest(),
        "raw_proposal_sha256": sha256(raw.encode("utf-8")).hexdigest(),
        "raw_proposal_chars": len(raw),
    }
    llm_meta = dict(state.get("llm_meta") or {})
    llm_meta["parse_ok"] = parse_ok
    if schema_version:
        llm_meta.update(schema_version=schema_version, parse_failed=not parse_ok)
        extra.update(schema_version=schema_version, parse_failed=not parse_ok)
    extra["llm_call"] = llm_meta
    if state.get("revision_feedback"):
        extra["revision_feedback"] = state["revision_feedback"]

    # Attempt to extract a structured edit so claw_style_worker can apply
    # the proposal via its deterministic patch path (no coding agent required).
    current_constants = getattr(status, "current_constants", {}) or {}
    structured_edit = _extract_structured_edit(parsed, objective, current_constants, node_spec, strict=bool(schema_version))
    if structured_edit:
        extra["deterministic_patch"] = True
        extra["structured_edit"] = structured_edit

    if schema_version and not structured_edit:
        extra["proposal_precondition_failed"] = True

    if not structured_edit and getattr(node_spec, "active_keys", None):
        # D25: keep the key the manager named when it resolves to an editable
        # symbol but not to a current (active) constant, so the control plane
        # can reject the inactive key before any worker run.  Ungated nodes
        # are byte-identical: nothing is added to their proposals.
        named_raw = str(parsed.get("param") or "").strip()
        if not named_raw:
            match = re.search(
                r"change\s+([A-Za-z_][A-Za-z0-9_]*)\s+from\s+\S+\s+to\s+\S+",
                objective,
                re.IGNORECASE,
            )
            named_raw = match.group(1) if match else ""
        canonical = _canonical_editable_symbol(named_raw, node_spec)
        if canonical:
            extra["named_param"] = canonical

    proposal = ManagerProposal(
        manager_mode=LangGraphManager.mode,
        proposal_summary=summary,
        proposal_rationale=rationale,
        target_files=node_spec.editable_paths,
        objective=objective,
        extra=extra,
    )
    return {**state, "proposal": proposal}


def _canonical_editable_symbol(raw: str, node_spec: NodeSpec) -> str | None:
    """Resolve *raw* case-insensitively against the node's editable symbols."""
    if not raw:
        return None
    editable = tuple(getattr(node_spec, "editable_symbols", ()) or ())
    if raw in editable:
        return raw
    lookup = {symbol.lower(): symbol for symbol in editable}
    return lookup.get(raw.lower())


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
    deterministic = False

    def __init__(
        self,
        llm: Any = None,
        model: str = "ollama/qwen2.5-coder:7b",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        repetition: int = 0,
        transport_retry: TransportRetryPolicy | None = DEFAULT_POLICY,
        sleep: Any = None,
        proposal_schema: str | None = None,
        seed_key: str | None = None,
    ) -> None:
        if proposal_schema is not None:
            manager_response_format(proposal_schema)
        self.schema_version = proposal_schema
        self._injected_llm = llm
        self._model = model
        self._host = host
        self._temperature = temperature
        # D26: repetition index used in the deterministic per-call seed.  The
        # default 0 is the root campaign; branch repetitions set it.
        self._repetition = int(repetition)
        # Track S (29 Oct 2026; opt-in): the key of the D26 seed in place of the
        # campaign id, so campaigns of one instance under different policies
        # draw the same seeds at the same repetition.  ``None`` = the campaign id.
        self._seed_key = seed_key
        # Bounded transport retry (19 Oct 2026): up to 3 attempts for
        # transport/server-loss errors only.  ``None`` or ``max_attempts=1``
        # is the historical no-retry behaviour.
        self._transport_retry = transport_retry
        self._sleep = sleep if sleep is not None else time.sleep
        self._graph: Any = None

    @classmethod
    def from_config(cls, config_path: str) -> "LangGraphManager":
        cfg = json.loads(__import__("pathlib").Path(config_path).read_text())
        return cls(
            model=cfg.get("model", "qwen2.5-coder:7b"),
            host=cfg.get("host", "http://localhost:11434"),
            temperature=cfg.get("temperature", 0.2),
            proposal_schema=cfg.get("proposal_schema"),
        )

    # ------------------------------------------------------------------
    # Public interface — same signature as BaselineManager / PromptManager
    # ------------------------------------------------------------------

    def propose_next_trial(
        self,
        status: ManagerStatus,
        memory_context: MemoryContext,
        node_spec: NodeSpec,
        *,
        revision_feedback: str | None = None,
        revision_attempt: int | None = None,
    ) -> ManagerProposal:
        """Generate one proposal.

        ``revision_feedback`` is set only by the control plane's E1 ``revise``
        action or the D25 inactive-key revision loop.  It is added to the
        prompt and recorded as ``extra["revision_feedback"]``; when it is
        ``None`` (the default) the prompt and the proposal are exactly as
        before.

        ``revision_attempt`` (D26) is the zero-based count of revisions already
        made for this trial; it enters the deterministic sampling seed so a
        revision is not a repeat of the rejected call under the same seed.
        """
        seed = manager_seed(
            self._seed_key if self._seed_key is not None else status.campaign_id,
            status.budget_index,
            self._repetition,
            int(revision_attempt or 0),
        )
        if self._injected_llm is not None:
            call_llm = self._injected_llm
            seed_as_kwarg = True
        else:
            # Real backends take the seed at construction time (ChatOllama
            # supports a per-call ``seed`` option); the graph default LLM is
            # only a fallback for callers that bypass propose_next_trial.
            call_llm = self._resolve_llm(seed=seed)
            seed_as_kwarg = False
        graph = self._get_graph()
        try:
            result = self._invoke_graph(graph, status, memory_context, node_spec, revision_feedback,
                                        seed, call_llm, seed_as_kwarg)
        except Exception as exc:
            if getattr(exc, "llm_call_attempts", None) is not None:
                try:
                    exc.llm_call_seed = seed  # type: ignore[attr-defined]
                    # R1-CACHEPROMPT-FALSE (31 Oct 2026): the failed-call record states the setting too.
                    if self._injected_llm is None:
                        exc.llm_call_cache_prompt = resolve_llm_config(self._model).cache_prompt  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover
                    pass
            raise
        return result["proposal"]

    def _invoke_graph(self, graph, status, memory_context, node_spec, revision_feedback,
                      seed, call_llm, seed_as_kwarg):
        return graph.invoke(
            {
                "status": status,
                "memory_context": memory_context,
                "node_spec": node_spec,
                "context_text": "",
                "raw_proposal": "",
                "proposal": None,
                "revision_feedback": revision_feedback,
                "llm_seed": seed,
                "llm": call_llm,
                "llm_seed_as_kwarg": seed_as_kwarg,
                "transport_retry": self._transport_retry,
                "proposal_schema": self.schema_version,
            }
        )

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
        # R1-CACHEPROMPT-FALSE (31 Oct 2026): the value the real backend sends, recorded per call.
        llm_cache_prompt = None if self._injected_llm is not None else resolve_llm_config(self._model).cache_prompt

        def _generate_proposal(state: _PlanState) -> _PlanState:
            from langchain_core.messages import HumanMessage
            call_llm = state.get("llm") or llm
            seed = state.get("llm_seed")
            messages = [HumanMessage(content=state["context_text"])]
            policy = state.get("transport_retry") or TransportRetryPolicy(max_attempts=1, backoff_seconds=())

            def _call() -> Any:
                # The same messages and the same D26 seed on every attempt.
                if (
                    state.get("llm_seed_as_kwarg")
                    and seed is not None
                    and _accepts_seed_kwarg(call_llm)
                ):
                    return call_llm.invoke(messages, seed=seed)
                return call_llm.invoke(messages)

            started = time.perf_counter()
            response, attempts = invoke_with_retry(_call, policy, sleep=self._sleep)
            # Latency of the successful attempt (the pre-retry meaning of the field).
            latency_ms = (attempts[-1]["seconds"] * 1000.0) if policy.enabled else (
                (time.perf_counter() - started) * 1000.0
            )
            prompt_tokens, completion_tokens = _usage_from_response(response)
            content = getattr(response, "content", "") or ""
            # Reasoning never enters the final JSON parser. Separate server reasoning
            # fields are not part of content; inline thinking is removed only on the
            # opt-in schema path, preserving the historical free path.
            if self.schema_version:
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
            llm_meta = {
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response_chars": len(content),
                "seed": seed,
            }
            # R1-CACHEPROMPT-FALSE: recorded only when the backend actually carries the setting, so a
            # provider that does not (or an injected stub) does not gain a misleading null.
            if llm_cache_prompt is not None:
                llm_meta["cache_prompt"] = llm_cache_prompt
            if self.schema_version:
                usage = getattr(response, "usage_metadata", None) or {}
                token_usage = (getattr(response, "response_metadata", None) or {}).get("token_usage") or {}
                details = token_usage.get("completion_tokens_details") or {}
                reasoning = (usage.get("output_token_details") or {}).get("reasoning", details.get("reasoning_tokens"))
                llm_meta.update(schema_version=self.schema_version, reasoning_tokens=reasoning)
            if policy.enabled:
                # Every attempt is logged (attempt index, error class); a
                # no-retry policy keeps the historical payload unchanged.
                llm_meta["transport_retry_policy"] = POLICY_VERSION
                llm_meta["transport_attempts"] = attempts
            return {**state, "raw_proposal": content, "llm_meta": llm_meta}

        builder = StateGraph(_PlanState)
        builder.add_node("prepare_context", _prepare_context)
        builder.add_node("generate_proposal", _generate_proposal)
        builder.add_node("validate_proposal", _validate_proposal)
        builder.add_edge(START, "prepare_context")
        builder.add_edge("prepare_context", "generate_proposal")
        builder.add_edge("generate_proposal", "validate_proposal")
        builder.add_edge("validate_proposal", END)
        return builder.compile()

    def _resolve_llm(self, seed: int | None = None) -> Any:
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
        if config.provider == "anthropic":
            # D26 is an Ollama sampling seed; Anthropic exposes no seed.  Keep
            # the pre-D26 unseeded behaviour for that provider (review F6).
            seed = None
        kwargs = {"response_format": manager_response_format(self.schema_version)} if self.schema_version else {}
        return _build_chat_model(config, self._temperature, seed=seed, **kwargs)
