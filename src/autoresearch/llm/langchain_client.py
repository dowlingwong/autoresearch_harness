from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment]

from autoresearch.llm.providers import LLMConfig, resolve_llm_config
from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.memory.summarizer import MemoryContext
from autoresearch.nodes.spec import NodeSpec


class ExperimentProposal(BaseModel):  # type: ignore[misc,valid-type]
    summary: str = Field(..., min_length=1) if Field else ""  # type: ignore[assignment]
    rationale: str = Field(..., min_length=1) if Field else ""  # type: ignore[assignment]
    objective: str = Field(..., min_length=1) if Field else ""  # type: ignore[assignment]


@dataclass(frozen=True)
class CampaignState:
    status: ManagerStatus
    memory_context: MemoryContext
    node_spec: NodeSpec


class LangChainProposalBackend:
    """LangChain-backed proposal generator with the Manager interface.

    The backend only produces ``ManagerProposal`` objects. It does not own
    budget, lifecycle state, worker execution, decisions, or ledger writes.
    """

    mode = "langchain_backend"

    def __init__(
        self,
        model_id: str,
        *,
        artifacts_dir: str | Path | None = None,
        llm: Any = None,
        temperature: float = 0.2,
    ) -> None:
        self.config = resolve_llm_config(model_id)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.temperature = temperature
        self._llm = llm

    def propose(self, campaign_state: CampaignState) -> ManagerProposal:
        return self.propose_next_trial(
            campaign_state.status,
            campaign_state.memory_context,
            campaign_state.node_spec,
        )

    def propose_next_trial(
        self,
        status: ManagerStatus,
        memory_context: MemoryContext,
        node_spec: NodeSpec,
    ) -> ManagerProposal:
        prompt = _proposal_prompt(status, memory_context, node_spec)
        response_text = self._invoke(prompt)
        parsed = _parse_structured_proposal(response_text)
        artifact_refs = self._write_artifacts(status, prompt, response_text, parsed)
        return ManagerProposal(
            manager_mode=self.mode,
            proposal_summary=parsed.summary.strip(),
            proposal_rationale=parsed.rationale.strip(),
            target_files=node_spec.editable_paths,
            objective=parsed.objective.strip(),
            extra={
                "llm_backend": "langchain",
                "provider": self.config.provider,
                "model_name": self.config.model_name,
                "base_url": self.config.base_url,
                "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                "raw_response_sha256": sha256(response_text.encode("utf-8")).hexdigest(),
                "raw_response_chars": len(response_text),
                **artifact_refs,
            },
        )

    def _invoke(self, prompt: str) -> str:
        llm = self._llm if self._llm is not None else _build_chat_model(self.config, self.temperature)
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
        except TypeError:
            response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

    def _write_artifacts(
        self,
        status: ManagerStatus,
        prompt: str,
        response_text: str,
        proposal: ExperimentProposal,
    ) -> dict[str, str]:
        if self.artifacts_dir is None:
            return {}
        trial_dir = self.artifacts_dir / f"trial-{status.budget_index:03d}" / "langchain"
        trial_dir.mkdir(parents=True, exist_ok=True)
        prompt_ref = trial_dir / "prompt.txt"
        response_ref = trial_dir / "response.txt"
        proposal_ref = trial_dir / "proposal.json"
        prompt_ref.write_text(prompt, encoding="utf-8")
        response_ref.write_text(response_text, encoding="utf-8")
        proposal_ref.write_text(
            json.dumps(
                {
                    "summary": proposal.summary,
                    "rationale": proposal.rationale,
                    "objective": proposal.objective,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "langchain_prompt_ref": str(prompt_ref),
            "langchain_response_ref": str(response_ref),
            "langchain_structured_proposal_ref": str(proposal_ref),
        }


def _build_chat_model(config: LLMConfig, temperature: float) -> Any:
    if config.provider in {"legacy", "ollama"}:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError("langchain-ollama is required for Ollama models") from exc
        return ChatOllama(
            model=config.model_name,
            base_url=config.base_url or "http://localhost:11434",
            temperature=temperature,
        )

    if config.provider in {"vllm", "lm_studio", "llamacpp", "openai", "deepseek"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for OpenAI-compatible providers "
                "(vllm, lm_studio, llamacpp, openai, deepseek)"
            ) from exc
        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "base_url": config.base_url or None,
            "temperature": temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.provider == "deepseek":
            thinking = os.environ.get("DEEPSEEK_THINKING", "").strip().lower()
            if thinking:
                if thinking not in {"enabled", "disabled"}:
                    raise ValueError("DEEPSEEK_THINKING must be 'enabled' or 'disabled'")
                kwargs["extra_body"] = {"thinking": {"type": thinking}}
            reasoning_effort = os.environ.get("DEEPSEEK_REASONING_EFFORT", "").strip().lower()
            if reasoning_effort:
                if reasoning_effort not in {"high", "max"}:
                    raise ValueError("DEEPSEEK_REASONING_EFFORT must be 'high' or 'max'")
                kwargs["reasoning_effort"] = reasoning_effort
        return ChatOpenAI(**kwargs)

    if config.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError("langchain-anthropic is required for Anthropic models") from exc
        kwargs = {"model": config.model_name, "temperature": temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatAnthropic(**kwargs)

    raise ValueError(f"unsupported provider for LangChain backend: {config.provider}")


def _proposal_prompt(status: ManagerStatus, memory_context: MemoryContext, node_spec: NodeSpec) -> str:
    memory = memory_context.context_text.strip() or "No prior trial memory."
    return "\n".join(
        [
            "You are a research manager proposing one bounded ML experiment.",
            f"Campaign: {status.campaign_id}",
            f"Trial budget index: {status.budget_index}",
            f"Node: {node_spec.name}",
            f"Metric: {node_spec.metric_name} ({node_spec.metric_direction})",
            f"Current best metric: {status.current_best_metric}",
            f"Editable files: {', '.join(node_spec.editable_paths)}",
            "",
            "Prior memory:",
            memory,
            "",
            "Respond with only JSON matching this schema:",
            '{"summary": "...", "rationale": "...", "objective": "..."}',
            "The objective must instruct the worker to edit only the editable files, "
            "make exactly one bounded change, and not run the experiment manually.",
        ]
    )


def _parse_structured_proposal(raw: str) -> ExperimentProposal:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    payload = json.loads(cleaned)
    if hasattr(ExperimentProposal, "model_validate"):
        return ExperimentProposal.model_validate(payload)
    return ExperimentProposal.parse_obj(payload)
