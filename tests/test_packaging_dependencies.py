from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_langgraph_and_langchain_backends_are_packaged() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = set(payload["project"]["dependencies"])
    names = {dep.split(">=", 1)[0] for dep in deps}
    assert {
        "langgraph",
        "langchain-core",
        "langchain-ollama",
        "langchain-openai",
        "langchain-anthropic",
    }.issubset(names)
