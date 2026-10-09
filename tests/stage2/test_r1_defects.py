"""R1 build-batch defects (BUILD_RUNBOOK L3).

Each test in this module is written to fail against the harness as it stood
before the corresponding fix and pass after it.  unittest style follows the
repository convention.
"""
from __future__ import annotations

import unittest

from autoresearch.manager.prompt_manager import PromptManager


class TestPromptManagerIsDeterministic(unittest.TestCase):
    """L3.1: prompt_manager makes no LLM call and must say so."""

    def test_prompt_manager_declares_deterministic(self) -> None:
        self.assertTrue(getattr(PromptManager, "deterministic", False))

    def test_prompt_manager_holds_no_llm(self) -> None:
        manager = PromptManager()
        self.assertFalse(hasattr(manager, "_injected_llm"))
        self.assertFalse(hasattr(manager, "llm"))


if __name__ == "__main__":
    unittest.main()
