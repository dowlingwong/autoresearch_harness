from __future__ import annotations

from pathlib import Path

from autoresearch.worker.claw_worker import (
    _extract_training_seed,
    _hash_editable_state,
    _hash_training_config,
)


def test_editable_state_hash_changes_with_train_file(tmp_path: Path) -> None:
    train = tmp_path / "train.py"
    train.write_text("SEED = 123\n", encoding="utf-8")
    first = _hash_editable_state(tmp_path, ("train.py",))
    train.write_text("SEED = 456\n", encoding="utf-8")
    second = _hash_editable_state(tmp_path, ("train.py",))
    assert first != second


def test_training_config_hash_ignores_proposal_text() -> None:
    base = {
        "description": "trial one",
        "objective": "try one thing",
        "train_command": "RESNET_TRIGGER_FAST_SEARCH=1 uv run train.py",
        "timeout_seconds": 600,
        "log_path": "run.log",
        "results_tsv": "results.tsv",
        "syntax_check_command": "python3 -m py_compile train.py",
    }
    changed_proposal = dict(base, description="trial two", objective="try another thing")
    assert _hash_training_config(base) == _hash_training_config(changed_proposal)


def test_extract_training_seed(tmp_path: Path) -> None:
    train = tmp_path / "train.py"
    train.write_text("\nSEED = 123\n", encoding="utf-8")
    assert _extract_training_seed(train) == "123"
