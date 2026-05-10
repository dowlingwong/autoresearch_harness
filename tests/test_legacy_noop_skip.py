from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = ROOT / "harness" / "claw-code"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

autoresearch_worker = importlib.import_module("src.autoresearch_worker")


class _FakeCreatedWorker:
    worker_id = "worker-1"


class _FakeWorkerRun:
    last_result = {"changed_files": ["train.py"]}

    def to_dict(self):
        return {
            "worker_id": "worker-1",
            "last_result": self.last_result,
        }


def test_legacy_noop_patch_skips_experiment(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "prepare.py").write_text("", encoding="utf-8")
    (tmp_path / "program.md").write_text("program", encoding="utf-8")
    (tmp_path / "train.py").write_text("SEED = 123\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "prepare.py", "program.md", "train.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)

    packet = autoresearch_worker.AutoresearchExperimentPacket(
        objective="edit train.py but make no effective change",
        description="noop",
        train_command="false",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("run_experiment should not be called for a no-op patch")

    monkeypatch.setattr(autoresearch_worker, "create_worker", lambda **kwargs: _FakeCreatedWorker())
    monkeypatch.setattr(autoresearch_worker, "run_worker", lambda *args, **kwargs: _FakeWorkerRun())
    monkeypatch.setattr(autoresearch_worker, "run_experiment", fail_if_called)

    result = autoresearch_worker.run_autoresearch_packet(packet, root=tmp_path)

    assert result["error"] == "no_op_patch"
    assert result["no_op_patch"] is True
    assert result["experiment"] is None
    assert result["recommended_status"] == "discard"


def test_stage2_legacy_mode_runs_without_internal_commit(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "prepare.py").write_text("", encoding="utf-8")
    (tmp_path / "program.md").write_text("program", encoding="utf-8")
    (tmp_path / "train.py").write_text("SEED = 123\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "prepare.py", "program.md", "train.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    head_before = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    packet = autoresearch_worker.AutoresearchExperimentPacket(
        objective="edit train.py",
        description="stage2 no commit",
        train_command="true",
    )

    def fake_run_worker(*args, **kwargs):
        (tmp_path / "train.py").write_text("SEED = 456\n", encoding="utf-8")
        return _FakeWorkerRun()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_commit_train_change should not be called in Stage 2 legacy mode")

    monkeypatch.setenv(autoresearch_worker.NO_LEGACY_COMMITS_ENV, "1")
    monkeypatch.setattr(autoresearch_worker, "create_worker", lambda **kwargs: _FakeCreatedWorker())
    monkeypatch.setattr(autoresearch_worker, "run_worker", fake_run_worker)
    monkeypatch.setattr(autoresearch_worker, "_commit_train_change", fail_if_called)
    monkeypatch.setattr(
        autoresearch_worker,
        "run_experiment",
        lambda **kwargs: autoresearch_worker.ExperimentMetrics(
            success=True,
            log_path=str(tmp_path / "run.log"),
            timed_out=False,
            return_code=0,
            val_bpb=0.5,
        ),
    )

    result = autoresearch_worker.run_autoresearch_packet(packet, root=tmp_path)
    head_after = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert result["commit"] == f"{head_before}-dirty"
    assert head_after == head_before
    assert result["recommended_status"] == "keep"


def test_legacy_discard_only_reverts_train_py(tmp_path: Path) -> None:
    (tmp_path / "prepare.py").write_text("", encoding="utf-8")
    (tmp_path / "program.md").write_text("program", encoding="utf-8")
    (tmp_path / "train.py").write_text("SEED = 123\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("keep me\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    (tmp_path / "train.py").write_text("SEED = 456\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("user work stays\n", encoding="utf-8")
    state = autoresearch_worker.AutoresearchState(
        root=str(tmp_path),
        branch="main",
        best_bpb=0.5,
        pending_experiment=autoresearch_worker.PendingExperiment(
            commit="candidate",
            base_commit=base_commit,
            description="candidate",
            packet=autoresearch_worker.AutoresearchExperimentPacket(
                objective="candidate",
                description="candidate",
            ).to_dict(),
            worker={},
            experiment={
                "success": True,
                "log_path": "run.log",
                "timed_out": False,
                "return_code": 0,
                "val_bpb": 0.6,
            },
            recommended_status="discard",
            results_tsv="results.tsv",
            log_path="run.log",
            created_at="2026-01-01T00:00:00Z",
        ).to_dict(),
        updated_at="2026-01-01T00:00:00Z",
    )
    autoresearch_worker.save_autoresearch_state(state, tmp_path)

    autoresearch_worker.discard_autoresearch_candidate(tmp_path)

    assert (tmp_path / "train.py").read_text(encoding="utf-8") == "SEED = 123\n"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "user work stays\n"
