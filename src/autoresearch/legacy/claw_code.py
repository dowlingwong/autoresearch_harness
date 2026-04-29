from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from autoresearch.control_plane.lifecycle import build_trial_records_from_legacy_loop_result
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.nodes.spec import load_node_spec


class ClawCodeAdapterError(RuntimeError):
    """Raised when the legacy claw-code autoresearch CLI fails."""


class ClawCodeAutoresearchAdapter:
    """Thin subprocess wrapper around the existing Stage 1 autoresearch CLI.

    The Stage 2 package treats `harness/claw-code` as a legacy worker/control
    substrate for now. This adapter avoids moving that code while giving the
    new framework a stable boundary to call.
    """

    def __init__(self, repo_root: str | Path, harness_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.harness_root = (
            Path(harness_root).resolve()
            if harness_root is not None
            else self.repo_root / "harness" / "claw-code"
        )

    def setup(self, node_root: str | Path, initialize_results: bool = True) -> dict[str, Any]:
        args = ["autoresearch", "setup", "--root", str(Path(node_root).resolve())]
        if not initialize_results:
            args.append("--no-init-results")
        return self._run_json(args)

    def status(self, node_root: str | Path) -> dict[str, Any]:
        return self._run_json(["autoresearch", "status", "--root", str(Path(node_root).resolve())])

    def parse_log(self, node_root: str | Path, log_path: str | Path) -> dict[str, Any]:
        return self._run_json(
            [
                "autoresearch",
                "parse-log",
                "--root",
                str(Path(node_root).resolve()),
                str(log_path),
            ]
        )

    def loop(
        self,
        node_root: str | Path,
        packet_path: str | Path,
        model: str,
        host: str,
        iterations: int = 1,
        retry_limit: int = 1,
        allow_any_branch: bool = False,
    ) -> dict[str, Any]:
        args = [
            "autoresearch",
            "loop",
            "--root",
            str(Path(node_root).resolve()),
            "--packet",
            str(Path(packet_path).resolve()),
            "--model",
            model,
            "--host",
            host,
            "--iterations",
            str(iterations),
            "--retry-limit",
            str(retry_limit),
        ]
        if allow_any_branch:
            args.append("--allow-any-branch")
        return self._run_json(args)

    def loop_and_record(
        self,
        node_root: str | Path,
        packet_path: str | Path,
        node_spec_path: str | Path,
        records_path: str | Path,
        campaign_id: str,
        manager_mode: str,
        worker_mode: str,
        memory_mode: str,
        model: str,
        host: str,
        iterations: int = 1,
        retry_limit: int = 1,
        allow_any_branch: bool = False,
    ) -> dict[str, Any]:
        legacy_result = self.loop(
            node_root=node_root,
            packet_path=packet_path,
            model=model,
            host=host,
            iterations=iterations,
            retry_limit=retry_limit,
            allow_any_branch=allow_any_branch,
        )
        node_spec = load_node_spec(node_spec_path)
        records = build_trial_records_from_legacy_loop_result(
            legacy_loop_result=legacy_result,
            node_spec=node_spec,
            campaign_id=campaign_id,
            manager_mode=manager_mode,
            worker_mode=worker_mode,
            memory_mode=memory_mode,
        )
        store = TrialAppendStore(records_path)
        store.append_many(records)
        return {
            "legacy_result": legacy_result,
            "records_path": str(Path(records_path)),
            "stage2_records": [record.to_dict() for record in records],
        }

    def _run_json(self, args: list[str]) -> dict[str, Any]:
        command = ["python3", "-m", "src.main", *args]
        run = subprocess.run(
            command,
            cwd=self.harness_root,
            env={**self._pythonpath_env(), "PYTHONPATH": str(self.harness_root)},
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode != 0:
            raise ClawCodeAdapterError(run.stderr.strip() or run.stdout.strip())
        try:
            payload = json.loads(run.stdout)
        except json.JSONDecodeError as error:
            raise ClawCodeAdapterError(f"legacy CLI did not return JSON: {run.stdout[:500]}") from error
        if not isinstance(payload, dict):
            raise ClawCodeAdapterError("legacy CLI returned non-object JSON")
        return payload

    @staticmethod
    def _pythonpath_env() -> dict[str, str]:
        # Keep this method isolated so later phases can merge caller env vars
        # without changing the adapter interface.
        return dict(os.environ)
