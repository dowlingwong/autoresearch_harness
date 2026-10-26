"""Exam random-valid anchor. No inference; uniform active key, then uniform declared value."""
from __future__ import annotations
import random
from autoresearch.manager.base import ManagerProposal
from autoresearch.manager.langgraph_manager import manager_seed
from autoresearch.nodes.active_keys import resolve_active_keys


class RandomValidManager:
    mode = "random_valid"
    deterministic = True

    def __init__(self, value_grid: dict[str, list[str]]) -> None:
        converted = {str(k): tuple(str(x) for x in values) for k, values in value_grid.items()}
        if not converted or any(not v or len(set(v)) != len(v) for v in converted.values()):
            raise ValueError("declare a nonempty unique value grid for every key")
        self.grid = converted

    def propose_next_trial(self, status, memory_context, node_spec):
        constants = {**status.effective_config, **status.current_constants}
        active = resolve_active_keys(node_spec, constants)
        if active is None:
            if node_spec.active_keys:
                raise ValueError("cannot resolve the declared active keys for this configuration")
            active = tuple(node_spec.editable_symbols)
        keys = tuple(active)
        if any(k not in constants or k not in node_spec.editable_symbols for k in keys):
            raise ValueError('every active editable key needs a current value')
        if not keys or any(k not in self.grid for k in keys):
            raise ValueError("active keys/current values and their complete declared grids are required")
        rng = random.Random(manager_seed(status.campaign_id, status.budget_index))
        key = rng.choice(keys)
        old, new = constants[key], rng.choice(self.grid[key])
        path = node_spec.editable_paths[0]
        return ManagerProposal(manager_mode=self.mode, proposal_summary=f"{key} {old} -> {new}",
            proposal_rationale="Uniform active key and uniform declared value grid, seeded by instance/trial.",
            target_files=node_spec.editable_paths, objective=f"Change {key} from {old} to {new} in {path}.",
            extra={"structured_edit": {"type": "config_value" if path.endswith((".yaml", ".yml")) else "python_constant",
                   "path": path, "symbol": key, "old": old, "new": new}})
