"""DT2: emitted decisions must carry explicit zero and nonzero LLM charges."""
from pathlib import Path
from autoresearch.control_plane.campaign import run_dry_campaign
from autoresearch.control_plane.controller import ControlDecision
from autoresearch.nodes.registry import load_registered_node

ROOT = Path(__file__).resolve().parents[2]

class SideQueryController:
    def decide(self, view, mask):
        return ControlDecision("continue", controller_version="dt2/test", input_hash=view.input_hash(),
                               llm_calls=int(view.epoch == "E1"))

def test_every_emitted_control_decision_has_explicit_llm_calls(tmp_path):
    events = []
    run_dry_campaign(node_spec=load_registered_node("lr_synthetic", repo_root=ROOT),
                     campaign_id="dt2", budget=1, manager_mode="baseline_manager",
                     memory_mode="none", records_path=tmp_path / "trials.jsonl",
                     controller=SideQueryController(), event_store=events)
    decisions = [e.payload for e in events if e.event_type == "control_decision"]
    assert len(decisions) == 3
    assert [e["llm_calls"] for e in decisions] == [0, 1, 0]
