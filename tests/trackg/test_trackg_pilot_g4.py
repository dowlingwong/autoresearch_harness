"""G.4 pilot: pilot-file validation, the run loop (fake campaign), and the measurements from ledgers (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.trackg import pilot_g4 as pg
from autoresearch.trackg.noise_floor import check_effective

HARNESS = Path(__file__).resolve().parents[2]


def pilot(**over):
    from autoresearch.trackg import isolation, noise_floor as nf
    from autoresearch.trackg.node_template import load_trackg_node_spec
    spec = load_trackg_node_spec(HARNESS / "configs/trackg/nodes/mlab_cifar10.yaml")
    pin = nf.frozen_pin_sha256(isolation.frozen_file_hashes(HARNESS / "nodes/mlab_cifar10", spec.frozen_paths))
    data = {"schema": pg.PILOT_SCHEMA, "label": "ENGINEERING_ONLY", "node": "mlab_cifar10",
            "spec_path": "configs/trackg/nodes/mlab_cifar10.yaml", "node_dir": "nodes/mlab_cifar10",
            "node_spec_sha256": spec.raw_sha256, "committed_pin_sha256": pin, "data_manifest_sha256": None,
            "proposal_schema": "manager_proposal/v2-diff",
            "extra_payload": {"cache_prompt": False}, "budget": 3,
            "instances": [{"instance": 1, "train_seed": 4001, "split_seed": 4001, "manager_seed": 4001},
                          {"instance": 2, "train_seed": 4002, "split_seed": 4002, "manager_seed": 4002}],
            "measured_run_seconds": 12.297, "base_url": "http://127.0.0.1:1/v1", "model": "gemma4_31b",
            "cgroup_parent": "/sys/fs/cgroup/job", "interpreter": "/env/bin/python", "memory_limit": "m",
            "isolation_waivers": {}}
    data.update(over)
    return data


@pytest.mark.parametrize("over, needle", [
    ({"node": "mlab_imdb"}, "not a G.4 pilot node"), ({"extra_payload": {}}, "cache_prompt"),
    ({"proposal_schema": "manager_proposal/v1"}, "v2-diff"), ({"budget": 0}, "budget"),
    ({"isolation_waivers": {"root": "x"}}, "no isolation waiver"),
    ({"instances": [{"instance": 1, "train_seed": 1005, "split_seed": 4001, "manager_seed": 1}]}, "reserved"),
    ({"instances": [{"instance": 1, "train_seed": 7003, "split_seed": 4001, "manager_seed": 1}]}, "reserved"),
])
def test_pilot_refusals(over, needle):
    with pytest.raises(pg.PilotStop, match=needle):
        pg.validate_pilot(pilot(**over))


def _ledger_entries():
    usage = lambda p: {"prompt_tokens": p, "completion_tokens": 300, "wall_seconds": 20.0}
    run = {"executed": True, "elapsed_seconds_info_only": 12.0}
    return [
        {"step": 0, "decision": "baseline", "runs": [run]},
        {"step": 1, "decision": "kept", "manager_call": {"usage": usage(5000)}, "runs": [run], "final_outcome": {"kind": "ok"}},
        {"step": 2, "decision": "invalid_proposal", "manager_call": {"usage": usage(5100)}, "runs": []},
        {"step": 3, "decision": "discarded_divergent", "manager_call": {"usage": usage(5200)}, "runs": [run],
         "final_outcome": {"kind": "divergent", "reason": "timeout"}},
    ]


def test_run_pilot_with_a_fake_campaign_and_summary(tmp_path):
    seen = []

    def fake_campaign(spec, node_dir, runs_root, manager, limits, budget, **kw):
        assert kw["enable_v2_diff"] is True and kw["record_gate"] is check_effective
        assert manager.extra_payload == {"cache_prompt": False} and manager.model == "gemma4_31b"
        assert limits.isolation_waivers == {} and limits.timeout()[0] == pytest.approx(3 * 12.297)
        seen.append((kw["train_seed"], kw["split_seed"], manager.seed))
        Path(runs_root).mkdir(parents=True)
        (Path(runs_root) / "campaign_ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in _ledger_entries()))
        return []

    rec = pg.run_pilot(pilot(), HARNESS, tmp_path / "out", gpu_uuids=("GPU-12345678-aaaa",), cpu_list=(0,),
                       chat=lambda payload: ("{}", {}), campaign=fake_campaign)
    assert rec["stop"] is None and seen == [(4001, 4001, 4001), (4002, 4002, 4002)]
    s = pg.summarise(tmp_path / "out", census_divergence_rate=0.25)
    assert s["proposals"] == 6 and s["decisions"]["kept"] == 2 and s["decisions"]["invalid_proposal"] == 2
    assert s["rates"]["invalid"] == round(2 / 6, 4) and s["rates"]["divergent_of_ran"] == 0.5
    assert s["rates"]["timeouts_of_ran"] == 0.5 and s["diff_liveness"]["ran_of_valid"] == 1.0
    assert s["prompt_tokens"]["median"] == 5100 and s["seconds"]["llm_total"] == 120.0
    assert s["divergence_vs_census"]["ratio"] == 2.0


def test_run_pilot_stops_on_a_pin_mismatch(tmp_path):
    rec = pg.run_pilot(pilot(committed_pin_sha256="f" * 64), HARNESS, tmp_path / "out", gpu_uuids=("GPU-12345678-aaaa",),
                       cpu_list=(0,), chat=lambda p: ("{}", {}), campaign=lambda *a, **k: [])
    assert "committed-file pin" in rec["stop"]


def test_timed_chat_records_wall_seconds():
    ticks = iter([1.0, 3.5])
    chat = pg.timed_chat(lambda payload: ("x", {"prompt_tokens": 1}), clock=lambda: next(ticks))
    assert chat({}) == ("x", {"prompt_tokens": 1, "wall_seconds": 2.5})
