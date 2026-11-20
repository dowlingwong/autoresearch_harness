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
            "isolation_waivers": {}, "max_tokens": 4096, "temperature": 0.2, "memory_mode": "history"}
    data.update(over)
    return data


@pytest.mark.parametrize("over, needle", [
    ({"node": "mlab_imdb"}, "not a G.4 pilot node"), ({"extra_payload": {}}, "cache_prompt"),
    ({"extra_payload": {"cache_prompt": False, "x": 1}}, "cache_prompt"),
    ({"proposal_schema": "manager_proposal/v1"}, "v2-diff"), ({"budget": 0}, "budget"),
    ({"isolation_waivers": {"root": "x"}}, "no isolation waiver"),
    ({"instances": [{"instance": 1, "train_seed": 1005, "split_seed": 4001, "manager_seed": 4001}]}, "seed block"),
    ({"instances": [{"instance": 1, "train_seed": 7003, "split_seed": 4001, "manager_seed": 4001}]}, "seed block"),
    ({"instances": [{"instance": 1, "train_seed": 9001, "split_seed": 4001, "manager_seed": 4001}]}, "seed block"),
    ({"instances": [{"instance": 1, "train_seed": 8001, "split_seed": 4001, "manager_seed": 4001}]}, "seed block"),
    ({"instances": [{"instance": 1, "train_seed": 5101, "split_seed": 4001, "manager_seed": 4001}]}, "seed block"),
    ({"instances": [{"instance": 1, "train_seed": 4001, "split_seed": 4001, "manager_seed": 4001},
                    {"instance": 2, "train_seed": 4001, "split_seed": 4001, "manager_seed": 4001}]}, "unique"),
    ({"instances": [{"instance": 1, "train_seed": 4001, "split_seed": 4001, "manager_seed": 4001, "x": 1}]},
     "an instance has"),
    ({"max_tokens": 10 ** 9}, "completion cap"), ({"max_tokens": "lots"}, "completion cap"),
    ({"temperature": "0.2"}, "temperature"), ({"temperature": 3.0}, "temperature"),
    ({"memory_mode": "append_only_summary"}, "memory_mode"),
    ({"spec_path": "../x.yaml"}, "relative path"), ({"node_dir": "/abs"}, "relative path"),
])
def test_pilot_refusals(over, needle):
    with pytest.raises(pg.PilotStop, match=needle):
        pg.validate_pilot(pilot(**over))


def _call(p, completion=300, cache=False):
    return {"usage": {"prompt_tokens": p, "completion_tokens": completion, "wall_seconds": 20.0},
            "request": {"cache_prompt": cache, "temperature": 0.2}}


def _ledger_entries():
    run = {"executed": True, "elapsed_seconds_info_only": 12.0, "incumbent_code_state_sha256": "i0",
           "code_state_sha256": "c1"}
    return [
        {"step": 0, "decision": "baseline", "runs": [run]},
        {"step": 1, "decision": "kept", "manager_call": _call(5000), "runs": [run], "final_outcome": {"kind": "ok"}},
        {"step": 2, "decision": "invalid_proposal", "manager_call": _call(5100), "runs": []},
        {"step": 3, "decision": "discarded_divergent", "manager_call": _call(5200), "runs": [run],
         "final_outcome": {"kind": "divergent", "reason": "timeout"}},
        {"step": 4, "decision": "discarded", "manager_call": _call(5300), "runs": [run], "final_outcome": {"kind": "ok"}},
        {"step": 5, "decision": "not_run_refused", "manager_call": _call(5400),
         "runs": [{"executed": False, "diff": {"refusal": "scope_violation"}}]},
        {"step": 6, "decision": "not_run_refused", "manager_call": _call(5500),
         "runs": [{"executed": False, "diff": {"refusal": "context_mismatch"}}]},
        {"step": 7, "decision": "not_run_inert", "manager_call": _call(5600), "runs": [{"executed": False}]},
        {"step": 8, "decision": "not_run_duplicate", "manager_call": _call(5700), "runs": [{"executed": False}]},
        {"step": 9, "decision": "infrastructure_failure_final", "manager_call": _call(5800, completion=4096),
         "runs": [run]},
    ]


def test_run_pilot_with_a_fake_campaign_and_summary(tmp_path):
    seen = []

    def fake_campaign(spec, node_dir, runs_root, manager, limits, budget, **kw):
        assert kw["enable_v2_diff"] is True and kw["record_gate"] is check_effective
        assert manager.extra_payload == {"cache_prompt": False} and manager.model == "gemma4_31b"
        assert manager.max_tokens == 4096 and manager.temperature == 0.2 and manager.include_history is True
        assert limits.isolation_waivers == {} and limits.timeout()[0] == pytest.approx(3 * 12.297)
        seen.append((kw["train_seed"], kw["split_seed"], manager.seed))
        Path(runs_root).mkdir(parents=True)
        (Path(runs_root) / "campaign_ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in _ledger_entries()))
        return []

    rec = pg.run_pilot(pilot(), HARNESS, tmp_path / "out", gpu_uuids=("GPU-12345678-aaaa",), cpu_list=(0,),
                       chat=lambda payload: ("{}", {}), campaign=fake_campaign)
    assert rec["stop"] is None and seen == [(4001, 4001, 4001), (4002, 4002, 4002)]
    s = pg.summarise(tmp_path / "out")
    n = 18   # 9 proposals x 2 instances
    assert s["proposals"] == n and s["decisions"]["kept"] == 2 and s["decisions"]["invalid_proposal"] == 2
    r = s["rates"]
    assert r == {"invalid": round(2 / n, 4), "refused": round(4 / n, 4), "inert": round(2 / n, 4),
                 "duplicate": round(2 / n, 4), "divergent_of_ran": round(2 / 6, 4), "kept_of_ran": round(2 / 6, 4),
                 "divergent_of_proposals": round(2 / n, 4), "infrastructure_final": round(2 / n, 4),
                 "timeouts_of_ran": round(2 / 6, 4), "timeouts_of_proposals": round(2 / n, 4),
                 "ran_of_proposals": round(6 / n, 4)}
    assert s["refusal_reasons"] == {"context_mismatch": 2, "scope_violation": 2}
    # both instances ran the same (incumbent, candidate) states: one cross-instance group with all 6 ran proposals
    assert s["cross_instance_duplicates"] == {"groups": 1, "ran_proposals_in_groups": 6, "share_of_ran": 1.0,
                                              "keyed_ran": 2}
    assert s["diff_liveness"] == {"valid_of_proposals": round(16 / n, 4), "ran_of_valid": round(6 / 16, 4),
                                  "ok_of_proposals": round(4 / n, 4), "ok_of_valid": round(4 / 16, 4)}
    assert s["fit"] == {"context": 32768, "max_tokens": 4096, "all_fit_ctx": True, "calls_at_completion_cap": 2,
                        "cache_prompt_false_on_every_call": True, "calls": 18}
    assert s["prompt_tokens"]["median"] == 5400 and s["seconds"]["llm_total"] == 360.0
    assert s["seconds"]["run_total"] == 12.0 * 2 * 4 and "divergence_vs_census" not in s
    wall = sum(i["wall_seconds"] for i in rec["instances"])
    assert s["throughput_proposals_per_hour"] == (round(n / (wall / 3600.0), 2) if wall > 0 else None)


def test_summary_flags_cache_and_fit(tmp_path):
    out = tmp_path / "o"
    (out / "instance_1").mkdir(parents=True)
    entries = [{"step": 1, "decision": "invalid_proposal", "manager_call": _call(30000, cache=True), "runs": []}]
    (out / "instance_1" / "campaign_ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    s = pg.summarise(out)
    assert s["fit"]["all_fit_ctx"] is False and s["fit"]["cache_prompt_false_on_every_call"] is False


def test_stopped_instance_keeps_its_wall_time(tmp_path):
    from autoresearch.trackg.campaign_v2_diff import V2DiffCampaignError

    def failing(spec, node_dir, runs_root, manager, limits, budget, **kw):
        raise V2DiffCampaignError("STOP: run s001 failed the record gate")
    rec = pg.run_pilot(pilot(), HARNESS, tmp_path / "out", gpu_uuids=("GPU-12345678-aaaa",), cpu_list=(0,),
                       chat=lambda payload: ("{}", {}), campaign=failing)
    assert "record gate" in rec["stop"] and "wall_seconds" in rec["instances"][0]


def test_run_pilot_refuses_paths_outside_the_checkout(tmp_path):
    data = pilot()
    data["node_dir"] = "nodes/../../outside"
    rec = pg.run_pilot(data, HARNESS, tmp_path / "out", gpu_uuids=("GPU-12345678-aaaa",), cpu_list=(0,),
                       chat=lambda p: ("{}", {}), campaign=lambda *a, **k: [])
    assert "inside the harness checkout" in rec["stop"]


def test_main_returns_3_on_a_stop(tmp_path, monkeypatch):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(pilot(committed_pin_sha256="f" * 64)))
    monkeypatch.setattr(pg.gpu_identity, "resolve_from_job", lambda env: (("GPU-12345678-aaaa",), {}))
    assert pg.main(["run", "--pilot", str(path), "--harness-root", str(HARNESS), "--out", str(tmp_path / "o")]) == 3
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(pilot(budget=0)))
    assert pg.main(["run", "--pilot", str(bad), "--harness-root", str(HARNESS), "--out", str(tmp_path / "o2")]) == 2


def test_run_pilot_stops_on_a_pin_mismatch(tmp_path):
    rec = pg.run_pilot(pilot(committed_pin_sha256="f" * 64), HARNESS, tmp_path / "out", gpu_uuids=("GPU-12345678-aaaa",),
                       cpu_list=(0,), chat=lambda p: ("{}", {}), campaign=lambda *a, **k: [])
    assert "committed-file pin" in rec["stop"]


def test_timed_chat_records_wall_seconds():
    ticks = iter([1.0, 3.5])
    chat = pg.timed_chat(lambda payload: ("x", {"prompt_tokens": 1}), clock=lambda: next(ticks))
    assert chat({}) == ("x", {"prompt_tokens": 1, "wall_seconds": 2.5})


def test_manager_temperature_and_memory_mode(tmp_path):
    from autoresearch.trackg.manager_v2_diff import V2DiffManager, build_prompt
    from autoresearch.trackg.node_template import load_trackg_node_spec
    spec = load_trackg_node_spec(HARNESS / "configs/trackg/nodes/mlab_cifar10.yaml")
    hist = [{"step": 1, "summary": "lr up", "decision": "kept", "metric": 0.5}]
    assert "lr up" in build_prompt(spec, HARNESS / "nodes/mlab_cifar10", hist)
    none = build_prompt(spec, HARNESS / "nodes/mlab_cifar10", hist, include_history=False)
    assert "lr up" not in none and "No earlier steps shown." in none
    payloads = []
    m = V2DiffManager(chat=lambda p: payloads.append(p) or ("{}", {}), model="m", seed=4001, temperature=0.2,
                      include_history=False, extra_payload={"cache_prompt": False})
    m.propose(spec, HARNESS / "nodes/mlab_cifar10", hist, 1)
    assert payloads[0]["temperature"] == 0.2 and "lr up" not in payloads[0]["messages"][0]["content"]
    assert V2DiffManager(chat=lambda p: ("{}", {}), model="m").temperature == 0.0   # the earlier default stays
    with pytest.raises(ValueError, match="temperature"):
        V2DiffManager(chat=lambda p: ("{}", {}), model="m", temperature=5)
    with pytest.raises(ValueError, match="include_history"):
        V2DiffManager(chat=lambda p: ("{}", {}), model="m", include_history="yes")
