"""G.5 engines (``autoresearch.trackg.audit_g5``): the contract, the seed guards, failure classification and the output
members.  Synthetic only: a fake isolated node, a scripted LLM, fake runners, fake ``memory.events``; no GPU, no server,
no label.

The real isolated runs here need waivers this container requires (root, no cgroup v2, no Yama, no worker sandbox); the
gate sees them only after the test runner wrapper marks the record effective, which stands in for the TASL-2 host.  The
cross-check against the paper side's readers (``g5_job.read_rows``, ``g5_seal.reduce_rows``, ``campaign_counts``)
imports them read-only from the G.5 paper worktree (``AR_G5_PAPER_ROOT``, default ``/tmp/pb/wt/g5m``) and is skipped
when that worktree is absent.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from autoresearch.trackg import audit_g5 as A
from autoresearch.trackg import isolation
from autoresearch.trackg import noise_floor as nf
from autoresearch.trackg import pilot_g4
from autoresearch.trackg.diff_support import apply_diff, code_state_hashes
from autoresearch.trackg.node_template import load_trackg_node_spec
from autoresearch.trackg.outcome import RunOutcome
from autoresearch.trackg.runner import ResourceLimits, RunResult, WorkingCopyRunner

from .isolated_fixture import TRAIN_PY, isolated_spec_mapping, make_isolated_node

HARNESS = Path(__file__).resolve().parents[2]
PAPER = Path(os.environ.get("AR_G5_PAPER_ROOT", "/tmp/pb/wt/g5m"))
PAPER_T1 = PAPER / "sweep2" / "t1_tasl"
NODE = "fake_isolated"
SPEC_REL = "configs/trackg/nodes/fake_isolated.yaml"
NODE_REL = "nodes/node"
BASE_TRAIN = TRAIN_PY
HALF_TRAIN = TRAIN_PY.replace('MODE = "ok"', 'MODE = "half"')


def udiff(old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "a/train.py", "b/train.py", n=1))


D_HALF = udiff(BASE_TRAIN, HALF_TRAIN)                                     # accuracy 0.5 (baseline 1.0)
D_EXTRA = udiff(HALF_TRAIN, HALF_TRAIN.replace('MODE = "half"', 'MODE = "half"\nEXTRA = 1'))
D_CRASH = udiff(HALF_TRAIN, HALF_TRAIN.replace('MODE = "half"', 'MODE = "crash"'))
D_NAN = udiff(HALF_TRAIN, HALF_TRAIN.replace('MODE = "half"', 'MODE = "nan"'))
D_EXTRA0 = udiff(BASE_TRAIN, BASE_TRAIN.replace('MODE = "ok"', 'MODE = "ok"\nEXTRA = 1'))   # on the baseline
D_INERT0 = udiff(BASE_TRAIN, BASE_TRAIN.replace('MODE = "ok"', 'MODE = "ok"  # a comment only'))


def paper():
    if not (PAPER_T1 / "g5_job.py").is_file():
        pytest.skip(f"the G.5 paper worktree is not at {PAPER}")
    if str(PAPER_T1) not in sys.path:
        sys.path.insert(0, str(PAPER_T1))
    import g5_job as J  # noqa: PLC0415
    import g5_seal as SE  # noqa: PLC0415
    import g5_tasl as S  # noqa: PLC0415
    return J, SE, S


# =============================================================================================== fake harness
def make_harness(root: Path, **spec_over) -> Path:
    """A harness checkout holding one isolated fake node (maximise accuracy unless overridden)."""
    (root / "configs" / "trackg" / "nodes").mkdir(parents=True)
    make_isolated_node(root / "nodes")
    (root / SPEC_REL).write_text(yaml.safe_dump(isolated_spec_mapping(**spec_over)), encoding="utf-8")
    return root


def pins(root: Path) -> dict:
    spec = load_trackg_node_spec(root / SPEC_REL)
    pin = isolation.frozen_file_hashes(root / NODE_REL, spec.frozen_paths)
    return {"node_spec_sha256": spec.raw_sha256, "committed_pin_sha256": nf.frozen_pin_sha256(nf.committed_view(pin)),
            "data_manifest_sha256": None}


def state_of(root: Path, diffs: list[str], tmp: Path) -> str:
    spec = load_trackg_node_spec(root / SPEC_REL)
    work = tmp / f"st_{len(list(tmp.glob('st_*')))}"
    shutil.copytree(root / NODE_REL, work)
    for d in diffs:
        assert apply_diff(work, d, spec.to_base_node_spec()).applied
    return code_state_hashes(work, spec.to_base_node_spec())[0]


def chunk_file(root: Path, units: list[dict], seeds, *, k=10, **over) -> dict:
    data = {"schema": A.CHUNK_SCHEMA, "label": "EVIDENCE", "run_id": "c3_t1_labels_20261201", "node": NODE,
            "family": "G5-7", "wave": 1, "chunk": 0, "K": k, "spec_path": SPEC_REL, "node_dir": NODE_REL,
            **pins(root), "measured_run_seconds": 20.0, "timeout_factor": 3.0, "split_policy": "vary",
            "units": units, "runs": [{"unit": u["unit"], "seed": s, "train_seed": s, "split_seed": s}
                                     for u in units for s in seeds],
            "interpreter": sys.executable, "cgroup_parent": "/sys/fs/cgroup/fake.scope", "isolation_waivers": {},
            "record_gate": "noise_floor.check_effective", "deadline_epoch": time.time() + 3600,
            "memory_limit": "external: systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0"}
    data.update(over)
    return data


def campaign_file(root: Path, *, wave=1, modes=("history",), budget=2, **over) -> dict:
    data = {"schema": A.CAMPAIGN_SCHEMA, "label": "EVIDENCE", "run_id": "c3_t1_labels_20261201", "node": NODE,
            "family": "G5-7", "wave": wave, "spec_path": SPEC_REL, "node_dir": NODE_REL, **pins(root),
            "proposal_schema": "manager_proposal/v2-diff", "extra_payload": {"cache_prompt": False}, "budget": budget,
            "w_semantics": "manager_calls",
            "instances": [{"instance": k, "train_seed": 5100 + k, "split_seed": 5100 + k, "manager_seed": 5100 + k,
                           "memory_mode": m} for k in A.WAVE_INSTANCES[wave] for m in modes],
            "max_tokens": 4096, "temperature": 0.2, "measured_run_seconds": 20.0, "timeout_factor": 3.0,
            "base_url": "http://127.0.0.1:1/v1", "model": "gemma4_31b", "cgroup_parent": "/sys/fs/cgroup/fake.scope",
            "interpreter": sys.executable, "isolation_waivers": {}, "deadline_epoch": time.time() + 3600,
            "memory_limit": "external: systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0"}
    data.update(over)
    return data


def stop(match, fn, *a, **kw):
    with pytest.raises(A.G5EngineStop, match=match):
        fn(*a, **kw)


# =============================================================================================== contract
def test_contract_marker_schemas_and_paper_constants():
    src = Path(A.__file__).read_text()
    for needle in ("g5_tasl_engine_contract/1", "trackg_g5_campaign/v1", "trackg_g5_audit_chunk/v1"):
        assert needle in src                     # what g5_job.check_engines greps for in this file
    assert (A.ENGINE_CONTRACT, A.ROW_SCHEMA, A.INFRASTRUCTURE_RERUNS) == \
        ("g5_tasl_engine_contract/1", "trackg_g5_audit_row/v1", 2)
    assert Path(A.__file__).resolve().relative_to(HARNESS).as_posix() == "src/autoresearch/trackg/audit_g5.py"
    if (PAPER_T1 / "g5_job.py").is_file():
        J, _, S = paper()
        c = J.engine_contract()
        assert (c["marker"], c["file"], c["module"]) == (A.ENGINE_CONTRACT, "src/autoresearch/trackg/audit_g5.py",
                                                         "autoresearch.trackg.audit_g5")
        assert (J.CAMPAIGN_SCHEMA, J.CHUNK_SCHEMA, J.ROW_SCHEMA) == (A.CAMPAIGN_SCHEMA, A.CHUNK_SCHEMA, A.ROW_SCHEMA)
        assert c["infrastructure_reruns"] == A.INFRASTRUCTURE_RERUNS and c["file"] in S.G5_TRACKG_ADDITIONS
        assert c["subcommands"] == {"campaign": "campaign", "audit": "audit"}
        spec = json.loads((PAPER_T1 / "g5_tasl_spec.json").read_text())
        for k in (10, 6):
            seeds = S.audit_seeds(spec, k)
            assert (seeds["label"], seeds["evaluation"]) == A.audit_seeds(k)
        assert S.instance_seeds(spec) == list(A.INSTANCE_SEEDS)


def test_existing_callers_keep_their_guards():
    assert pilot_g4.PILOT_NODES == ("mlab_cifar10", "mlgym_fmnist", "autoresearch_fixedtokens",
                                    "mlgym_breakout_minatar")
    assert pilot_g4.PILOT_SEED_BLOCK == range(4001, 4100)
    assert nf.RESERVED_SEEDS == (range(7001, 7011), range(9001, 10000))
    import inspect  # noqa: PLC0415
    from autoresearch.trackg.campaign_v2_diff import run_v2_diff_campaign  # noqa: PLC0415
    assert inspect.signature(run_v2_diff_campaign).parameters["runner_factory"].default is WorkingCopyRunner  # G.4
    assert issubclass(A.G5CampaignRunner, WorkingCopyRunner)
    with pytest.raises(pilot_g4.PilotStop, match="4001-4099"):
        pilot_g4.validate_pilot({"schema": pilot_g4.PILOT_SCHEMA, "label": "ENGINEERING_ONLY", "node": "mlab_cifar10",
                                 "proposal_schema": "manager_proposal/v2-diff", "extra_payload": {"cache_prompt": False},
                                 "budget": 2, "instances": [{"instance": 1, "train_seed": 5101, "split_seed": 5101,
                                                             "manager_seed": 5101}]})
    with pytest.raises(pilot_g4.PilotStop, match="ENGINEERING_ONLY"):
        pilot_g4.validate_pilot({"schema": pilot_g4.PILOT_SCHEMA, "label": "EVIDENCE"})


# =============================================================================================== campaign file
def test_valid_campaign_files(tmp_path):
    root = make_harness(tmp_path / "h")
    A.validate_campaign(campaign_file(root))
    A.validate_campaign(campaign_file(root, wave=2, modes=("history", "none")))
    A.validate_campaign(campaign_file(root, temperature=0.0))


@pytest.mark.parametrize("over, needle", [
    ({"label": "ENGINEERING_ONLY"}, "EVIDENCE"),
    ({"extra": 1}, "keys differ"),
    ({"isolation_waivers": {"root": "x"}}, "no isolation waiver"),
    ({"extra_payload": {}}, "cache_prompt: false"),
    ({"w_semantics": "worker_runs"}, "G5-W B"),
    ({"max_tokens": 8192}, "completion cap 4096"),
    ({"timeout_factor": 2.0}, "3 x the census"),
    ({"measured_run_seconds": 301}, "measured_run_seconds"),
    ({"proposal_schema": "manager_proposal/v1"}, "v2-diff"),
    ({"spec_path": "../x.yaml"}, "relative path"),
    ({"temperature": True}, "temperature"),
    ({"budget": 0}, "budget"),
    ({"wave": 3}, "wave must be 1 or 2"),
    ({"committed_pin_sha256": "x"}, "sha256"),
])
def test_campaign_file_refusals(tmp_path, over, needle):
    root = make_harness(tmp_path / "h")
    f = campaign_file(root)
    f.update(over)
    stop(needle, A.validate_campaign, f)


def test_campaign_seed_guards(tmp_path):
    root = make_harness(tmp_path / "h")

    def with_inst(mutate):
        f = campaign_file(root)
        mutate(f["instances"])
        return f
    stop("STOP 10: train_seed 4001", A.validate_campaign,
         with_inst(lambda i: i[0].update(train_seed=4001, split_seed=4001, manager_seed=4001)))
    stop("STOP 10: train_seed 5113", A.validate_campaign,
         with_inst(lambda i: i[3].update(instance=4, train_seed=5113)))
    stop("not 5102", A.validate_campaign, with_inst(lambda i: i[1].update(split_seed=5103)))
    stop("STOP 10: instance 5 is not an instance of wave 1", A.validate_campaign,
         with_inst(lambda i: i[3].update(instance=5, train_seed=5105, split_seed=5105, manager_seed=5105)))
    stop("exactly the wave's instances", A.validate_campaign, with_inst(lambda i: i.pop()))
    stop("appears twice", A.validate_campaign, with_inst(lambda i: i.append(dict(i[0]))))
    stop("memory_mode", A.validate_campaign, with_inst(lambda i: i[0].update(memory_mode="summary")))
    stop("exactly", A.validate_campaign, with_inst(lambda i: i[0].update(extra=1)))
    twins = campaign_file(root, modes=("history", "none"))
    twins["instances"] = [i for i in twins["instances"] if not (i["memory_mode"] == "none" and i["instance"] == 2)]
    stop("memory mode none", A.validate_campaign, twins)


# =============================================================================================== truncation
class ScriptedChat:
    """Replies in order; each reply is (content, completion_tokens)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        content, completion = self.replies.pop(0)
        return content, {"prompt_tokens": 900, "completion_tokens": completion}


def reply(diff, summary="s"):
    return json.dumps({"rationale": "r", "summary": summary, "objective": "o", "diff": diff})


def test_truncation_is_an_invalid_proposal(tmp_path):
    root = make_harness(tmp_path / "h")
    spec = load_trackg_node_spec(root / SPEC_REL)
    chat = ScriptedChat([(reply(D_HALF), 4096), (reply(D_HALF), 4095), ("not json", 4096), (reply(D_HALF), 5000)])
    m = A.TruncationInvalidManager(chat=chat, model="m", seed=5101, max_tokens=4096, temperature=0.2,
                                   extra_payload={"cache_prompt": False})
    p, call = m.propose(spec, root / NODE_REL, [], 1)
    assert p is None and call["truncated"] is True and call["parse_error"].startswith("truncated")
    assert call["raw_reply"] == reply(D_HALF) and call["request"]["cache_prompt"] is False
    p, call = m.propose(spec, root / NODE_REL, [], 2)
    assert p is not None and call["truncated"] is False and "parse_error" not in call
    p, call = m.propose(spec, root / NODE_REL, [], 3)
    assert p is None and call["truncated"] is True and "truncated" not in call["parse_error"]   # its own parse error
    assert m.propose(spec, root / NODE_REL, [], 4)[0] is None


# =============================================================================================== chunk file
def units_of(root: Path, tmp: Path, chains: list[list[str]]) -> list[dict]:
    return [{"unit": f"unit-{i}", "code_state_sha256": state_of(root, c, tmp), "diffs": c} for i, c in enumerate(chains)]


def test_valid_chunk_files(tmp_path):
    root = make_harness(tmp_path / "h")
    units = units_of(root, tmp_path, [[], [D_HALF]])
    A.validate_chunk(chunk_file(root, units, range(7001, 7011)))
    A.validate_chunk(chunk_file(root, units, [7001, 7002, 7003, 7006, 7007, 7008], k=6))
    A.validate_chunk(chunk_file(root, [], []))         # an empty chunk (g5_job.chunk_plan's [[]])


def test_chunk_seed_guards(tmp_path):
    root = make_harness(tmp_path / "h")
    units = units_of(root, tmp_path, [[]])
    for seed, k in ((7004, 6), (7009, 6), (7010, 6), (7011, 10), (9001, 10), (5101, 10), (3001, 10), (4001, 10)):
        stop("STOP 10", A.validate_chunk, chunk_file(root, units, [seed], k=k))
    stop("K is 10 or 6", A.validate_chunk, chunk_file(root, units, [7001], k=8))
    bad = chunk_file(root, units, [7001])
    bad["runs"][0]["split_seed"] = 7002
    stop("training seed = split seed", A.validate_chunk, bad)


@pytest.mark.parametrize("mutate, needle", [
    (lambda c: c.update(split_policy="fixed"), "split_policy vary"),
    (lambda c: c.update(record_gate="none"), "record gate"),
    (lambda c: c.update(isolation_waivers={"no_cgroup": "x"}), "no isolation waiver"),
    (lambda c: c.update(extra=1), "keys differ"),
    (lambda c: c["units"][0].update(code_state_sha256="B"), "must be a sha256"),
    (lambda c: c["runs"].append(dict(c["runs"][0])), "repeats a planned"),
    (lambda c: c["runs"][0].update(unit="other"), "not in the chunk"),
    (lambda c: c["units"].append({"unit": "unused", "code_state_sha256": "a" * 64, "diffs": []}), "every unit"),
    (lambda c: c["units"].append(dict(c["units"][0])), "appears twice"),
    (lambda c: c.update(label="ENGINEERING_ONLY"), "EVIDENCE"),
    (lambda c: c.update(timeout_factor=1.5), "3 x the census"),
])
def test_chunk_file_refusals(tmp_path, mutate, needle):
    root = make_harness(tmp_path / "h")
    c = chunk_file(root, units_of(root, tmp_path, [[]]), [7001, 7002])
    mutate(c)
    stop(needle, A.validate_chunk, c)


# =============================================================================================== attribution
def events(oom=0, oom_kill=0):
    return {"low": 0, "high": 0, "max": 3, "oom": oom, "oom_kill": oom_kill}


def test_attribution_edit_caused_failures_are_divergent():
    killed9 = RunOutcome.infrastructure_failure("killed", signal=9)
    out, why = A.attribute(killed9, events(), events(oom=1, oom_kill=1))
    assert (out.kind, out.reason) == ("divergent", "oom_kill_memory_limit") and why["original_reason"] == "killed"
    out, _ = A.attribute(RunOutcome.infrastructure_failure("killed", returncode=137), events(2, 2), events(3, 3))
    assert (out.kind, out.reason) == ("divergent", "oom_kill_memory_limit")
    for sig in (signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT, signal.SIGFPE, signal.SIGILL):
        out, why = A.attribute(RunOutcome.infrastructure_failure("killed", signal=int(sig)), None, None)
        assert (out.kind, out.reason) == ("divergent", "crash_signal") and why["memory_events"] == "unavailable"


def test_attribution_keeps_host_failures_infrastructure():
    killed9 = RunOutcome.infrastructure_failure("killed", signal=9)
    assert A.attribute(killed9, events(), events(oom_kill=1))[0].kind == "infrastructure_failure"   # host-wide OOM
    assert A.attribute(killed9, events(), events())[0].kind == "infrastructure_failure"
    assert A.attribute(killed9, None, None)[0].kind == "infrastructure_failure"
    term = RunOutcome.infrastructure_failure("killed", returncode=143)
    assert A.attribute(term, events(), events())[0].kind == "infrastructure_failure"
    for o in (RunOutcome.infrastructure_failure("launch_error", error="x"),
              RunOutcome.infrastructure_failure("infrastructure_log_pattern", pattern="No space left on device")):
        out, why = A.attribute(o, events(), events(oom=1, oom_kill=1))
        assert out is o and why["rule"].startswith("infrastructure")
    for o in (RunOutcome.ok(0.5), RunOutcome.divergent("timeout"), RunOutcome.divergent("non_finite_loss"),
              RunOutcome.divergent("no_metric_at_budget"), RunOutcome.divergent("nonzero_exit", returncode=4)):
        assert A.attribute(o, events(), events(oom=1, oom_kill=1)) == (o, None)


def test_memory_limit_cgroup(tmp_path):
    scope = tmp_path / "user.slice" / "job.scope"
    sub = scope / "runs"
    sub.mkdir(parents=True)
    (tmp_path / "user.slice" / "memory.max").write_text("max\n")
    (tmp_path / "user.slice" / "memory.events").write_text("oom 0\n")
    assert A.memory_limit_cgroup(str(sub)) is None
    (scope / "memory.max").write_text("8589934592\n")
    (scope / "memory.events").write_text("low 0\nhigh 0\nmax 4\noom 1\noom_kill 1\noom_group_kill 0\n")
    assert A.memory_limit_cgroup(str(sub)) == scope
    assert A.read_memory_events(scope) == {"low": 0, "high": 0, "max": 4, "oom": 1, "oom_kill": 1, "oom_group_kill": 0}
    assert A.read_memory_events(None) is None and A.read_memory_events(tmp_path / "absent") is None


# =============================================================================================== audit, fake runner
class FakeFactory:
    """Runner factory whose runs copy the incumbent and apply the diff for real (so code-state hashes are real), and
    whose outcome comes from ``behaviour(state, seed, attempt) -> (RunOutcome, isolation overrides)``."""

    def __init__(self, behaviour, on_run=None):
        self.behaviour, self.on_run = behaviour, on_run
        self.calls: list[tuple[str, int]] = []
        self.attempt: dict[tuple, int] = {}

    def __call__(self, spec, node_root, runs_root, *, frozen_pin=None, protect_paths=()):
        return _FakeRunner(self, spec, Path(node_root), Path(runs_root))


class _FakeRunner:
    def __init__(self, factory, spec, node_root, runs_root):
        self.f, self.spec, self.node_root, self.runs_root = factory, spec, node_root, runs_root

    def run(self, run_id, limits, *, train_seed, split_seed, diff_text, evaluated, register_evaluated,
            keep_code_snapshot, interpreter):
        assert train_seed == split_seed and evaluated is None and not register_evaluated and not keep_code_snapshot
        assert limits.isolation_waivers == {} and limits.timeout() == (60.0, "3 x measured 20 s")
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True)
        work = run_dir / "work"
        shutil.copytree(self.node_root, work, symlinks=True)
        base = self.spec.to_base_node_spec()
        if diff_text is not None:
            assert apply_diff(work, diff_text, base).applied
        state = code_state_hashes(work, base)[0]
        key = (state, train_seed)
        n = self.f.attempt[key] = self.f.attempt.get(key, -1) + 1
        self.f.calls.append((state, train_seed))
        if self.f.on_run:
            self.f.on_run(state, train_seed, n)
        outcome, iso_over = self.f.behaviour(state, train_seed, n)
        iso = {"effective": True, "waivers": {}, "exceptions": [], "log_intact": True, "parent_traced_by": None,
               "parent_dumpable_at_accept": False, "argv": ["python", "parent.py"]}
        iso.update(iso_over or {})
        record = {"run_id": run_id, "executed": True, "code_state_sha256": state, "isolation": iso,
                  "outcome": outcome.to_dict(), "elapsed_seconds_info_only": 1.5, "frozen_pin_sha256": "b" * 64,
                  "sealed_test": {"test_accuracy": 0.4242}}
        (run_dir / "run_record.json").write_text(json.dumps(record))
        (work / "model.bin").write_bytes(bytes(range(256)))
        return RunResult(run_id, run_dir, work, outcome, record)


def _rows(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines() if x.strip()]


def run_audit(chunk, root, out, factory, **kw):
    return A.run_audit(A.validate_chunk(chunk), root, out, gpu_uuids=("GPU-12345678-aaaa-bbbb-cccc-1234567890ab",),
                       cpu_list=(0,), runner_factory=factory, **kw)


def value(state, seed):
    return round(0.25 + (int(state[:6], 16) % 1000) / 4000 + (seed - 7000) / 1000, 6)


def test_audit_rows_one_per_planned_run_with_values(tmp_path):
    root = make_harness(tmp_path / "h")
    units = units_of(root, tmp_path, [[], [D_HALF], [D_HALF, D_EXTRA]])
    c = chunk_file(root, units, range(7001, 7011))
    f = FakeFactory(lambda st, seed, n: (RunOutcome.ok(value(st, seed)), None))
    rec = run_audit(c, root, tmp_path / "out", f)
    assert rec["stop"] is None and rec["written_rows"] == rec["expected_runs"] == 30
    rows = _rows(tmp_path / "out")
    assert [(r["unit"], r["seed"]) for r in rows] == [(r["unit"], r["seed"]) for r in c["runs"]]
    states = {u["unit"]: u["code_state_sha256"] for u in units}
    for r in rows:
        assert r["schema"] == "trackg_g5_audit_row/v1" and r["executed"] is True
        assert r["code_state_sha256"] == states[r["unit"]] and r["train_seed"] == r["split_seed"] == r["seed"]
        assert r["outcome"] == {"kind": "ok", "reason": None, "value": value(states[r["unit"]], r["seed"])}
        assert r["isolation"] == {"effective": True, "waivers": {}, "exceptions": [], "log_intact": True,
                                  "parent_traced_by": None, "parent_dumpable_at_accept": False}
        assert r["attempts"] == 1 and r["earlier_attempts"] == [] and "sealed_test" not in json.dumps(r)
    assert not (tmp_path / "out" / "states").exists() and not list((tmp_path / "out" / "runs").glob("*/work"))
    crec = json.loads((tmp_path / "out" / "chunk_record.json").read_text())
    assert crec["stop"] is None and crec["units_rematerialised"] == 3 and crec["memory_events_cgroup"] is None


def test_audit_infrastructure_reruns_then_one_row(tmp_path):
    root = make_harness(tmp_path / "h")
    units = units_of(root, tmp_path, [[], [D_HALF]])
    half = units[1]["code_state_sha256"]
    launch = RunOutcome.infrastructure_failure("launch_error", error="x")

    def behaviour(st, seed, n):
        if st == half and seed == 7002:
            return launch, None                                          # every attempt
        if st == half and seed == 7003 and n < 2:
            return launch, None                                          # recovers on the second rerun
        return RunOutcome.ok(0.5), None
    f = FakeFactory(behaviour)
    rec = run_audit(chunk_file(root, units, [7001, 7002, 7003]), root, tmp_path / "out", f)
    assert rec["stop"] is None and rec["written_rows"] == 6
    rows = {(r["unit"], r["seed"]): r for r in _rows(tmp_path / "out")}
    bad = rows[("unit-1", 7002)]
    assert bad["outcome"] == {"kind": "infrastructure_failure", "reason": "launch_error"}
    assert bad["attempts"] == 3 and bad["infrastructure_reruns"] == 2 and len(bad["earlier_attempts"]) == 2
    assert f.calls.count((half, 7002)) == 3 and "value" not in bad["outcome"]
    ok = rows[("unit-1", 7003)]
    assert ok["outcome"]["kind"] == "ok" and ok["attempts"] == 3 and ok["run_id"].endswith("_a2")
    assert [a["reason"] for a in ok["earlier_attempts"]] == ["launch_error", "launch_error"]


def test_audit_runner_refusal_is_infrastructure_after_reruns(tmp_path):
    root = make_harness(tmp_path / "h")
    units = units_of(root, tmp_path, [[]])

    class Refusing:
        def __init__(self, *a, **kw):
            pass

        def run(self, run_id, *a, **kw):
            from autoresearch.trackg.runner import RunnerError
            raise RunnerError("this host refuses the worker sandbox")
    rec = run_audit(chunk_file(root, units, [7001]), root, tmp_path / "out", Refusing)
    row = _rows(tmp_path / "out")[0]
    assert rec["stop"] is None
    assert row["executed"] is False and "isolation" not in row and row["outcome"]["kind"] == "infrastructure_failure"
    assert row["attempts"] == 3 and row["outcome"]["reason"] == "runner_error"


@pytest.mark.parametrize("failure, reason", [
    (RunOutcome.divergent("timeout", returncode=None), "timeout"),
    (RunOutcome.divergent("non_finite_metric", token="nan"), "non_finite_metric"),
    (RunOutcome.divergent("non_finite_loss", token="nan"), "non_finite_loss"),
    (RunOutcome.divergent("no_metric_at_budget"), "no_metric_at_budget"),
    (RunOutcome.divergent("nonzero_exit", returncode=4), "nonzero_exit"),
    (RunOutcome.infrastructure_failure("killed", signal=int(signal.SIGSEGV)), "crash_signal"),
])
def test_audit_edit_caused_failures_are_divergent_rows_without_rerun(tmp_path, failure, reason):
    root = make_harness(tmp_path / "h")
    units = units_of(root, tmp_path, [[D_HALF]])
    f = FakeFactory(lambda st, seed, n: (failure, None))
    rec = run_audit(chunk_file(root, units, [7001, 7006]), root, tmp_path / "out", f)
    rows = _rows(tmp_path / "out")
    assert rec["stop"] is None and len(f.calls) == 2
    assert all(r["outcome"] == {"kind": "divergent", "reason": reason} and r["attempts"] == 1 for r in rows)
    assert all(r["isolation"]["effective"] is True for r in rows)


def test_audit_oom_kill_from_the_edit_is_divergent_a_host_oom_is_rerun(tmp_path):
    root = make_harness(tmp_path / "h")
    units = units_of(root, tmp_path, [[], [D_HALF]])
    half = units[1]["code_state_sha256"]
    scope = tmp_path / "cg" / "job.scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text("8589934592\n")
    counters = {"oom": 0, "oom_kill": 0}

    def write():
        (scope / "memory.events").write_text("".join(f"{k} {v}\n" for k, v in counters.items()))
    write()

    def on_run(st, seed, n):
        if st == half:                                  # the edit's run reaches the scope's MemoryMax: killed
            counters["oom"] += 1
            counters["oom_kill"] += 1
        elif seed == 7002 and n == 0:                   # a host-wide OOM kills the baseline's run once
            counters["oom_kill"] += 1
        write()

    def behaviour(st, seed, n):
        if st == half or (seed == 7002 and n == 0):
            return RunOutcome.infrastructure_failure("killed", signal=9), None
        return RunOutcome.ok(1.0), None
    f = FakeFactory(behaviour, on_run)
    rec = run_audit(chunk_file(root, units, [7001, 7002], cgroup_parent=str(scope)), root, tmp_path / "out", f)
    rows = {(r["unit"], r["seed"]): r for r in _rows(tmp_path / "out")}
    assert rec["stop"] is None and rec["memory_events_cgroup"] == str(scope)
    for seed in (7001, 7002):
        r = rows[("unit-1", seed)]
        assert r["outcome"] == {"kind": "divergent", "reason": "oom_kill_memory_limit"} and r["attempts"] == 1
        assert r["attribution"]["memory_events"] == {"oom": 1, "oom_kill": 1} and r["attribution"]["signal"] == 9
    host = rows[("unit-0", 7002)]
    assert host["outcome"]["kind"] == "ok" and host["attempts"] == 2
    assert host["earlier_attempts"][0]["reason"] == "killed"


def test_audit_unisolated_record_writes_its_row_and_stops(tmp_path):
    root = make_harness(tmp_path / "h")
    units = units_of(root, tmp_path, [[], [D_HALF]])
    half = units[1]["code_state_sha256"]
    f = FakeFactory(lambda st, seed, n: (RunOutcome.ok(0.737373), {"exceptions": ["shared_dev_shm"],
                                                                   "effective": False} if st == half else None))
    c = chunk_file(root, units, [7001, 7002])
    rec = run_audit(c, root, tmp_path / "out", f)
    rows = _rows(tmp_path / "out")
    assert rec["stop"].startswith("G5EngineStop: STOP: stop_unisolated") and len(rows) == 3
    assert rows[-1]["isolation"]["effective"] is False and "value" not in rows[-1]["outcome"]
    assert "0.737373" not in rec["stop"] and "7001" not in rec["stop"]          # no value, no seed in the stop text
    if (PAPER_T1 / "g5_job.py").is_file():
        J, _, _ = paper()
        with pytest.raises(J.Stop, match="STOP 2"):
            J.read_rows((tmp_path / "out" / "rows.jsonl").read_text().splitlines(), c)


def test_audit_integrity_stops_before_any_run(tmp_path):
    root = make_harness(tmp_path / "h")
    units = units_of(root, tmp_path, [[], [D_HALF]])
    units[1]["code_state_sha256"] = units[0]["code_state_sha256"]           # not the state D_HALF gives
    f = FakeFactory(lambda st, seed, n: (RunOutcome.ok(0.5), None))
    rec = run_audit(chunk_file(root, units, [7001]), root, tmp_path / "out", f)
    assert "integrity: unit 1" in rec["stop"] and f.calls == [] and _rows(tmp_path / "out") == []
    units = [{"unit": "u", "code_state_sha256": "e" * 64, "diffs": [D_EXTRA]}]   # D_EXTRA needs D_HALF first
    rec = run_audit(chunk_file(root, units, [7001]), root, tmp_path / "out2", f)
    assert "integrity: unit 0" in rec["stop"] and f.calls == []
    units = units_of(root, tmp_path, [[]])
    units[0]["code_state_sha256"] = "c" * 64
    rec = run_audit(chunk_file(root, units, [7001]), root, tmp_path / "out3", f)
    assert "the node kit's code state differs" in rec["stop"]


def test_audit_pins_and_deadline(tmp_path):
    root = make_harness(tmp_path / "h")
    units = units_of(root, tmp_path, [[]])
    f = FakeFactory(lambda st, seed, n: (RunOutcome.ok(0.5), None))
    rec = run_audit(chunk_file(root, units, [7001], node_spec_sha256="d" * 64), root, tmp_path / "a", f)
    assert "spec sha256" in rec["stop"]
    rec = run_audit(chunk_file(root, units, [7001], committed_pin_sha256="d" * 64), root, tmp_path / "b", f)
    assert "committed-file pin" in rec["stop"]
    rec = run_audit(chunk_file(root, units, [7001, 7002], deadline_epoch=1000.0), root, tmp_path / "c", f,
                    clock=lambda: 900.0)
    assert "deadline: run 1 of 2" in rec["stop"] and f.calls == []
    unisolated = make_harness(tmp_path / "h2", isolation=None)
    rec = run_audit(chunk_file(unisolated, units_of(unisolated, tmp_path, [[]]), [7001]), unisolated, tmp_path / "d", f)
    assert "no isolation block" in rec["stop"]


def test_audit_empty_chunk(tmp_path):
    root = make_harness(tmp_path / "h")
    f = FakeFactory(lambda st, seed, n: (RunOutcome.ok(0.5), None))
    rec = run_audit(chunk_file(root, [], []), root, tmp_path / "out", f)
    assert rec["stop"] is None and rec["written_rows"] == 0 and (tmp_path / "out" / "rows.jsonl").read_text() == ""


# =============================================================================================== CLI and GPUs
def test_cli_refusals_leave_a_record(tmp_path, monkeypatch):
    root = make_harness(tmp_path / "h")
    c = chunk_file(root, units_of(root, tmp_path, [[]]), [7001])
    path = tmp_path / "chunk.json"
    path.write_text(json.dumps(c))
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("_CONDOR_AssignedGPUs", raising=False)
    assert A.main(["audit", "--chunk", str(path), "--harness-root", str(root), "--out", str(tmp_path / "o")]) == 2
    crec = json.loads((tmp_path / "o" / "chunk_record.json").read_text())
    assert "needs --gpu-uuid" in crec["stop"] and crec["refused_before_running"] is True
    path.write_text(json.dumps(dict(c, runs=[dict(c["runs"][0], seed=9001)])))
    assert A.main(["audit", "--chunk", str(path), "--harness-root", str(root), "--out", str(tmp_path / "o2")]) == 2
    assert "STOP 10" in json.loads((tmp_path / "o2" / "chunk_record.json").read_text())["stop"]
    camp = tmp_path / "campaign.json"
    camp.write_text(json.dumps(campaign_file(root, label="ENGINEERING_ONLY")))
    assert A.main(["campaign", "--campaign", str(camp), "--harness-root", str(root), "--out", str(tmp_path / "o3")]) == 2
    assert "EVIDENCE" in json.loads((tmp_path / "o3" / "campaign_record.json").read_text())["stop"]


def test_resolve_gpus():
    uuid = "GPU-1a2b3c4d-0000-1111-2222-333344445555"

    def smi(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, f"GPU 0: NVIDIA GeForce RTX 4090 (UUID: {uuid})\n", "")
    assert A.resolve_gpus("mlab_imdb", uuid, {"CUDA_VISIBLE_DEVICES": ""}, runner=smi) == (uuid,)
    stop("needs --gpu-uuid", A.resolve_gpus, "mlab_imdb", None, {})
    assert A.resolve_gpus("mlgym_houseprice", None, {"CUDA_VISIBLE_DEVICES": ""}) == ()
    stop("CUDA_VISIBLE_DEVICES empty", A.resolve_gpus, "mlgym_houseprice", None, {"CUDA_VISIBLE_DEVICES": "0"})
    stop("runs without --gpu-uuid", A.resolve_gpus, "mlgym_houseprice", uuid, {})
    with pytest.raises(Exception, match="scheduler variables"):
        A.resolve_gpus("mlab_imdb", uuid, {"NVIDIA_VISIBLE_DEVICES": "GPU-1a2b3c4d"}, runner=smi)


# =============================================================================================== real runs end to end
def _test_waivers() -> dict:
    out = {"no_cgroup": "test container: no delegated cgroup v2", "no_worker_sandbox": "test container"}
    if os.geteuid() == 0:
        out["root"] = "test container runs as root"
    scope = isolation.ptrace_scope()
    if scope is None or scope < 1:
        out["ptrace_scope"] = f"test host ptrace_scope={scope}"
    return out


REAL_BUILD_LIMITS = A.build_limits


def _test_limits(data, gpu_uuids, cpu_list):
    """The engine's limits with this container's waivers (the engine itself never builds a waiver)."""
    real = REAL_BUILD_LIMITS(data, gpu_uuids, cpu_list)
    return dataclasses.replace(real, cgroup_parent=None, kill_grace_seconds=0.5, isolation_waivers=_test_waivers())


def _waiver_gate(record):
    """Stands in for check_effective on this container: the only gaps are the declared test waivers."""
    iso = record["isolation"]
    assert iso["waivers"] == _test_waivers() and iso["exceptions"] == [] and iso["log_intact"] is True


class HostRunner(WorkingCopyRunner):
    """The real runner; the record's isolation verdict is set to what TASL-2 gives (effective, no waiver) after the
    run, so the real gate and the paper's row checks can be exercised here."""

    def run(self, run_id, limits, **kw):
        result = super().run(run_id, _test_limits_from(limits), **kw)
        _waiver_gate(result.record)
        result.record["isolation"].update(effective=True, waivers={}, exceptions=[])
        return result


def _test_limits_from(limits: ResourceLimits) -> ResourceLimits:
    assert limits.isolation_waivers == {} and limits.cgroup_parent is not None
    return dataclasses.replace(limits, cgroup_parent=None, kill_grace_seconds=0.5, isolation_waivers=_test_waivers())


@pytest.fixture(scope="module")
def real_campaign(tmp_path_factory):
    """One real wave-1 campaign (4 instances, W = 3) on the isolated fake node, minimising accuracy so that D_HALF is
    kept.  Instance 1 keeps D_HALF, discards D_EXTRA (a chain of two diffs), then an unparseable reply (invalid);
    2: D_HALF, D_CRASH (divergent), then D_HALF again in a reply at the cap (invalid, not run); 3: D_HALF, D_NAN
    (divergent), unparseable; 4: D_EXTRA0 on the baseline (discarded), the same diff again (not-run duplicate), and a
    comment-only diff (not-run inert)."""
    tmp = tmp_path_factory.mktemp("g5camp")
    root = make_harness(tmp / "h", metric_direction="minimize")
    data = A.validate_campaign(campaign_file(root, budget=3))
    chat = ScriptedChat([(reply(D_HALF, "half"), 300), (reply(D_EXTRA, "extra"), 300), ("not json", 100),
                         (reply(D_HALF, "half"), 300), (reply(D_CRASH, "crash"), 300), (reply(D_HALF, "half"), 4096),
                         (reply(D_HALF, "half"), 300), (reply(D_NAN, "nan"), 300), ("not json", 100),
                         (reply(D_EXTRA0, "extra0"), 300), (reply(D_EXTRA0, "extra0"), 300),
                         (reply(D_INERT0, "comment"), 300)])
    mp = pytest.MonkeyPatch()
    mp.setattr(A, "build_limits", _test_limits)
    try:
        rec = A.run_campaign(data, root, tmp / "out" / "campaign", gpu_uuids=(), cpu_list=(0,), chat=chat,
                             gate=_waiver_gate)
    finally:
        mp.undo()
    return {"root": root, "data": data, "record": rec, "out": tmp / "out" / "campaign", "chat": chat, "tmp": tmp}


def _ledgers(out: Path) -> dict[str, list[str]]:
    return {p.parent.name: p.read_text().splitlines() for p in sorted(out.glob("instance_*/campaign_ledger.jsonl"))}


def test_real_campaign_ledgers_and_record(real_campaign):
    rc = real_campaign
    assert rc["record"]["stop"] is None and rc["chat"].replies == []
    ledgers = _ledgers(rc["out"])
    assert sorted(ledgers) == [f"instance_{k}_history" for k in (1, 2, 3, 4)]
    decisions = {name: [json.loads(x)["decision"] for x in lines] for name, lines in ledgers.items()}
    assert decisions == {"instance_1_history": ["baseline", "kept", "discarded", "invalid_proposal"],
                         "instance_2_history": ["baseline", "kept", "discarded_divergent", "invalid_proposal"],
                         "instance_3_history": ["baseline", "kept", "discarded_divergent", "invalid_proposal"],
                         "instance_4_history": ["baseline", "discarded", "not_run_duplicate", "not_run_inert"]}
    for lines in ledgers.values():
        for e in map(json.loads, lines):
            assert e["schema"] == "trackg_v2_diff_campaign_ledger/v1"
            for run in e.get("runs", []):
                assert run["executed"] is (e["decision"] not in ("not_run_duplicate", "not_run_inert"))
                if run["executed"]:
                    assert isinstance(run["elapsed_seconds_info_only"], float) and "code_state_source" not in run
                assert "sealed" not in run and A._SHA.fullmatch(run["code_state_sha256"])   # top level, every run
            if e["step"] >= 1:
                assert e["manager_call"]["request"]["cache_prompt"] is False
                assert e["manager_call"]["request"]["max_tokens"] == 4096
                assert e["manager_call"]["request"]["temperature"] == 0.2
    inst4 = [json.loads(x) for x in ledgers["instance_4_history"]]
    base_state, ran = inst4[0]["runs"][-1]["code_state_sha256"], inst4[1]["runs"][-1]
    dup, inert = inst4[2]["runs"][-1], inst4[3]["runs"][-1]
    # what g5_assemble reads for a not-run step: the last run record's top-level code_state_sha256
    assert dup["code_state_sha256"] == ran["code_state_sha256"] == dup["diff"]["code_state_sha256"] != base_state
    assert dup["code_state_source"] == "diff_provenance" and dup["diff"]["duplicate"] is True
    assert inert["code_state_sha256"] == base_state and inert["code_state_source"] == "diff_provenance"
    trunc = json.loads(ledgers["instance_2_history"][3])["manager_call"]
    assert trunc["truncated"] is True and trunc["usage"]["completion_tokens"] == 4096
    seeds = [p["seed"] for p in rc["chat"].payloads]
    assert len(set(seeds)) == 12                      # D26 seed per instance seed and step
    runs = [json.loads(ledgers["instance_1_history"][0])["runs"][0]]
    assert runs[0]["seeds"] == {"train": 5101, "split": 5101}
    crec = json.loads((rc["out"] / "campaign_record.json").read_text())
    assert crec["stop"] is None and [i["dir"] for i in crec["instances"]] == sorted(ledgers)


def test_real_campaign_then_audit_through_the_paper_readers(real_campaign, tmp_path):
    """Ledgers -> g5_job.audit_plan / chunk files -> the real audit engine -> g5_job.read_rows and
    g5_seal.reduce_rows: re-materialised normalised hashes equal the ledgers', ok rows carry outcome.value."""
    J, SE, S = paper()
    rc = real_campaign
    spec = json.loads((PAPER_T1 / "g5_tasl_spec.json").read_text())
    ledgers = _ledgers(rc["out"])
    counts = J.campaign_counts(ledgers, max_tokens=4096, ctx=32768, timeout_basis=20.0)
    assert counts["counts"]["truncated"] == 1 and counts["counts"]["truncated_not_invalid"] == 0
    assert counts["counts"]["invalid"] == 3 and counts["checks"]["requests_without_cache_prompt_false"] == 0
    assert (counts["counts"]["duplicate"], counts["counts"]["inert"]) == (1, 1)
    for lines in ledgers.values():                    # the expression g5_assemble.node_rows uses for not-run steps
        for e in map(json.loads, lines):
            if e["decision"] in ("not_run_duplicate", "not_run_inert"):
                assert (e.get("runs") or [{}])[-1].get("code_state_sha256")
    plan = J.audit_plan(spec, NODE, 1, ledgers, k=6)
    by_diffs = {tuple(u["diffs"]): u for u in plan["units"]}
    assert set(by_diffs) == {(), (D_HALF,), (D_HALF, D_EXTRA), (D_HALF, D_CRASH), (D_HALF, D_NAN), (D_EXTRA0,)}
    node = {"node": NODE, "family": "G5-7", "spec_path": SPEC_REL, "node_dir": NODE_REL, **pins(rc["root"]),
            "slowest_census_seconds": 20.0, "run_seconds": J.CHUNK_LIMIT_S / 1.875 / 6.5}   # chunks of 6 runs (one unit)
    chunks = J.chunk_plan(plan, node)
    assert all(len(c) == 6 for c in chunks)
    wanted = {by_diffs[d]["unit"]: d for d in ((), (D_HALF, D_EXTRA), (D_HALF, D_CRASH))}
    seen = {}
    for index, runs in enumerate(chunks):
        if runs[0]["unit"] not in wanted:
            continue
        chunk = J.audit_chunk_file(spec, node, plan, index, runs, interpreter=sys.executable,
                                   cgroup_parent="/sys/fs/cgroup/fake.scope", deadline_epoch=time.time() + 3600)
        out = tmp_path / f"chunk-{index}" / "out"
        (out.parent).mkdir()
        (out / "chunk.json").parent.mkdir()
        (out / "chunk.json").write_text(json.dumps(chunk))
        rec = A.run_audit(A.validate_chunk(chunk), rc["root"], out / "audit", gpu_uuids=(), cpu_list=(0,),
                          runner_factory=HostRunner)
        assert rec["stop"] is None, rec["stop"]
        lines = (out / "audit" / "rows.jsonl").read_text().splitlines()
        got = J.read_rows(lines, chunk)                     # schema, plan, seeds, isolation, code states
        reduced = SE.reduce_rows(spec, lines, chunk, "traindev")
        assert reduced["evaluation_rows_dropped"] == 3
        seen[wanted[runs[0]["unit"]]] = ({r["kind"] for r in got}, {v for _, v in reduced["rows"].values()},
                                         {json.loads(x)["outcome"].get("reason") for x in lines})
        # the job reads exactly these two members from out.tar; everything else under out/ stays unread
        assert (out / "audit" / "chunk_record.json").is_file() and list((out / "audit" / "runs").iterdir())
    assert seen[()] == ({"ok"}, {1.0}, {None})
    assert seen[(D_HALF, D_EXTRA)] == ({"ok"}, {0.5}, {None})
    assert seen[(D_HALF, D_CRASH)] == ({"divergent"}, {None}, {"nonzero_exit"})
