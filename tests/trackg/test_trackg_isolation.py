"""R1-HARNESS-ISOLATION: isolated runs (real subprocesses, CPU only).

The cgroup tests need a writable cgroup v2 directory: ``AR_TEST_CGROUP_PARENT``, or (as root) a fresh
directory under ``/sys/fs/cgroup/unified`` or ``/sys/fs/cgroup``; they are skipped otherwise.  The
same-user forgery test (``test_unprivileged_end_to_end``) needs root to drop to uid 65534.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from autoresearch.trackg import isolation
from autoresearch.trackg.node_template import TrackGNodeSpec, TrackGNodeSpecError, WallClockBudgetError
from autoresearch.trackg.outcome import DIVERGENT, INFRASTRUCTURE_FAILURE, OK, ProcessExit, classify_run
from autoresearch.trackg.runner import ResourceLimits, RunnerError, WorkingCopyRunner

from .fixtures import make_node, spec_mapping
from .isolated_fixture import NODE_LIB, isolated_spec_mapping, make_isolated_node

ROOT = os.geteuid() == 0
ROOT_REASON = {"root_isolation_reason": "test container runs as root"} if ROOT else {}


def _cgroup_root() -> Path | None:
    env = os.environ.get("AR_TEST_CGROUP_PARENT")
    if env:
        return Path(env)
    if not ROOT:
        return None
    for base in (Path("/sys/fs/cgroup/unified"), Path("/sys/fs/cgroup")):
        if (base / "cgroup.controllers").is_file() and os.access(base, os.W_OK):
            return base
    return None


@pytest.fixture
def cgroup_parent():
    base = _cgroup_root()
    if base is None:
        pytest.skip("no writable cgroup v2 directory")
    path = base / f"ar_test_{os.getpid()}_{time.monotonic_ns() % 10**9}"
    try:
        path.mkdir()
    except OSError as error:
        pytest.skip(f"cannot create a test cgroup: {error}")
    if not (path / "cgroup.kill").is_file():
        path.rmdir()
        pytest.skip("cgroup v2 without cgroup.kill")
    yield path
    for child in sorted(path.glob("*/"), reverse=True):
        try:
            (child / "cgroup.kill").write_text("1")
            time.sleep(0.05)
            child.rmdir()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass


def limits(cgroup_parent=None, **kw):
    base = dict(uncapped_reason="test fixture", measured_run_seconds=20.0, kill_grace_seconds=3.0, **ROOT_REASON)
    if cgroup_parent is None:
        base["no_cgroup_reason"] = "test without cgroup"
    else:
        base["cgroup_parent"] = str(cgroup_parent)
    base.update(kw)
    return ResourceLimits(**base)


def run(tmp_path, mode="ok", lim=None, run_id="r1", spec_over=None, env=None):
    node = make_isolated_node(tmp_path / mode, mode)
    spec = TrackGNodeSpec.from_mapping(isolated_spec_mapping(**(spec_over or {})))
    runner = WorkingCopyRunner(spec, node, tmp_path / mode / "runs")
    old = {}
    for k, v in (env or {}).items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        return runner.run(run_id, lim or limits())
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------------------------------- node spec

def test_isolation_block_is_validated():
    spec = TrackGNodeSpec.from_mapping(isolated_spec_mapping())
    assert spec.isolation.entry == "parent.py" and spec.isolation.argv("py", 5) == ["py", "parent.py", "5"]
    with pytest.raises(TrackGNodeSpecError, match="covered by frozen_paths"):
        TrackGNodeSpec.from_mapping(isolated_spec_mapping(isolation={"entry": "train.py"}))
    with pytest.raises(TrackGNodeSpecError, match="AR_RESULT_SOCKET"):
        TrackGNodeSpec.from_mapping(isolated_spec_mapping(isolation={"entry": "parent.py",
                                                                     "env": {"AR_RESULT_SOCKET": "3"}}))
    with pytest.raises(TrackGNodeSpecError, match="unknown keys"):
        TrackGNodeSpec.from_mapping(isolated_spec_mapping(isolation={"entry": "parent.py", "shell": True}))
    with pytest.raises(WallClockBudgetError):
        TrackGNodeSpec.from_mapping(isolated_spec_mapping(isolation={"entry": "parent.py",
                                                                     "args": ["--timeout", "5"]}))
    assert TrackGNodeSpec.from_mapping(spec_mapping()).isolation is None


def test_isolated_run_needs_a_cgroup_or_a_reason_and_refuses_root_without_one(tmp_path):
    node = make_isolated_node(tmp_path)
    runner = WorkingCopyRunner(TrackGNodeSpec.from_mapping(isolated_spec_mapping()), node, tmp_path / "runs")
    with pytest.raises(RunnerError, match="cgroup_parent"):
        runner.run("a", ResourceLimits(uncapped_reason="t", measured_run_seconds=5, **ROOT_REASON))
    if ROOT:
        with pytest.raises(RunnerError, match="as root"):
            runner.run("b", ResourceLimits(uncapped_reason="t", measured_run_seconds=5, no_cgroup_reason="t"))
    with pytest.raises(RunnerError, match="no_cgroup_namespace_reason"):
        ResourceLimits(uncapped_reason="t", measured_run_seconds=5, cgroup_namespace=False)


# ------------------------------------------------------------------------------------------- outcome rules

def test_classify_refuses_more_than_one_metric_line_and_takes_the_isolated_metric_from_the_channel():
    ok = ProcessExit(0)
    rx = r"^m=(\S+)"
    assert classify_run("m=1\nm=2\n", ok, metric_regex=rx).reason == "multiple_metric_lines"
    assert classify_run("m=1\n", ok, metric_regex=rx).value == 1.0
    assert classify_run("log\n", ok, metric_regex=rx, result_text="m=3\n").value == 3.0
    assert classify_run("m=9\n", ok, metric_regex=rx, result_text="m=3\n").reason == "metric_line_in_run_output"
    assert classify_run("", ok, metric_regex=rx, result_text="m=3\nm=4\n").reason == "result_channel_metric_lines"
    assert classify_run("", ok, metric_regex=rx, result_text="").reason == "no_metric_at_budget"
    timed = ProcessExit(-9, timed_out=True)
    assert classify_run("", timed, metric_regex=rx).kind == INFRASTRUCTURE_FAILURE
    out = classify_run("", timed, metric_regex=rx, timeout_is_proposal_failure=True)
    assert (out.kind, out.reason) == (DIVERGENT, "timeout")


# ------------------------------------------------------------------------------------------- isolated runs

def test_ok_isolated_run_takes_the_metric_from_the_channel(tmp_path):
    result = run(tmp_path)
    rec = result.record
    assert result.outcome.kind == OK and result.outcome.value == pytest.approx(1.0), rec
    iso = rec["isolation"]
    assert rec["command"] == ["python3", "parent.py", "5"] and iso["argv"] == rec["command"]
    assert iso["log_intact"] and iso["frozen_files_checked"] == 3 and not iso["frozen_files_changed"]
    assert iso["runner_dumpable"] is False and iso["effective"] is (not ROOT)
    assert rec["sealed"] == {"test_accuracy": 0.4242}
    assert rec["environment_applied"]["FAKE_NODE"] == "1"
    log = (result.run_dir / "run.log").read_text()
    assert "parent shared buffer sum=10" in log and "step 5 loss" in log
    assert "val_accuracy" not in log and log.rstrip().splitlines()[-1].startswith("AR_RUNNER result_channel_sha256=")
    channel = (result.run_dir / "result_channel.txt").read_text()
    assert channel == "val_accuracy: 1.0000\ntest_accuracy: 0.4242\n"
    assert iso["result_channel_sha256"] == __import__("hashlib").sha256(channel.encode()).hexdigest()


@pytest.mark.parametrize("mode,reason", [
    ("forge_stdout", "metric_line_in_run_output"),
    ("tamper_log", "log_tampered"),
    ("frozen", "frozen_file_changed"),
    ("crash", "nonzero_exit"),
    ("nan", "non_finite_loss"),
])
def test_forged_or_broken_isolated_runs_are_divergent(tmp_path, mode, reason):
    result = run(tmp_path, mode)
    assert (result.outcome.kind, result.outcome.reason) == (DIVERGENT, reason), result.record["outcome"]


def test_forging_through_proc_never_gives_an_accepted_forged_metric(tmp_path):
    """As root the worker can open the parent's pipe (the record says isolation is not effective) and the
    second metric line refuses the run; as a normal user the open fails and the run is ok."""
    result = run(tmp_path, "forge_proc")
    log = (result.run_dir / "run.log").read_text()
    assert "forge report:" in log
    if ROOT:
        assert result.record["isolation"]["effective"] is False
        if ":written" in log.split("forge report:")[1].splitlines()[0]:
            assert result.outcome.kind == DIVERGENT
    else:
        assert result.outcome.kind == OK and result.outcome.value == pytest.approx(1.0)


def test_measured_timeout_is_a_proposal_failure_and_kills_the_cgroup(tmp_path, cgroup_parent):
    lim = limits(cgroup_parent, measured_run_seconds=0.7, timeout_factor=3.0)
    start = time.monotonic()
    result = run(tmp_path, "sleep", lim)
    assert time.monotonic() - start < 30
    assert (result.outcome.kind, result.outcome.reason) == (DIVERGENT, "timeout")
    rec = result.record
    assert rec["process"]["killed_with"] == "cgroup.kill"
    cg = rec["isolation"]["cgroup"]
    assert cg["removed"] and not Path(cg["path"]).exists()
    assert rec["isolation"]["cgroup_namespace"] is True


def test_provisional_timeout_stays_infrastructure(tmp_path, cgroup_parent):
    lim = ResourceLimits(uncapped_reason="t", provisional_timeout_seconds=2.0, kill_grace_seconds=2.0,
                         cgroup_parent=str(cgroup_parent), **ROOT_REASON)
    result = run(tmp_path, "sleep", lim)
    assert (result.outcome.kind, result.outcome.reason) == (INFRASTRUCTURE_FAILURE, "timeout")


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2] != "Z"


def test_escaped_daemon_is_killed_by_the_cgroup_and_cannot_migrate_out(tmp_path, cgroup_parent):
    target = cgroup_parent / "cgroup.procs"
    result = run(tmp_path, "escape", limits(cgroup_parent), env={"FAKE_CGROUP_TARGET": str(target)})
    assert result.outcome.kind == OK, result.record["outcome"]
    pid = int((result.run_dir / "escaped.pid").read_text())
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)
    migration = (result.run_dir / "migration.txt").read_text()
    if result.record["isolation"]["nsdelegate"]:
        assert migration != "moved", migration      # the cgroup namespace refuses the move (nsdelegate)
    assert result.record["process"]["leftover_processes_killed"] is True
    # killed whether or not it left the cgroup: re-parented to the runner (subreaper) and swept
    assert result.record["isolation"]["cgroup"]["kill"]["orphans_killed"] >= (1 if migration == "moved" else 0)


def test_plain_run_with_a_cgroup_kills_it_on_timeout(tmp_path, cgroup_parent):
    node = make_node(tmp_path, "sleep")
    runner = WorkingCopyRunner(TrackGNodeSpec.from_mapping(spec_mapping()), node, tmp_path / "runs")
    lim = ResourceLimits(uncapped_reason="t", measured_run_seconds=0.3, kill_grace_seconds=2.0,
                         cgroup_parent=str(cgroup_parent))
    result = runner.run("r1", lim)
    assert (result.outcome.kind, result.outcome.reason) == (DIVERGENT, "timeout")
    assert result.record["process"]["killed_with"] == "cgroup.kill"
    assert result.record["process"]["cgroup"]["removed"]


def test_node_library_protocol_refuses_bad_arrays(tmp_path):
    sys.path.insert(0, str(NODE_LIB.parent))
    try:
        import ar_isolated as iso
    finally:
        sys.path.pop(0)
    import socket
    import numpy as np
    a, b = socket.socketpair()
    ca, cb = iso.Connection(a), iso.Connection(b)
    ca.send("x", {"k": 1}, {"v": np.arange(6, dtype=np.float32).reshape(2, 3)}, [])
    op, meta, arrays, fds = cb.recv()
    assert op == "x" and meta == {"k": 1} and arrays["v"].shape == (2, 3) and float(arrays["v"].sum()) == 15.0
    with pytest.raises(iso.IsolationProtocolError, match="not allowed"):
        ca.send("x", {}, {"o": np.array([object()])}, [])
    import json as _json
    import struct
    header = _json.dumps({"op": "x", "meta": {}, "arrays": [{"name": "v", "dtype": "float32", "shape": [2],
                                                              "nbytes": 999}]}).encode()
    a.sendall(iso.MAGIC + struct.pack("<Q", len(header)) + header)
    with pytest.raises(iso.IsolationProtocolError, match="size mismatch"):
        cb.recv()
    a.close()
    b.close()


def test_node_library_copy_matches_the_master():
    # kits copy the library byte-identically; the fixture does the same
    assert NODE_LIB.is_file() and "parent_start" in NODE_LIB.read_text()


# ------------------------------------------------------------------------------------------- same user, not root

UNPRIV_SCRIPT = textwrap.dedent('''\
    import json, sys
    from autoresearch.trackg.node_template import TrackGNodeSpec
    from autoresearch.trackg.runner import ResourceLimits, WorkingCopyRunner
    sys.path.insert(0, sys.argv[4])
    from isolated_fixture import isolated_spec_mapping
    spec = TrackGNodeSpec.from_mapping(isolated_spec_mapping())
    runner = WorkingCopyRunner(spec, sys.argv[1], sys.argv[2])
    lim = ResourceLimits(uncapped_reason="t", measured_run_seconds=20.0, cgroup_parent=sys.argv[3])
    result = runner.run("u1", lim)
    print(json.dumps({"outcome": result.record["outcome"], "isolation": result.record["isolation"],
                      "log": (result.run_dir / "run.log").read_text()}))
    ''')


@pytest.mark.skipif(not ROOT, reason="needs root to drop to uid 65534 and delegate a cgroup")
def test_unprivileged_end_to_end_forgery_through_proc_fails(tmp_path, cgroup_parent):
    """A same-user worker cannot write into the parent's or the runner's pipes (non-dumpable processes), and the
    run under a delegated cgroup with a cgroup namespace is ok with the honest metric."""
    # pytest's own temp root is private to root: use a fresh directory uid 65534 can reach (removed at the end)
    work = Path(tempfile.mkdtemp(prefix="ar_unpriv_", dir=os.environ.get("AR_TEST_UNPRIV_DIR", "/tmp")))
    os.chmod(work, 0o755)
    try:
        _unprivileged_run(work, cgroup_parent)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _unprivileged_run(work, cgroup_parent):
    node = make_isolated_node(work, "forge_proc")
    runs = work / "runs"
    (work / "script.py").write_text(UNPRIV_SCRIPT)
    shutil.copyfile(Path(__file__).with_name("isolated_fixture.py"), work / "isolated_fixture.py")
    runner_cg = cgroup_parent / "runner"
    runner_cg.mkdir()
    delegated = cgroup_parent / "runs"
    delegated.mkdir()
    for path in [work, *work.rglob("*")]:
        os.chown(path, 65534, 65534)
    for cg in (cgroup_parent, runner_cg, delegated):
        os.chown(cg, 65534, 65534)
        for name in ("cgroup.procs", "cgroup.subtree_control", "cgroup.threads"):
            if (cg / name).exists():
                os.chown(cg / name, 65534, 65534)

    def drop():
        (runner_cg / "cgroup.procs").write_text("0")
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)

    src = Path(isolation.__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(src), "HOME": str(work), "TMPDIR": str(work)}
    proc = subprocess.run([sys.executable, str(work / "script.py"), str(node), str(runs), str(delegated),
                           str(work)], preexec_fn=drop, env=env, capture_output=True, text=True, timeout=120,
                          cwd=str(work))
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "forge report:" in out["log"], (out["outcome"], out["log"][-3000:])
    report = out["log"].split("forge report:")[1].splitlines()[0]
    assert ":written" not in report and "PermissionError" in report, report
    assert out["isolation"]["effective"] is True and out["isolation"]["cgroup_namespace"] is True
    assert out["outcome"]["kind"] == "ok" and out["outcome"]["value"] == pytest.approx(1.0), out["outcome"]
