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


def host_waivers(extra=None) -> dict:
    """Waivers this test host needs: root (the container runs as root), ptrace_scope (no Yama here)."""
    out = {}
    if ROOT:
        out["root"] = "test container runs as root"
    scope = isolation.ptrace_scope()
    if scope is None or scope < 1:
        out["ptrace_scope"] = f"test host ptrace_scope={scope}"
    out.update(extra or {})
    return out


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


def limits(cgroup_parent=None, waive=None, **kw):
    extra = dict(waive or {})
    base = dict(uncapped_reason="test fixture", measured_run_seconds=20.0, kill_grace_seconds=3.0)
    if cgroup_parent is None:
        extra["no_cgroup"] = "test without cgroup"
    else:
        base["cgroup_parent"] = str(cgroup_parent)
    base["isolation_waivers"] = host_waivers(extra)
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
    assert spec.isolation.entry == "parent.py" and spec.isolation.argv("py", 5) == ["py", "-E", "-s", "parent.py", "5"]
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


def test_isolated_run_refuses_each_missing_host_condition_without_a_waiver(tmp_path):
    node = make_isolated_node(tmp_path)
    runner = WorkingCopyRunner(TrackGNodeSpec.from_mapping(isolated_spec_mapping()), node, tmp_path / "runs")
    with pytest.raises(RunnerError, match="cgroup_parent"):
        runner.run("a", ResourceLimits(uncapped_reason="t", measured_run_seconds=5, isolation_waivers=host_waivers()))
    if ROOT:
        with pytest.raises(RunnerError, match="as root"):
            runner.run("b", ResourceLimits(uncapped_reason="t", measured_run_seconds=5,
                                           isolation_waivers={"no_cgroup": "t", "ptrace_scope": "t"}))
    scope = isolation.ptrace_scope()
    if scope is None or scope < 1:
        with pytest.raises(RunnerError, match="ptrace_scope"):
            runner.run("c", ResourceLimits(uncapped_reason="t", measured_run_seconds=5,
                                           isolation_waivers={"no_cgroup": "t", "root": "t"}))
    with pytest.raises(RunnerError, match="isolation_waivers"):
        ResourceLimits(uncapped_reason="t", measured_run_seconds=5, isolation_waivers={"everything": "x"})
    with pytest.raises(RunnerError, match="timeout_factor"):
        ResourceLimits(uncapped_reason="t", measured_run_seconds=5, timeout_factor=2.0)
    ResourceLimits(uncapped_reason="t", measured_run_seconds=5, timeout_factor=2.0,
                   isolation_waivers={"timeout_factor": "test"})


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
    assert rec["command"] == ["python3", "-E", "-s", "parent.py", "5"] and iso["argv"] == rec["command"]
    assert iso["log_intact"] and iso["frozen_files_checked"] == 3 and not iso["frozen_files_changed"]
    assert iso["runner_dumpable"] is False and iso["effective"] is False   # waivers on this test host
    assert iso["worker_sandbox"]["writable"] == [str(result.work_dir.resolve())] and "_verified_log" not in iso
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
    ("tamper_log", "nonzero_exit"),          # the run directory is read-only in the sandbox: the write fails
    ("frozen", "frozen_file_changed"),       # the working copy is writable, and checked after the run
    ("crash", "nonzero_exit"),
    ("nan", "non_finite_loss"),
])
def test_forged_or_broken_isolated_runs_are_divergent(tmp_path, mode, reason):
    result = run(tmp_path, mode)
    assert (result.outcome.kind, result.outcome.reason) == (DIVERGENT, reason), result.record["outcome"]


def test_without_the_sandbox_a_tampered_log_is_refused(tmp_path):
    result = run(tmp_path, "tamper_log", limits(waive={"no_worker_sandbox": "test"}))
    assert (result.outcome.kind, result.outcome.reason) == (DIVERGENT, "log_tampered"), result.record["outcome"]
    assert result.record["isolation"]["worker_sandbox"] is None and result.record["isolation"]["effective"] is False


def test_the_sandbox_confines_the_worker(tmp_path, monkeypatch):
    """Review round 1 (B1-B4, M3): the worker is PID 1 of its own namespace with no capability and no network;
    the run directory, the kit and the home directory are read-only, the working copy is writable; it cannot
    signal the runner."""
    kit = tmp_path / "kit_marker"
    monkeypatch.setenv("FAKE_RUNNER_PID", str(os.getpid()))
    monkeypatch.setenv("FAKE_REAL_HOME", os.environ.get("HOME", "/root"))
    node = make_isolated_node(tmp_path / "sb", "sandbox_report")
    monkeypatch.setenv("FAKE_KIT", str(node))
    spec = TrackGNodeSpec.from_mapping(isolated_spec_mapping())
    result = WorkingCopyRunner(spec, node, tmp_path / "sb" / "runs").run("r1", limits())
    assert result.outcome.kind == OK, result.record["outcome"]
    log = (result.run_dir / "run.log").read_text()
    report = json.loads(log.split("sandbox report:")[1].splitlines()[0])
    assert report["pid"] == 1 and report["ppid"] == 0 and report["procs"] <= 3, report
    assert report["run_log"] != "writable" and report["kit"] != "writable" and report["work"] == "writable", report
    assert report["home"] != "writable" and report["net"] != "connected", report
    assert report["FAKE_RUNNER_PID"] != "signal_ok" and report["caps"] == "0000000000000000", report
    assert report["setuid"] != "ok" or report["uid"] == 0
    del kit


def test_a_tampered_kit_or_incumbent_is_an_integrity_stop(tmp_path):
    """Review round 1, B1: the working copy's frozen files must match the pin taken from the kit."""
    from autoresearch.trackg.runner import IntegrityError
    node = make_isolated_node(tmp_path)
    runner = WorkingCopyRunner(TrackGNodeSpec.from_mapping(isolated_spec_mapping()), node, tmp_path / "runs")
    (node / "parent.py").write_text((node / "parent.py").read_text() + "# changed outside a diff\n")
    with pytest.raises(IntegrityError, match="parent.py"):
        runner.run("r1", limits())
    fresh = WorkingCopyRunner(TrackGNodeSpec.from_mapping(isolated_spec_mapping()), node, tmp_path / "runs2")
    with pytest.raises(IntegrityError, match="code state"):
        fresh.run("r2", limits(), expected_incumbent_state="0" * 64)


def test_elapsed_time_beyond_the_timeout_is_a_timeout_even_after_a_normal_exit(tmp_path):
    """Review round 1, B4: a run that ends by itself after the deadline (here the runner is stopped meanwhile by an
    outside helper, as a worker could once do) is a timeout: the runner's monotonic clock keeps running."""
    node = make_isolated_node(tmp_path)
    runner = WorkingCopyRunner(TrackGNodeSpec.from_mapping(isolated_spec_mapping()), node, tmp_path / "runs")
    helper = subprocess.Popen(["/bin/sh", "-c", f"sleep 0.3; kill -STOP {os.getpid()}; sleep 4; kill -CONT {os.getpid()}"])
    try:
        result = runner.run("r1", limits(measured_run_seconds=1.0))
    finally:
        helper.wait()
    assert (result.outcome.kind, result.outcome.reason) == (DIVERGENT, "timeout"), result.record["outcome"]


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
    assert rec["isolation"]["worker_sandbox"] is not None


def test_provisional_timeout_stays_infrastructure(tmp_path, cgroup_parent):
    lim = ResourceLimits(uncapped_reason="t", provisional_timeout_seconds=2.0, kill_grace_seconds=2.0,
                         cgroup_parent=str(cgroup_parent), isolation_waivers=host_waivers())
    result = run(tmp_path, "sleep", lim)
    assert (result.outcome.kind, result.outcome.reason) == (INFRASTRUCTURE_FAILURE, "timeout")


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2] != "Z"


def _processes_with_arg(token: str) -> list[int]:
    out = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if token.encode() in (entry / "cmdline").read_bytes().split(b"\0"):
                    out.append(int(entry.name))
            except OSError:
                pass
    return [pid for pid in out if _alive(pid)]


def test_escaped_daemon_is_killed_whether_or_not_it_leaves_the_cgroup(tmp_path, cgroup_parent):
    """Review round 1, M2: a daemon that double-forks, calls setsid and tries to move to another cgroup dies with
    the run: the worker's PID namespace is torn down when its PID 1 ends (and the cgroup is killed)."""
    token = f"{300 + (os.getpid() % 97)}.{time.monotonic_ns() % 10**6}"
    target = cgroup_parent / "cgroup.procs"
    result = run(tmp_path, "escape", limits(cgroup_parent),
                 env={"FAKE_CGROUP_TARGET": str(target), "FAKE_ESCAPE_TOKEN": token})
    assert result.outcome.kind == OK, result.record["outcome"]
    assert "escape migration=" in (result.run_dir / "run.log").read_text()
    deadline = time.monotonic() + 5
    while _processes_with_arg(token) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _processes_with_arg(token)


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


def test_node_library_needs_only_the_standard_library_and_numpy():
    """Kits copy the library into nodes whose environments do not have the harness: no harness import."""
    import ast
    tree = ast.parse(NODE_LIB.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names - {"numpy", "torch", "__future__"} <= set(sys.stdlib_module_names), names


def test_shared_buffer_is_sealed_against_resizing():
    """Review round 1, M1: the worker gets the memfd; it must not shrink it under the parent's mapping (SIGBUS)."""
    sys.path.insert(0, str(NODE_LIB.parent))
    try:
        import ar_isolated as iso
    finally:
        sys.path.pop(0)
    buf = iso.SharedBuffer((8,), "float32")
    with pytest.raises(PermissionError):
        os.ftruncate(buf.fd, 0)
    buf.close()


# ------------------------------------------------------------------------------------------- same user, not root

UNPRIV_SCRIPT = textwrap.dedent('''\
    import json, sys
    from autoresearch.trackg.node_template import TrackGNodeSpec
    from autoresearch.trackg.runner import ResourceLimits, WorkingCopyRunner
    sys.path.insert(0, sys.argv[4])
    from isolated_fixture import isolated_spec_mapping
    spec = TrackGNodeSpec.from_mapping(isolated_spec_mapping())
    runner = WorkingCopyRunner(spec, sys.argv[1], sys.argv[2])
    from autoresearch.trackg import isolation
    scope = isolation.ptrace_scope()
    waivers = {} if scope is not None and scope >= 1 else {"ptrace_scope": f"test host ptrace_scope={scope}"}
    lim = ResourceLimits(uncapped_reason="t", measured_run_seconds=20.0, cgroup_parent=sys.argv[3],
                         isolation_waivers=waivers)
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
    assert ":written" not in report, report
    assert out["isolation"]["euid"] == 65534 and out["isolation"]["worker_sandbox"] is not None
    assert set(out["isolation"]["waivers"]) <= {"ptrace_scope"}
    assert out["outcome"]["kind"] == "ok" and out["outcome"]["value"] == pytest.approx(1.0), out["outcome"]
