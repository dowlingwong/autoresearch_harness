"""Run isolation for Track G nodes (R1-HARNESS-ISOLATION; AI-assisted, Claude).

PLAN ``R1-HARNESS-ISOLATION``: before any T1 label run and any RL training, the editable code runs in a
separate process; the metric log is append-only; a timeout kills the run's whole cgroup; a run with more
than one metric line is refused.  The threat it answers (Track G port kits, READY 56-58): editable code
that runs in the frozen wrapper's process can forge the metric line.

How a node with an ``isolation`` block (``trackg_node/v1``) is run by :class:`WorkingCopyRunner`:

* **No shell; a frozen parent.**  The runner starts ``[python, -s, entry, *args]`` itself (no user site
  directory; the environment keeps the determinism settings).  ``entry`` is a frozen file (the *parent*) that never imports editable
  code; it starts the editable code in a *worker* process (node library ``ar_isolated.py``) and computes the
  metric from what the worker returns.
* **Worker sandbox.**  The worker runs in its own session and in new user, mount, PID, network and cgroup
  namespaces, inside a new root file system that holds only an allow-list (``ar_isolated``'s module docstring,
  review round 2): system directories, the run interpreter, the data and the shim read-only; its working copy
  writable except the frozen files; a private ``/tmp``; ``/sys`` without any cgroup file system; ``/dev``.  It is
  PID 1 of its own PID namespace with a fresh ``/proc``, no capability and ``no_new_privs``, and has no
  network.  So it cannot see or signal the runner or the parent, cannot read or write the runs tree (other
  runs' result channels), the incumbent, the kit, the home directory or the Python environment, cannot connect
  to a daemon's socket outside its root, and when the run ends the kernel kills every process of its PID
  namespace.  The runner checks before the run that this
  host allows the sandbox (unprivileged user namespaces with mount, PID and network namespaces); without it the
  run is refused unless the limits name a waiver (``no_worker_sandbox``), which the record shows.
* **Result channel.**  The runner listens on a Unix socket with a random abstract name (``AR_RESULT_SOCKET``,
  visible in ``/proc/net/unix`` of the runner's network namespace, not the worker's) and accepts one
  connection, only from the parent's own pid (``SO_PEERCRED``); the parent connects after it made itself
  non-dumpable, and the worker never inherits the connection.  The metric (and, for test-scored nodes, the
  test metric) is taken **only** from this channel.  The run is refused (``divergent``) unless the channel
  holds exactly one metric line (and at most one test-metric line), and if any line of the run's output
  (``run.log``) matches the metric regex.
* **Non-dumpable processes.**  The runner marks itself non-dumpable (``prctl(PR_SET_DUMPABLE, 0)``) before it
  starts an isolated run, and the parent does so when it imports ``ar_isolated`` (before NumPy); then
  ``/proc/<pid>/fd``, ``pidfd_getfd`` and ``ptrace`` of these processes need ``CAP_SYS_PTRACE`` in their user
  namespace (the parent stays in the runner's).  Two conditions: the run must not run as root (refused unless
  waived: ``root``), and Yama's ``ptrace_scope`` must be at least 1, so that an unrelated process of the same
  user cannot attach to the parent during its interpreter start-up, before the flag is set (refused unless
  waived: ``ptrace_scope``).  At the accept the runner also requires the parent's ``TracerPid`` to be 0 and the parent to be
  non-dumpable (its ``/proc/<pid>/status`` owned by root; not checkable as root); otherwise the record is written
  and the runner raises :class:`IntegrityError` (STOP): a process outside the run interfered.
* **Append-only log.**  The run's standard output and error go to a pipe the runner reads; the runner alone
  appends to ``run.log`` (``O_APPEND``) and keeps a running sha256.  A run whose log file differs from what
  the runner wrote is ``divergent`` (``log_tampered``); the classification uses the verified bytes.  The
  runner's own last line records the result channel's sha256.
* **cgroup.**  The run's processes join a fresh cgroup v2 directory under ``limits.cgroup_parent`` (joined in
  the child before ``exec``).  On a timeout and after every normal exit the runner writes ``1`` to
  ``cgroup.kill`` (falling back to SIGKILL of every listed pid until the cgroup is empty), then removes the
  directory; the worker's cgroup namespace is rooted in the run cgroup.  The runner is also a child subreaper
  (``PR_SET_CHILD_SUBREAPER``) and kills every child it did not have before the run, with its descendants.
  A run without a cgroup is refused unless waived (``no_cgroup``).
* **Timeout.**  A run whose wall time (the runner's monotonic clock) exceeds the timeout counts as timed out
  even if it ended by itself (a stopped runner's clock keeps running; so does a suspension of the whole host or
  batch job, which then also counts as the proposal's timeout), and R1-TRACKG-PORT-DECISIONS (4)
  requires the factor 3 for a measured timeout (other factors need the ``timeout_factor`` waiver).
* **Frozen files and incumbent pinned.**  The sha256 of every regular file under the frozen paths (symbolic
  links by target) of the working copy must equal the pin taken from the node kit when the runner (or the
  campaign) was set up, before the run, and must be unchanged after it (``frozen_file_changed``); a campaign
  also passes the incumbent's code-state hash, which the copy must match.  A mismatch before the run is an
  integrity STOP (:class:`IntegrityError`), not a proposal outcome.  The pin is taken from the kit as it is when
  the runner or campaign starts; every run record carries its hash (``frozen_pin_sha256``), which the census
  should commit, so a kit changed before a later campaign is caught by comparing it.

Not closed here (stated in the record's ``isolation.limits`` and the kits' threat model): the worker can read
every allow-listed path: its working copy (the kit's files included) and the data, so the validation and test
labels wherever a node keeps them there (a read-only mount does not hide them); it computes the training itself, so the budget accounting and the loss lines stay in the
editable process; it can answer the parent's evaluation requests differently from a fixed model (the kits'
probes narrow this); a node that sets ``shared_dev_shm`` (needed for torch's CUDA IPC reference counters) gives the
worker the host's ``/dev/shm``, through which it can keep state between runs or reach concurrent runs;
the parent's environment and Python installation are trusted as the run's uid left them before the campaign
(the worker cannot write them; another process of that uid outside the run could); CUDA IPC across the PID
namespace is exercised by the kits' GPU smoke only; its output
can contain an infrastructure log pattern, which makes the run an ``infrastructure_failure`` (rerun, not
registered as evaluated) -- an RL reward must not treat that class as neutral without the author's decision.
"""
from __future__ import annotations

import ctypes
import errno
import json
import hashlib
import os
import re
import secrets
import selectors
import signal
import socket
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PR_SET_DUMPABLE = 4
PR_GET_DUMPABLE = 3
PR_SET_CHILD_SUBREAPER = 36
RESULT_SOCKET_ENV = "AR_RESULT_SOCKET"
SANDBOX_ENV = "AR_ISOLATION_SANDBOX"
RESULT_CHANNEL_MAX_BYTES = 64 * 1024
_CGROUP_NAME_RE = re.compile(r"[^A-Za-z0-9_.\-]")


class IsolationError(RuntimeError):
    """Raised when an isolated run cannot be set up (before anything runs)."""


# ---------------------------------------------------------------------------------------------- dumpable


def _prctl(option: int, arg: int = 0) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(ctypes.c_int(option), ctypes.c_ulong(arg), ctypes.c_ulong(0), ctypes.c_ulong(0),
                        ctypes.c_ulong(0))
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result


def set_non_dumpable() -> None:
    """Mark this process non-dumpable: ``/proc/<pid>/fd``, ``/proc/<pid>/mem`` and ``ptrace`` then need
    ``CAP_SYS_PTRACE`` (only for a process that does not run as root)."""
    _prctl(PR_SET_DUMPABLE, 0)


def is_dumpable() -> bool:
    return bool(_prctl(PR_GET_DUMPABLE))


def set_child_subreaper() -> None:
    _prctl(PR_SET_CHILD_SUBREAPER, 1)


def _proc_table() -> dict[int, tuple[int, str]]:
    """pid -> (ppid, state) from /proc."""
    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rfind(")") + 2:].split()
        if len(fields) > 1:
            table[int(entry.name)] = (int(fields[1]), fields[0])
    return table


def children_of(pid: int) -> set[int]:
    return {p for p, (pp, _state) in _proc_table().items() if pp == pid}


def kill_new_orphans(before: set[int], exclude: set[int], rounds: int = 50) -> int:
    """Kill every child of this process that is not in *before* or *exclude*, with all its descendants, until
    none is left; reap them.  Returns how many processes were killed."""
    me = os.getpid()
    killed: set[int] = set()
    for _ in range(rounds):
        table = _proc_table()
        roots = {p for p, (pp, st) in table.items() if pp == me and p not in before and p not in exclude}
        live_roots = {p for p in roots if table[p][1] != "Z"}
        tree, frontier = set(live_roots), set(live_roots)
        while frontier:
            frontier = {p for p, (pp, st) in table.items() if pp in frontier and p not in tree}
            tree |= frontier
        for pid in tree:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.add(pid)
            except ProcessLookupError:
                pass
        for pid in roots:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        if not live_roots:
            break
        time.sleep(0.02)
    return len(killed)


def cgroup2_nsdelegate(path: str | Path) -> bool | None:
    """Whether the cgroup2 mount that holds *path* has the ``nsdelegate`` option (``None``: not found)."""
    target = str(Path(path).resolve())
    best, opts = "", None
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        left, _, right = line.partition(" - ")
        parts, rparts = left.split(), right.split()
        if len(parts) < 6 or not rparts or rparts[0] != "cgroup2":
            continue
        mount_point = parts[4]
        if (target == mount_point or target.startswith(mount_point.rstrip("/") + "/")) and len(mount_point) > len(best):
            best = mount_point
            opts = set(parts[5].split(",")) | set((rparts[2] if len(rparts) > 2 else "").split(","))
    return None if opts is None else "nsdelegate" in opts


# ---------------------------------------------------------------------------------------------- frozen files


def frozen_file_hashes(work: Path, frozen_paths: tuple[str, ...]) -> dict[str, str]:
    """sha256 of every regular file under the frozen paths of *work*; symbolic links are recorded by their
    target (``link:<target>``) and not followed (the ``data`` link points outside the node)."""
    out: dict[str, str] = {}
    work = Path(work)

    def visit(path: Path) -> None:
        rel = path.relative_to(work).as_posix()
        if path.is_symlink():
            out[rel] = "link:" + os.readlink(path)
        elif path.is_dir():
            for child in sorted(path.iterdir()):
                if child.name == "__pycache__":
                    continue
                visit(child)
        elif path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            out[rel] = digest.hexdigest()

    for entry in frozen_paths:
        target = work / entry.rstrip("/").removesuffix("/*")
        if target.exists() or target.is_symlink():
            visit(target)
    return out


# ---------------------------------------------------------------------------------------------- cgroup


@dataclass
class RunCgroup:
    """A cgroup v2 directory for one run."""

    path: Path
    migration_possible: bool

    @classmethod
    def create(cls, parent: str | Path, run_id: str) -> "RunCgroup":
        parent = Path(parent)
        if not (parent / "cgroup.procs").is_file():
            raise IsolationError(f"cgroup parent {parent} is not a cgroup directory")
        if not (parent / "cgroup.controllers").is_file():
            raise IsolationError(f"cgroup parent {parent} is not cgroup v2")
        name = "ar_" + _CGROUP_NAME_RE.sub("_", run_id)[:80] + f"_{os.getpid()}_{time.monotonic_ns() % 10**9}"
        path = parent / name
        try:
            path.mkdir()
        except OSError as error:
            raise IsolationError(f"cannot create the run cgroup {path}: {error}") from error
        if not (path / "cgroup.kill").is_file():
            os.rmdir(path)
            raise IsolationError(f"{path} has no cgroup.kill (cgroup v2 with kernel >= 5.14 needed)")
        if not os.access(path / "cgroup.procs", os.W_OK):
            os.rmdir(path)
            raise IsolationError(f"{path}/cgroup.procs is not writable")
        # a process in the run cgroup could move itself to a sibling if it can write the parent's procs file
        return cls(path=path, migration_possible=os.access(parent / "cgroup.procs", os.W_OK))

    def procs(self) -> list[int]:
        try:
            text = (self.path / "cgroup.procs").read_text()
        except OSError:
            return []
        return [int(x) for x in text.split()]

    def populated(self) -> bool:
        try:
            for line in (self.path / "cgroup.events").read_text().splitlines():
                key, _, value = line.partition(" ")
                if key == "populated":
                    return value.strip() == "1"
        except OSError:
            pass
        return bool(self.procs())

    def kill(self, grace: float = 10.0) -> dict[str, Any]:
        """Kill every process in the cgroup; wait until it is empty (at most *grace* seconds)."""
        before = self.procs()
        method = "cgroup.kill"
        try:
            (self.path / "cgroup.kill").write_text("1")
        except OSError:
            method = "sigkill_each"
        deadline = time.monotonic() + grace
        while self.populated() and time.monotonic() < deadline:
            for pid in self.procs():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.02)
        return {"method": method, "processes_before": len(before), "empty_after": not self.populated()}

    def remove(self) -> bool:
        for _ in range(100):
            try:
                os.rmdir(self.path)
                return True
            except OSError as error:
                if error.errno not in (errno.EBUSY, errno.ENOTEMPTY):
                    return False
                time.sleep(0.02)
        return False


CLONE_NEWUSER = 0x10000000
CLONE_NEWCGROUP = 0x02000000


def _unshare(flags: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(ctypes.c_int(flags)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"unshare: {os.strerror(err)}")


def _write_proc(name: str, text: str) -> None:
    fd = os.open(f"/proc/self/{name}", os.O_WRONLY)
    try:
        os.write(fd, text.encode("ascii"))
    finally:
        os.close(fd)


def join_cgroup_preexec(cgroup: RunCgroup):
    """Child setup (between fork and exec): join *cgroup* (the parent stays in the runner's namespaces)."""
    procs = str(cgroup.path / "cgroup.procs")

    def join() -> None:
        try:
            fd = os.open(procs, os.O_WRONLY)
            try:
                os.write(fd, b"0")
            finally:
                os.close(fd)
        except OSError as error:
            raise OSError(error.errno, f"join {procs}: {error.strerror}") from None

    return join


NODE_LIB = Path(__file__).resolve().parent / "node_lib" / "ar_isolated.py"
_SANDBOX_PROBE = 'import importlib.util, json, os, sys\nspec = importlib.util.spec_from_file_location("ar_isolated", sys.argv[1])\niso = importlib.util.module_from_spec(spec); spec.loader.exec_module(iso)\nw = iso.Worker(os.path.join(sys.argv[2], "probe_worker.py"), os.path.dirname(sys.argv[1]), cwd=sys.argv[2])\nmeta, _ = w.call("ping")\nw.close()\nprint(json.dumps(meta))\n'
_SANDBOX_PROBE_WORKER = 'import sys\nsys.path.insert(0, sys.argv[1])\nimport ar_isolated as iso\nc = iso.worker_connect()\nimport os\niso.serve({"ping": lambda m, a, f: ({"pid": os.getpid(), "uid": os.getuid()}, {})}, c)\n'


def worker_sandbox_available(interpreter: str, config: dict, probe_dir: Path) -> tuple[bool, str]:
    """Whether the run interpreter can start a worker in the sandbox *config* on this host: a short real worker
    from *probe_dir* (a new directory the runner owns, readable inside the sandbox)."""
    probe_dir = Path(probe_dir)
    files = probe_dir / "files"
    files.mkdir(parents=True)
    (files / "probe_worker.py").write_text(_SANDBOX_PROBE_WORKER)
    (probe_dir / "root").mkdir()
    probe_config = {**config, "root": str(probe_dir / "root"), "rw": [],
                    "ro": [*config.get("ro", []), str(files), str(NODE_LIB.parent)], "ro_in_rw": []}
    env = dict(os.environ)
    env[SANDBOX_ENV] = json.dumps(probe_config)
    try:
        proc = subprocess.run([interpreter, "-s", "-c", _SANDBOX_PROBE, str(NODE_LIB), str(files)],
                              capture_output=True, text=True, env=env, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        return False, repr(error)
    if proc.returncode != 0:
        text = proc.stdout + proc.stderr
        errors = [line for line in text.splitlines() if "AR_ERROR" in line]
        return False, " | ".join(errors[:3]) or text[-600:]
    try:
        meta = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, proc.stdout[-300:]
    if meta.get("pid") != 1:
        return False, f"the probe worker is not PID 1 of its namespace: {meta}"
    return True, "ok"


def ptrace_scope() -> int | None:
    try:
        return int(Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip())
    except (OSError, ValueError):
        return None


def non_dumpable(pid: int) -> bool:
    """Whether process *pid* is non-dumpable: its ``/proc/<pid>/status`` then belongs to root.  As root this cannot
    be told apart (every such file is root's): ``True`` then."""
    if os.getuid() == 0:
        return True
    try:
        return os.stat(f"/proc/{pid}/status").st_uid == 0
    except OSError:
        return False


def tracer_pid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("TracerPid:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


# ---------------------------------------------------------------------------------------------- the run


@dataclass
class IsolatedExit:
    returncode: int | None
    timed_out: bool
    launch_error: str | None
    log_sha256_written: str
    log_bytes_written: int
    result_text: str
    result_overflow: bool
    kill: dict[str, Any] = field(default_factory=dict)


def _peer_pid(conn: socket.socket) -> int:
    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return struct.unpack("3i", creds)[0]


def execute(argv: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, timeout: float,
            cgroup: Any, preexec, drain_grace: float = 10.0, sweep=None) -> IsolatedExit:
    """Start *argv*, accept its result channel, pump its output into *log_path* (append-only, hashed), enforce
    *timeout* by killing the cgroup, and kill the cgroup after a normal exit too.  *sweep(pid)* (the plain
    runner's process-group and session sweep) runs after each cgroup kill and returns whether it killed
    anything.

    The result channel is a Unix socket in the abstract namespace with a random name (``AR_RESULT_SOCKET``);
    the runner accepts exactly one connection, and only from the parent process itself (its pid, checked
    with ``SO_PEERCRED``); the parent connects after it made itself non-dumpable, so no process ever holds the
    channel while it can be reached through ``/proc``.  Other connections are closed and counted."""
    out_r, out_w = os.pipe()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
    name = "\0ar_result_" + secrets.token_hex(16)
    listener.bind(name)
    listener.listen(8)
    listener.setblocking(False)
    env = dict(env)
    env[RESULT_SOCKET_ENV] = name[1:]
    set_child_subreaper()
    children_before = children_of(os.getpid())
    hasher = hashlib.sha256()
    written = 0
    result = bytearray()
    overflow = False
    rejected = 0
    channel_error = None
    parent_traced = 0
    parent_dumpable = False
    started = time.monotonic()
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o644)

    def child_setup() -> None:
        try:
            preexec()
        except BaseException as error:  # noqa: BLE001 - subprocess reports only "Exception occurred in preexec_fn"
            os.write(2, f"AR_RUNNER child setup failed: {error!r}\n".encode("utf-8", "replace"))
            raise

    channel: socket.socket | None = None
    try:
        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out_w,
                                    stderr=out_w, start_new_session=True, preexec_fn=child_setup)
        except (OSError, subprocess.SubprocessError) as error:
            os.close(out_w)
            os.set_blocking(out_r, False)
            try:
                chunk = os.read(out_r, 1 << 16)      # the child setup's message, if any
            except BlockingIOError:
                chunk = b""
            if chunk:
                os.write(log_fd, chunk)
                hasher.update(chunk)
                written += len(chunk)
            os.close(out_r)
            return IsolatedExit(None, False, repr(error), hasher.hexdigest(), written, "", False,
                                {"method": None, "processes_before": 0, "empty_after": True})
        os.close(out_w)
        sel = selectors.DefaultSelector()
        sel.register(out_r, selectors.EVENT_READ, "out")
        sel.register(listener, selectors.EVENT_READ, "listen")
        out_open = True
        deadline = time.monotonic() + timeout
        timed_out = False
        exited = False
        drain_deadline = None
        kill_info: dict[str, Any] = {}
        while True:
            now = time.monotonic()
            if not exited and proc.poll() is not None:
                exited = True
                kill_info = cgroup.kill()      # leftovers after a normal exit
                kill_info["session_sweep_killed"] = bool(sweep(proc.pid)) if sweep else None
                drain_deadline = time.monotonic() + drain_grace
            if not exited and now >= deadline:
                timed_out = True
                kill_info = cgroup.kill()
                kill_info["session_sweep_killed"] = bool(sweep(proc.pid)) if sweep else None
                proc.wait()
                exited = True
                drain_deadline = time.monotonic() + drain_grace
            if exited and not out_open and channel is None:
                break
            if drain_deadline is not None and now >= drain_deadline:
                break                          # a writer outside the cgroup keeps a pipe open: stop reading
            wait = 0.2 if not exited else max(0.0, min(0.2, drain_deadline - now))
            for key, _ in sel.select(timeout=wait):
                if key.data == "listen":
                    try:
                        conn, _addr = listener.accept()
                    except (BlockingIOError, InterruptedError):
                        continue
                    if channel is None and not exited and _peer_pid(conn) == proc.pid:
                        conn.setblocking(False)
                        channel = conn
                        sel.register(conn, selectors.EVENT_READ, "res")
                        traced = tracer_pid(proc.pid)   # the parent is non-dumpable now: no new tracer can attach
                        if traced:
                            parent_traced = traced
                        if not non_dumpable(proc.pid):
                            parent_dumpable = True
                    else:
                        rejected += 1
                        conn.close()
                    continue
                if key.data == "out":
                    chunk = os.read(out_r, 1 << 16)
                    if not chunk:
                        sel.unregister(out_r)
                        os.close(out_r)
                        out_open = False
                        continue
                    os.write(log_fd, chunk)
                    hasher.update(chunk)
                    written += len(chunk)
                    continue
                try:
                    chunk = channel.recv(1 << 16)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError as error:       # e.g. reset: the parent closed with unread data sent to it
                    channel_error = repr(error)
                    chunk = b""
                if not chunk:
                    sel.unregister(channel)
                    channel.close()
                    channel = None
                    continue
                if len(result) + len(chunk) <= RESULT_CHANNEL_MAX_BYTES:
                    result.extend(chunk)
                else:
                    overflow = True
        if not exited:
            proc.wait()
        sel.close()
        if out_open:
            os.close(out_r)
        if not kill_info.get("empty_after", True) or cgroup.populated():
            kill_info = {**kill_info, **{"second": cgroup.kill()}}
        kill_info["orphans_killed"] = kill_new_orphans(children_before, {proc.pid})
        kill_info["rejected_result_connections"] = rejected
        kill_info["result_channel_error"] = channel_error
        kill_info["parent_traced_by"] = parent_traced or None
        kill_info["parent_dumpable_at_accept"] = parent_dumpable
        elapsed = time.monotonic() - started
        if not timed_out and elapsed > timeout:
            timed_out = True                   # e.g. the runner was stopped (SIGSTOP) while the run went on
            kill_info["timed_out_by_elapsed"] = round(elapsed, 3)
        return IsolatedExit(proc.returncode, timed_out, None, hasher.hexdigest(), written,
                            bytes(result).decode("utf-8", errors="replace"), overflow, kill_info)
    finally:
        os.close(log_fd)
        if channel is not None:
            channel.close()
        listener.close()


def verify_log(log_path: Path, sha256: str, size: int) -> bytes | None:
    """The bytes of *log_path* if they are exactly those the runner wrote, else ``None``."""
    try:
        fd = os.open(log_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    return data if len(data) == size and hashlib.sha256(data).hexdigest() == sha256 else None


def append_runner_line(log_path: Path, line: str) -> None:
    fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    try:
        os.write(fd, ("\n" + line + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def limits_text() -> list[str]:
    return [
        "the worker can read every allow-listed path: its working copy (the kit's files included), the data "
        "(validation and test labels included, where a node keeps them there), the interpreter and system directories",
        "budget accounting and loss lines stay in the worker (editable) process",
        "a worker can answer the parent's evaluation requests differently from a fixed model (kit probes only)",
        "with shared_dev_shm (CUDA IPC), /dev/shm is shared with the worker and can carry state between runs",
        "the Python installation, the data, the harness checkout and the kit before the pin are trusted as the uid "
        "left them (the worker cannot write them; another process of that uid can; frozen_pin_sha256 should be "
        "committed by the census)",
        "/dev is the host's (a private /dev/shm unless shared_dev_shm); CUDA IPC across the PID namespace is "
        "exercised only by the kits' GPU smoke",
        "an infrastructure log pattern printed by the worker makes the run an infrastructure_failure",
        "the guarantees hold only with the worker sandbox, a non-root uid, a cgroup and ptrace_scope >= 1 "
        "(the record lists any waiver)",
    ]
