"""Frozen node library: parent/worker split of an isolated Track G node (R1-HARNESS-ISOLATION).

FROZEN: copied byte-identically into every isolated node and listed in its ``frozen_paths``; the master copy
is ``autoresearch/trackg/node_lib/ar_isolated.py`` in the harness (the port tests compare the copies).
Standard library and NumPy only (imported on first use); ``torch`` is imported lazily by the CUDA helpers.

The runner starts the node's frozen **parent** script directly (no shell, ``python -s``); the name of the
runner's result socket is in ``AR_RESULT_SOCKET``.  Importing this module in that process makes it
non-dumpable at once (before NumPy is imported), so another process of the same user can neither open its file
descriptors through ``/proc`` nor ``ptrace`` it from then on.  The parent:

1. calls :func:`parent_start` as its first statement: it connects to the result socket (the runner accepts only
   the parent's pid) with a close-on-exec socket, and ``AR_RESULT_SOCKET`` is removed from the environment;
2. imports everything it needs (it never imports editable code);
3. starts the editable code with :class:`Worker` -- a separate process running a frozen worker script that
   calls :func:`worker_connect` and then :func:`serve`; the worker inherits only one end of a socket pair (the
   result socket is close-on-exec, and closed in a forked child before anything else).  The parent's script directory is removed from
   its ``sys.path`` first, so nothing it imports later can come from the working copy;
4. drives the worker with requests (:meth:`Worker.call`) and computes the metric from the arrays it returns;
5. writes the metric line with :meth:`ResultChannel.write` (the runner refuses a run whose channel holds more
   or fewer than one metric line, or whose output holds a metric line), and exits.

**Worker sandbox** (when the runner sets ``AR_ISOLATION_SANDBOX``, a JSON object): the worker is started through a
launcher -- a fresh, single-threaded ``python -I -S`` running this file -- in its own session and in new user,
mount, PID, network and cgroup namespaces.  (The parent may already run thread pools, whose fork handlers would
start threads in a forked child, and ``unshare(CLONE_NEWUSER)`` refuses a multithreaded caller.)  The launcher
maps the run's uid and gid to 0 in the new user namespace, makes the mount tree private and builds a new root
file system that holds only an allow-list: a private ``/tmp``; read-only binds of the ``ro`` paths (system
directories, the run interpreter's installation and ``sys.path``, the data, the determinism shim); ``/sys``
read-only with an empty read-only ``tmpfs`` over ``/sys/fs/cgroup`` and every cgroup file system below it; the
``rw`` paths (the run's working copy) writable, with the ``ro_in_rw`` paths (its frozen files) read-only again;
``/dev``, with a private ``/dev/shm`` unless the node sets ``shared_dev_shm`` (torch's CUDA IPC reference
counters live there); and the host ``/proc`` for the moment.  It then enters that root with ``pivot_root``,
detaches the old one and makes the new root read-only.  Nothing else exists for the worker: no home directory,
no runs tree (other runs' results), no ``/run``, no socket of a daemon outside.  The launcher then forks PID 1 of
the new PID namespace, which mounts a fresh ``/proc`` over the host one, drops every capability (bounding set,
ambient set, permitted / effective / inheritable sets, locked secure bits), sets ``no_new_privs`` and execs the
worker; the launcher only waits and passes on the exit code.  The worker's environment is an allow-list
(``AR_*``, CUDA, threading and framework variables, ``PATH``, ``PYTHONPATH``, ``PYTHONHASHSEED``, locale; any name containing TOKEN, KEY, SECRET,
PASSWORD, CREDENTIAL, AUTH, COOKIE or SESSION is dropped whatever its prefix) plus the config's ``env`` (a home directory, cache directories and the determinism record inside the working copy): the
campaign's credentials are not passed on.  So the editable code cannot see or signal any process outside its PID
namespace and process group (the runner, the parent), cannot reach the network (abstract Unix sockets included)
or any pathname socket outside its root, cannot read or write anything outside the allow-list, cannot change its
own frozen files, cannot keep state in ``/dev/shm`` between runs (unless shared), cannot gain privileges through
set-uid programs or regain capabilities (mounts made in a more privileged namespace stay locked in a nested
one), and when PID 1 dies the kernel kills every process of the namespace.

Messages are a fixed magic, an 8-byte header length, a JSON header and the raw bytes of each array (no
pickle).  Arrays are checked against a dtype whitelist and their declared shape.  For large outputs the parent
can hand the worker a shared buffer: a CPU buffer (``memfd``, sealed against shrinking and growing, passed over
the socket with ``SCM_RIGHTS``) or, on a GPU, a CUDA tensor shared by CUDA IPC (:func:`cuda_share` /
:func:`cuda_open`); the parent reads a private copy of either after each answer.  The CUDA path needs a GPU and
is exercised by the kits' GPU smoke only; it relies on torch's IPC reference counter files in ``/dev/shm``,
which is therefore not made private.
"""
from __future__ import annotations

import ctypes
import os

_PR_SET_DUMPABLE = 4
RESULT_SOCKET_ENV = "AR_RESULT_SOCKET"


def _prctl(option: int, arg: int, arg3: int = 0) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(ctypes.c_int(option), ctypes.c_ulong(arg), ctypes.c_ulong(arg3), ctypes.c_ulong(0),
                  ctypes.c_ulong(0)) < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


if os.environ.get(RESULT_SOCKET_ENV):
    _prctl(_PR_SET_DUMPABLE, 0)  # the parent: non-dumpable before anything else is imported

import base64  # noqa: E402
import json  # noqa: E402
import mmap  # noqa: E402
import socket  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402



class _LazyNumpy:
    """NumPy, imported on first use: the worker's PID-1 fork (:func:`worker_connect`) must happen before any library
    starts threads (OpenBLAS starts its pool when NumPy is imported; a fork of a threaded process can deadlock)."""

    def __getattr__(self, name):
        import numpy
        return getattr(numpy, name)


np = _LazyNumpy()

MAGIC = b"ARI1"
WORKER_FD_ENV = "AR_WORKER_FD"
SANDBOX_ENV = "AR_ISOLATION_SANDBOX"
SANDBOXED_ENV = "AR_WORKER_SANDBOXED"
MAX_HEADER_BYTES = 1 << 20
MAX_MESSAGE_BYTES = 1 << 30   # large outputs go through shared buffers
_LAUNCH_FLAG = "--ar-sandbox-launch"
_LAUNCH_CONFIG_ENV = "AR_SANDBOX_LAUNCH_CONFIG"
# Environment variables a sandboxed worker keeps (review round 3, F1): anything else -- credentials of the
# campaign's LLM manager, for instance -- is not passed on.
_ENV_KEEP_PREFIXES = ("AR_", "CUDA_", "NVIDIA_", "CUBLAS_", "CUDNN_", "NCCL_", "OMP_", "MKL_", "OPENBLAS_",
                      "PYTORCH_", "TORCH", "XLA_", "JAX_", "TF_", "LC_", "HF_", "TRANSFORMERS_", "TOKENIZERS_")
_ENV_KEEP = {"PATH", "LANG", "LANGUAGE", "TZ", "PYTHONPATH", "PYTHONHASHSEED", "SOURCE_DATE_EPOCH",
             "TOKENIZERS_PARALLELISM"}   # names kept as they are; the credential filter below applies to prefixes
# Names that look like credentials are dropped whatever their prefix (HF_TOKEN, TORCH_HUB_TOKEN, AR_*_API_KEY).
_ENV_DROP_PARTS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "COOKIE", "SESSION")
_RESULT_FD: list[int] = []    # the parent's result socket, closed in the worker's child before anything else
DTYPES = ("float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool")


class IsolationProtocolError(RuntimeError):
    """A malformed message, a refused dtype or shape, or a worker that went away."""


# ---------------------------------------------------------------------------------------------- parent side


class ResultChannel:
    """The parent's connection to the runner's result socket."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def write(self, line: str) -> None:
        self._sock.sendall((line.rstrip("\n") + "\n").encode("utf-8"))

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def parent_start() -> ResultChannel:
    """First statement of a frozen parent: (non-dumpable since this module's import) connect the result channel."""
    _prctl(_PR_SET_DUMPABLE, 0)
    name = os.environ.pop(RESULT_SOCKET_ENV, None)
    if not name:
        raise SystemExit("AR_ERROR isolation: no result channel (AR_RESULT_SOCKET); the runner starts this script")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
    sock.connect("\0" + name)
    _RESULT_FD.append(sock.fileno())
    return ResultChannel(sock)


# ---------------------------------------------------------------------------------------------- worker sandbox

_CLONE_NEWNS, _CLONE_NEWCGROUP, _CLONE_NEWUSER = 0x00020000, 0x02000000, 0x10000000
_CLONE_NEWPID, _CLONE_NEWNET = 0x20000000, 0x40000000
_MS_RDONLY, _MS_NOSUID, _MS_NODEV, _MS_NOEXEC = 1, 2, 4, 8
_MS_REMOUNT, _MS_BIND, _MS_REC, _MS_PRIVATE = 32, 4096, 0x4000, 1 << 18


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def _check(result: int, what: str) -> None:
    if result != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{what}: {os.strerror(err)}")


def _write_proc(name: str, text: str) -> None:
    fd = os.open(f"/proc/self/{name}", os.O_WRONLY)
    try:
        os.write(fd, text.encode("ascii"))
    finally:
        os.close(fd)


_MS_FLAG_NAMES = {"nosuid": _MS_NOSUID, "nodev": _MS_NODEV, "noexec": _MS_NOEXEC}


def _mount_flags_from_mountinfo(lines, path: str) -> int:
    """The nosuid/nodev/noexec flags of the mount that holds *path*, from ``/proc/self/mountinfo`` lines.

    A bind mount made in a user namespace inherits the source mount's flags **locked**: a later read-only
    remount that does not repeat every one of them is refused with EPERM.  The unit used to fix the remount at
    ``NOSUID`` and so dropped ``NODEV``/``NOEXEC``; on a host whose ``/sys`` is ``nosuid,nodev,noexec`` the
    bind of ``/sys`` therefore failed.  The longest matching mount point wins, so a submount's flags are taken
    from the submount."""
    real = os.path.realpath(path)
    best, flags = "", 0
    for line in lines:
        parts = line.partition(" - ")[0].split()
        if len(parts) < 6:
            continue
        point = parts[4]
        if (real == point or real.startswith(point.rstrip("/") + "/")) and len(point) > len(best):
            best, flags = point, 0
            for name in parts[5].split(","):
                flags |= _MS_FLAG_NAMES.get(name, 0)
    return flags


def _source_mount_flags(path: str) -> int:
    """The locked access flags of the mount that holds *path* (0 when the mount table cannot be read)."""
    try:
        with open("/proc/self/mountinfo") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return 0
    return _mount_flags_from_mountinfo(lines, path)


def _bind_remount_flags(locked: int, read_only: bool) -> int:
    """The remount flags of a bind mount: ``nosuid`` always (hardening, allowed even when not inherited), the
    flags the source mount carries (a user namespace locks them), and read-only when asked."""
    return _MS_BIND | _MS_REMOUNT | _MS_NOSUID | locked | (_MS_RDONLY if read_only else 0)


def _sandbox_preexec(config: dict):
    """Child setup before exec (see the module docstring): namespaces, then a new root file system that holds only
    the allow-listed paths, then ``pivot_root`` into it.  Every failure is reported on fd 2 and raised."""
    root = str(config["root"])
    ro = [str(p) for p in config.get("ro", []) if os.path.lexists(p)]
    ro_flat = [str(p) for p in config.get("ro_flat", []) if os.path.isdir(p)]
    rw = [str(p) for p in config.get("rw", []) if os.path.isdir(p)]
    ro_in_rw = [str(p) for p in config.get("ro_in_rw", []) if os.path.lexists(p)]
    dev = str(config.get("dev", "/dev"))
    uid, gid = os.getuid(), os.getgid()
    result_fds = list(_RESULT_FD)

    def target(path: str) -> bytes:
        return (root + path).encode()

    def make_parents(path: str, is_dir: bool) -> None:
        full = root + path
        os.makedirs(full if is_dir else os.path.dirname(full), exist_ok=True)
        if not is_dir and not os.path.lexists(full):
            os.close(os.open(full, os.O_WRONLY | os.O_CREAT, 0o600))

    def setup() -> None:
        step = "close result socket"
        cwd = os.getcwd()   # the child's working directory (subprocess entered it before this runs)
        try:
            for fd in result_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            libc = _libc()
            step = "dumpable"   # needed to write this child's own uid_map; it holds nothing secret now
            _prctl(_PR_SET_DUMPABLE, 1)
            step = "unshare"
            _check(libc.unshare(ctypes.c_int(_CLONE_NEWUSER | _CLONE_NEWNS | _CLONE_NEWPID | _CLONE_NEWNET
                                             | _CLONE_NEWCGROUP)), "unshare")
            step = "id maps"
            _write_proc("setgroups", "deny")
            _write_proc("uid_map", f"0 {uid} 1")
            _write_proc("gid_map", f"0 {gid} 1")
            step = "private mounts"
            _check(libc.mount(b"none", b"/", None, ctypes.c_ulong(_MS_REC | _MS_PRIVATE), None), "make / private")
            step = "new root"
            _check(libc.mount(b"tmpfs", root.encode(), b"tmpfs", ctypes.c_ulong(_MS_NOSUID | _MS_NODEV),
                              b"mode=0755"), "tmpfs root")

            def bind(path: str, read_only: bool, recursive: bool = True) -> None:
                src = os.path.realpath(path) if not os.path.islink(path) else path
                if os.path.islink(path):          # e.g. /lib -> usr/lib on merged-/usr systems
                    full = root + path
                    os.makedirs(os.path.dirname(full), exist_ok=True)
                    if not os.path.lexists(full):
                        os.symlink(os.readlink(path), full)
                    return
                make_parents(path, os.path.isdir(src))
                flags = _MS_BIND | (_MS_REC if recursive else 0)
                _check(libc.mount(src.encode(), target(path), None, ctypes.c_ulong(flags), None), "bind")
                # keep every locked flag of the source mount (see _mount_flags_from_mountinfo)
                remount = _bind_remount_flags(_source_mount_flags(src), read_only)
                _check(libc.mount(None, target(path), None, ctypes.c_ulong(remount), None), "remount")

            # a private /tmp first: allow-listed paths below /tmp are bound on top of it
            step = "private /tmp"
            make_parents("/tmp", True)
            _check(libc.mount(b"tmpfs", target("/tmp"), b"tmpfs", ctypes.c_ulong(_MS_NOSUID | _MS_NODEV),
                              b"mode=1777,size=8g"), "tmpfs /tmp")
            for path in ro:
                step = f"read-only {path}"
                bind(path, True)
            for path in ro_flat:
                # with its submounts (a user namespace cannot split them off), then an empty read-only tmpfs over
                # each cgroup file system below it, so the worker cannot reach any cgroup.kill or cgroup.procs
                step = f"read-only {path}"
                bind(path, True)
                for line in open("/proc/self/mountinfo").read().splitlines():
                    left, _, right = line.partition(" - ")
                    point = left.split()[4] if len(left.split()) > 4 else ""
                    if right.split()[:1] in (["cgroup"], ["cgroup2"]) and (point == path or point.startswith(path + "/")):
                        step = f"hide {point}"
                        if os.path.isdir(root + point):
                            _check(libc.mount(b"tmpfs", target(point), b"tmpfs",
                                              ctypes.c_ulong(_MS_NOSUID | _MS_NODEV | _MS_RDONLY), b"size=4k"),
                                   "tmpfs over a cgroup file system")
                if os.path.isdir(root + path + "/fs/cgroup"):      # and over the directory that holds them
                    step = f"hide {path}/fs/cgroup"
                    _check(libc.mount(b"tmpfs", target(path + "/fs/cgroup"), b"tmpfs",
                                      ctypes.c_ulong(_MS_NOSUID | _MS_NODEV | _MS_RDONLY), b"size=4k"),
                           "tmpfs over /sys/fs/cgroup")
            for path in rw:
                step = f"writable {path}"
                bind(path, False)
            for path in ro_in_rw:
                step = f"read-only inside the working copy {path}"
                bind(path, True)
            step = f"devices {dev}"
            make_parents("/dev", True)
            _check(libc.mount(dev.encode(), target("/dev"), None, ctypes.c_ulong(_MS_BIND | _MS_REC), None),
                   "bind /dev")
            if not config.get("shared_dev_shm") and os.path.isdir(root + "/dev/shm"):
                # a private /dev/shm (review round 3, F2): no state kept between runs or shared with other runs; a
                # node that shares CUDA tensors (torch's IPC reference counters live in /dev/shm) opts out
                step = "private /dev/shm"
                _check(libc.mount(b"tmpfs", target("/dev/shm"), b"tmpfs", ctypes.c_ulong(_MS_NOSUID | _MS_NODEV),
                                  b"mode=1777,size=8g"), "tmpfs /dev/shm")
            make_parents("/proc", True)
            # the host /proc for the moment (PID 1 mounts a fresh one over it before any editable code runs; the
            # kernel lets a user namespace mount proc only where a fully visible proc mount already exists)
            _check(libc.mount(b"/proc", target("/proc"), None, ctypes.c_ulong(_MS_BIND | _MS_REC), None), "bind /proc")
            step = "pivot_root"
            os.makedirs(root + "/.old_root", exist_ok=True)
            number = {"x86_64": 155, "aarch64": 41}.get(os.uname().machine)
            if number is None:
                raise OSError(f"pivot_root: unknown syscall number on {os.uname().machine}")
            _check(libc.syscall(ctypes.c_long(number), root.encode(), (root + "/.old_root").encode()), "pivot_root")
            os.chdir("/")
            _check(libc.umount2(b"/.old_root", ctypes.c_int(2)), "detach the old root")   # MNT_DETACH
            os.rmdir("/.old_root")
            # the root tmpfs was mounted nosuid,nodev: keep those flags when it is remounted read-only
            _check(libc.mount(None, b"/", None,
                              ctypes.c_ulong(_bind_remount_flags(_source_mount_flags("/"), True))),
                   "read-only root")
            step = "chdir"
            os.chdir(cwd)
        except BaseException as error:  # noqa: BLE001 - subprocess reports only "Exception occurred in preexec_fn"
            os.write(2, f"AR_ERROR isolation: worker sandbox setup failed at {step}: {error!r}\n".encode())
            raise

    return setup


def _drop_capabilities() -> None:
    """PID 1 of the worker's namespaces, after mounting /proc: no capability left, none to regain."""
    libc = _libc()
    for cap in range(64):
        libc.prctl(ctypes.c_int(24), ctypes.c_ulong(cap), ctypes.c_ulong(0), ctypes.c_ulong(0),
                   ctypes.c_ulong(0))                                   # PR_CAPBSET_DROP (EINVAL past the last)
    _prctl(47, 4)                                                       # PR_CAP_AMBIENT, CLEAR_ALL
    _prctl(28, 1 | 2 | 4 | 8 | 32)  # PR_SET_SECUREBITS: NOROOT(+LOCKED), NO_SETUID_FIXUP(+LOCKED), KEEP_CAPS_LOCKED

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]

    header, data = Header(0x20080522, 0), (Data * 2)()
    import platform
    number = {"x86_64": 126, "aarch64": 91}.get(platform.machine())
    if number is None:
        raise OSError(f"capset: unknown syscall number on {platform.machine()}")
    _check(libc.syscall(ctypes.c_long(number), ctypes.byref(header), ctypes.byref(data)), "capset")
    _prctl(38, 1)                                                       # PR_SET_NO_NEW_PRIVS
    status = open("/proc/self/status").read()
    caps = {line.split(":")[0]: int(line.split()[1], 16) for line in status.splitlines()
            if line.startswith(("CapEff", "CapPrm", "CapBnd", "CapAmb"))}
    if any(caps.values()):
        raise OSError(f"capabilities left after the drop: {caps}")


class Worker:
    """A worker process that runs the editable code; the parent talks to it over a socket pair.  With
    ``AR_ISOLATION_SANDBOX`` set (by the runner), the worker runs in the sandbox of the module docstring."""

    def __init__(self, script: str, *args: str, env: dict | None = None, cwd: str | None = None) -> None:
        here = os.path.dirname(os.path.abspath(sys.argv[0]))
        sys.path[:] = [p for p in sys.path if p not in ("", here) and os.path.abspath(p) != here]
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        child_env = dict(os.environ if env is None else env)
        child_env.pop(RESULT_SOCKET_ENV, None)
        raw = child_env.pop(SANDBOX_ENV, None)
        sandbox = json.loads(raw) if raw else None
        preexec = None
        argv = [sys.executable, script, *args]
        if sandbox is not None:
            child_env = _worker_environment(child_env)
            child_env.update({str(k): str(v) for k, v in (sandbox.get("env") or {}).items()})
            child_env[_LAUNCH_CONFIG_ENV] = json.dumps(sandbox)
            # A fresh single-threaded interpreter (no site, no sitecustomize) sets the sandbox up, forks PID 1 of the
            # new PID namespace, which mounts /proc, drops every capability and execs the worker: the parent may
            # run thread pools (torch, OpenMP) whose fork handlers start threads in a forked child, and
            # unshare(CLONE_NEWUSER) refuses a multithreaded caller; nothing is forked after the worker starts.
            argv = [sys.executable, "-I", "-S", os.path.abspath(__file__), _LAUNCH_FLAG, *argv]
        child_env[WORKER_FD_ENV] = str(child_sock.fileno())
        # its own session and process group: kill(0, ...) in the worker cannot reach the parent
        self.process = subprocess.Popen(argv, env=child_env, cwd=cwd, stdin=subprocess.DEVNULL,
                                        pass_fds=(child_sock.fileno(),), close_fds=True, preexec_fn=preexec,
                                        start_new_session=True)
        child_sock.close()
        self.conn = Connection(parent_sock)
        self.sandboxed = sandbox is not None

    def call(self, op: str, meta: dict | None = None, arrays: dict | None = None, fds: list[int] | None = None):
        """Send one request and return the worker's reply ``(meta, arrays)``; a worker error is raised."""
        self.conn.send(op, meta or {}, arrays or {}, fds or [])
        reply_op, reply_meta, reply_arrays, reply_fds = self.conn.recv()
        for fd in reply_fds:      # the parent accepts no descriptors from the worker
            os.close(fd)
        if reply_op == "error":
            raise IsolationProtocolError(f"worker error in {op!r}: {reply_meta.get('error', '')[:2000]}")
        if reply_op != "ok":
            raise IsolationProtocolError(f"unexpected reply {reply_op!r} to {op!r}")
        return reply_meta, reply_arrays

    def close(self, grace: float = 30.0) -> int | None:
        try:
            self.conn.send("exit", {}, {}, [])
        except OSError:
            pass
        try:
            return self.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return self.process.wait()
        finally:
            self.conn.close()


# ---------------------------------------------------------------------------------------------- worker side


def worker_connect() -> "Connection":
    """First statement of a frozen worker entry: take the socket (in a sandbox the launcher has already made this
    process PID 1 of its namespaces, with no capability)."""
    raw = os.environ.pop(WORKER_FD_ENV, None)
    if raw is None or not raw.isdigit():
        raise SystemExit("AR_ERROR isolation: no worker socket (AR_WORKER_FD); the parent starts this script")
    return Connection(socket.socket(fileno=int(raw)))


def _worker_environment(env: dict) -> dict:
    """The allow-listed part of *env* for a sandboxed worker (review round 3, F1)."""
    return {k: v for k, v in env.items()
            if k in _ENV_KEEP
            or (k.startswith(_ENV_KEEP_PREFIXES) and not any(p in k.upper() for p in _ENV_DROP_PARTS))}


def serve(handlers: dict, conn: "Connection | None" = None) -> None:
    """Worker loop: answer each request with ``handlers[op](meta, arrays, fds) -> (meta, arrays)`` until
    ``exit``.  An exception in a handler is sent back as an error reply."""
    import traceback

    conn = conn or worker_connect()
    while True:
        try:
            op, meta, arrays, fds = conn.recv()
        except IsolationProtocolError:
            return
        if op == "exit":
            conn.close()
            return
        handler = handlers.get(op)
        if handler is None:
            conn.send("error", {"error": f"unknown op {op!r}"}, {}, [])
            continue
        try:
            reply_meta, reply_arrays = handler(meta, arrays, fds)
        except Exception:  # noqa: BLE001 - reported to the parent, which decides
            conn.send("error", {"error": traceback.format_exc()[-4000:]}, {}, [])
            continue
        conn.send("ok", reply_meta or {}, reply_arrays or {}, [])


# ---------------------------------------------------------------------------------------------- messages


class Connection:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def send(self, op: str, meta: dict, arrays: dict, fds: list[int]) -> None:
        specs, blobs = [], []
        for name, value in arrays.items():
            arr = np.ascontiguousarray(value)
            if arr.dtype.name not in DTYPES:
                raise IsolationProtocolError(f"array {name!r}: dtype {arr.dtype.name} not allowed")
            specs.append({"name": str(name), "dtype": arr.dtype.name, "shape": list(arr.shape),
                          "nbytes": int(arr.nbytes)})
            blobs.append(arr)
        header = json.dumps({"op": op, "meta": meta, "arrays": specs}, allow_nan=True).encode("utf-8")
        if len(header) > MAX_HEADER_BYTES:
            raise IsolationProtocolError("header too large")
        prefix = MAGIC + struct.pack("<Q", len(header))
        if fds:
            socket.send_fds(self.sock, [prefix], list(fds))
        else:
            self.sock.sendall(prefix)
        self.sock.sendall(header)
        for arr in blobs:
            if arr.nbytes:
                self.sock.sendall(memoryview(arr.reshape(-1).view(np.uint8)))

    def _recv_exact(self, n: int, into: memoryview | None = None) -> bytes | None:
        buf = into if into is not None else memoryview(bytearray(n))
        got = 0
        while got < n:
            try:
                k = self.sock.recv_into(buf[got:], n - got)
            except (ConnectionResetError, BrokenPipeError) as error:
                raise IsolationProtocolError(f"connection closed ({error})") from None
            if k == 0:
                raise IsolationProtocolError("connection closed")
            got += k
        return None if into is not None else bytes(buf)

    def recv(self):
        try:
            msg, fds, _flags, _addr = socket.recv_fds(self.sock, 12, 16)
        except (ConnectionResetError, BrokenPipeError) as error:
            raise IsolationProtocolError(f"connection closed ({error})") from None
        if not msg:
            raise IsolationProtocolError("connection closed")
        prefix = msg
        while len(prefix) < 12:
            more = self.sock.recv(12 - len(prefix))
            if not more:
                raise IsolationProtocolError("connection closed")
            prefix += more
        if prefix[:4] != MAGIC:
            raise IsolationProtocolError("bad magic")
        (length,) = struct.unpack("<Q", prefix[4:])
        if length > MAX_HEADER_BYTES:
            raise IsolationProtocolError("header too large")
        header = json.loads(self._recv_exact(length))
        if not isinstance(header, dict) or not isinstance(header.get("op"), str):
            raise IsolationProtocolError("malformed header")
        arrays, total = {}, 0
        for spec in header.get("arrays", []):
            dtype = spec.get("dtype")
            shape = spec.get("shape")
            if dtype not in DTYPES or not isinstance(shape, list) or \
                    not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape):
                raise IsolationProtocolError(f"refused array spec {spec!r}")
            count = 1
            for dim in shape:
                count *= dim            # Python integers: no overflow
            nbytes = count * np.dtype(dtype).itemsize
            total += nbytes
            if spec.get("nbytes") != nbytes or total > MAX_MESSAGE_BYTES:
                raise IsolationProtocolError(f"array {spec.get('name')!r}: size mismatch or too large")
            arr = np.empty(shape, dtype=dtype)
            if nbytes:
                self._recv_exact(nbytes, memoryview(arr.reshape(-1).view(np.uint8)))
            arrays[str(spec.get("name"))] = arr
        meta = header.get("meta", {})
        if not isinstance(meta, dict):
            raise IsolationProtocolError("meta must be a mapping")
        return header["op"], meta, arrays, list(fds)


# ---------------------------------------------------------------------------------------------- shared buffers


class SharedBuffer:
    """A CPU buffer the parent creates (``memfd``) and lends to the worker for one large output."""

    def __init__(self, shape: tuple, dtype: str, fd: int | None = None) -> None:
        if dtype not in DTYPES:
            raise IsolationProtocolError(f"dtype {dtype} not allowed")
        self.shape, self.dtype = tuple(int(d) for d in shape), dtype
        count = 1
        for dim in self.shape:
            count *= dim
        nbytes = count * np.dtype(dtype).itemsize
        self.nbytes = nbytes
        if fd is None:
            fd = os.memfd_create("ar_shared", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            os.ftruncate(fd, max(nbytes, 1))
            # the worker gets the descriptor: it must not shrink the file under the parent's mapping (SIGBUS)
            import fcntl
            seals = getattr(fcntl, "F_SEAL_SHRINK", 2) | getattr(fcntl, "F_SEAL_GROW", 4) | getattr(fcntl, "F_SEAL_SEAL", 1)
            fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), seals)   # Linux values where fcntl lacks them
        elif os.fstat(fd).st_size < nbytes:
            raise IsolationProtocolError("shared buffer smaller than its declared shape")
        self.fd = fd
        self._map = mmap.mmap(fd, max(nbytes, 1))
        self.array = np.frombuffer(self._map, dtype=dtype, count=count).reshape(self.shape)

    def describe(self) -> dict:
        return {"shape": list(self.shape), "dtype": self.dtype}

    def snapshot(self) -> np.ndarray:
        """A private copy (the worker keeps its mapping and could change the buffer later)."""
        return np.array(self.array, copy=True)

    def close(self) -> None:
        del self.array
        self._map.close()
        os.close(self.fd)


def cuda_share(tensor) -> dict:
    """CUDA IPC description of a contiguous CUDA tensor (parent side); the parent keeps *tensor* alive."""
    storage = tensor.untyped_storage()
    (device, handle, size, offset, ref_handle, ref_offset, event_handle, event_sync) = storage._share_cuda_()
    enc = lambda b: base64.b64encode(b).decode("ascii") if isinstance(b, (bytes, bytearray)) else b  # noqa: E731
    return {"device": int(device), "handle": enc(handle), "size": int(size), "offset": int(offset),
            "ref_handle": enc(ref_handle), "ref_offset": int(ref_offset), "event_handle": enc(event_handle),
            "event_sync": bool(event_sync), "shape": list(tensor.shape), "dtype": str(tensor.dtype).split(".")[-1],
            "storage_offset": int(tensor.storage_offset())}


def cuda_open(desc: dict):
    """Open a tensor described by :func:`cuda_share` (worker side)."""
    import torch

    dec = lambda s: base64.b64decode(s) if isinstance(s, str) else s  # noqa: E731
    storage = torch.UntypedStorage._new_shared_cuda(
        desc["device"], dec(desc["handle"]), desc["size"], desc["offset"], dec(desc["ref_handle"]),
        desc["ref_offset"], dec(desc["event_handle"]), desc["event_sync"])
    dtype = getattr(torch, desc["dtype"])
    t = torch.empty(0, dtype=dtype, device=f"cuda:{desc['device']}")
    t.set_(storage, desc["storage_offset"], desc["shape"])
    return t


def _launch(argv: list[str]) -> None:
    """``python -I -S ar_isolated.py --ar-sandbox-launch <worker argv>`` (a fresh, single-threaded process): set the
    sandbox up, fork PID 1 of the new PID namespace, which mounts a fresh /proc, drops every capability and execs
    the worker; this outer process only waits and passes on the exit code."""
    config = json.loads(os.environ.pop(_LAUNCH_CONFIG_ENV))
    try:
        _sandbox_preexec(config)()
    except BaseException:  # noqa: BLE001 - already reported on fd 2
        os._exit(13)
    child = os.fork()
    if child == 0:
        try:
            _check(_libc().mount(b"proc", b"/proc", b"proc", ctypes.c_ulong(_MS_NOSUID | _MS_NODEV | _MS_NOEXEC),
                                 None), "mount /proc")
            _drop_capabilities()
            os.execve(argv[0], argv, dict(os.environ))
        except BaseException as error:  # noqa: BLE001
            os.write(2, f"AR_ERROR isolation: worker sandbox (PID 1) failed: {error!r}\n".encode())
            os._exit(13)
    raw = os.environ.get(WORKER_FD_ENV)
    if raw and raw.isdigit():
        os.close(int(raw))         # only PID 1 keeps the worker socket
    while True:
        try:
            _pid, status = os.waitpid(child, 0)
            break
        except InterruptedError:
            continue
    os._exit(os.waitstatus_to_exitcode(status) & 0xFF if os.WIFEXITED(status) else 1)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == _LAUNCH_FLAG:
    _launch(sys.argv[2:])
