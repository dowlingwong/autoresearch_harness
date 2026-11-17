"""Frozen node library: parent/worker split of an isolated Track G node (R1-HARNESS-ISOLATION).

FROZEN: copied byte-identically into every isolated node and listed in its ``frozen_paths``; the master copy
is ``autoresearch/trackg/node_lib/ar_isolated.py`` in the harness (the port tests compare the copies).
Standard library and NumPy only; ``torch`` is imported lazily by the CUDA helpers.

The runner starts the node's frozen **parent** script directly (no shell); the name of the runner's result
socket is in ``AR_RESULT_SOCKET``.  The parent:

1. calls :func:`parent_start` as its **first** statement: the process becomes non-dumpable (another process of
   the same user can then neither open its file descriptors through ``/proc`` nor ``ptrace`` it), then it
   connects to the result socket (the runner accepts only the parent's pid) with a close-on-exec socket, and
   ``AR_RESULT_SOCKET`` is removed from the environment;
2. imports everything it needs (it never imports editable code);
3. starts the editable code with :class:`Worker` -- a separate process running a frozen worker script that
   calls :func:`serve`; the worker inherits only one end of a socket pair (never the result channel);
4. drives the worker with requests (:meth:`Worker.call`) and computes the metric from the arrays it returns;
5. writes the metric line with :meth:`ResultChannel.write` (the runner refuses a run whose channel holds more
   or fewer than one metric line, or whose output holds a metric line), and exits.

Messages are a fixed magic, an 8-byte header length, a JSON header and the raw bytes of each array (no
pickle).  Arrays are checked against a dtype whitelist and their declared shape.  For large outputs the parent
can hand the worker a shared buffer: a CPU buffer (``memfd``, passed over the socket with ``SCM_RIGHTS``) or,
on a GPU, a CUDA tensor shared by CUDA IPC (:func:`cuda_share` / :func:`cuda_open`); the CUDA path needs a
GPU and is exercised by the kits' GPU smoke only.
"""
from __future__ import annotations

import base64
import ctypes
import json
import mmap
import os
import socket
import struct
import subprocess
import sys

import numpy as np

MAGIC = b"ARI1"
WORKER_FD_ENV = "AR_WORKER_FD"
RESULT_SOCKET_ENV = "AR_RESULT_SOCKET"
MAX_HEADER_BYTES = 1 << 20
MAX_MESSAGE_BYTES = 16 << 30
DTYPES = ("float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool")
_PR_SET_DUMPABLE = 4


class IsolationProtocolError(RuntimeError):
    """A malformed message, a refused dtype or shape, or a worker that went away."""


def _prctl(option: int, arg: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(ctypes.c_int(option), ctypes.c_ulong(arg), ctypes.c_ulong(0), ctypes.c_ulong(0),
                  ctypes.c_ulong(0)) < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


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
    """First statement of a frozen parent: non-dumpable, then connect the result channel (module docstring)."""
    _prctl(_PR_SET_DUMPABLE, 0)
    name = os.environ.pop(RESULT_SOCKET_ENV, None)
    if not name:
        raise SystemExit("AR_ERROR isolation: no result channel (AR_RESULT_SOCKET); the runner starts this script")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
    sock.connect("\0" + name)
    return ResultChannel(sock)


class Worker:
    """A worker process that runs the editable code; the parent talks to it over a socket pair."""

    def __init__(self, script: str, *args: str, env: dict | None = None, cwd: str | None = None) -> None:
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        child_env = dict(os.environ if env is None else env)
        child_env.pop(RESULT_SOCKET_ENV, None)
        child_env[WORKER_FD_ENV] = str(child_sock.fileno())
        self.process = subprocess.Popen([sys.executable, script, *args], env=child_env, cwd=cwd,
                                        stdin=subprocess.DEVNULL, pass_fds=(child_sock.fileno(),),
                                        close_fds=True)
        child_sock.close()
        self.conn = Connection(parent_sock)

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
    raw = os.environ.pop(WORKER_FD_ENV, None)
    if raw is None or not raw.isdigit():
        raise SystemExit("AR_ERROR isolation: no worker socket (AR_WORKER_FD); the parent starts this script")
    return Connection(socket.socket(fileno=int(raw)))


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
            k = self.sock.recv_into(buf[got:], n - got)
            if k == 0:
                raise IsolationProtocolError("connection closed")
            got += k
        return None if into is not None else bytes(buf)

    def recv(self):
        msg, fds, _flags, _addr = socket.recv_fds(self.sock, 12, 16)
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
            count = int(np.prod(shape, dtype=np.int64)) if shape else 1
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
        nbytes = int(np.prod(self.shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        self.nbytes = nbytes
        if fd is None:
            fd = os.memfd_create("ar_shared", os.MFD_CLOEXEC)
            os.ftruncate(fd, max(nbytes, 1))
        elif os.fstat(fd).st_size < nbytes:
            raise IsolationProtocolError("shared buffer smaller than its declared shape")
        self.fd = fd
        self._map = mmap.mmap(fd, max(nbytes, 1))
        self.array = np.frombuffer(self._map, dtype=dtype, count=int(np.prod(self.shape, dtype=np.int64))
                                   ).reshape(self.shape)

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
