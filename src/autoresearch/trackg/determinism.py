"""Deterministic settings for Track G runs, with a record of what could not be forced.

Two parts:

- Environment variables the runner always sets (``CUBLAS_WORKSPACE_CONFIG``,
  ``PYTHONHASHSEED``, thread counts).
- Interpreter-level settings (``torch.backends.cudnn.benchmark = False``,
  ``torch.use_deterministic_algorithms(True)``, ``torch.set_num_threads``)
  applied by a ``sitecustomize`` shim that the runner puts first on the run's
  ``PYTHONPATH``.  The shim writes a JSON record of each setting:
  ``forced`` or ``not_forced`` with the reason (for example ``torch_absent``).

Limits recorded rather than hidden:

- ``torch.use_deterministic_algorithms(True)`` makes an op without a
  deterministic implementation raise, which the base parser would classify
  as ``divergent``; a node whose baseline needs it may declare
  ``determinism.deterministic_warn_only: true`` (recorded as ``warn_only``),
  decided in the port's review and the census determinism check;
- the shim imports ``torch`` at start-up in every Python interpreter of the
  run: the import time is part of the measured run time (the timeout basis),
  and environment variables a task sets in-script before ``import torch``
  (for example ``PYTORCH_CUDA_ALLOC_CONF``) no longer take effect -- set them
  in the run command or the wrapper's environment instead;
- the data-loader worker count is only advertised (``AR_NUM_WORKERS``); it is
  always listed as ``not_forced``;
- the shim runs at interpreter start; a task script that later sets
  ``cudnn.benchmark = True`` overrides it (the port checklist forbids that);
- a non-Python run command, or an environment whose own ``sitecustomize``
  shadows the shim, leaves no shim record; the runner then records every
  interpreter-level setting as ``not_forced`` with reason ``shim_not_loaded``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DETERMINISM_RECORD_ENV = "AR_DETERMINISM_RECORD"
WARN_ONLY_ENV = "AR_DETERMINISTIC_WARN_ONLY"
THREADS_ENV = "AR_NUM_THREADS"
WORKERS_ENV = "AR_NUM_WORKERS"
DETERMINISM_RECORD_SCHEMA = "trackg_determinism/v1"
INTERPRETER_SETTINGS = ("cudnn.benchmark=False", "torch.use_deterministic_algorithms(True)",
                        "torch.set_num_threads")

SHIM_SOURCE = r'''
# Written by autoresearch.trackg.determinism; applied at interpreter start.
# The settings are applied in every Python interpreter of the run (including
# spawned data-loader workers); the record is written by the first one only.
def _ar_trackg_determinism():
    import json, os
    path = os.environ.get("AR_DETERMINISM_RECORD")
    warn_only = os.environ.get("AR_DETERMINISTIC_WARN_ONLY", "0") == "1"
    rec = {"schema": "trackg_determinism/v1", "settings": [], "warn_only": warn_only}
    def note(setting, status, reason=""):
        rec["settings"].append({"setting": setting, "status": status, "reason": reason})
    try:
        import torch
    except Exception as error:  # torch absent: record, never fail the run
        for s in ("cudnn.benchmark=False", "torch.use_deterministic_algorithms(True)",
                  "torch.set_num_threads"):
            note(s, "not_forced", "torch_absent: %s" % type(error).__name__)
    else:
        rec["torch_version"] = getattr(torch, "__version__", None)
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            note("cudnn.benchmark=False", "forced")
        except Exception as error:
            note("cudnn.benchmark=False", "not_forced", repr(error))
        try:
            if warn_only:
                torch.use_deterministic_algorithms(True, warn_only=True)
                note("torch.use_deterministic_algorithms(True)", "forced", "warn_only=True")
            else:
                torch.use_deterministic_algorithms(True)
                note("torch.use_deterministic_algorithms(True)", "forced")
        except Exception as error:
            note("torch.use_deterministic_algorithms(True)", "not_forced", repr(error))
        try:
            n = int(os.environ.get("AR_NUM_THREADS", "0"))
            if n > 0:
                torch.set_num_threads(n)
                note("torch.set_num_threads", "forced", str(n))
            else:
                note("torch.set_num_threads", "not_forced", "AR_NUM_THREADS unset")
        except Exception as error:
            note("torch.set_num_threads", "not_forced", repr(error))
    if not path or os.path.exists(path):
        return
    try:
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(rec, handle, sort_keys=True)
    except Exception:
        pass
try:
    _ar_trackg_determinism()
except Exception:
    pass
del _ar_trackg_determinism
'''


def write_shim(shim_dir: Path) -> Path:
    """Write the ``sitecustomize.py`` shim into *shim_dir* and return its path."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    target = shim_dir / "sitecustomize.py"
    target.write_text(SHIM_SOURCE.lstrip("\n"), encoding="utf-8")
    return target


def environment_settings(settings: Any, threads: int, workers: int) -> dict[str, str]:
    """Environment variables for deterministic, fixed-thread runs."""
    value = str(threads)
    return {
        "CUBLAS_WORKSPACE_CONFIG": settings.cublas_workspace_config,
        "PYTHONHASHSEED": settings.pythonhashseed,
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
        THREADS_ENV: value,
        WORKERS_ENV: str(workers),
        WARN_ONLY_ENV: "1" if settings.deterministic_warn_only else "0",
    }


def read_record(record_path: Path, env: dict[str, str]) -> dict[str, Any]:
    """Combine the shim's record (or its absence) with the environment settings."""
    record: dict[str, Any] = {"schema": DETERMINISM_RECORD_SCHEMA, "environment": {}, "settings": []}
    for key in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS", THREADS_ENV, WORKERS_ENV):
        record["environment"][key] = env.get(key)
    shim: dict[str, Any] | None = None
    if record_path.exists():
        try:
            shim = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            shim = None
    if shim is None:
        record["shim_loaded"] = False
        record["settings"] = [
            {"setting": s, "status": "not_forced", "reason": "shim_not_loaded"} for s in INTERPRETER_SETTINGS
        ]
    else:
        record["shim_loaded"] = True
        record["settings"] = list(shim.get("settings", []))
        record["warn_only"] = bool(shim.get("warn_only", False))
        if "torch_version" in shim:
            record["torch_version"] = shim["torch_version"]
    # The data-loader worker count is only advertised to the task; nothing enforces it.
    record["settings"].append({"setting": "dataloader_workers", "status": "not_forced",
                               "reason": "advertised via AR_NUM_WORKERS only"})
    record["not_forced"] = sorted(s["setting"] for s in record["settings"] if s.get("status") != "forced")
    return record
