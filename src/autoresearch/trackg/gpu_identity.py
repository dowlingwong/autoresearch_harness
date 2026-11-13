"""Resolve the GPU an HTCondor job was assigned to full UUIDs for the M10 runner (AI-assisted draft).

HTCondor sets ``NVIDIA_VISIBLE_DEVICES`` (and ``_CONDOR_AssignedGPUs``) to a **short** UUID such as
``GPU-1a2b3c4d`` (the first 8 hex digits). The runner takes GPUs only by UUID (``ResourceLimits.gpu_uuids``);
this module turns the assignment into full UUIDs by matching it against ``nvidia-smi -L`` inside the job,
and refuses anything ambiguous. It only reads the environment: ``NVIDIA_VISIBLE_DEVICES`` is never
changed (the pool sets it; the runner later sets ``CUDA_VISIBLE_DEVICES`` for its child to the full UUID).

Rules:

* identity sources are ``NVIDIA_VISIBLE_DEVICES`` and ``_CONDOR_AssignedGPUs``: at least one must be
  set; every entry must be a UUID form (``GPU-…`` or ``MIG-…``); index forms (``0``), ``all``, ``none``,
  ``void`` and empty values are refused; when both are set they must name the same GPUs;
* ``CUDA_VISIBLE_DEVICES``, if it holds UUID forms, must agree as well; if it holds indices it is not an
  identity source and is reported (``cuda_visible_devices_ignored``), since inside a container the
  visible GPUs are renumbered from 0;
* each assigned id must be a prefix of exactly one UUID listed by ``nvidia-smi -L``; no match or more
  than one match is refused; two ids resolving to the same GPU are refused.
"""
from __future__ import annotations

import re
import subprocess
from typing import Callable, Mapping, Sequence

_UUID_FORM = re.compile(r"^(GPU|MIG)-[0-9A-Za-z][0-9A-Za-z/\-]{7,}$")
_SMI_UUID = re.compile(r"\(UUID: ((?:GPU|MIG)-[0-9A-Za-z/\-]+)\)")
_REFUSED = {"", "all", "none", "void", "nodevfiles"}
SOURCES = ("NVIDIA_VISIBLE_DEVICES", "_CONDOR_AssignedGPUs")


class GpuIdentityError(ValueError):
    """The assigned GPU cannot be resolved to full UUIDs unambiguously."""


def _split(value: str) -> list[str]:
    return [x.strip() for x in value.split(",")]


def _uuid_list(name: str, value: str) -> list[str]:
    items = _split(value)
    for item in items:
        if item.lower() in _REFUSED or item.isdigit():
            raise GpuIdentityError(f"{name}={value!r}: index or placeholder forms are refused; a GPU UUID is required")
        if not _UUID_FORM.match(item):
            raise GpuIdentityError(f"{name}={value!r}: {item!r} is not a GPU-/MIG- UUID form")
    if len(set(items)) != len(items):
        raise GpuIdentityError(f"{name}={value!r} names a GPU twice")
    return items


def parse_nvidia_smi_l(text: str) -> list[str]:
    """Full UUIDs in ``nvidia-smi -L`` order (GPU and MIG lines)."""
    uuids = _SMI_UUID.findall(text)
    if not uuids:
        raise GpuIdentityError("nvidia-smi -L lists no GPU UUID")
    if len(set(uuids)) != len(uuids):
        raise GpuIdentityError("nvidia-smi -L lists a UUID twice")
    return uuids


def assigned_ids(env: Mapping[str, str]) -> tuple[list[str], dict]:
    present = {name: env[name] for name in SOURCES if name in env}
    if not present:
        raise GpuIdentityError(f"none of {SOURCES} is set: no HTCondor GPU assignment to resolve")
    lists = {name: _uuid_list(name, value) for name, value in present.items()}
    first = next(iter(lists.values()))
    for name, items in lists.items():
        if set(items) != set(first):
            raise GpuIdentityError(f"GPU assignment sources disagree: {present}")
    note = {"sources": present}
    cuda = env.get("CUDA_VISIBLE_DEVICES")
    if cuda is not None:
        parts = _split(cuda)
        if all(_UUID_FORM.match(p) for p in parts):
            if set(parts) != set(first):
                raise GpuIdentityError(f"CUDA_VISIBLE_DEVICES={cuda!r} disagrees with {present}")
            note["cuda_visible_devices"] = cuda
        else:
            note["cuda_visible_devices_ignored"] = cuda
    return first, note


def resolve(env: Mapping[str, str], smi_text: str) -> tuple[tuple[str, ...], dict]:
    ids, note = assigned_ids(env)
    listed = parse_nvidia_smi_l(smi_text)
    full = []
    for gid in ids:
        matches = [u for u in listed if u.startswith(gid)]
        if len(matches) != 1:
            raise GpuIdentityError(f"assigned GPU {gid!r} matches {len(matches)} UUIDs in nvidia-smi -L ({listed})")
        full.append(matches[0])
    if len(set(full)) != len(full):
        raise GpuIdentityError(f"assigned ids {ids} resolve to the same GPU")
    note.update(assigned=ids, resolved=full, nvidia_smi_uuids=listed)
    return tuple(full), note


def resolve_from_job(env: Mapping[str, str], runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                     nvidia_smi: Sequence[str] = ("nvidia-smi", "-L")) -> tuple[tuple[str, ...], dict]:
    """Run ``nvidia-smi -L`` inside the job and resolve the assignment (for ``ResourceLimits.gpu_uuids``)."""
    try:
        res = runner(list(nvidia_smi), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuIdentityError(f"nvidia-smi -L failed: {exc}") from exc
    if res.returncode != 0:
        raise GpuIdentityError(f"nvidia-smi -L exited {res.returncode}: {res.stderr.strip()[:200]}")
    return resolve(env, res.stdout)
