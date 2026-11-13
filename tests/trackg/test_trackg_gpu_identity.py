"""HTCondor GPU assignment -> full UUIDs for the M10 runner (synthetic; no GPU needed)."""
import subprocess

import pytest

from autoresearch.trackg.gpu_identity import GpuIdentityError, assigned_ids, parse_nvidia_smi_l, resolve, resolve_from_job
from autoresearch.trackg.runner import ResourceLimits

SMI = """GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-1a2b3c4d-1111-2222-3333-444455556666)
GPU 1: NVIDIA A100-PCIE-40GB (UUID: GPU-9f8e7d6c-aaaa-bbbb-cccc-ddddeeeeffff)
"""
SHORT = "GPU-1a2b3c4d"
FULL = "GPU-1a2b3c4d-1111-2222-3333-444455556666"


def test_short_uuid_resolves_to_the_full_uuid_and_feeds_the_runner():
    uuids, note = resolve({"NVIDIA_VISIBLE_DEVICES": SHORT}, SMI)
    assert uuids == (FULL,) and note["assigned"] == [SHORT] and note["resolved"] == [FULL]
    limits = ResourceLimits(gpu_uuids=uuids, cpu_list=(0,), memory_bytes=1 << 30, provisional_timeout_seconds=60)
    assert limits.gpu_uuids == (FULL,)


def test_both_sources_must_agree():
    env = {"NVIDIA_VISIBLE_DEVICES": SHORT, "_CONDOR_AssignedGPUs": SHORT}
    assert resolve(env, SMI)[0] == (FULL,)
    with pytest.raises(GpuIdentityError, match="disagree"):
        resolve({"NVIDIA_VISIBLE_DEVICES": SHORT, "_CONDOR_AssignedGPUs": "GPU-9f8e7d6c"}, SMI)


@pytest.mark.parametrize("value", ["0", "0,1", "all", "none", "void", "", "1a2b3c4d"])
def test_index_and_placeholder_forms_are_refused(value):
    with pytest.raises(GpuIdentityError):
        assigned_ids({"NVIDIA_VISIBLE_DEVICES": value})


def test_no_assignment_is_refused():
    with pytest.raises(GpuIdentityError, match="none of"):
        assigned_ids({"CUDA_VISIBLE_DEVICES": SHORT})


def test_cuda_visible_devices_uuid_and_index_forms_are_checked():
    uuids, note = resolve({"NVIDIA_VISIBLE_DEVICES": SHORT, "CUDA_VISIBLE_DEVICES": "0"}, SMI)
    assert uuids == (FULL,) and note["cuda_visible_devices_index_checked"] == "0"
    with pytest.raises(GpuIdentityError, match="nvidia-smi order"):
        resolve({"NVIDIA_VISIBLE_DEVICES": SHORT, "CUDA_VISIBLE_DEVICES": "1"}, SMI)
    with pytest.raises(GpuIdentityError, match="nvidia-smi order"):
        resolve({"NVIDIA_VISIBLE_DEVICES": SHORT, "CUDA_VISIBLE_DEVICES": "5"}, SMI)
    with pytest.raises(GpuIdentityError, match="other GPUs"):
        resolve({"NVIDIA_VISIBLE_DEVICES": SHORT, "CUDA_VISIBLE_DEVICES": "GPU-9f8e7d6c"}, SMI)
    with pytest.raises(GpuIdentityError, match="mixes"):
        resolve({"NVIDIA_VISIBLE_DEVICES": SHORT, "CUDA_VISIBLE_DEVICES": f"{SHORT},1"}, SMI)


def test_short_and_full_forms_of_one_gpu_agree():
    env = {"NVIDIA_VISIBLE_DEVICES": SHORT, "_CONDOR_AssignedGPUs": FULL, "CUDA_VISIBLE_DEVICES": FULL}
    assert resolve(env, SMI)[0] == (FULL,)


def test_ambiguous_or_missing_prefix_is_refused():
    twins = "GPU 0: X (UUID: GPU-1a2b3c4d-aaaa)\nGPU 1: X (UUID: GPU-1a2b3c4d-bbbb)\n"
    with pytest.raises(GpuIdentityError, match="matches 2"):
        resolve({"NVIDIA_VISIBLE_DEVICES": SHORT}, twins)
    with pytest.raises(GpuIdentityError, match="matches 0"):
        resolve({"NVIDIA_VISIBLE_DEVICES": "GPU-00000000"}, SMI)


def test_mig_lines_and_duplicates():
    mig = SMI + "  MIG 1g.5gb Device 0: (UUID: MIG-abcdef01-2222-3333-4444-555566667777)\n"
    assert parse_nvidia_smi_l(mig)[-1].startswith("MIG-abcdef01")
    with pytest.raises(GpuIdentityError, match="twice"):
        assigned_ids({"NVIDIA_VISIBLE_DEVICES": f"{SHORT},{SHORT}"})
    with pytest.raises(GpuIdentityError, match="no GPU UUID"):
        parse_nvidia_smi_l("No devices were found\n")


def test_resolve_from_job_runs_nvidia_smi_and_never_writes_the_environment():
    env = {"NVIDIA_VISIBLE_DEVICES": SHORT}
    before = dict(env)
    calls = []

    def fake(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, SMI, "")

    assert resolve_from_job(env, runner=fake)[0] == (FULL,)
    assert calls == [["nvidia-smi", "-L"]] and env == before


def test_resolve_from_job_refuses_a_failing_nvidia_smi():
    with pytest.raises(GpuIdentityError, match="exited 9"):
        resolve_from_job({"NVIDIA_VISIBLE_DEVICES": SHORT},
                         runner=lambda argv, **kw: subprocess.CompletedProcess(argv, 9, "", "boom"))
