# R1-TRACKG-AR-COND7 option A — code-unit report (kit branch, 20 November 2026)

**Status.** The harness/kit change implementing option A for `autoresearch_fixedtokens` is **coded, committed
locally and unit-checked on the Mac** (CPU, no GPU, no torch). The worker now hands logits to the frozen parent
through the existing CPU `memfd` buffer (`iso.SharedBuffer` + `bind_cpu` + `SCM_RIGHTS`) on every host, GPU
included; the CUDA-IPC branch is gone from the node's result path, and `ar_isolated.cuda_open` initializes CUDA
before opening a shared storage. The node spec no longer sets `shared_dev_shm`, so an isolated run is expected
to record `effective: true`, `waivers {}`, `exceptions: []`. **Not pushed, not run on a GPU**: the isolated GPU
smoke on Ada, the slowdown measurement and the fresh independent review are still to be done by the
coordinator/reviewer. ENGINEERING_ONLY; AI-assisted.

## 1. Authority and inputs

- Authority row **R1-TRACKG-AR-COND7** (option A) in `docs/r1/PLAN.md` of published paper main `de3de54f`
  (`git -C .claude-worktrees/ops-tasl-20261116 show origin/main:docs/r1/PLAN.md`, row at line 1317): remove CUDA
  IPC from the result path (`torch.cuda.init()` added to `cuda_open` in the same change), keep isolation
  `effective:true` with no waivers (no `shared_dev_shm`), update kit pins and `frozen_pin_sha256`, run one
  isolated smoke, then U2; measure and disclose the logits-transfer slowdown.
- Condition C3 of `docs/r1/reviews/HARNESS_ISOLATION_FIX_REVIEW_20261119.md` (ACCEPTED; Ada host-check
  `6a700509…`): no autoresearch GPU run until the logits transfer design decision is recorded (socket/CPU
  buffer) **and** `cuda_open`'s missing `torch.cuda.init()` is fixed in that same reviewed change.
- Kit branch `mac-trackg-iso-ar-20261117` at `448f6820f885d70ee6872b692284fe861afd3a33` (ar kit after the
  accepted isolation core fix `5403392`, merge `2e65922`); kit-update report
  `docs/r1/reports/trackg_g0_20261117/TRACKG_ISO_KIT_UPDATE_20261119.md` (autoresearch GPU smoke BLOCKED by
  condition 7; old pin `b8482875…` → `b5f0ec2f…` from the isolation fix).
- This worktree: `/Users/wongdowling/Documents/agent_paper_B/.claude-worktrees/harness_ar_cond7_20261120`,
  branch `mac-trackg-iso-ar-cond7a-20261120`, base `448f682`, commit recorded in §7 below. Local only — no push,
  no force push, no deletion, no GPU/TASL/HTCondor, no main-paper-checkout edit.

## 2. Exact changes

Diff is 7 files, +85/−63 (`git diff --numstat`); the frozen result-path files and the isolation master are:

| file | old sha256 | new sha256 |
|---|---|---|
| `nodes/autoresearch_fixedtokens/run_fixed_budget.py` | `93778238d9e468e617e4c8bd8e126bc32a3a8740aa1132c2e28f921fa0479e9a` | `58d80b5224ac5e3981f60ac7b7e251a8c97eaf31d0a88a4e8811e920699a33ed` |
| `nodes/autoresearch_fixedtokens/ar_worker.py` | `fd838dac623d23e5d82cdf7927b2e7470de1fa5f3a675ae5f424905b7f689573` | `65a4927c54a566baa5efd7c26104e66eeb96ccfd6eddf92b59766b70d75613eb` |
| `nodes/autoresearch_fixedtokens/ar_isolated.py` | `37d99ab092c73d885b95daa443d0b145148f0b02a6ff464e055704c39e48b610` | `3ba34c67673585e5858c5d5c2b4f0bc1b6b6be11c9898a62f71d12a5b60ddc36` |
| `src/autoresearch/trackg/node_lib/ar_isolated.py` (master) | `37d99ab0…` | `3ba34c67…` (byte-identical to the node copy) |
| `configs/trackg/nodes/autoresearch_fixedtokens.yaml` | `953a081d…` (file) | `5bdce00a…` (file) |
| `nodes/autoresearch_fixedtokens/representative_edits/manifest.json` | `02e05d06…` | `55f2666a…` |
| `tests/trackg/test_port_autoresearch.py` | `0c2e6114…` | `2b9f3692…` |

1. **`run_fixed_budget.py` — the parent uses only the CPU buffer.** `LogitSource.__init__` now always allocates
   `iso.SharedBuffer((B, T, vocab), "float32")` and hands the sealed `memfd` descriptor to the worker with a
   `bind_cpu` request; `__call__` returns `torch.from_numpy(self.shared.snapshot()).to(self.device)`. The
   `device.type == "cuda"` branch is deleted: no `torch.cuda.memory._set_allocator_settings(...)`, no
   `iso.cuda_share`, no `bind_cuda`, no `torch.cuda.synchronize`. `device` may still be `cuda`: the parent
   snapshots CPU memory (a private copy of the shared mapping) and moves it to its own device for the
   cross-entropy, which it alone computes.
2. **`ar_worker.py` — the worker only knows the CPU buffer.** The `bind_cuda` handler is removed from `serve`;
   `bind_cpu` maps the parent's `memfd` and `logits` copies `model(x)` (bf16 autocast, unchanged) to that CPU
   buffer as float32 (`out.copy_(result.float())`, a synchronous device→host copy); the CUDA-only
   `torch.cuda.synchronize` is gone. Shape/dtype checks, the model-out-of-train extraction, the eval-mode
   transition, the weight fingerprint and the exit codes are unchanged.
3. **`ar_isolated.py` — latent init bug fixed, helper retained.** `cuda_open` now calls `torch.cuda.init()`
   before `torch.UntypedStorage._new_shared_cuda` (a fresh consumer process otherwise segfaults). `cuda_share` /
   `cuda_open` stay in the frozen library for compatibility, but **no node uses them** (verified: the mlab and
   mlgym kits' parent/worker scripts never call them either). The master docstring now states that the CPU
   `memfd` buffer is the path for every isolated node, GPU included, and why CUDA IPC is not used. The node copy
   is byte-identical to the master (the port tests assert this).
4. **Node spec** `configs/trackg/nodes/autoresearch_fixedtokens.yaml`: `isolation.shared_dev_shm: true` and its
   CUDA-IPC comment are replaced by `shared_dev_shm: false` with a comment naming option A. That is the only
   semantic change; `effective` is computed as `not waivers and not exceptions and euid != 0 and cgroup is not
   None and sandbox is not None` (`runner.py`), so with this spec an isolated run records `exceptions: []`.
   Nothing in the core sandbox/probe was weakened — the private `/dev/shm` mount remains the default and only
   this node's opt-out is gone.
5. **Representative-edits manifest**: only `node_spec_sha256` changes (`5f48b86f…` → `5c1bdf3d…`), because
   `port_support.check_manifest` requires the manifest to name the current spec. The E1–E4 diffs, their
   `sha256`/`normalised_diff_sha256`/`files_touched`, the baseline raw hash and `measured: false` are unchanged
   (recomputed with the kit's own `declare_edits`: everything except the interpreter-dependent normalised
   code-state hashes and the spec hash is identical; the spec hash is interpreter-independent).
6. **Test** `tests/trackg/test_port_autoresearch.py::test_shared_dev_shm_for_cuda_ipc` is replaced by
   `test_result_path_uses_the_cpu_buffer_not_cuda_ipc` (static, no torch): spec `shared_dev_shm` is false; neither
   the frozen parent nor the frozen worker contains `cuda_share`, `bind_cuda` or `cuda_open`; the parent has
   `iso.SharedBuffer(`/`"bind_cpu"`/`snapshot()`, the worker has `iso.SharedBuffer(`/`def bind_cpu(`/
   `out.copy_(result.float())`; `torch.cuda.init()` is inside `cuda_open`. No other test needed changing — the
   other `shared_dev_shm` tests use the fixture spec, not this node.

## 3. Why no CUDA IPC — and why this satisfies condition 7

Condition 7 (host-check report §3 item 7 / §5.5, review C3): the worker runs in a new user, mount, PID, network
and cgroup namespace. CUDA IPC needs a shared `/dev/shm` (torch's IPC reference-counter files) and a CUDA IPC
handle to cross a PID-namespace boundary; on Ada (RTX 6000 Ada, driver 545.23.08) opening the parent tensor in
the worker fails with `CUDA error: invalid resource handle`, and with a private `/dev/shm` it cannot work at all.
Option B (a `shared_dev_shm` waiver) was rejected because it reopens a state channel that the sandbox closes
and would weaken the isolation label/RL path (PLAN row). Option A removes the need: the parent allocates a
sealed CPU `memfd`, passes the descriptor over the existing socket with `SCM_RIGHTS` (no namespace crossing, no
shared `/dev/shm`), the worker maps it as ordinary CPU memory and writes float32 logits, and the parent takes a
private snapshot and computes the metric on its own device. The result path no longer contains any CUDA IPC
call; `cuda_open` is retained only as an unused compatibility helper (with the init fix C3 requires).

## 4. Metric semantics — unchanged

The parent still builds every validation batch (loader + frozen tokenizer) **before** the worker starts, with its
own fixed `EVAL_BATCH_SIZE = 16`, holds the targets, and computes `cross_entropy(logits.reshape(-1, vocab),
y.reshape(-1))` and the token-byte counts itself; the worker still returns only `model(x)` logits under
`torch.no_grad()` + bf16 autocast, and still must return a plain `torch.Tensor` of shape `(B, T, vocab)`. The
causality probe (3 cuts, `CAUSAL_ATOL`, exit 9), the repeat check (exit 10), the weights fingerprint, the budget
closure (`ar_budget.py`), the data verify, the exit codes and the `AR_INFO`/`AR_METRIC` lines are untouched.
Only the transport of the logits changed. The values are the same float32 logits: an exact host copy of the
worker output, then an exact host→device copy. On a CPU host the code path is the previously-tested one; on a
GPU host the values are identical to what the (broken) IPC path would have carried, so a CPU smoke before/after
this change must produce the same metric for the same seeds.

## 5. Frozen pin (kit's own definition/loader)

Computed with `autoresearch.trackg.isolation.frozen_file_hashes(node_dir, spec.frozen_paths)` and
`sha256(json.dumps(pin, sort_keys=True))` (the runner's own `frozen_pin_sha256` definition, 20 entries):

| value | old | new |
|---|---|---|
| `frozen_pin_sha256` | `b5f0ec2f351d59369fb14996848fce540309393646cc20c36b6651387b1e4453` | `147f0d35d40df54b5902260f07bf25043f515dbc0066991ab05e3b8c64eda0fc` |
| spec `raw_sha256` (canonical payload) | `5f48b86f03e0224cc12104ef1d335460310c271e723704055dbf9948bc135ff9` | `5c1bdf3d84c0721818adc239186ac30ffa2c2bb3b6b373e56b236e47cbd61192` |

The old pin agrees with the accepted isolation-fix review's table (`b8482875…` → `b5f0ec2f…`) and the old spec
hash agrees with the committed manifest. Recompute:

```bash
cd <worktree> && PYTHONPATH=src python3 - <<'EOF'
import hashlib, json
from pathlib import Path
from autoresearch.trackg.node_template import load_trackg_node_spec
from autoresearch.trackg.isolation import frozen_file_hashes
spec = load_trackg_node_spec("configs/trackg/nodes/autoresearch_fixedtokens.yaml")
pin = frozen_file_hashes(Path("nodes/autoresearch_fixedtokens"), spec.frozen_paths)
print(spec.raw_sha256, hashlib.sha256(json.dumps(pin, sort_keys=True).encode()).hexdigest())
EOF
# -> 5c1bdf3d… 147f0d35…
```

The census/run records will carry the new pin; a fresh campaign must take its pin from this kit tip.

## 6. Tests (Mac CPU, no GPU, no torch)

Environment: macOS 26.6.2 arm64 (Darwin 25.6.0), Python 3.11.11 (`_harness_venv`; pytest 9.2.0, numpy, pyyaml,
tiktoken; **no torch/pyarrow/rustbpe**), `CUDA_VISIBLE_DEVICES=`. Tests that need a real runner sandbox,
`prctl`, `/proc` or `memfd` are Linux-only and cannot pass (or complete) on Darwin; the same selection was run
before and after the change.

| suite | before | after |
|---|---|---|
| `tests/trackg` with 19 Linux-only isolation tests deselected (see note) | **9 failed, 186 passed, 19 skipped, 19 deselected** | **9 failed, 186 passed, 19 skipped, 19 deselected** |
| failure set | identical (diff of `FAILED` lines is empty): `review_round1` 3 (`shadowing_module` ×2, `os.sched_getaffinity`), `review_round2` 5 (`/proc` shadow check ×4, session-kill), `runner` 1 (`os.sched_getaffinity`) | same 9, all Darwin platform failures |
| `test_trackg_isolation.py` non-e2e selection | **8 passed, 1 skipped, 19 deselected** | **8 passed, 1 skipped, 19 deselected** |
| `test_port_autoresearch.py` | **11 passed, 18 skipped** | **11 passed, 18 skipped** (incl. the new option-A test, `test_static`, `test_manifest_complete`) |

Note: the 19 deselected isolation tests are the real-subprocess end-to-end tests (they abort on `prctl`/`unshare`
on Darwin; `test_elapsed_time…` hangs at its helper `wait()` because the sandbox never starts) plus the
`memfd`-only `test_shared_buffer_is_sealed_against_resizing`. They are not evidence for or against this change;
they must run on Ada. The 18 skipped port tests are the torch/pyarrow/rustbpe synthetic smoke and wrapper-metric
probes — these are the tests that would exercise `bind_cpu` + `logits` dynamically and therefore **the Ada CPU
smoke is the first dynamic run of the new transport**.

`py_compile` passes for all changed Python files. `check_static`, the isolation entry/writer checks, the
frozen-path refusal, the manifest re-declaration check and the frozen pin recomputation all pass (the latter
checked in both the port test and the §5 command).

## 7. Commit

Branch `mac-trackg-iso-ar-cond7a-20261120`, base `448f6820f885d70ee6872b692284fe861afd3a33`. The
code/test/spec/manifest change is commit **`0e0b82093c6b12703f16b82239e981aee33ad56b`**; this report is added
in the docs commit that follows it. No push.

## 8. Expected behavior and expected slowdown (to be measured on Ada, not now)

- **Behavior**: isolated run → frozen parent allocates the CPU buffer after `train` and a `bind_cpu`; worker
  returns float32 logits; parent snapshots and (if a GPU is visible) moves them to its device; metric and probes
  as before; run record `isolation.effective: true`, `waivers {}`, `exceptions: []` (spec now has no
  `shared_dev_shm`); the worker's `/dev/shm` is the private tmpfs.
- **Slowdown mechanism**: the buffer is `B × T × vocab × 4 B` = 16 × 2048 × 8192 × 4 = **1 GiB** float32. Per
  evaluation request the worker writes 1 GiB into the CPU buffer (from a CUDA source: a device→host copy, plus a
  transient `.float()` cast of the bf16 output, i.e. ~0.5 GiB in + 1 GiB temp on the device), and the parent
  copies 1 GiB host-side (`snapshot`) and moves 1 GiB host→device. There are 8 logits requests per run (4
  validation batches + 3 causality-probe batches + 1 repeat). At a conservative sequential copy bandwidth of
  ~10 GB/s host and ~25 GB/s PCIe gen4 x16 that is roughly **0.15–0.25 s per request, ≈1.2–2 s per run`, plus a
  transient **≈2 GiB host RSS** (shared memfd page cache + parent snapshot) and **≈1 GiB extra VRAM** (logits
  temp) — small next to the 500-step training budget (estimated ~1 min on an L40S plus compilation), but real
  and to be measured/disclosed from the Ada smoke; the CPU-buffer page cache also counts against the run's
  memory cap (the isolated GPU smoke uses an 8 GiB cap, so this needs checking, not assuming).

## 9. Residual risks and review conditions

Residual risks (disclosed, none a reason to reject the unit on its own):

1. **The torch transport is not dynamically exercised on this Mac** (no torch). The Ada numbered smoke / U2 must
   be the first real run: it must confirm `bind_cpu` + `logits` + parent snapshot run isolated with
   `effective:true`, `waivers {}`, `exceptions: []`, and that the metric is finite and repeatable. Until then
   the GPU behavior is reasoned, not measured.
2. **Slowdown and memory**: the estimate above must be replaced by the Ada measurement (the PLAN row requires
   the slowdown to be disclosed; the row also says the U2 unit runs after one isolated smoke).
3. **`cuda_open` init fix is untested here**: no torch on the Mac; it is a two-line compatibility fix (not on the
   node's result path) and should be exercised by a reviewer's torch probe or the next GPU smoke of a node that
   ever uses IPC (none does now).
4. **Kit copies**: the core master change (`torch.cuda.init()`, docstrings) is on this ar kit branch only; the
   mlab/mlgym kits' `ar_isolated.py` copies stay at `37d99ab0…` until those kits are next updated. They do not
   use the helper, so this is a consistency item, not a behavior difference. The port tests compare each node
   copy to the master of its own branch, so no test is broken by this.
5. **Manifest spec hash**: `node_spec_sha256` was updated without re-declaring the edits under Python 3.12 (the
   Mac has 3.11/3.13). The spec hash is interpreter-independent and all other manifest fields were verified
   identical, so this equals what a 3.12 re-declaration would write; a reviewer can re-run
   `declare_edits` on Ada/3.12 to confirm.
6. **Paper-side documentation**: the paper repo's isolation report §3 item 7 / §5.5 and the port report still
   describe the autoresearch node's CUDA IPC result path; they must be corrected when this change is integrated
   (review C4 is an owned follow-up, out of this kit's scope). The generic `isolation.limits_text()` still
   mentions CUDA IPC as the reason a node may set `shared_dev_shm`; it is text-only (no node sets it now) and
   was deliberately not edited to keep this change out of the accepted isolation core.

Review conditions for the independent reviewer:

- R1: verify the diff (7 files, no frozen path edited outside the intent; node `ar_isolated.py` byte-equal to
  the master; `ar_parent.py`/`ar_data.py`/`prepare.py`/`ar_budget.py` untouched; no CUDA IPC call in
  `run_fixed_budget.py`/`ar_worker.py`; `torch.cuda.init()` present in `cuda_open`; spec has no
  `shared_dev_shm`).
- R2: recompute the frozen pin and spec hash independently (§5); confirm old `b5f0ec2f…` and new `147f0d35…`.
- R3: run the CPU tests on Ada as well (the torch synthetic smoke must pass: `test_port_autoresearch.py`
  wrapper-metric/wrapper-budget/smoke) and confirm the metric and the probe/exit behavior are unchanged.
- R4: run the one isolated GPU smoke on Ada (after review) and confirm `effective:true`, `waivers {}`,
  `exceptions: []`, no `shared_dev_shm` mount, no CUDA IPC error, and record the measured slowdown and memory
  against §8.
- R5: decide the unification of the other kits' `ar_isolated.py` copies (risk 4) and the paper-side text
  correction (risk 6) as part of integration.

## 10. Evidence index (in this kit branch)

`docs/r1/reports/trackg_g0_20261117/TRACKG_AR_COND7A_CODE_UNIT_20261120_captures/` (`SHA256SUMS` sha256
`e47141b2e3342f689ccdc435630e93d94237411e444475675711c60ad86aeef0`, 15 files):

- `before/` — the pre-change tree (detached worktree at `448f682`): `pytest_trackg_selection.txt`
  (9 failed / 186 passed / 19 skipped / 19 deselected), `pytest_isolation_unit_selection.txt`
  (8 passed / 1 skipped / 19 deselected), `pytest_port_autoresearch.txt` (11 passed / 18 skipped),
  `failures.txt`, `pins_and_spec.txt` (old pin `b5f0ec2f…`, spec `5f48b86f…`, `shared_dev_shm: True`),
  `sha256_files.txt`, `environment.txt`.
- `after/` — the committed change (`6f235c4`): same five logs (9/186/19/19, 8/1/19, 11/18, new pin
  `147f0d35…`, spec `5c1bdf3d…`, `shared_dev_shm: False`); `failures.txt` is byte-identical to `before/`'s.
- `code_change.diff` — `git diff 448f682..0e0b820` (the 7-file code/test/spec/manifest change).

Raw captures are macOS-only evidence for the static/unit checks; they are not GPU evidence.
