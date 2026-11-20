# R1-TRACKG-AR-COND7 option A — code-unit report (kit branch, 20 November 2026)

**Status.** The harness/kit change implementing option A for `autoresearch_fixedtokens` is **coded, committed
locally and unit-checked on the Mac** (CPU, no GPU, no torch). The worker now hands logits to the frozen parent
through the existing CPU `memfd` buffer (`iso.SharedBuffer` + `bind_cpu` + `SCM_RIGHTS`) on every host, GPU
included; the CUDA-IPC branch is gone from the node's result path, and `ar_isolated.cuda_open` initializes CUDA
before opening a shared storage. The node spec no longer sets `shared_dev_shm`, so an isolated run is expected
to record `effective: true`, `waivers {}`, `exceptions: []`. The independent code review
(`docs/r1/reviews/TRACKG_AR_COND7A_CODE_REVIEW_20261120.md`, sha256 `261a03e9…`) returned **ACCEPTED for the
code change: 0 BLOCKING, 1 MAJOR (report-only), 6 MINOR**; this revision applies M1 (corrected §8 slowdown) and
m1–m6 (report text/attribution), with **no code change**. **Not pushed, not run on a GPU**: the one isolated GPU
smoke on Ada and its measured slowdown remain the coordinator's/reviewer's C2 step. ENGINEERING_ONLY;
AI-assisted.

## 1. Authority and inputs

- Authority row **R1-TRACKG-AR-COND7** (option A) in `docs/r1/PLAN.md` of the current published paper main
  **`df58bcd5bdeeb7d1cb1cd1c3beddc29376b3575e`** (`git -C .claude-worktrees/ops-tasl-20261116 show
  origin/main:docs/r1/PLAN.md`, row at line 1317), added by commit `bf1bcad6` ("author decisions 19 Nov (noon)
  — … R1-TRACKG-AR-COND7 (option A)"); the row does **not** exist at `de3de54f`, which predates it (a docs
  ancestor of `bf1bcad6`). Row: remove CUDA
  IPC from the result path (`torch.cuda.init()` added to `cuda_open` in the same change), keep isolation
  `effective:true` with no waivers (no `shared_dev_shm`), update kit pins and `frozen_pin_sha256`, run one
  isolated smoke, then U2; measure and disclose the logits-transfer slowdown.
- Condition C3 of `docs/r1/reviews/HARNESS_ISOLATION_FIX_REVIEW_20261119.md` (sha256 `c8e468c7…`, ACCEPTED;
  Ada host-check `6a700509…`): no autoresearch GPU run until the logits transfer design decision is recorded
  (socket/CPU buffer) **and** `cuda_open`'s missing `torch.cuda.init()` is fixed in that same reviewed change.
- Kit branch `mac-trackg-iso-ar-20261117` at `448f6820f885d70ee6872b692284fe861afd3a33` (ar kit after the
  accepted isolation core fix `5403392`, merge `2e65922`); kit-update report
  `docs/r1/reports/trackg_g0_20261117/TRACKG_ISO_KIT_UPDATE_20261119.md` (autoresearch GPU smoke BLOCKED by
  condition 7; old pin `b8482875…` → `b5f0ec2f…` from the isolation fix).
- This worktree: `/Users/wongdowling/Documents/agent_paper_B/.claude-worktrees/harness_ar_cond7_20261120`,
  branch `mac-trackg-iso-ar-cond7a-20261120`, base `448f682`, commit recorded in §7 below. Local only — no push,
  no force push, no deletion, no GPU/TASL/HTCondor, no main-paper-checkout edit.

## 2. Exact changes

The code change is **7 files, +85/−63** (`git diff --numstat 448f682 0e0b820 -- <the 7 paths below>`); the
commit that carries it, `0e0b820`, has **8** files because this report's first version was committed with it
(m1). The frozen result-path files and the isolation master are:

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
   `cuda_open` stay in the frozen library for compatibility. No script of the isolated kits' nodes calls them:
   the ar kit's two former call sites are removed by this change, and a read-only scan of the other two kit
   branches in this clone (`git grep -l 'cuda_share\|bind_cuda\|cuda_open' origin/mac-trackg-iso-{mlab,mlgym}-20261117`
   — `captures/other_kits_cuda_grep.txt`) finds the three identifiers **only inside the frozen `ar_isolated.py`
   definitions**, no parent/worker call site (consistent with the accepted port/isolation reports; this ar-only
   worktree cannot execute the other kits). The ar kit master gets the init fix/docstring; the mlab/mlgym kit
   masters stay at `37d99ab0…` — unused there, so no runtime risk — and propagating the same two-line
   fix/docstring to them is a coordinator-owned integration decision (R5/C5a). The master docstring now states
   that the CPU `memfd` buffer is the path for every isolated node, GPU included, and why CUDA IPC is not used.
   The node copy is byte-identical to the master (the port tests assert this).
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

**Committed-file pin vs deployed pin (m3).** The value above (`147f0d35…`, **20 entries**) is the
*committed-file* pin, recomputed from the kit tree that has no `data/` directory. A run record does **not** carry
this value: `WorkingCopyRunner` takes its pin from the node root it is given (`runner.py:189–190`), and in the
deployed kit that root already contains the `data/ -> $AR_TRACKG_DATA_ROOT/autoresearch_fixedtokens` symlink made
by `setup_command`; `frozen_file_hashes` records that link as `data: link:<absolute target>`, i.e. a **21-entry
deployed pin** whose sha256 depends on the absolute data path. Reproduced with a demo link in
`captures/pins_deployed_note.txt`: committed demo = `147f0d35…` (20 entries); demo copy with
`data -> /tmp/cond7_deployed_data` = `bbe1ca49…` (21 entries, `data: link:/tmp/cond7_deployed_data`, a
target-dependent value, **not** a committed pin). Rule for U2/census: compare a record's `frozen_pin_sha256`
with the **deployed** pin recomputed on the same host/data root (or across runs with the same data path), and
compare the kit against the committed 20-entry `147f0d35…`; do not compare the record to `147f0d35…` directly.
The campaign must take its pin from this kit tip; if a U2 record shows `147f0d35…` it means the data link was
missing/not deployed.

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

Branch `mac-trackg-iso-ar-cond7a-20261120`, base `448f6820f885d70ee6872b692284fe861afd3a33`, four linear
docs/code commits and no merges:

| commit | content |
|---|---|
| `0e0b82093c6b12703f16b82239e981aee33ad56b` | the 7-file code/test/spec/manifest change **and this report's first version** (8 files total) |
| `6f235c432b99e6e4f103c071a34b1441f55798fb` | edits report §7 only (+3/−2) |
| `13786d1fe740e9bb77d10f88685feb88f28a5980` | the before/after evidence captures |
| (this review-fix revision) | docs-only: review M1 (corrected §8) and m1–m6 in this report plus two capture files; no code change. Its sha is in the implementer's hand-off to the coordinator (a commit cannot name its own sha). |

`code_change.diff` in the captures is `git diff 448f682..0e0b820` = **8 files / 526 lines** (the 7 code files
plus report v1), not a 7-file diff (m1). No push; the remote `mac-trackg-iso-ar-*` refs are unchanged.

## 8. Expected behavior and expected slowdown (corrected; still an estimate — Ada smoke must measure it)

- **Behavior**: isolated run → frozen parent allocates the CPU buffer after `train` and a `bind_cpu`; worker
  returns float32 logits; parent snapshots and (if a GPU is visible) moves them to its device; metric and probes
  as before; run record `isolation.effective: true`, `waivers {}`, `exceptions: []` (spec now has no
  `shared_dev_shm`); the worker's `/dev/shm` is the private tmpfs.
- **Real evaluation configuration** (`prepare.py:42,44,60`, `run_fixed_budget.py:138–143`): `MAX_SEQ_LEN = 2048`,
  `EVAL_TOKENS = 40 × 524288 = 20,971,520`, `VOCAB_SIZE = 8192`, `EVAL_BATCH_SIZE = 16` →
  `rows = 10,240`, `steps = rows // 16 = **640**` validation requests, plus the **3** causality-probe requests and
  the **1** repeat request → **644 logits requests per run**, each over one `(16, 2048, 8192)` float32 buffer =
  `16×2048×8192×4 = 1,073,741,824 B = **1 GiB**`. One-way logits volume per run ≈ **640 GiB**; with the three
  copies per request (worker device→host write, parent host snapshot, parent host→device move) the process moves
  ≈ **1.9 TiB per run**. The earlier version of this report wrongly used the *synthetic CPU smoke's* reduced
  config (`MAX_SEQ_LEN 256`, `EVAL_TOKENS = 4×16×256` → 4 batches of a 128 MiB buffer, 8 requests total);
  that is a test-only configuration, not the node's run (`review M1`).
- **Slowdown estimate (to be replaced by the Ada measurement, C2)**: per request the three ~1 GiB host transfers
  are ≈0.04 s (1 GiB D2H at ~25 GB/s PCIe gen4 x16) + 0.05–0.10 s (1 GiB host memcpy snapshot at ~10–20 GB/s) +
  ≈0.04 s (1 GiB H2D) plus sync/allocation overhead → **≈0.15–0.25 s per request**, so **≈97–161 s of transfer
  time per run** (644 × 0.15–0.25 s) — about **80× the volume and ~50–130× the time** of the previously reported
  "≈1.2–2 s". Against the only measured GPU run of this node (17 Nov U2 B1
  `elapsed_seconds_info_only = 67.901 s`, B2 repeat 64.096 s, other good runs 60.9–71.0 s, E2 110.81 s — paper
  repo `docs/r1/reports/trackg_g0_20261117/TRACKG_U2_AR_RESULT_20261117.md`, read in the ops worktree) this
  implies **≈165–229 s per run if the rest is unchanged, i.e. ≈2.4–3.4× B1**, not a small overhead. The 17 Nov
  smoke (wall 564 s, 8 executed runs) would become roughly **≈22–31 min** if the estimate holds
  (564 s + 8 × (97–161 s) ≈ 1,340–1,850 s). **These numbers are an estimate, not a measurement**: the Ada smoke
  must replace them with the measured per-run and per-request wall time against the 67.901 s baseline, and if
  the measured overhead is large it must be carried in the U2/census cost and a mitigation decided (e.g.
  registering/pinning the host buffer, copying H2D directly from the mapping, or reducing the number of
  round-trips) before the U2 (C2/C6).
- **Memory (also to be measured)**: per request the run holds the 1 GiB `memfd` mapping (file-backed shmem, the
  same pages shared by parent and worker) plus the parent's anonymous 1 GiB snapshot (transient) plus device
  tensors — worker: `result` bf16 512 MiB + `.float()` temp 1 GiB; parent: one 1 GiB logits tensor per batch
  and up to three during the probe (`base`, `first_logits`, `after`). The **mechanism matters** (`review m5`):
  the runner's default memory cap is `memory_method="rlimit_data"` (`ResourceLimits`), which does **not** count
  file-backed shared mappings — only the parent's anonymous snapshot is RLIMIT_DATA-accounted — while the 17 Nov
  GPU smoke enforced its cap externally (`systemd-run --user --scope -p MemoryMax=64G …`, recorded as
  `external_limit`), which does count shmem/page cache. The Ada smoke must record the enforcing mechanism, the
  peak host memory under it (the 1 GiB memfd shmem + 1 GiB snapshot included) and the peak VRAM, and compare the
  VRAM with the 17 Nov baseline worker peak `5,211.3 MB`.

## 9. Residual risks and review conditions

Residual risks (disclosed, none a reason to reject the unit on its own):

1. **The torch transport is not dynamically exercised on this Mac** (no torch). The Ada numbered smoke / U2 must
   be the first real run: it must confirm `bind_cpu` + `logits` + parent snapshot run isolated with
   `effective:true`, `waivers {}`, `exceptions: []`, that the metric is **finite** and that the **B2 repeat
   equals B1 within `CAUSAL_ATOL`** (the §9-risk-1 / R4 requirement), and it must measure the wall time and
   memory. Until then the GPU behavior and the §8 slowdown are reasoned, not measured.
2. **Slowdown and memory**: the corrected §8 estimate (644 requests × 1 GiB, ≈640 GiB one-way, ≈97–161 s of
   transfer per run, plausibly 2.4–3.4× the measured 67.901 s B1) is an estimate based on copy bandwidths, not a
   measurement. The Ada smoke/U2 must replace it with measured per-run/per-request wall time and peak host/VRAM
   (and the memory-cap mechanism), and the U2/census cost must carry the measured number (PLAN row; C2/C6).
3. **`cuda_open` init fix is untested here**: no torch on the Mac; it is a two-line compatibility fix (not on the
   node's result path) and should be exercised by a reviewer's torch probe or the next GPU smoke of a node that
   ever uses IPC (none does now). It exists only in this ar kit master: the mlab/mlgym kit masters stay at
   `37d99ab0…` (unused there, no runtime risk); propagation is a coordinator decision recorded under C5a/R5.
4. **Kit copies**: the core master change (`torch.cuda.init()`, docstrings) is on this ar kit branch only. The
   claim that the other kits do not call the helper is backed by the read-only scan captured in
   `captures/other_kits_cuda_grep.txt` (`git grep` on `origin/mac-trackg-iso-mlab-20261117` `fb6e7d7` and
   `origin/mac-trackg-iso-mlgym-20261117` `f07875d`: the identifiers occur only in the frozen `ar_isolated.py`
   definitions, no parent/worker call site). That matches the accepted port/isolation reports; the ar-only
   worktree cannot execute those kits. The port tests compare each node copy to the master of its own branch,
   so no test is broken.
5. **Manifest spec hash**: `node_spec_sha256` was updated without re-declaring the edits under Python 3.12 (the
   Mac has 3.11/3.13). The spec hash is interpreter-independent and all other manifest fields were verified
   identical, so this equals what a 3.12 re-declaration would write; a reviewer can re-run
   `declare_edits` on Ada/3.12 to confirm.
6. **Paper-side documentation and stale limits text (C5b/C5c):** the paper repo's isolation report §3 item 7 /
   §5.5 and the port report still describe the autoresearch node's CUDA IPC result path; they must be corrected
   when this change is integrated (review C4 of the isolation-fix review is an owned follow-up, out of this
   kit's scope). Separately, the accepted core's `isolation.limits_text()` still contains the CUDA-IPC sentences
   (`isolation.py:617` "with shared_dev_shm (CUDA IPC), /dev/shm is shared…" and `:621` "CUDA IPC across the PID
   namespace is exercised only by the kits' GPU smoke"), and `runner.py:544` copies that text into **every
   isolated run record's `isolation.limits`**, so the AR records will carry a now-stale sentence even though no
   node uses CUDA IPC. Text-only; deliberately left in the accepted core to keep this change out of it; the
   coordinator owns the decision/record for fixing it (C5c/m6).

Review conditions for the independent reviewer:

- R1: verify the code diff (7 code files; commit `0e0b820` has 8 files because it also carries report v1 —
  reviewer m1); no frozen path edited outside the intent; node `ar_isolated.py` byte-equal to the
  master; `ar_parent.py`/`ar_data.py`/`prepare.py`/`ar_budget.py` untouched; no CUDA IPC call in
  `run_fixed_budget.py`/`ar_worker.py`; `torch.cuda.init()` present in `cuda_open`; spec has no
  `shared_dev_shm`.
- R2: recompute the frozen pin and spec hash independently (§5); confirm old `b5f0ec2f…` and new `147f0d35…`
  (committed, 20 entries) and use the deployed (21-entry) pin rule for any record comparison.
- R3: run on Ada the CPU/Linux tests that cannot run on Darwin, with the 19 Linux-only isolation tests
  **included** (in particular `test_dev_shm_is_private_unless_the_node_opts_out` and
  `test_shared_buffer_is_sealed_against_resizing`), and the full `test_port_autoresearch.py` including the 18
  torch/pyarrow wrapper-metric/budget/smoke tests; confirm no new failures vs base `448f682` and that the
  metric, probe, exit and dynamic `bind_cpu`/`logits` behaviour are unchanged.
- R4: run the one isolated GPU smoke on Ada (after C1) and confirm `effective:true`, `waivers {}`,
  `exceptions: []`, no `shared_dev_shm`, no CUDA IPC error, the metric is **finite**, and the **B2 repeat equals
  B1 within `CAUSAL_ATOL`**; record the measured per-run and per-request wall time against the 17 Nov B1
  `67.901 s`, the peak host memory **and the mechanism that enforces the cap** (`rlimit_data`/`rlimit_as` does
  not count the file-backed memfd; the smoke's external `systemd-run MemoryMax` does), and the peak VRAM
  (baseline `5,211.3 MB`); replace §8's estimate with these values.
- R5: record the ownership/decision for (a) propagating the `cuda_open` init fix/docstring to the mlab/mlgym
  kit masters (`37d99ab0…`, unused → no runtime risk), (b) the paper-side text correction (isolation-report §3
  item 7/§5.5 and port report; C4 of the isolation-fix review), and (c) the stale `isolation.limits_text()`
  CUDA-IPC sentences that land in every AR run record's `isolation.limits` (`isolation.py:617,621`,
  `runner.py:544`).

## 10. Evidence index (in this kit branch)

`docs/r1/reports/trackg_g0_20261117/TRACKG_AR_COND7A_CODE_UNIT_20261120_captures/` (`SHA256SUMS` sha256
`49d4c3a89339fa43f0654d708faf5ba4c89acc9c0836f0bdbbeff3b75e5cbd8d`, 17 files):

- `before/` — the pre-change tree (detached worktree at `448f682`): `pytest_trackg_selection.txt`
  (9 failed / 186 passed / 19 skipped / 19 deselected), `pytest_isolation_unit_selection.txt`
  (8 passed / 1 skipped / 19 deselected), `pytest_port_autoresearch.txt` (11 passed / 18 skipped),
  `failures.txt`, `pins_and_spec.txt` (old pin `b5f0ec2f…`, spec `5f48b86f…`, `shared_dev_shm: True`),
  `sha256_files.txt`, `environment.txt`.
- `after/` — the changed tree (captured at `6f235c4`; code content identical to `0e0b820`): same five logs
  (9/186/19/19, 8/1/19, 11/18, new pin `147f0d35…`, spec `5c1bdf3d…`, `shared_dev_shm: False`);
  `failures.txt` is byte-identical to `before/`'s.
- `code_change.diff` — `git diff 448f682..0e0b820` = **8 files / 526 lines** (the 7 code/test/spec/manifest
  files plus report v1; m1), not a 7-file diff.
- `pins_deployed_note.txt` — committed 20-entry pin `147f0d35…` vs the deployed demo 21-entry pin with a
  `data -> /tmp/cond7_deployed_data` link (`bbe1ca49…`, target-dependent; m3).
- `other_kits_cuda_grep.txt` — read-only `git grep` provenance for the claim that the mlab/mlgym kit node
  scripts do not call `cuda_share`/`bind_cuda`/`cuda_open` (only the frozen `ar_isolated.py` definitions; m4).

Raw captures are macOS-only evidence for the static/unit checks; they are not GPU evidence.

## 11. Review-findings disposition (review `261a03e9…`)

| finding | severity | disposition in this revision |
|---|---|---|
| M1 slowdown wrong by ~80× | MAJOR | §8 rewritten to the real config: 640+3+1 = 644 requests, 640 GiB one-way / ≈1.9 TiB moved, ≈97–161 s/run, ≈2.4–3.4× the 67.901 s B1, ≈22–31 min for the 17 Nov smoke; the old "8 requests / ≈1.2–2 s / small" claim is gone; the figures are marked as an estimate that the Ada smoke must replace (C2). |
| m1 commit attribution / `code_change.diff` | MINOR | §2 (commit has 8 files; code change 7 files), §7 (commit table: report v1 is in `0e0b820`, `6f235c4` edits §7 only), §10 (`code_change.diff` = 8 files/526 lines). |
| m2 authority commit | MINOR | §1: row at origin/main `df58bcd5…` line 1317, added by `bf1bcad6`; explicitly not at `de3de54f`. |
| m3 committed vs deployed pin | MINOR | §5: committed 20-entry `147f0d35…` vs deployed 21-entry pin (`data: link:<abs target>`, target-dependent); comparison rule for U2/census; demo in `captures/pins_deployed_note.txt`. |
| m4 other-kit "verified" claim / propagation | MINOR | §2 item 3 and §9 risks 3–4: provenance is the read-only `git grep` of the two kit branches, captured in `captures/other_kits_cuda_grep.txt`; mlab/mlgym masters not touched (`37d99ab0…`, unused), propagation is a coordinator decision (C5a/R5). |
| m5 memory-cap wording + R4 | MINOR | §8 memory bullet: `rlimit_data` does not count file-backed shmem, only the anonymous snapshot; the 17 Nov smoke uses external `systemd-run MemoryMax` (recorded `external_limit`), which does. R4 now requires metric finiteness, B2 repeat within `CAUSAL_ATOL`, the measured wall time/memory and the enforcing mechanism. |
| m6 stale `limits_text()` | MINOR | §9 risk 6: exact lines (`isolation.py:617,621`, `runner.py:544`), that it lands in every AR run record's `isolation.limits`, text-only, left in the accepted core; coordinator decision recorded under C5c/R5. |
