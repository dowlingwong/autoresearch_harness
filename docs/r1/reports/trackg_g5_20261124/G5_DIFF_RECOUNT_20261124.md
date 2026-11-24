# Diff recount at the G.5 harness commits (R1-G5-DIFF-RECOUNT, option A), 24 November 2026

**Status.** Coded, committed locally on two branches and unit-tested in a CPU container. Nothing is pushed. Nothing
ran on a GPU, on TASL-2 or on TASL-7. No label, evaluation value, sealed file, G.1b or G.5 output was read. The
report is ENGINEERING_ONLY. AI-assisted (Claude). The unit is awaiting independent review.

**Authority.** PLAN row `R1-G5-DIFF-RECOUNT` (paper `main` `3024db3f`, `docs/r1/PLAN.md` line 1508), option A: the
G.5 harness commits apply a manager diff after recomputing each hunk header's line counts from the hunk body
(`git apply --recount` semantics); the context must still match; structurally malformed diffs stay refused. Also the
user's task text for this unit, and the coordinator's constraint (below) that the unit must keep P5
(`g5_tasl.measurement_path_check`).

## 1. Commits

| Family | Base (reviewed G.5 tip, READY 95) | Branch | Code commit | Report commit |
|---|---|---|---|---|
| G5-7 | `52e3c0b0` | `mac-trackg-g5-7-20261124` | `9eb0eb40` | this file's commit, on top |
| G5-AR | `00829da1` | `mac-trackg-g5-ar-20261124` | `197d3ecb` (`cherry-pick -x` of `9eb0eb40`) | cherry-pick of the report commit |

The files the unit adds or changes are byte-identical in both families:

| File | sha256 (both families) |
|---|---|
| `src/autoresearch/trackg/campaign_v2_diff.py` (changed; was `904e498b…`) | `6a2beb4df3a2d31fd16688f3ab9d4a29209a07ccc4a476a1bd82ddcdae3c3fbc` |
| `src/autoresearch/trackg/audit_g5.py` (changed; was `1089dbac…`) | `d25ff41285111657b4399dd90051b1de77bc4602f6934412745f19f716567e99` |
| `scripts/trackg_g4_reclassify.py` (new) | `8c1e1619b5ce13776f7132324e45bbfcf0eba8287a9942baca37d4895d9f2c43` |
| `tests/trackg/test_trackg_diff_recount.py` (new) | `1071397534cd0347d121e1ec4cda17aeed995e1a352ec409dfd9a61b1d884efb` |
| `tests/trackg/test_trackg_g4_reclassify.py` (new) | `2b17410b095af1322a3d81c4f9f5248d23236d6bfaaf8ad7f94bd905b8384d6c` |
| `docs/r1/reports/trackg_g5_20261124/G5_DIFF_RECOUNT_20261124.md` (new) | this file |

## 2. Where the recount lives, and why (P5)

**The constraint.** The paper's P5 check, `g5_tasl.measurement_path_check` (paper `main`, `sweep2/t1_tasl/g5_tasl.py`
lines 1359-1458), works per family against that family's G.1b pin (`c025f89a` for G5-7, `2d09f9a9` for G5-AR).
It STOPs when any of these holds:

- a file of the measurement path differs from the pin. The measurement path is all of `src/autoresearch/trackg/`
  except the four engine files (`manager_v2_diff.py`, `campaign_v2_diff.py`, `proposal_v2_diff.py`, `pilot_g4.py`),
  plus `nodes/`, `third_party/`, `configs/trackg/` and every module of `noise_floor`'s import closure;
- a new file appears under `trackg/` other than `audit_g5.py` or `diagnostics_g5.py`, or anywhere under `nodes/`,
  `third_party/`, `configs/trackg/` or `trackg/node_lib/`;
- `noise_floor`'s import closure differs between the pin and the G.5 commit.

The check protects the G.1b noise-floor measurement code: G.5's labels are compared with the noise floor measured
on the pin, so the code that measured it must be the code that runs. `diff_support.py` holds `parse_unified_diff`
and `apply_diff`. It is in that path twice: as a `trackg/` file, and as a member of `noise_floor`'s closure (with
`runner.py`). Editing it would STOP every G.5 job at P5, and it would change the noise-floor measurement code.

**The choice: option (A).** `diff_support.py` and every other measurement-path file stay byte-identical. The recount
is a pre-pass, `recount_hunk_headers`, in the engine file `campaign_v2_diff.py`. It rewrites only the two counts in
the `@@` lines of a copy of the manager's diff. The unchanged `diff_support.apply_diff`, called by the unchanged
`runner.py`, then parses and applies the copy. `audit_g5.py`, an allowed file, imports the same function for its
re-materialisation. I put the function in `campaign_v2_diff.py`, not in a new module, for three reasons:

- it is the engine that hands each diff to the runner;
- it adds no module, so no new file has to be placed and justified;
- it adds no new `autoresearch` import, because `campaign_v2_diff` already imported `diff_support`.

The pre-pass reads three names from `diff_support`: `_HUNK_RE`, `_ALLOWED_GIT_HEADERS` and `_REFUSED_GIT_HEADERS`.
It reads them so that its view of a diff's structure is the strict parser's own. They are private names, but the
file is frozen by P5, so they cannot drift. Option (B), changing `diff_support.py`, would need a paper-side change
to P5 and an author decision. It was not built.

**Evidence** (run from a scratch detached paper worktree at `origin/main` `3024db3f`):

- `noise_floor`'s import closure (`g5_tasl.import_closure`) is the same 15 files at all six commits: both pins
  (`c025f89a`, `2d09f9a9`), both old tips (`52e3c0b0`, `00829da1`) and both new code tips (`9eb0eb40`, `197d3ecb`).
  The files are `autoresearch/__init__`, `control_plane/__init__`, `control_plane/permissions`, `nodes/__init__`,
  `nodes/spec`, and in `trackg/`: `__init__`, `determinism`, `diff_support`, `gpu_identity`, `isolation`,
  `node_template`, `noise_floor`, `outcome`, `runner`, `selection`. No engine file is in it.
- `git diff --quiet` from each pin to its new tip shows `diff_support.py`, `runner.py`, `noise_floor.py`,
  `pilot_g4.py`, `manager_v2_diff.py` and `declared_edits.py` unchanged.
- `measurement_path_check` passes for both families (§7).

## 3. The rule

`campaign_v2_diff.recount_hunk_headers(diff_text) -> DiffRecount` (rule id `trackg_diff_recount/v1`). The scan
follows `parse_unified_diff`'s structure: blank and allowed git header lines are skipped; a `--- `/`+++ ` pair is a
file header; `@@` lines are hunks. Per hunk:

1. **The header first.** The pre-pass reads the hunk with the header's counts, as `_parse_hunk` does. If that reading
   succeeds and what follows is not more hunk body, the hunk is left exactly as it is. "Not more body" means one of:
   the end of the diff; an `@@` line; after blank lines, the end, a `---`/`+++` pair, or any line that is not a body
   line. Every diff that the strict parser accepts is therefore returned byte for byte. So is every refusal that
   does not come from wrong counts. The tests check this on a seeded corpus of over 300 accepted diffs.
2. **Otherwise the body decides.** The body runs from the line after the header to the first of these:
   - a line starting with `@@`;
   - a file-level git line: `diff --git`, `index`, `new file mode`, `deleted file mode`, or a refused rename, copy,
     mode or binary line (no body line can start with these);
   - in a file patch without a `diff --git` line only, a `--- ` line immediately followed by a `+++ ` line;
   - the end of the diff.

   The old count is the context lines plus the removed (`-`) lines. The new count is the context lines plus the
   added (`+`) lines. A context line starts with a space or is empty. `\ No newline at end of file` markers are not
   counted. The header's start lines and its trailing text (for example a function name) are kept; only the counts
   are rewritten, as `start,count`.
3. **Blank lines.**
   - Before the next file header or the end of the diff, empty lines and whitespace lines that do not start with a
     space are separators. That is how the strict parser treats them between file patches.
   - A line of one space is always empty context (the canonical form).
   - Blank lines before an `@@` line are empty context, as git counts them. The strict parser refused that layout, so
     this widens nothing it accepted.
4. **No safe recount: the hunk is left as it is**, the scan stops, and the strict parser refuses the diff with the
   reason it gave before. That happens for:
   - `bad_line_tag`: a body line that starts with none of space, `-`, `+` or `\`, and is not a terminator. A tab line
     counts here, as it does for the strict parser. Refused `malformed_hunk` (or `malformed_diff`, as before).
   - `empty_body`: no counted line. Refused `malformed_hunk`.
   - `no_old_lines`: a body with no context and no removed line, whose header has old lines. Only the header's start
     line places such an insertion, and `diff_support` locates a hunk with an old count of 0 one line later than one
     with an old count of 1 or more. A recount would move it, so it is refused (`malformed_hunk`). A header whose old
     count is already 0 is recounted: only its new count changes, and the position stays the same.
   - `ambiguous_file_header_pair`: see the next point.
   - A hunk header that does not match the strict pattern (`bad_hunk_header`, then `malformed_hunk`), or a line the
     strict parser refuses between file patches.
5. **The `-- `/`++ ` case.** Removing a line whose text begins with `-- ` writes `--- …`, and adding one that begins
   with `++ ` writes `+++ …`. Such a pair looks like a file header.
   - **The header decides first** (point 1). With correct counts the pair is body, as before.
   - **Patch with a `diff --git` line:** inside a recounted body the pair is body, as in git, whose own recount ends a
     body only at `@@ ` or `diff `. If the pair was really a second file's header that lacked its `diff --git` line,
     the removed line `-- a/<path>` must match the first file exactly. So the diff is refused
     (`context_mismatch`), never applied elsewhere (tested).
   - **Patch without a `diff --git` line** (the prompt asks for `--- a/`/`+++ b/` headers): a pair directly followed
     by an `@@` line is a file header, and the strict parser checks it as one (safe path, no rename, each file once).
     A pair that is not followed by `@@` could be either a removed `-- …` line plus an added `++ …` line, or a header
     with no hunks. That is ambiguous, so it is left and refused (tested).
   - Remaining risk: a body pair followed by `@@` is read as a header. That needs a removed line `-- a/P` and an added
     line `++ b/P` naming an editable file P, other than the current file, whose next hunk's context then matches P
     exactly. Then the edit would apply to P. I judge this negligible, but it is not impossible.
6. **Zero hunks.** A correctly counted `@@ -0,0 +0,0 @@` with no body is a no-op hunk that the strict parser accepts.
   The header fits, so it is left as before. An empty body under a non-zero header is refused (`empty_body`).
7. **Matching is not loosened.** Context matching, the unique-offset search, ambiguity refusal, the scope check, path
   safety, new-file and shadowing checks, and inert and duplicate detection are all `diff_support`'s, unchanged. The
   expected position uses the old count only to decide between "at" and "after" the start line. That is why
   `no_old_lines` is refused.

## 4. Comparison with `git apply --recount`

Git's `recount_diff` (`apply.c`) starts after the header line. It counts `' '` and `'\n'` lines as context (both
counts), `'-'` lines as old, `'+'` lines as new, and skips `'\\'` lines. It stops at a line that starts with `@@ ` or
`diff `. On any other line it warns `recount: unexpected line` and keeps the header's counts. On an empty hunk it
warns `recount: ignore empty hunk` and keeps the header. `parse_fragment` then reads the hunk with those counts, and
`apply_one_fragment` matches it with git's own rules: the start guess comes from `newpos`; with zero leading or
trailing context it must match the file's beginning or end, unless `--unidiff-zero`; whitespace and fuzz options
apply.

| Point | git `--recount` | this rule |
|---|---|---|
| Counting | context ` `/empty, `-`, `+`; `\` skipped | the same |
| Body ends at | `@@ ` or `diff ` | `@@`, any file-level git line, end; a `---`/`+++` pair followed by `@@` only in patches without `diff --git` |
| Header counts | ignored | used only where they fit the body exactly (point 1); then identical to the old rule |
| `---`/`+++` pair in the body | body (multi-file patches without `diff --git` then break in git) | body with `diff --git`; file header (or an ambiguity refusal) without |
| Blank lines before the next file or the end | context | separators (the strict parser's convention); a one-space line is context |
| Unexpected line, empty hunk | header kept (usually a later failure) | header kept, strict parser refuses (same in effect) |
| Context-free insertion whose header had old lines | header start used | refused (`no_old_lines`) |
| Matching | git's (newpos guess, beginning/end anchoring, options) | `diff_support`'s, unchanged (zero fuzz, exact context, unique offset) |

## 5. Scope: what changes, what records gain

**Where the recount applies:**

- **Campaign path** (`run_v2_diff_campaign`): every proposal's diff. This covers G.5 campaigns (`audit_g5 campaign`)
  and **G.4's pilot**: `pilot_g4.run_pilot` calls `run_v2_diff_campaign`, so a G.4b on these commits recounts. The
  G.4 behaviour therefore changes at these pins, although `pilot_g4.py` itself is byte-identical.
- **Audit re-materialisation** (`audit_g5.prepare_states`): every chain diff and the unit's last diff. The runner of
  each audit run gets the recounted text, as the campaign's runner did.
- **Inert and duplicate code-state hashing:** these run inside `apply_diff` on the text it is given, so they see the
  recounted copy, as the campaign did.
- **Not changed:** `diff_support` itself; `noise_floor` and `declared_edits` (G.1b's declared edits stay on the strict
  rule; G.1b is unaffected); the T0 path; `manager_v2_diff.py` (the prompt is the same).

**Engines that change:** `campaign_v2_diff.py` (both the G.4 and the G.5 campaign path) and `audit_g5.py`
(re-materialisation, and two record fields). Pinned unchanged: `pilot_g4.py`, `noise_floor.py`, `manager_v2_diff.py`,
`proposal_v2_diff.py`, `diff_support.py`, `runner.py`.

**Records:**

- **Campaign ledger line** (every proposal with a diff): a new block, `diff_recount`:

  ```
  {"rule": "trackg_diff_recount/v1", "changed": bool, "hunks_seen": n, "hunks_recounted": k,
   "recounted": [{"line": header line in the diff (1-based), "header": [old, new], "body": [old, new]}, ...],
   "left_to_strict_parser": null | "bad_line_tag" | "empty_body" | "no_old_lines" | "ambiguous_file_header_pair"
                            | "bad_hunk_header" | "not_a_file_header" | "unsupported_file_line",
   "left_at_line": null | n, "raw_diff_sha256": sha256(proposal.diff), "applied_diff_sha256": sha256(applied text)}
  ```

  No diff text is in the block.
- **Run record `diff`** (on disk and in the ledger's `runs[]`): `diff_text` and `diff_sha256` are those of the text the
  runner applied. That is the recounted copy when `changed` is true, and the manager's text otherwise. This is the
  runner's own field, written by the unchanged `runner.py`.
- **`campaign_record.json`:** `diff_rule`.
- **`chunk_record.json`:** `diff_rule` and `diffs_recounted`.
- **Audit rows** (`trackg_g5_audit_row/v1`): unchanged.

**Paper readers checked:**

- `g5_job.parse_ledger` and `audit_plan` read `proposal.diff`, `runs[].code_state_sha256` and
  `incumbent_code_state_sha256`.
- `g5_features._states` reads `runs[].diff.code_state_sha256` / `code_state_raw_sha256`.
- `g5_assemble` reads `runs[].code_state_sha256` or `runs[].diff.code_state_sha256`.
- `g5_job.campaign_diagnostics` counts decisions.
- `g5_job` reads `campaign_record.json`/`chunk_record.json` with `.get`. The only fixed key set, `RECORD_KEYS`, is the
  job's own record, which this unit does not write.
- None of them reads `diff_sha256` or `diff_text`, or rejects an extra key.

The end-to-end test of `test_trackg_audit_g5.py` passes 63 of 63 at both tips with the paper worktree present
(`AR_G5_PAPER_ROOT` = the scratch `origin/main` worktree). It pipes real engine output through the paper's
`campaign_file`, `campaign_diagnostics`, `tar_members`, `audit_plan`, `audit_chunk_file`, `read_rows` and
`reduce_rows`.

**Hashes:**

- **Code state** (`code_state_sha256`, `code_state_raw_sha256`): the hash of the resulting editable files. A recounted
  diff yields the same parsed hunks as the correctly counted diff, so the same files and the same hashes (tested).
  `normalised_diff_sha256` hashes the parsed hunk lines, so it is also the same (tested).
- **`diff_sha256`** (in the run record): the hash of the text given to `apply_diff`. That is the recounted copy when
  headers were rewritten, so it is **not** the hash of the manager's raw text in that case. The raw hash is in
  `diff_recount.raw_diff_sha256`. `diff_sha256` could not be kept as the raw hash without changing `runner.py` or
  `diff_support.py`, which P5 forbids.

## 6. O1 compliance (D7-DIFF B review, item O1)

- `proposal.diff` in the ledger is the manager's text as written. The campaign stores it before the recount and
  passes only the copy to the runner (tested: `ledger[1]["proposal"]["diff"] == LR_WRONG`).
- The audit re-materialises from the ledger's `proposal.diff` texts (the paper's `audit_plan` chains them) through
  the same `recount_hunk_headers` as the campaign. So rebuilt hashes equal the ledger's (tested: a two-step chain of
  wrong-count diffs re-materialises to both ledger states). Without the recount the chain would not apply (tested),
  which is why the audit must use it.

## 7. Verification

**Tests** (each file run alone, `TMPDIR` under `/tmp/hrcdev/t`, `/tmp/tgport312/bin/python -B -m pytest -q`; the same
counts on both worktrees):

| File | Result |
|---|---|
| `tests/trackg/test_trackg_diff_recount.py` (new) | 54 passed |
| `tests/trackg/test_trackg_g4_reclassify.py` (new) | 5 passed |
| `test_trackg_diff_support.py` | 27 passed |
| `test_trackg_campaign_v2_diff.py` | 16 passed |
| `test_trackg_pilot_g4.py` | 31 passed |
| `test_trackg_noise_floor.py` | 33 passed |
| `test_trackg_isolation.py` | 28 passed |
| `test_trackg_audit_g5.py` | 61 passed, 2 skipped without the paper worktree; 63 passed with `AR_G5_PAPER_ROOT` set |
| `test_trackg_declared_edits.py` | 3 passed |
| `test_trackg_runner.py` | 12 passed |
| `test_trackg_review_round1.py`, `_round2.py`, `_proposal_v2_diff.py`, `_smoke.py` (G5-7 worktree) | 42, 16, 3, 1 passed |
| `test_trackg_t0_identity.py` (G5-7 worktree) | 2 passed, 1 failed; the same failure with this unit's changes stashed, on the unchanged tip. The T0 probe subprocess (`scripts/trackg_t0_identity_probe.py`) exits 1 in this container. Not caused by this unit; not investigated further. |

**What the new tests cover:**

- **Wrong counts accepted:** the report's example shape, `@@ -21,18 +21,25 @@` over 17 old and 21 new lines, plus six
  other wrong pairs, give the correct text and the same application as the correctly counted diff. A too-small
  header (formerly `malformed_diff`) is accepted. Omitted counts and header trailing text are handled.
- **Matching not loosened:** wrong context refused; a wrong removed line refused; offsets equal to the correctly
  counted diff at four start lines; ambiguity preserved.
- **Malformed stays refused (15 cases):** bad headers, bad tags, a tab line, `..`, absolute and quoted paths, no
  hunks, rename, mode change, binary, leading prose, `---` without `+++`, a file twice, a scope violation. Also a bad
  header after a recounted hunk; empty bodies; context-free insertions; the zero-zero hunk.
- **The `-- `/`++ ` cases:** right counts; the ambiguous pair; a pair before `@@` in three variants, each refused;
  git style; a missing `diff --git` line refused.
- **Structure:** several hunks and files; blank-line separators and empty context; three "No newline" cases.
- **Identity:** byte identity on more than 300 accepted random diffs; more than 300 perturbed headers restored
  exactly.
- **Campaign:** ledger and record fields; a refusal left to the strict parser.
- **Audit:** re-materialisation; the chunk record fields; `diff_support` still strict.

**Paper checks** (scratch detached worktree of paper `origin/main` `3024db3f` at `/tmp/hrcdev/paper`; harness git dir
`/tmp/pb/harness`; code tips `9eb0eb40`/`197d3ecb`; repeated on the final tips, see the hand-off):

- `g5_pin_check.py /tmp/pb/harness <G5-7> <G5-AR>` exits 0. `g5_changes` are the two G.5 reports already there,
  `audit_g5.py`, `campaign_v2_diff.py`, `test_trackg_audit_g5.py`, and this unit's script, two test files and
  report. The pin delta is unchanged.
- `g5_job.check_engines` passes at both tips. The marker, schemas and truncation class are unchanged in `audit_g5.py`.
- `g5_tasl.measurement_path_check` passes for both families: 308 measurement-path files unchanged;
  `engine_files_in_closure` `[]`; closure 15 files, unchanged. `g5_added` is the reports, `audit_g5.py`, the test
  files and `scripts/trackg_g4_reclassify.py`. `scripts/` is outside the measurement directories.

## 8. Offline re-classification of the G.4 refusals

**What is committed.** On paper branch `origin/ops-tasl-20261116`,
`docs/r1/reports/tasl_chain_20261122/G4_TASL_PILOT_RESULT_20261122_captures/` holds, per node, `SEAL_RECEIPT.json`,
`SHA256SUMS`, `sealed_SHA256SUMS`, `attempt.json`, `job_record.json` and `pilot_summary.json`. The rest is preflight
records, the server record, the dates recompute and `MANIFEST.sha256` (`git ls-tree -r -l`). **No manager reply and
no ledger is committed**: the report says "no `out.tar`", and the summaries hold only counts (`refusal_reasons`:
cifar10 53/6, fmnist 45/3, breakout 24/24). So the 122 `malformed_hunk` replies, and the 33 `malformed_diff`, can
only be re-classified on TASL-2, from `sealed_g4/g4-tasl-20261121-a/<node>/attempt-1/out.tar`.

**The script:** `scripts/trackg_g4_reclassify.py`. It is outside `trackg/`, because P5 allows no other new file there.

1. **It verifies each attempt against the committed captures.**
   - The sha256 of `out.tar` must equal the captures' `SHA256SUMS` and `sealed_SHA256SUMS`. The report gives cifar10
     `09566310…`, fmnist `fecf36e4…`, breakout `bf19d401…`.
   - The attempt's `SHA256SUMS` and `SEAL_RECEIPT.json` must be byte-identical to the captures' copies, and every
     file `SHA256SUMS` lists must verify.
   - `job_record.json` must match the receipt's `job_record_json_sha256`.
   - Any mismatch STOPs (exit 2).
2. **It reads only `out/pilot.json` and `out/pilot/instance_*/campaign_ledger.jsonl`**, in memory, as regular files. The
   output lists the members read. It reads no run record (a run record on disk keeps the sealed test block), no log,
   no result channel, no summary, no label, no evaluation data, and nothing from G.1b or G.5.
3. **It rebuilds every incumbent from the pinned task code.** The start is the node kit of `--harness-root`, whose
   node spec must have the pilot's `node_spec_sha256`. P5 keeps `nodes/` and `configs/trackg/` byte-identical to
   `c025f89a`, the G.4 harness. The ledger's kept diffs are then applied in order. Each rebuilt incumbent is checked
   against the code-state hashes the ledger recorded: the baseline's `code_state_*`, each refused run's
   `incumbent_code_state_*`, each kept run's `code_state_*`, normalised and raw. The sealed `out.tar` also holds each
   refused run's working copy (`instance_*/<run_id>/work/`, the runner's layout, unchanged by a refusal). The script
   does not use it: the rebuild needs only the ledger and the pinned kit, and it is checked against the ledger's
   hashes.
4. **It re-runs only the parse/apply step,** for every `not_run_refused` proposal: `apply_diff(..., write=False)`
   against the rebuilt incumbent, with the duplicate registry rebuilt from the ledger's executed runs. It does this
   twice: with the G.4 rule (the manager's text) and with the new rule (the recounted copy). Nothing is trained or
   run, and no model is called. For every other proposal it checks that the recount leaves the text unchanged.
5. **Output** (JSON, schema `trackg_g4_reclassify/v1`, per node and total):
   - `refused_before` (by the ledger's reason);
   - `g4_rule_replay` (matches with the ledger, plus mismatches);
   - `new_rule` (`applied`, `duplicate`, `inert`, `refused:<reason>`) and `new_rule_runnable`;
   - `by_reason` (before -> after);
   - `with_recounted_headers` and `left_to_strict_parser`;
   - `accepted_by_g4_rule` and `accepted_text_unchanged`;
   - `incumbent_unverified` and per-step rows (instance, step, before, old, new; no diff text and no metric).

   Exit 0 when complete and consistent; 3 when an instance stopped, an incumbent did not verify or the G.4-rule
   replay disagreed with the ledger. `--out` writes a new file, never an existing one.

**Tested on a synthetic sealed attempt** built as the G.4 job builds one. A G.4-rule campaign (the recount patched
out, as on `c025f89a`) writes `out/pilot/instance_1/`, then `out.tar` is made of `out`. `SHA256SUMS` covers
`out.tar`, `job_record.json` and `server_record.json`; the seal copies them and writes `SEAL_RECEIPT.json`; a captures
directory holds the sums and the receipt. The test expects:

- 5 refusals re-classified: 4 `malformed_hunk` and 1 `malformed_diff`;
- after: 2 applied, 1 duplicate, 1 `context_mismatch`, 1 `malformed_hunk` kept;
- G.4-rule replay 5 of 5 matching the ledger;
- the accepted proposal's text unchanged;
- only the two allowed members read;
- no diff text or metric in the output.

Also tested: a tampered receipt or tar STOPs; another node spec STOPs; a changed kit is reported and classifies
nothing.

**The command on TASL-2.** It needs a harness checkout at the final G5-7 tip (for example `harness_g5`, which the
G.5 spec names), the torch312 environment (which has the harness's dependencies) and the committed captures:

```
cd /home/dowling/r1_tracks/t1_tasl
H=/home/dowling/r1_tracks/t1_tasl/harness_g5          # detached at the final G5-7 tip, clean
git -C "$H" rev-parse HEAD; git -C "$H" status --porcelain
git -C paper_g4 fetch origin ops-tasl-20261116
mkdir -p /tmp/g4caps && git -C paper_g4 archive origin/ops-tasl-20261116 \
  docs/r1/reports/tasl_chain_20261122/G4_TASL_PILOT_RESULT_20261122_captures | tar -x -C /tmp/g4caps
C=/tmp/g4caps/docs/r1/reports/tasl_chain_20261122/G4_TASL_PILOT_RESULT_20261122_captures
S=sealed_g4/g4-tasl-20261121-a
envs/torch312/bin/python -B "$H/scripts/trackg_g4_reclassify.py" --harness-root "$H" \
  --pair "$S/mlab_cifar10/attempt-1"           "$C/mlab_cifar10" \
  --pair "$S/mlgym_fmnist/attempt-1"           "$C/mlgym_fmnist" \
  --pair "$S/mlgym_breakout_minatar/attempt-1" "$C/mlgym_breakout_minatar" \
  --out records/g4-tasl-20261121-a/g4_reclassify_20261124.json
echo rc=$?
```

The script reads the sealed attempt and writes only the new `--out` file. A temporary directory holds the rebuilt
incumbents and is removed afterwards. The new-file shadowing check uses the pilot's recorded interpreter when it
exists on the host (it does on TASL-2), else `python3`; the output names the interpreter used. The breakout node ran
under jax312, but the parse/apply step does not import the task code, so torch312 can run all three nodes.

**Expected on TASL-2** (a smoke check, not a result):

- `refused_before` totals `malformed_hunk` 122 and `malformed_diff` 33;
- `g4_rule_replay.matches_ledger` 155 with no mismatch;
- `incumbent_unverified` 0;
- `accepted_by_g4_rule` 25 (22 ran + 3 duplicates) with `accepted_text_unchanged` 25;
- `new_rule` and `by_reason` are the measurement.

The 33 `malformed_diff` are included because a header smaller than its body makes the strict parser stop early and
refuse the rest as `malformed_diff`; some of the 33 may be count errors too.

## 9. What the paper side must update

1. **G.5 spec harness commits.** `sweep2/t1_tasl/g5_tasl_spec.json` `harness_families.G5-7.g5_commit` and
   `G5-AR.g5_commit` (now `PENDING`) should name the final tips of `mac-trackg-g5-7-20261124` and
   `mac-trackg-g5-ar-20261124` once this unit is reviewed, instead of `52e3c0b0`/`00829da1`. The checkouts at
   `harness_g5` and `harness_g5_ar` must be at those tips.
2. **`g5_job.check_engines` needles.** No change is needed: its four needles (`g5_tasl_engine_contract/1`, the two
   input schemas, `class TruncationInvalidManager(`) are unchanged in `audit_g5.py`. Optionally, a fifth check could
   require the recount: `trackg_diff_recount/v1` in `campaign_v2_diff.py`, or `DIFF_RECOUNT_RULE` in `audit_g5.py`.
3. **`g5_tasl.measurement_path_check`.** No change. It passes, and `G5_TRACKG_ADDITIONS` is unchanged.
4. **G.4b spec and launcher (`sweep2/t1_tasl/g4_tasl.py`).** A G.4b re-pilot on these pins needs:
   - a new spec with a new `run_id`, W = 20 and `harness.commit` = the final G5-7 tip;
   - a `harness_dir` checked out at that tip (G.4's `harness` is at `c025f89a`).

   Today `g4_tasl.validate_spec` (line 152) STOPs unless `harness.commit` equals the G.1b spec's (`c025f89a`), "one
   harness pin for the T1 table". `launch` checks the TASL-2 host record's `host_check_harness_commit` against
   `harness.commit` (`G.check_host_record`, line 863; the G.1b record names `c025f89a`). Both need a reviewed
   paper-side rule. For example: allow a commit that passes `measurement_path_check` against the G.1b pin, and check
   the host record against the G.1b pin, as G.5's families do (`g1b_pin` beside `g5_commit`). The job record and
   receipt checks (`harness_commit == spec`) follow the spec.
5. **Paper code that counts `malformed_hunk`.** `git grep` on paper `origin/main` finds no Python or JSON file that
   names `malformed_hunk` or `malformed_diff`. The reasons reach the paper only through
   `pilot_g4.summarise().refusal_reasons` (a free dictionary) and the reports. The G.4b report should give both
   counts. Optionally the diagnostics could count `diff_recount.changed` per node (a paper-side choice).
6. **Disclosure.** The G.4 pilot's strict-rule liveness stays disclosed (the PLAN row says so). G.5 and G.4b ledgers
   carry `diff_recount` per proposal, and both engine records carry `diff_rule`.

## 10. TASL-2 smoke items (before G.4b or G.5 use these tips)

1. `git -C <harness_g5> rev-parse HEAD` is the final G5-7 tip and `status --porcelain` is empty; the same for
   `harness_g5_ar` and the G5-AR tip.
2. `g5_pin_check`, `g5_job.check_engines` and `g5_tasl.measurement_path_check` pass from the paper checkout on the
   host, with the host's harness git dir.
3. In torch312 and jax312 on TASL-2 (Python 3.12.16), from the harness checkout:
   `PYTHONPATH=src python -B -m pytest -p no:cacheprovider -q tests/trackg/test_trackg_diff_recount.py
   tests/trackg/test_trackg_g4_reclassify.py` passes (CPU only, about 3 s; no GPU, no network).
4. The re-classification command in §8 exits 0 with the expected smoke values (155 replay matches, 0 unverified
   incumbents, 25 of 25 accepted texts unchanged). Its JSON is kept with the G.4b records.
5. After the first G.4b node: every proposal line of its ledger has `diff_recount` with `rule:
   trackg_diff_recount/v1`, and `proposal.diff` differs from `runs[0].diff.diff_text` only where
   `diff_recount.changed` is true. Where `changed` is true, `runs[0].diff.diff_sha256` equals
   `diff_recount.applied_diff_sha256`.

## 11. Open points for the reviewer

- **Header-first** (point 1) departs from "the header's counts are ignored", by design: it keeps every
  strict-accepted diff byte-identical, including the `-- `/`++ ` body case and trailing blank context. A pure
  git-style recount would change some diffs the strict parser accepted.
- **Trailing blank lines** before a file header or the end are separators here, but context in git. Choosing git's
  convention would refuse more diffs (a context line that must match a blank line), and never apply one wrongly.
- **The residual risk in point 5** (a body pair followed by `@@`, in a patch without `diff --git`) is documented, not
  eliminated.
- **`diff_sha256`** in run records is the applied copy's hash when headers were rewritten (§5).
- **The pre-pass imports three private names** from `diff_support` (frozen by P5).
- **The paper worktree** used for the checks was added with `git worktree add --detach` from `/tmp/pb/repo`, which
  writes worktree metadata under that repository's `.git`. It was not removed.
